"""Gymnasium environment: Unitree A1 velocity tracking in MuJoCo.

The policy runs at 50 Hz. Each action is a residual around the standing pose and
is tracked by the position actuators already defined in the Menagerie MJCF.
Physics stays on the CPU. Domain randomization covers trunk/link mass, ground
friction, joint damping, sensor noise, and action latency.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import yaml

from src.envs.reward_functions import DEFAULT_WEIGHTS, compute_total_reward

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENE = ROOT / "assets" / "unitree_a1" / "scene.xml"
DEFAULT_ROBOT_CONFIG = ROOT / "configs" / "robot" / "quadruped.yaml"

N_JOINTS = 12
OBS_DIM = 48


def _load_yaml(source: str | Path | dict | None) -> dict:
    if source is None:
        return {}
    if isinstance(source, dict):
        return source
    path = Path(source)
    with path.open() as handle:
        data = yaml.safe_load(handle) or {}
    return data


def _quat_rotate_inverse(quat_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate ``vector`` from world frame into the frame of ``quat_wxyz``."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    qv = np.array([-x, -y, -z], dtype=np.float64)
    t = 2.0 * np.cross(qv, vector)
    return vector + w * t + np.cross(qv, t)


class QuadrupedEnv(gym.Env):
    """12-DoF A1 on a flat floor. Command is body-frame ``(vx, vy, yaw_rate)``."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        robot_config: str | Path | dict | None = None,
        dr_config: str | Path | dict | None = None,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.render_mode = render_mode
        robot = _load_yaml(robot_config if robot_config is not None else DEFAULT_ROBOT_CONFIG)
        self._dr = _load_yaml(dr_config)

        scene = Path(robot.get("scene", DEFAULT_SCENE))
        if not scene.is_absolute():
            scene = ROOT / scene
        if not scene.is_file():
            raise FileNotFoundError(f"MuJoCo scene not found: {scene}")

        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self._home_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if self._home_id < 0:
            raise RuntimeError("The A1 model has no 'home' keyframe.")

        self._home_qpos = self.model.key_qpos[self._home_id].copy()
        self._home_ctrl = self.model.key_ctrl[self._home_id].copy()
        self._joint_qpos = self._home_qpos[7:].copy()
        self._ctrl_low = self.model.actuator_ctrlrange[:, 0].copy()
        self._ctrl_high = self.model.actuator_ctrlrange[:, 1].copy()

        self.control_dt = float(robot.get("control_dt", 0.02))
        sim_dt = float(self.model.opt.timestep)
        self._decimation = max(1, int(round(self.control_dt / sim_dt)))
        self.action_scale = float(robot.get("action_scale", 0.25))
        self.episode_steps = int(robot.get("episode_steps", 1000))
        self._reset_joint_noise = float(robot.get("reset_joint_noise", 0.02))
        self._fall_height = float(robot.get("fall_height", 0.15))
        self._fall_gravity_z = float(robot.get("fall_gravity_z", -0.4))

        command = robot.get("command", {})
        self._vx_range = tuple(command.get("vx", [0.3, 0.8]))
        self._vy_range = tuple(command.get("vy", [0.0, 0.0]))
        self._yaw_range = tuple(command.get("yaw", [0.0, 0.0]))
        self._reward_weights = dict(DEFAULT_WEIGHTS)
        self._reward_weights.update(robot.get("rewards", {}))

        self._trunk_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self._floor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self._foot_ids = np.flatnonzero(
            (self.model.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE) & (self.model.geom_size[:, 0] > 0.015)
        )
        self._mass0 = self.model.body_mass.copy()
        self._inertia0 = self.model.body_inertia.copy()
        self._friction0 = self.model.geom_friction.copy()
        self._damping0 = self.model.dof_damping.copy()

        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(N_JOINTS,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-10.0, high=10.0, shape=(OBS_DIM,), dtype=np.float32)

        self._command = np.zeros(3, dtype=np.float64)
        self._last_action = np.zeros(N_JOINTS, dtype=np.float64)
        self._step_count = 0
        self._delay = 0
        self._action_history: deque[np.ndarray] = deque()
        self._renderer: mujoco.Renderer | None = None
        self._reset_action_history()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._restore_nominal_physics()
        self._apply_domain_randomization()
        mujoco.mj_resetDataKeyframe(self.model, self.data, self._home_id)
        noise = self.np_random.uniform(-self._reset_joint_noise, self._reset_joint_noise, size=N_JOINTS)
        self.data.qpos[7:] = self._joint_qpos + noise
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self._home_ctrl
        mujoco.mj_forward(self.model, self.data)
        for _ in range(self._decimation):
            self.data.ctrl[:] = self._home_ctrl
            mujoco.mj_step(self.model, self.data)

        self._command = np.array(
            [
                self.np_random.uniform(*self._vx_range),
                self.np_random.uniform(*self._vy_range),
                self.np_random.uniform(*self._yaw_range),
            ],
            dtype=np.float64,
        )
        self._last_action[:] = 0.0
        self._step_count = 0
        self._sample_latency()
        self._reset_action_history()
        obs = self._observation()
        return obs, self._info(fallen=False, terms={})

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._action_history.append(action.copy())
        applied = self._action_history.popleft()
        target = np.clip(self._home_ctrl + applied * self.action_scale, self._ctrl_low, self._ctrl_high)
        for _ in range(self._decimation):
            self.data.ctrl[:] = target
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1
        fallen = self._is_fallen()
        lin_vel, ang_vel, gravity = self._body_signals()
        reward, terms = compute_total_reward(
            lin_vel=lin_vel,
            ang_vel=ang_vel,
            command=self._command,
            projected_gravity=gravity,
            torques=self.data.actuator_force,
            joint_vel=self.data.qvel[6:],
            action=applied,
            last_action=self._last_action,
            fallen=fallen,
            weights=self._reward_weights,
        )
        self._last_action[:] = applied
        terminated = bool(fallen)
        truncated = self._step_count >= self.episode_steps
        obs = self._observation()
        return obs, reward, terminated, truncated, self._info(fallen, terms)

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data)
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _observation(self) -> np.ndarray:
        lin_vel, ang_vel, gravity = self._body_signals()
        obs = np.concatenate(
            [
                self.data.qpos[7:] - self._joint_qpos,
                self.data.qvel[6:],
                lin_vel,
                ang_vel,
                gravity,
                self._command,
                self._last_action,
            ]
        ).astype(np.float32)
        obs = self._add_sensor_noise(obs)
        return np.clip(obs, -10.0, 10.0)

    def _body_signals(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        quat = self.data.qpos[3:7]
        lin_vel = _quat_rotate_inverse(quat, self.data.qvel[0:3])
        ang_vel = _quat_rotate_inverse(quat, self.data.qvel[3:6])
        gravity = _quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        return lin_vel, ang_vel, gravity

    def _is_fallen(self) -> bool:
        height = float(self.data.qpos[2])
        _, _, gravity = self._body_signals()
        return height < self._fall_height or float(gravity[2]) > self._fall_gravity_z

    def _info(self, fallen: bool, terms: dict) -> dict:
        lin_vel, ang_vel, gravity = self._body_signals()
        return {
            "linear_vel": lin_vel.astype(np.float32),
            "angular_vel": ang_vel.astype(np.float32),
            "projected_gravity": gravity.astype(np.float32),
            "torques": self.data.actuator_force.astype(np.float32).copy(),
            "joint_vel": self.data.qvel[6:].astype(np.float32).copy(),
            "command": self._command.astype(np.float32).copy(),
            "fallen": fallen,
            "reward_terms": terms,
            "trunk_mass": float(self.model.body_mass[self._trunk_id]),
        }

    def _restore_nominal_physics(self) -> None:
        self.model.body_mass[:] = self._mass0
        self.model.body_inertia[:] = self._inertia0
        self.model.geom_friction[:] = self._friction0
        self.model.dof_damping[:] = self._damping0

    def _apply_domain_randomization(self) -> None:
        mass_on, mass_cfg = self._dr_block("randomize_mass")
        if mass_on:
            low, high = mass_cfg.get("range", [-0.2, 0.2])
            scale = 1.0 + float(self.np_random.uniform(low, high))
            moving = self._mass0 > 0.0
            self.model.body_mass[moving] = self._mass0[moving] * scale
            self.model.body_inertia[moving] = self._inertia0[moving] * scale

        friction_on, friction_cfg = self._dr_block("randomize_friction")
        if friction_on:
            low, high = friction_cfg.get("range", [0.5, 1.5])
            scale = float(self.np_random.uniform(low, high))
            ids = [self._floor_id, *self._foot_ids.tolist()]
            for geom_id in ids:
                if geom_id < 0:
                    continue
                self.model.geom_friction[geom_id, 0] = self._friction0[geom_id, 0] * scale

        damping_on, damping_cfg = self._dr_block("randomize_damping")
        if damping_on:
            low, high = damping_cfg.get("range", [0.75, 1.25])
            scale = float(self.np_random.uniform(low, high))
            self.model.dof_damping[:] = self._damping0 * scale

        mujoco.mj_setConst(self.model, self.data)

    def _sample_latency(self) -> None:
        enabled, cfg = self._dr_block("action_latency")
        if not enabled:
            self._delay = 0
            return
        low, high = cfg.get("range_ms", [0, 0])
        latency_s = float(self.np_random.uniform(low, high)) / 1000.0
        self._delay = int(round(latency_s / self.control_dt))

    def _reset_action_history(self) -> None:
        # ``delay`` zeros sit ahead of the first real action, so delay 0 applies it now.
        self._action_history = deque(np.zeros(N_JOINTS, dtype=np.float64) for _ in range(self._delay))

    def _add_sensor_noise(self, obs: np.ndarray) -> np.ndarray:
        enabled, cfg = self._dr_block("sensor_noise")
        if not enabled:
            return obs
        noisy = obs.copy()
        joint_pos_std = float(cfg.get("joint_pos_std", 0.0))
        joint_vel_std = float(cfg.get("joint_vel_std", 0.0))
        imu_std = float(cfg.get("imu_noise_std", 0.0))
        if joint_pos_std:
            noisy[0:12] += self.np_random.normal(0.0, joint_pos_std, size=12).astype(np.float32)
        if joint_vel_std:
            noisy[12:24] += self.np_random.normal(0.0, joint_vel_std, size=12).astype(np.float32)
        if imu_std:
            noisy[27:33] += self.np_random.normal(0.0, imu_std, size=6).astype(np.float32)
        return noisy

    def _dr_block(self, key: str) -> tuple[bool, dict]:
        block = self._dr.get(key) or {}
        return bool(block.get("enabled", False)), block
