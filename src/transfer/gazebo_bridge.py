"""Run a MuJoCo actor in Gazebo Sim and compare the two simulators.

Gazebo is reached through gz-transport: joint state and odometry in, position
targets out. When ``rclpy`` imports, the same process is a ROS 2 node that
mirrors joint state and odometry. The actor is the ONNX file when onnxruntime
is installed, and otherwise the same deterministic PyTorch module the exporter
writes into that file.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

from src.evaluation.metrics import forward_velocity_error, mean_forward_velocity, survival_time
from src.training.train import ROOT
from src.transfer.a1_description import (
    CONTROL_DT,
    HOME,
    JOINTS,
    build_observation,
    gazebo_command,
    gazebo_sdf,
    projected_gravity,
)
from src.utils.torch_runtime import assert_cpu_pytorch

GZ_PYTHON = [
    "/opt/ros/lyrical/opt/gz_msgs_vendor/lib/python",
    "/opt/ros/lyrical/opt/gz_transport_vendor/lib/python",
]
SCENE_PATH = ROOT / "assets" / "unitree_a1" / "a1_gazebo.sdf"
JOINT_TOPIC = "/world/a1_flat/model/a1/joint_state"
ODOM_TOPIC = "/model/a1/odometry"
FALL_HEIGHT = 0.15
FALL_GRAVITY_Z = -0.4


def _import_gz():
    for path in GZ_PYTHON:
        if path not in sys.path:
            sys.path.insert(0, path)
    from gz.msgs.boolean_pb2 import Boolean
    from gz.msgs.double_pb2 import Double
    from gz.msgs.model_pb2 import Model
    from gz.msgs.odometry_pb2 import Odometry
    from gz.msgs.world_control_pb2 import WorldControl
    from gz.transport import Node

    return Boolean, Double, Model, Odometry, WorldControl, Node


class Actor:
    """Deterministic action from a raw 48-vector. Normalization is inside the graph."""

    def __init__(self, model_path: Path):
        candidate = model_path.with_suffix(".onnx") if model_path.suffix == ".zip" else model_path
        self.path = candidate
        try:
            import onnxruntime as ort

            if not candidate.is_file():
                raise FileNotFoundError(candidate)
            self.session = ort.InferenceSession(str(candidate), providers=["CPUExecutionProvider"])
            self.module = None
            print(f"Actor: ONNX {candidate.name}")
            return
        except Exception as exc:
            print(f"ONNX runtime unavailable ({exc}). Using the PyTorch actor graph.")
        from src.transfer.export_policy import _ActorMean, _normalization
        from stable_baselines3 import PPO
        import torch

        source = model_path if model_path.suffix == ".zip" else model_path.with_suffix(".zip")
        policy = PPO.load(str(source), device="cpu")
        mean, var = _normalization(str(source))
        module = _ActorMean(policy.policy, mean, var)
        module.eval()
        self.session = None
        self.module = module
        self.torch = torch
        print(f"Actor: PyTorch graph of {source.name}")

    def __call__(self, observation: np.ndarray) -> np.ndarray:
        row = np.asarray(observation, dtype=np.float32).reshape(1, -1)
        if self.session is not None:
            return np.asarray(self.session.run(None, {"observation": row})[0]).reshape(-1)
        with self.torch.no_grad():
            action = self.module(self.torch.from_numpy(row))
        return action.detach().cpu().numpy().reshape(-1)


class GazeboA1:
    def __init__(self):
        Boolean, Double, Model, Odometry, WorldControl, Node = _import_gz()
        self.Boolean = Boolean
        self.Double = Double
        self.Model = Model
        self.Odometry = Odometry
        self.WorldControl = WorldControl
        self.node = Node()
        self.joint_pos: dict[str, float] = {}
        self.joint_vel: dict[str, float] = {}
        self.sim_time = 0.0
        self.position = np.zeros(3)
        self.quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])
        # Twist from OdometryPublisher is already in the trunk frame when dimensions is 3.
        self.twist_linear = np.zeros(3)
        self.twist_angular = np.zeros(3)
        self.pubs = {}
        self._seen_joint = False
        self._seen_odom = False

    def connect(self, timeout: float = 20.0) -> None:
        self.node.subscribe(self.Model, JOINT_TOPIC, self._on_joint)
        self.node.subscribe(self.Odometry, ODOM_TOPIC, self._on_odom)
        for name in JOINTS:
            self.pubs[name] = self.node.advertise(f"/a1/{name}/cmd", self.Double)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._seen_joint and self._seen_odom and len(self.joint_pos) >= len(JOINTS):
                return
            time.sleep(0.05)
        raise TimeoutError(
            f"Gazebo topics did not arrive. joints={sorted(self.joint_pos)} odom={self._seen_odom}"
        )

    def _on_joint(self, msg) -> None:
        for joint in msg.joint:
            if not joint.name:
                continue
            self.joint_pos[joint.name] = float(joint.axis1.position)
            self.joint_vel[joint.name] = float(joint.axis1.velocity)
        self._seen_joint = True
        self.sim_time = float(msg.header.stamp.sec) + float(msg.header.stamp.nsec) * 1e-9

    def _on_odom(self, msg) -> None:
        pos = msg.pose.position
        quat = msg.pose.orientation
        self.position = np.array([pos.x, pos.y, pos.z], dtype=np.float64)
        self.quat_wxyz = np.array([quat.w, quat.x, quat.y, quat.z], dtype=np.float64)
        self.twist_linear = np.array([msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z])
        self.twist_angular = np.array([msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z])
        self._seen_odom = True

    def state(self) -> dict:
        # Gazebo joint zero is the standing pose. The policy uses absolute angles.
        q = np.array([self.joint_pos[name] for name in JOINTS]) + HOME
        qd = np.array([self.joint_vel.get(name, 0.0) for name in JOINTS])
        linear = self.twist_linear.copy()
        angular = self.twist_angular.copy()
        return {
            "q": q,
            "qd": qd,
            "linear": linear,
            "angular": angular,
            "gravity": projected_gravity(self.quat_wxyz),
            "height": float(self.position[2]),
        }

    def send(self, targets: np.ndarray) -> None:
        for name, value in zip(JOINTS, targets):
            msg = self.Double()
            msg.data = float(value)
            self.pubs[name].publish(msg)

    def publish_ros(self) -> None:
        mirror = getattr(self, "ros", None)
        if mirror is not None:
            mirror.publish(self)

    def reset_world(self) -> None:
        request = self.WorldControl()
        request.reset.all = True
        self.node.request("/world/a1_flat/control", request, self.WorldControl, self.Boolean, 2000)
        time.sleep(0.3)
        self.send(np.zeros(len(JOINTS)))


def start_gazebo(log_path: Path, with_camera: bool = False) -> subprocess.Popen:
    SCENE_PATH.write_text(gazebo_sdf(with_camera=with_camera))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w")
    command = f"source /opt/ros/lyrical/setup.bash && gz sim -s -r -v 2 {SCENE_PATH}"
    return subprocess.Popen(["bash", "-lc", command], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)


def stop_gazebo(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)


class RosMirror:
    """ROS 2 publishers for the joint state and the trunk odometry.

    Joint positions are the Gazebo angles, whose zero is the standing pose.
    The mirror stays off when rclpy cannot be imported, so a measurement run
    without a sourced ROS workspace still controls the robot.
    """

    def __init__(self) -> None:
        self.enabled = False
        self.node = None
        self.rclpy = None
        try:
            import sys

            # python3-empy and python3-lark are Debian packages. The venv does not see them.
            dist_packages = "/usr/lib/python3/dist-packages"
            if dist_packages not in sys.path:
                sys.path.append(dist_packages)
            import rclpy
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import JointState

            if not rclpy.ok():
                rclpy.init(args=None)
            self.rclpy = rclpy
            self.JointState = JointState
            self.Odometry = Odometry
            self.node = rclpy.create_node("a1_policy_bridge")
            self.joint_pub = self.node.create_publisher(JointState, "/a1/joint_states", 10)
            self.odom_pub = self.node.create_publisher(Odometry, "/a1/odom", 10)
            self.enabled = True
            print("ROS 2 node a1_policy_bridge publishing /a1/joint_states and /a1/odom", flush=True)
        except Exception as exc:
            print(f"ROS 2 mirror off ({exc}).", flush=True)

    def publish(self, robot: GazeboA1) -> None:
        if not self.enabled or self.node is None:
            return
        stamp = self.node.get_clock().now().to_msg()
        joint = self.JointState()
        joint.header.stamp = stamp
        joint.name = list(JOINTS)
        joint.position = [float(robot.joint_pos.get(name, 0.0)) for name in JOINTS]
        joint.velocity = [float(robot.joint_vel.get(name, 0.0)) for name in JOINTS]
        self.joint_pub.publish(joint)
        odom = self.Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = "world"
        odom.child_frame_id = "trunk"
        odom.pose.pose.position.x = float(robot.position[0])
        odom.pose.pose.position.y = float(robot.position[1])
        odom.pose.pose.position.z = float(robot.position[2])
        odom.pose.pose.orientation.w = float(robot.quat_wxyz[0])
        odom.pose.pose.orientation.x = float(robot.quat_wxyz[1])
        odom.pose.pose.orientation.y = float(robot.quat_wxyz[2])
        odom.pose.pose.orientation.z = float(robot.quat_wxyz[3])
        odom.twist.twist.linear.x = float(robot.twist_linear[0])
        odom.twist.twist.linear.y = float(robot.twist_linear[1])
        odom.twist.twist.linear.z = float(robot.twist_linear[2])
        odom.twist.twist.angular.x = float(robot.twist_angular[0])
        odom.twist.twist.angular.y = float(robot.twist_angular[1])
        odom.twist.twist.angular.z = float(robot.twist_angular[2])
        self.odom_pub.publish(odom)

    def close(self) -> None:
        if self.node is not None and self.rclpy is not None:
            self.node.destroy_node()
            self.node = None


def fallen(state: dict) -> bool:
    return state["height"] < FALL_HEIGHT or float(state["gravity"][2]) > FALL_GRAVITY_Z


def _wall_deadline(seconds: float) -> float:
    # gz sim often runs below the requested real-time factor on this CPU.
    return time.monotonic() + seconds * 3.0 + 10.0


def hold_pose(robot: GazeboA1, seconds: float) -> dict:
    robot.send(np.zeros(len(JOINTS)))
    start = robot.sim_time
    last = start
    wall_deadline = _wall_deadline(seconds)
    while robot.sim_time < start + seconds and time.monotonic() < wall_deadline:
        if robot.sim_time - last >= CONTROL_DT:
            robot.send(np.zeros(len(JOINTS)))
            robot.publish_ros()
            last = robot.sim_time
        time.sleep(0.002)
    state = robot.state()
    state["held_s"] = robot.sim_time - start
    return state


def run_gazebo_episode(robot: GazeboA1, actor: Actor, vx: float, delay_steps: int, seconds: float) -> dict:
    robot.reset_world()
    robot.send(np.zeros(len(JOINTS)))
    time.sleep(0.2)
    command = np.array([vx, 0.0, 0.0])
    history: deque[np.ndarray] = deque(np.zeros(len(JOINTS)) for _ in range(delay_steps))
    last_action = np.zeros(len(JOINTS))
    vx_hist: list[float] = []
    cmd_hist: list[float] = []
    start = robot.sim_time
    last = start - CONTROL_DT
    info_fallen = False
    wall_deadline = _wall_deadline(seconds)
    while robot.sim_time < start + seconds and time.monotonic() < wall_deadline:
        if robot.sim_time - last < CONTROL_DT:
            time.sleep(0.001)
            continue
        last = robot.sim_time
        state = robot.state()
        info_fallen = fallen(state)
        observation = build_observation(
            state["q"], state["qd"], state["linear"], state["angular"], state["gravity"], command, last_action
        )
        action = np.clip(actor(observation), -1.0, 1.0)
        if delay_steps:
            history.append(action.copy())
            applied = history.popleft()
        else:
            applied = action
        last_action = applied
        robot.send(gazebo_command(applied))
        robot.publish_ros()
        vx_hist.append(float(state["linear"][0]))
        cmd_hist.append(vx)
        if info_fallen:
            break
    elapsed = float(robot.sim_time - start)
    summary = _summary("gazebo", vx, delay_steps, seconds, vx_hist, cmd_hist, info_fallen)
    # The sim clock is the horizon. Counting control samples under-reports it when
    # a physics step skips past a 50 Hz boundary.
    summary["survival_s"] = elapsed
    summary["success"] = (not info_fallen) and summary["vx_error"] < 0.3 and elapsed >= seconds - 0.05
    return summary


def run_mujoco_episode(actor: Actor, vx: float, delay_steps: int, seconds: float, seed: int) -> dict:
    from src.envs.quadruped_env import QuadrupedEnv

    dr = None
    if delay_steps:
        ms = delay_steps * CONTROL_DT * 1000.0
        dr = {"action_latency": {"enabled": True, "range_ms": [ms, ms]}}
    env = QuadrupedEnv(dr_config=dr)
    try:
        env.reset(seed=seed)
        env._command[:] = np.array([vx, 0.0, 0.0])
        obs = env._observation()
        steps = int(seconds / env.control_dt)
        vx_hist: list[float] = []
        cmd_hist: list[float] = []
        done_fallen = False
        for _ in range(steps):
            action = np.clip(actor(obs), -1.0, 1.0)
            obs, _reward, terminated, truncated, info = env.step(action)
            vx_hist.append(float(info["linear_vel"][0]))
            cmd_hist.append(vx)
            done_fallen = bool(info["fallen"])
            if terminated or truncated:
                break
    finally:
        env.close()
    return _summary("mujoco", vx, delay_steps, seconds, vx_hist, cmd_hist, done_fallen)


def _summary(
    simulator: str,
    vx: float,
    delay_steps: int,
    seconds: float,
    vx_hist: list[float],
    cmd_hist: list[float],
    did_fall: bool,
) -> dict:
    error = forward_velocity_error(vx_hist, cmd_hist)
    survival = survival_time(CONTROL_DT, len(vx_hist))
    second_half = vx_hist[len(vx_hist) // 2 :]
    return {
        "simulator": simulator,
        "command_vx": vx,
        "delay_steps": delay_steps,
        "survival_s": survival,
        "vx": mean_forward_velocity(vx_hist),
        "vx_second_half": mean_forward_velocity(second_half),
        "vx_error": error,
        "fallen": did_fall,
        "success": (not did_fall) and error < 0.3 and survival >= seconds - CONTROL_DT,
    }


def compare(commands: list[float], seconds: float, seed: int) -> list[dict]:
    assert_cpu_pytorch()
    policies = [
        ("walk_baseline", ROOT / "models" / "walk_baseline.zip", 0),
        ("walk_dr", ROOT / "models" / "walk_dr.zip", 1),
    ]
    rows: list[dict] = []
    proc = None
    try:
        proc = start_gazebo(ROOT / "results" / "gazebo_sim.log")
        robot = GazeboA1()
        robot.connect()
        robot.ros = RosMirror()
        held = hold_pose(robot, 2.0)
        print(
            f"Hold: height {held['height']:.3f} m, joints max |q-home| "
            f"{np.max(np.abs(held['q'] - HOME)):.3f} rad, fallen={fallen(held)}",
            flush=True,
        )
        for name, path, delay in policies:
            actor = Actor(path)
            for vx in commands:
                gazebo_row = run_gazebo_episode(robot, actor, vx, delay, seconds)
                mujoco_row = run_mujoco_episode(actor, vx, delay, seconds, seed)
                gazebo_row["policy"] = name
                mujoco_row["policy"] = name
                rows.append(gazebo_row)
                rows.append(mujoco_row)
                print(
                    f"{name} delay {delay} cmd {vx:.2f}: "
                    f"gazebo {gazebo_row['survival_s']:.1f}s vx {gazebo_row['vx']:.2f} "
                    f"late {gazebo_row['vx_second_half']:.2f} err {gazebo_row['vx_error']:.3f} "
                    f"fall {gazebo_row['fallen']} "
                    f"mujoco {mujoco_row['survival_s']:.1f}s vx {mujoco_row['vx']:.2f} "
                    f"err {mujoco_row['vx_error']:.3f} fall {mujoco_row['fallen']}",
                    flush=True,
                )
    finally:
        ros = getattr(locals().get("robot", None), "ros", None)
        if ros is not None:
            ros.close()
        stop_gazebo(proc)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Zero-shot MuJoCo actor in Gazebo.")
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--commands", type=float, nargs="*", default=[0.4, 0.55, 0.7])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=str, default="results/gazebo_transfer.json")
    args = parser.parse_args()
    rows = compare(args.commands, args.seconds, args.seed)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
