"""Unitree A1 kinematics shared by MuJoCo and the Gazebo scene.

Joint order matches the MuJoCo actuators: FR, FL, RR, RL, each hip, thigh, calf.
The standing pose is the Menagerie ``home`` keyframe. Gazebo uses primitive
collisions with the same joint axes, limits, and link masses. It is not a
copy of the MJCF contact model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MESH_DIR = ROOT / "assets" / "unitree_a1" / "assets"

JOINTS: tuple[str, ...] = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)

# hip, thigh, calf, repeated for the four legs.
HOME = np.array([0.0, 0.9, -1.8] * 4, dtype=np.float64)
LOWER = np.array([-0.802851, -1.0472, -2.69653] * 4, dtype=np.float64)
UPPER = np.array([0.802851, 4.18879, -0.916298] * 4, dtype=np.float64)
DAMPING = np.array([1.0, 2.0, 2.0] * 4, dtype=np.float64)

ACTION_SCALE = 0.25
CONTROL_DT = 0.02
P_GAIN = 100.0
# MuJoCo position actuators have kp and no kv. Joint damping stays on the joint.
ACTUATOR_D_GAIN = 0.0
FORCE_LIMIT = 33.5
# World offset from the trunk origin for the recording camera. Roll stays 0.
DEMO_CAMERA_OFFSET = (-1.15, -0.85, 0.42)
DEMO_CAMERA_PITCH = 0.28
DEMO_CAMERA_YAW = 0.637
TRUNK_HEIGHT = 0.27

# Center of mass in the link frame, from the Menagerie inertial positions.
INERTIAL_COM: dict[str, tuple[float, float, float]] = {
    "trunk": (0.0, 0.0041, -0.0005),
    "FR_hip": (-0.003311, -0.000635, 0.000031),
    "FL_hip": (-0.003311, 0.000635, 0.000031),
    "RR_hip": (0.003311, -0.000635, 0.000031),
    "RL_hip": (0.003311, 0.000635, 0.000031),
    "FR_thigh": (-0.003237, 0.022327, -0.027326),
    "FL_thigh": (-0.003237, -0.022327, -0.027326),
    "RR_thigh": (-0.003237, 0.022327, -0.027326),
    "RL_thigh": (-0.003237, -0.022327, -0.027326),
    "FR_calf": (0.004727, 0.0, -0.131975),
    "FL_calf": (0.004727, 0.0, -0.131975),
    "RR_calf": (0.004727, 0.0, -0.131975),
    "RL_calf": (0.004727, 0.0, -0.131975),
}

LEGS: tuple[tuple[str, tuple[float, float, float], tuple[float, float, float]], ...] = (
    ("FR", (0.183, -0.047, 0.0), (0.0, -0.08505, 0.0)),
    ("FL", (0.183, 0.047, 0.0), (0.0, 0.08505, 0.0)),
    ("RR", (-0.183, -0.047, 0.0), (0.0, -0.08505, 0.0)),
    ("RL", (-0.183, 0.047, 0.0), (0.0, 0.08505, 0.0)),
)


def position_targets(action: np.ndarray) -> np.ndarray:
    """Map a policy action in [-1, 1] to actuator positions, in radians."""
    residual = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
    return np.clip(HOME + residual * ACTION_SCALE, LOWER, UPPER)


def build_observation(
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    linear_vel: np.ndarray,
    angular_vel: np.ndarray,
    projected_gravity: np.ndarray,
    command: np.ndarray,
    last_action: np.ndarray,
) -> np.ndarray:
    obs = np.concatenate(
        [
            np.asarray(joint_pos, dtype=np.float64) - HOME,
            np.asarray(joint_vel, dtype=np.float64),
            np.asarray(linear_vel, dtype=np.float64),
            np.asarray(angular_vel, dtype=np.float64),
            np.asarray(projected_gravity, dtype=np.float64),
            np.asarray(command, dtype=np.float64),
            np.asarray(last_action, dtype=np.float64),
        ]
    )
    return np.clip(obs, -10.0, 10.0).astype(np.float32)


def projected_gravity(quat_wxyz: np.ndarray) -> np.ndarray:
    """Gravity direction in the trunk frame. Upright is about ``[0, 0, -1]``."""
    from src.envs.quadruped_env import _quat_rotate_inverse

    return _quat_rotate_inverse(np.asarray(quat_wxyz, dtype=np.float64), np.array([0.0, 0.0, -1.0]))


def world_to_body(quat_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    from src.envs.quadruped_env import _quat_rotate_inverse

    return _quat_rotate_inverse(np.asarray(quat_wxyz, dtype=np.float64), np.asarray(vector, dtype=np.float64))


def ry(pitch: float, vector: tuple[float, float, float]) -> tuple[float, float, float]:
    c = float(np.cos(pitch))
    s = float(np.sin(pitch))
    x, y, z = vector
    return (c * x + s * z, y, -s * x + c * z)


def standing_layout() -> dict[str, tuple[tuple[float, float, float], float]]:
    """Link origin and pitch in the trunk frame, at the standing pose.

    Gazebo joint zero is this pose. Policy angles are absolute, so the bridge
    commands ``target - home``.
    """
    layout: dict[str, tuple[tuple[float, float, float], float]] = {"trunk": ((0.0, 0.0, 0.0), 0.0)}
    for leg, hip_pos, thigh_offset in LEGS:
        hip_origin = hip_pos
        thigh_origin = (
            hip_origin[0] + thigh_offset[0],
            hip_origin[1] + thigh_offset[1],
            hip_origin[2] + thigh_offset[2],
        )
        calf_origin = tuple(np.add(thigh_origin, ry(0.9, (0.0, 0.0, -0.2))))
        layout[f"{leg}_hip"] = (hip_origin, 0.0)
        layout[f"{leg}_thigh"] = (thigh_origin, 0.9)
        layout[f"{leg}_calf"] = (calf_origin, -0.9)
    return layout


def foot_height() -> float:
    """World z of a front foot center when the trunk origin is at ``TRUNK_HEIGHT``."""
    calf, pitch = standing_layout()["FR_calf"]
    foot = np.add(calf, ry(pitch, (0.0, 0.0, -0.2)))
    return float(foot[2] + TRUNK_HEIGHT)


def gazebo_command(action: np.ndarray) -> np.ndarray:
    """Joint targets in the Gazebo frame, where zero is the standing pose."""
    return position_targets(action) - HOME


def gazebo_sdf(with_camera: bool = False) -> str:
    """Flat-ground world. Link poses are the standing configuration.

    ``with_camera`` adds a world-fixed camera and a visual checker on the floor
    for a headless recording. The checker has no collision. The transfer
    measurement leaves both off so rendering does not slow the step.
    """
    parts = [
        '<?xml version="1.0"?>',
        '<sdf version="1.9">',
        '  <world name="a1_flat">',
        "    <physics name=\"physics\" type=\"ignored\">",
        "      <max_step_size>0.001</max_step_size>",
        "      <real_time_factor>1.0</real_time_factor>",
        "    </physics>",
        '    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"/>',
        '    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"/>',
        '    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"/>',
        *(
            [
                '    <plugin filename="gz-sim-sensors-system" name="gz::sim::systems::Sensors">',
                "      <render_engine>ogre2</render_engine>",
                "    </plugin>",
            ]
            if with_camera
            else []
        ),
        "    <light type=\"directional\" name=\"sun\">",
        "      <cast_shadows>false</cast_shadows>",
        "      <pose>0 0 10 0 0 0</pose>",
        "      <diffuse>0.8 0.8 0.8 1</diffuse>",
        "      <direction>-0.3 0.2 -0.9</direction>",
        "    </light>",
        '    <model name="ground">',
        "      <static>true</static>",
        '      <link name="link">',
        '        <collision name="collision">',
        "          <geometry><plane><normal>0 0 1</normal><size>20 20</size></plane></geometry>",
        "          <surface><friction><ode><mu>0.8</mu><mu2>0.8</mu2></ode></friction></surface>",
        "        </collision>",
        '        <visual name="visual">',
        "          <geometry><plane><normal>0 0 1</normal><size>20 20</size></plane></geometry>",
        "          <material><ambient>0.2 0.3 0.4 1</ambient><diffuse>0.2 0.3 0.4 1</diffuse></material>",
        "        </visual>",
        "      </link>",
        "    </model>",
        '    <model name="a1">',
        f"      <pose>0 0 {TRUNK_HEIGHT} 0 0 0</pose>",
        "      <self_collide>false</self_collide>",
    ]
    layout = standing_layout()
    parts.append(
        _link(
            "trunk",
            layout["trunk"][0],
            pitch=layout["trunk"][1],
            mass=4.713,
            inertia=(0.0158533, 0.0377999, 0.0456542),
            visual=_box(0.36, 0.14, 0.1),
            collision=_box(0.36, 0.14, 0.1),
            com=INERTIAL_COM["trunk"],
        )
    )
    for leg, _hip_pos, _thigh_offset in LEGS:
        parts.append(
            _link(
                f"{leg}_hip",
                layout[f"{leg}_hip"][0],
                pitch=layout[f"{leg}_hip"][1],
                mass=0.696,
                inertia=(0.00081, 0.00055, 0.00047),
                visual=_cylinder(0.04, 0.06),
                collision=_cylinder(0.04, 0.06),
                com=INERTIAL_COM[f"{leg}_hip"],
            )
        )
        parts.append(
            _link(
                f"{leg}_thigh",
                layout[f"{leg}_thigh"][0],
                pitch=layout[f"{leg}_thigh"][1],
                mass=1.013,
                inertia=(0.00556, 0.00514, 0.00134),
                visual=_box(0.04, 0.04, 0.2, center=(0.0, 0.0, -0.1)),
                collision=_box(0.04, 0.04, 0.2, center=(0.0, 0.0, -0.1)),
                com=INERTIAL_COM[f"{leg}_thigh"],
            )
        )
        parts.append(
            _link(
                f"{leg}_calf",
                layout[f"{leg}_calf"][0],
                pitch=layout[f"{leg}_calf"][1],
                mass=0.226,
                inertia=(0.00340, 0.00339, 0.00004),
                visual=_box(0.03, 0.03, 0.2, center=(0.0, 0.0, -0.1)),
                collision=_sphere(0.02, center=(0.0, 0.0, -0.2)),
                com=INERTIAL_COM[f"{leg}_calf"],
            )
        )
        parts.append(_joint(f"{leg}_hip_joint", "trunk", f"{leg}_hip", "1 0 0", 0, damping=1.0))
        parts.append(_joint(f"{leg}_thigh_joint", f"{leg}_hip", f"{leg}_thigh", "0 1 0", 1, damping=2.0))
        parts.append(_joint(f"{leg}_calf_joint", f"{leg}_thigh", f"{leg}_calf", "0 1 0", 2, damping=2.0))
    scenery = (_ground_grid() + _level_camera()) if with_camera else ""
    parts.append(
        f"""
      <plugin filename="gz-sim-joint-state-publisher-system" name="gz::sim::systems::JointStatePublisher"/>
      <plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher">
        <odom_frame>world</odom_frame>
        <robot_base_frame>trunk</robot_base_frame>
        <odom_topic>/model/a1/odometry</odom_topic>
        <dimensions>3</dimensions>
      </plugin>
    </model>
{scenery}
  </world>
</sdf>
"""
    )
    return "\n".join(parts) + "\n"


def _pose(origin: tuple[float, float, float], pitch: float = 0.0) -> str:
    return f"{origin[0]:.5f} {origin[1]:.5f} {origin[2]:.5f} 0 {pitch:.5f} 0"


def _box(sx: float, sy: float, sz: float, center: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> str:
    pose = f"<pose>{center[0]} {center[1]} {center[2]} 0 0 0</pose>" if center != (0.0, 0.0, 0.0) else ""
    return f"{pose}<geometry><box><size>{sx} {sy} {sz}</size></box></geometry>"


def _cylinder(radius: float, length: float) -> str:
    return f"<geometry><cylinder><radius>{radius}</radius><length>{length}</length></cylinder></geometry>"


def _sphere(radius: float, center: tuple[float, float, float]) -> str:
    return (
        f"<pose>{center[0]} {center[1]} {center[2]} 0 0 0</pose>"
        f"<geometry><sphere><radius>{radius}</radius></sphere></geometry>"
    )


def _link(
    name: str,
    origin: tuple[float, float, float],
    pitch: float,
    mass: float,
    inertia: tuple[float, float, float],
    visual: str,
    collision: str,
    com: tuple[float, float, float] = (0.0, 0.0, 0.0),
    extra: str = "",
) -> str:
    ixx, iyy, izz = inertia
    return f"""
      <link name="{name}">
        <pose>{_pose(origin, pitch)}</pose>
        <inertial>
          <pose>{com[0]:.5f} {com[1]:.5f} {com[2]:.5f} 0 0 0</pose>
          <mass>{mass}</mass>
          <inertia><ixx>{ixx}</ixx><ixy>0</ixy><ixz>0</ixz><iyy>{iyy}</iyy><iyz>0</iyz><izz>{izz}</izz></inertia>
        </inertial>
        <visual name="visual">{visual}<material><ambient>0.25 0.25 0.25 1</ambient><diffuse>0.25 0.25 0.25 1</diffuse></material></visual>
        <collision name="collision">{collision}<surface><friction><ode><mu>0.8</mu><mu2>0.8</mu2></ode></friction></surface></collision>
        {extra}
      </link>"""


def _ground_grid() -> str:
    """Visual checker only. No collision, so the contact plane stays the measurement floor."""
    tiles: list[str] = []
    size = 0.5
    for ix in range(-6, 18):
        for iy in range(-8, 6):
            x = (ix + 0.5) * size
            y = (iy + 0.5) * size
            color = "0.55 0.62 0.68 1" if (ix + iy) % 2 == 0 else "0.86 0.87 0.88 1"
            tiles.append(
                f'<visual name="t{ix}_{iy}">'
                f"<pose>{x:.2f} {y:.2f} 0.001 0 0 0</pose>"
                f"<geometry><box><size>0.48 0.48 0.002</size></box></geometry>"
                f"<material><ambient>{color}</ambient><diffuse>{color}</diffuse></material>"
                "</visual>"
            )
    body = "\n        ".join(tiles)
    return f"""
    <model name="ground_grid">
      <static>true</static>
      <link name="link">
        {body}
      </link>
    </model>
"""


def _level_camera() -> str:
    """Level camera aimed at the trunk. Roll is 0. The recorder slides it in the world.

    Gazebo cameras look along +X. The pose below matches a robot standing at the origin.
    """
    ox, oy, oz = DEMO_CAMERA_OFFSET
    z = TRUNK_HEIGHT + oz
    return f"""
    <model name="demo_view">
      <static>true</static>
      <pose>{ox:.3f} {oy:.3f} {z:.3f} 0 {DEMO_CAMERA_PITCH:.3f} {DEMO_CAMERA_YAW:.3f}</pose>
      <link name="link">
        <sensor name="demo_cam" type="camera">
          <camera>
            <horizontal_fov>1.15</horizontal_fov>
            <image><width>480</width><height>360</height></image>
            <clip><near>0.05</near><far>30</far></clip>
          </camera>
          <always_on>1</always_on>
          <update_rate>10</update_rate>
          <visualize>false</visualize>
          <topic>/a1/demo_cam</topic>
        </sensor>
      </link>
    </model>
"""


def _joint(name: str, parent: str, child: str, axis: str, kind: int, damping: float) -> str:
    # Joint zero is the standing pose, so the limits shift by the home angle.
    lower = LOWER[kind] - HOME[kind]
    upper = UPPER[kind] - HOME[kind]
    return f"""
      <joint name="{name}" type="revolute">
        <parent>{parent}</parent>
        <child>{child}</child>
        <axis>
          <xyz>{axis}</xyz>
          <limit><lower>{lower:.5f}</lower><upper>{upper:.5f}</upper><effort>{FORCE_LIMIT}</effort></limit>
          <dynamics><damping>{damping}</damping></dynamics>
        </axis>
      </joint>
      <plugin filename="gz-sim-joint-position-controller-system" name="gz::sim::systems::JointPositionController">
        <joint_name>{name}</joint_name>
        <topic>/a1/{name}/cmd</topic>
        <p_gain>{P_GAIN}</p_gain>
        <i_gain>0</i_gain>
        <d_gain>{ACTUATOR_D_GAIN}</d_gain>
        <cmd_min>{-FORCE_LIMIT}</cmd_min>
        <cmd_max>{FORCE_LIMIT}</cmd_max>
      </plugin>"""
