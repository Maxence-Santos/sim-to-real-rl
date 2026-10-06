import numpy as np

from src.transfer.a1_description import (
    HOME,
    JOINTS,
    build_observation,
    foot_height,
    gazebo_command,
    gazebo_sdf,
    position_targets,
    projected_gravity,
)


def test_position_targets_are_the_standing_pose_at_zero_action():
    np.testing.assert_allclose(position_targets(np.zeros(12)), HOME)


def test_position_targets_respect_the_calf_upper_limit():
    targets = position_targets(np.ones(12))
    assert targets[2] == HOME[2] + 0.25


def test_observation_matches_the_mujoco_layout():
    obs = build_observation(
        joint_pos=HOME,
        joint_vel=np.zeros(12),
        linear_vel=np.array([0.5, 0.0, 0.0]),
        angular_vel=np.zeros(3),
        projected_gravity=np.array([0.0, 0.0, -1.0]),
        command=np.array([0.5, 0.0, 0.0]),
        last_action=np.zeros(12),
    )
    assert obs.shape == (48,)
    assert obs[24] == np.float32(0.5)
    assert obs[33] == np.float32(0.5)


def test_upright_gravity_points_down_in_the_trunk():
    np.testing.assert_allclose(projected_gravity(np.array([1.0, 0.0, 0.0, 0.0])), [0.0, 0.0, -1.0])


def test_gazebo_scene_lists_every_joint_and_the_standing_pose():
    sdf = gazebo_sdf()
    for name in JOINTS:
        assert f'<joint name="{name}"' in sdf
        assert f"<topic>/a1/{name}/cmd</topic>" in sdf
    assert "<initial_position>" not in sdf
    assert 'robot_base_frame>trunk<' in sdf
    assert "<dimensions>3</dimensions>" in sdf
    assert 0.02 < foot_height() < 0.04
    np.testing.assert_allclose(gazebo_command(np.zeros(12)), np.zeros(12))
    # Calf mass sits near the foot, matching the Menagerie inertial position.
    assert "<pose>0.00473 0.00000 -0.13198 0 0 0</pose>" in sdf
    assert sdf.count("<d_gain>0.0</d_gain>") == 12
    assert "<damping>2.0</damping>" in sdf
    assert "demo_cam" not in sdf
    assert "ground_grid" not in sdf
    recorded = gazebo_sdf(with_camera=True)
    assert "demo_cam" in recorded
    assert "ground_grid" in recorded
    trunk = recorded.split('<link name="trunk">', 1)[1].split("</link>", 1)[0]
    assert "demo_cam" not in trunk
    grid = recorded.split('<model name="ground_grid">', 1)[1].split("</model>", 1)[0]
    assert "<collision" not in grid
    view_pose = recorded.split('<model name="demo_view">', 1)[1].split("</pose>", 1)[0]
    assert view_pose.split("<pose>", 1)[1].split()[3] == "0"
