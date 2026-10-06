import numpy as np

from src.envs.quadruped_env import N_JOINTS, QuadrupedEnv
from src.evaluation.compare_dr import axis_config, normalize_observation, run_episode


def test_axis_config_enables_only_the_named_block():
    full = {
        "randomize_mass": {"enabled": True, "range": [-0.2, 0.2]},
        "randomize_friction": {"enabled": True, "range": [0.5, 1.5]},
        "action_latency": {"enabled": True, "range_ms": [10, 40]},
        "sensor_noise": {"enabled": True, "joint_pos_std": 0.02},
    }
    assert axis_config(full, "nominal") is None
    mass = axis_config(full, "mass")
    assert list(mass) == ["randomize_mass"]
    assert mass["randomize_mass"]["enabled"] is True
    assert axis_config(full, "all") == full
    stacked = axis_config(full, "mass_given_latency")
    assert set(stacked) == {"randomize_mass", "action_latency"}


def test_normalize_observation_matches_vecnormalize():
    obs = np.array([2.0, -2.0], dtype=np.float32)
    mean = np.array([1.0, 0.0])
    var = np.array([1.0, 1.0])
    scaled = normalize_observation(obs, mean, var, clip=10.0, epsilon=1e-8)
    np.testing.assert_allclose(scaled, [1.0, -2.0])
    clipped = normalize_observation(np.array([100.0]), np.array([0.0]), np.array([1.0]), clip=10.0)
    assert float(clipped[0]) == 10.0


class _Standing:
    def predict(self, observation, deterministic=False):
        assert observation.shape == (48,)
        assert deterministic is True
        return np.zeros(N_JOINTS, dtype=np.float32), None


def test_run_episode_uses_the_seed_and_stops_when_truncated():
    env = QuadrupedEnv(robot_config={"episode_steps": 2}, dr_config=None)
    mean = np.zeros(48)
    var = np.ones(48)
    row = run_episode(env, _Standing(), seed=3, mean=mean, var=var)
    assert row["steps"] == 2
    assert row["seed"] == 3
    assert row["fallen"] is False
    again = QuadrupedEnv(robot_config={"episode_steps": 2}, dr_config=None)
    first, _ = env.reset(seed=3)
    second, _ = again.reset(seed=3)
    np.testing.assert_allclose(first, second)
    env.close()
    again.close()


def test_same_seed_repeats_the_mass_draw_after_a_step():
    config = {"randomize_mass": {"enabled": True, "range": [-0.2, 0.2]}}
    left = QuadrupedEnv(dr_config=config)
    right = QuadrupedEnv(dr_config=config)
    left_mass = left.reset(seed=7)[1]["trunk_mass"]
    left.step(np.zeros(N_JOINTS))
    right_mass = right.reset(seed=7)[1]["trunk_mass"]
    repeated = left.reset(seed=7)[1]["trunk_mass"]
    assert left_mass == right_mass == repeated
    left.close()
    right.close()
