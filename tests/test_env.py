import numpy as np

from src.envs.quadruped_env import N_JOINTS, OBS_DIM, QuadrupedEnv


def _disabled_dr(**overrides):
    config = {
        "randomize_mass": {"enabled": False},
        "randomize_friction": {"enabled": False},
        "randomize_damping": {"enabled": False},
        "action_latency": {"enabled": False},
        "sensor_noise": {"enabled": False},
    }
    config.update(overrides)
    return config


def test_reset_and_step_shapes():
    env = QuadrupedEnv(dr_config=_disabled_dr())
    obs, info = env.reset(seed=1)
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert info["command"].shape == (3,)
    next_obs, reward, terminated, truncated, _ = env.step(np.zeros(N_JOINTS))
    assert next_obs.shape == (OBS_DIM,)
    assert np.isfinite(reward)
    assert terminated is False
    assert truncated is False
    env.close()


def test_zero_action_stands_for_one_second():
    env = QuadrupedEnv(dr_config=_disabled_dr())
    env.reset(seed=0)
    terminated = False
    for _ in range(50):
        _, _, terminated, _, info = env.step(np.zeros(N_JOINTS))
        if terminated:
            break
    assert terminated is False
    assert info["fallen"] is False
    assert float(env.data.qpos[2]) > 0.2
    env.close()


def test_low_base_is_fallen():
    env = QuadrupedEnv(dr_config=_disabled_dr())
    env.reset(seed=0)
    env.data.qpos[2] = 0.02
    assert env._is_fallen()
    env.close()


def test_mass_randomization_scales_the_trunk():
    env = QuadrupedEnv(
        dr_config=_disabled_dr(randomize_mass={"enabled": True, "range": [-0.2, -0.2]})
    )
    nominal = float(env._mass0[env._trunk_id])
    _, info = env.reset(seed=0)
    assert info["trunk_mass"] == nominal * 0.8
    env.close()


def test_action_latency_holds_the_previous_command():
    env = QuadrupedEnv(
        dr_config=_disabled_dr(action_latency={"enabled": True, "range_ms": [40, 40]})
    )
    env.reset(seed=0)
    assert env._delay == 2
    env.step(np.ones(N_JOINTS))
    np.testing.assert_allclose(env.data.ctrl, env._home_ctrl)
    env.step(np.ones(N_JOINTS))
    np.testing.assert_allclose(env.data.ctrl, env._home_ctrl)
    env.step(np.ones(N_JOINTS))
    expected = np.clip(env._home_ctrl + env.action_scale, env._ctrl_low, env._ctrl_high)
    np.testing.assert_allclose(env.data.ctrl, expected)
    env.close()


def test_sensor_noise_perturbs_joint_position():
    env = QuadrupedEnv(
        dr_config=_disabled_dr(sensor_noise={"enabled": True, "joint_pos_std": 5.0})
    )
    obs, _ = env.reset(seed=2)
    clean = env.data.qpos[7:] - env._joint_qpos
    assert not np.allclose(obs[:12], clean, atol=0.5)
    env.close()
