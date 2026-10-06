import numpy as np

from src.envs.reward_functions import (
    action_rate_penalty,
    compute_total_reward,
    energy_penalty,
    stability_reward,
    velocity_tracking_reward,
    yaw_tracking_reward,
)


def test_perfect_tracking_is_one():
    assert velocity_tracking_reward(np.zeros(2), np.zeros(2)) == 1.0
    assert yaw_tracking_reward(0.2, 0.2) == 1.0


def test_velocity_reward_drops_when_the_base_is_stopped():
    tracked = velocity_tracking_reward(np.array([0.5, 0.0]), np.array([0.5, 0.0]))
    stopped = velocity_tracking_reward(np.zeros(2), np.array([0.5, 0.0]))
    assert tracked > stopped


def test_penalties_are_negative_and_upright_beats_tilted():
    assert energy_penalty(np.ones(12), np.ones(12)) < 0.0
    assert action_rate_penalty(np.ones(12), np.zeros(12)) < 0.0
    upright = stability_reward(np.array([0.0, 0.0, -1.0]))
    tilted = stability_reward(np.array([0.7, 0.0, -0.7]))
    assert upright == 1.0
    assert upright > tilted


def test_fall_is_subtracted_once():
    kwargs = dict(
        lin_vel=np.array([0.5, 0.0, 0.0]),
        ang_vel=np.zeros(3),
        command=np.array([0.5, 0.0, 0.0]),
        projected_gravity=np.array([0.0, 0.0, -1.0]),
        torques=np.zeros(12),
        joint_vel=np.zeros(12),
        action=np.zeros(12),
        last_action=np.zeros(12),
        weights={
            "alive": 0.0,
            "velocity": 1.0,
            "yaw": 0.0,
            "stability": 0.0,
            "energy": 0.0,
            "action_rate": 0.0,
            "fall": 2.0,
        },
    )
    standing, _ = compute_total_reward(fallen=False, **kwargs)
    fallen, terms = compute_total_reward(fallen=True, **kwargs)
    assert standing == 1.0
    assert fallen == standing - 2.0
    assert terms["fall"] == -1.0
