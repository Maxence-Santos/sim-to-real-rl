"""Reward terms for flat-ground quadruped velocity tracking.

Each term is a pure function. ``compute_total_reward`` applies the weights from
``configs/robot/quadruped.yaml``. Energy and action-rate terms are already negative.
"""

from __future__ import annotations

import numpy as np

DEFAULT_WEIGHTS = {
    "alive": 1.0,
    "velocity": 1.5,
    "yaw": 0.5,
    "stability": 0.3,
    "energy": 0.0005,
    "action_rate": 0.01,
    "fall": 2.0,
}


def velocity_tracking_reward(current_xy: np.ndarray, target_xy: np.ndarray, sigma: float = 0.25) -> float:
    """Exponential tracking reward for planar body-frame velocity. Range (0, 1]."""
    error = float(np.sum(np.square(current_xy - target_xy)))
    return float(np.exp(-error / sigma))


def yaw_tracking_reward(current_yaw_rate: float, target_yaw_rate: float, sigma: float = 0.25) -> float:
    error = (current_yaw_rate - target_yaw_rate) ** 2
    return float(np.exp(-error / sigma))


def energy_penalty(torques: np.ndarray, joint_vel: np.ndarray) -> float:
    """Negative mechanical power, so a positive weight reduces the return."""
    return float(-np.sum(np.abs(torques * joint_vel)))


def stability_reward(projected_gravity: np.ndarray) -> float:
    """Upright reward from the gravity vector expressed in the trunk frame.

    Upright is approximately ``[0, 0, -1]``. Roll and pitch show up in x and y.
    """
    tilt = float(np.sum(np.square(projected_gravity[:2])))
    return float(np.exp(-tilt / 0.1))


def action_rate_penalty(action: np.ndarray, last_action: np.ndarray) -> float:
    return float(-np.sum(np.square(action - last_action)))


def compute_total_reward(
    lin_vel: np.ndarray,
    ang_vel: np.ndarray,
    command: np.ndarray,
    projected_gravity: np.ndarray,
    torques: np.ndarray,
    joint_vel: np.ndarray,
    action: np.ndarray,
    last_action: np.ndarray,
    fallen: bool,
    weights: dict | None = None,
) -> tuple[float, dict[str, float]]:
    """Weighted sum of the locomotion terms. The fall term is subtracted once."""
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    terms = {
        "velocity": velocity_tracking_reward(lin_vel[:2], command[:2]),
        "yaw": yaw_tracking_reward(float(ang_vel[2]), float(command[2])),
        "stability": stability_reward(projected_gravity),
        "energy": energy_penalty(torques, joint_vel),
        "action_rate": action_rate_penalty(action, last_action),
        "alive": 0.0 if fallen else 1.0,
        "fall": -1.0 if fallen else 0.0,
    }
    total = (
        w["alive"] * terms["alive"]
        + w["velocity"] * terms["velocity"]
        + w["yaw"] * terms["yaw"]
        + w["stability"] * terms["stability"]
        + w["energy"] * terms["energy"]
        + w["action_rate"] * terms["action_rate"]
        + w["fall"] * terms["fall"]
    )
    return float(total), terms
