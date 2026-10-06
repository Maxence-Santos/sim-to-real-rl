"""Episode metrics for a velocity-tracking rollout."""

from __future__ import annotations

import numpy as np


def survival_time(control_dt: float, n_steps: int) -> float:
    return float(control_dt * n_steps)


def mean_forward_velocity(body_vx: list[float]) -> float:
    if not body_vx:
        return 0.0
    return float(np.mean(body_vx))


def forward_velocity_error(body_vx: list[float], command_vx: list[float]) -> float:
    if not body_vx:
        return 0.0
    return float(np.mean(np.abs(np.asarray(body_vx) - np.asarray(command_vx))))
