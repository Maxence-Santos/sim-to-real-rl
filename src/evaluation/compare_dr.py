"""Roll out one saved policy on each domain-randomization axis.

Episodes are seeded on the raw Gymnasium env. Two policies compared with the
same seeds therefore see the same mass, friction, damping, latency, command,
and sensor-noise draws. Observation normalization is applied in numpy from
the VecNormalize file saved next to the policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.quadruped_env import OBS_DIM, QuadrupedEnv
from src.evaluation.metrics import forward_velocity_error, mean_forward_velocity, survival_time
from src.training.train import ROOT, vecnormalize_path
from src.utils.torch_runtime import assert_cpu_pytorch

# Order matches the analysis table. ``None`` means that axis is off.
AXES: tuple[tuple[str, str | None], ...] = (
    ("nominal", None),
    ("mass", "randomize_mass"),
    ("friction", "randomize_friction"),
    ("damping", "randomize_damping"),
    ("latency", "action_latency"),
    ("sensor_noise", "sensor_noise"),
    ("all", "*"),
)

# Training draws latency on every episode, so these keep that block and add one other.
COMPOSITE_AXES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("mass_given_latency", ("randomize_mass", "action_latency")),
    ("friction_given_latency", ("randomize_friction", "action_latency")),
    ("damping_given_latency", ("randomize_damping", "action_latency")),
    ("noise_given_latency", ("sensor_noise", "action_latency")),
)

FIXED_AXES: dict[str, dict] = {
    "latency_20ms": {"action_latency": {"enabled": True, "range_ms": [20, 20]}},
    "latency_40ms": {"action_latency": {"enabled": True, "range_ms": [40, 40]}},
}

CONTROL_DT = 0.02
SUCCESS_VX_ERROR = 0.3
OBS_CLIP = 10.0
OBS_EPSILON = 1e-8


def load_dr_config(path: str | Path | None = None) -> dict:
    source = Path(path) if path is not None else ROOT / "configs" / "train" / "domain_randomization.yaml"
    if not source.is_absolute():
        source = ROOT / source
    with source.open() as handle:
        return yaml.safe_load(handle) or {}


def axis_config(full: dict, name: str) -> dict | None:
    """One enabled block, a latency-plus-one composite, the full file, or no randomization."""
    keys = dict(AXES)
    if name in keys:
        key = keys[name]
        if key is None:
            return None
        if key == "*":
            return dict(full)
        if key not in full:
            raise KeyError(f"Domain randomization config has no {key!r} block")
        return {key: full[key]}
    composites = dict(COMPOSITE_AXES)
    if name in composites:
        return {key: full[key] for key in composites[name]}
    if name in FIXED_AXES:
        return FIXED_AXES[name]
    known = ", ".join(
        [label for label, _ in AXES]
        + [label for label, _ in COMPOSITE_AXES]
        + list(FIXED_AXES)
    )
    raise KeyError(f"Unknown DR axis {name!r}. Known: {known}")


def normalize_observation(
    obs: np.ndarray,
    mean: np.ndarray,
    var: np.ndarray,
    clip: float = OBS_CLIP,
    epsilon: float = OBS_EPSILON,
) -> np.ndarray:
    scaled = (np.asarray(obs, dtype=np.float64) - mean) / np.sqrt(var + epsilon)
    return np.clip(scaled, -clip, clip).astype(np.float32)


def load_observation_stats(model_path: Path) -> tuple[np.ndarray, np.ndarray]:
    stats = model_path
    if stats.suffix == ".zip":
        stats = stats.with_suffix("")
    stats = vecnormalize_path(stats)
    if not stats.is_file():
        raise FileNotFoundError(f"Missing observation stats: {stats}")
    env = VecNormalize.load(str(stats), DummyVecEnv([lambda: QuadrupedEnv(dr_config=None)]))
    try:
        mean = np.asarray(env.obs_rms.mean, dtype=np.float64).reshape(-1)
        var = np.asarray(env.obs_rms.var, dtype=np.float64).reshape(-1)
    finally:
        env.close()
    if mean.shape != (OBS_DIM,) or var.shape != (OBS_DIM,):
        raise RuntimeError(f"Expected observation stats of length {OBS_DIM}, got {mean.shape} and {var.shape}")
    return mean, var


def run_episode(env: QuadrupedEnv, model, seed: int, mean: np.ndarray, var: np.ndarray) -> dict:
    obs, _info = env.reset(seed=seed)
    reward_sum = 0.0
    vx: list[float] = []
    cmd: list[float] = []
    steps = 0
    info: dict = {}
    done = False
    while not done:
        action, _state = model.predict(normalize_observation(obs, mean, var), deterministic=True)
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        obs, reward, terminated, truncated, info = env.step(action)
        reward_sum += float(reward)
        vx.append(float(info["linear_vel"][0]))
        cmd.append(float(info["command"][0]))
        steps += 1
        done = bool(terminated or truncated)
    error = forward_velocity_error(vx, cmd)
    survived = not bool(info.get("fallen", True))
    return {
        "seed": seed,
        "reward": reward_sum,
        "steps": steps,
        "time_s": survival_time(env.control_dt, steps),
        "vx": mean_forward_velocity(vx),
        "command_vx": float(np.mean(cmd)) if cmd else 0.0,
        "vx_error": error,
        "success": survived and error < SUCCESS_VX_ERROR,
        "fallen": bool(info.get("fallen", True)),
        "trunk_mass": float(info.get("trunk_mass", 0.0)),
        "delay_steps": int(env._delay),
    }


def summarize(name: str, rows: list[dict]) -> dict:
    def column(key: str) -> np.ndarray:
        return np.asarray([row[key] for row in rows], dtype=np.float64)

    return {
        "condition": name,
        "episodes": len(rows),
        "survival_s": float(np.mean(column("time_s"))),
        "vx_error": float(np.mean(column("vx_error"))),
        "vx": float(np.mean(column("vx"))),
        "command_vx": float(np.mean(column("command_vx"))),
        "reward": float(np.mean(column("reward"))),
        "success": float(np.mean(column("success"))),
        "fallen": float(np.mean(column("fallen"))),
        "rows": rows,
    }


def evaluate_axes(
    model_path: Path,
    episodes: int,
    seed: int,
    names: list[str] | None = None,
    dr_config: dict | None = None,
) -> list[dict]:
    full = load_dr_config() if dr_config is None else dr_config
    selected = [label for label, _key in AXES] if names is None else names
    mean, var = load_observation_stats(model_path)
    model = PPO.load(str(model_path), device="cpu")
    summaries = []
    try:
        for name in selected:
            config = axis_config(full, name)
            env = QuadrupedEnv(dr_config=config)
            try:
                rows = [run_episode(env, model, seed + index, mean, var) for index in range(episodes)]
            finally:
                env.close()
            summaries.append(summarize(name, rows))
    finally:
        del model
    return summaries


def format_table(summaries: list[dict]) -> str:
    header = f"{'condition':<14} {'survival_s':>10} {'vx_error':>10} {'success':>8} {'reward':>10} {'fallen':>8}"
    lines = [header]
    for row in summaries:
        lines.append(
            f"{row['condition']:<14} {row['survival_s']:10.2f} {row['vx_error']:10.3f} "
            f"{row['success']:8.0%} {row['reward']:10.1f} {row['fallen']:8.0%}"
        )
    return "\n".join(lines)


def main():
    assert_cpu_pytorch()
    parser = argparse.ArgumentParser(description="Evaluate one policy on each DR axis.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--conditions", nargs="*", default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    summaries = evaluate_axes(model_path, args.episodes, args.seed, args.conditions)
    print(f"Model: {model_path.name}")
    print(f"Episodes per condition: {args.episodes}, first seed: {args.seed}")
    print(format_table(summaries))
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": str(model_path),
            "episodes": args.episodes,
            "seed": args.seed,
            "conditions": summaries,
        }
        out.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"Wrote {out}")


if __name__ == "__main__":
    main()
