"""Roll out a saved policy without opening a window."""

from __future__ import annotations

import argparse

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.envs.quadruped_env import QuadrupedEnv
from src.evaluation.metrics import forward_velocity_error, mean_forward_velocity, survival_time
from src.training.train import vecnormalize_path
from src.utils.torch_runtime import assert_cpu_pytorch


def make_eval_env(stats_path: Path | None):
    env = DummyVecEnv([lambda: QuadrupedEnv(dr_config=None)])
    if stats_path is not None:
        env = VecNormalize.load(str(stats_path), env)
        env.training = False
        env.norm_reward = False
    return env


def run_episode(env, model: PPO) -> dict:
    obs = env.reset()
    done = False
    reward_sum = 0.0
    vx: list[float] = []
    cmd: list[float] = []
    steps = 0
    info: dict = {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        info = infos[0]
        reward_sum += float(rewards[0])
        vx.append(float(info["linear_vel"][0]))
        cmd.append(float(info["command"][0]))
        steps += 1
        done = bool(dones[0])
    control_dt = 0.02
    survived = not info.get("fallen", True)
    error = forward_velocity_error(vx, cmd)
    return {
        "reward": reward_sum,
        "steps": steps,
        "time_s": survival_time(control_dt, steps),
        "vx": mean_forward_velocity(vx),
        "vx_error": error,
        "success": survived and error < 0.3,
        "fallen": bool(info.get("fallen", True)),
    }


def main():
    assert_cpu_pytorch()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    model_path = Path(args.model)
    if model_path.suffix == ".zip":
        model_path = model_path.with_suffix("")
    stats = vecnormalize_path(model_path)
    env = make_eval_env(stats if stats.is_file() else None)
    env.seed(args.seed)
    model = PPO.load(args.model, device="cpu")
    rows = [run_episode(env, model) for _ in range(args.episodes)]
    env.close()

    print(f"Episodes: {args.episodes}")
    print(f"Mean reward: {np.mean([row['reward'] for row in rows]):.2f}")
    print(f"Mean survival: {np.mean([row['time_s'] for row in rows]):.2f} s")
    print(f"Mean body vx: {np.mean([row['vx'] for row in rows]):.3f} m/s")
    print(f"Mean |vx - command|: {np.mean([row['vx_error'] for row in rows]):.3f} m/s")
    print(f"Success (survived and error < 0.3 m/s): {np.mean([row['success'] for row in rows]):.0%}")


if __name__ == "__main__":
    main()
