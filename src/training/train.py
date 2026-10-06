"""Train a PPO locomotion policy on the CPU build of PyTorch."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from src.envs.quadruped_env import QuadrupedEnv
from src.utils.torch_runtime import assert_cpu_pytorch

ROOT = Path(__file__).resolve().parents[2]
ACTIVATIONS = {"ReLU": torch.nn.ReLU, "Tanh": torch.nn.Tanh}


def parse_args():
    parser = argparse.ArgumentParser(description="Train the A1 locomotion policy on CPU.")
    parser.add_argument("--config", type=str, default="configs/train/ppo_config.yaml")
    parser.add_argument("--dr-config", type=str, default="configs/train/domain_randomization.yaml")
    parser.add_argument("--no-dr", action="store_true", help="Train without domain randomization.")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total_timesteps.")
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save", type=str, default="models/latest_policy")
    return parser.parse_args()


def _resolve(path: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = ROOT / candidate
    return candidate


def vecnormalize_path(save_path: Path) -> Path:
    return save_path.with_name(save_path.name + "_vecnormalize.pkl")


def build_env(dr_config=None):
    """One MuJoCo instance. One torch thread so the processes do not oversubscribe the CPU."""
    torch.set_num_threads(1)
    return QuadrupedEnv(dr_config=dr_config)


def main():
    version = assert_cpu_pytorch()
    args = parse_args()
    config_path = _resolve(args.config)
    with config_path.open() as handle:
        config = yaml.safe_load(handle)

    n_envs = args.n_envs if args.n_envs is not None else int(config.get("n_envs", 4))
    timesteps = args.timesteps if args.timesteps is not None else int(config["total_timesteps"])
    dr_path = None if args.no_dr else str(_resolve(args.dr_config))
    policy_cfg = config["policy"]
    activation = ACTIVATIONS[policy_cfg.get("activation_fn", "ReLU")]
    net_arch = policy_cfg.get("net_arch", {"pi": [256, 256, 128], "vf": [256, 256, 128]})

    print(
        f"torch {version}, device=cpu, envs={n_envs}, timesteps={timesteps}, dr={dr_path is not None}",
        flush=True,
    )

    parallel = n_envs > 1
    env = make_vec_env(
        build_env,
        n_envs=n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv if parallel else DummyVecEnv,
        vec_env_kwargs={"start_method": "spawn"} if parallel else None,
        env_kwargs={"dr_config": dr_path},
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=float(config["gamma"]))
    save_path = _resolve(args.save)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = CheckpointCallback(
        save_freq=max(500_000 // n_envs, 1),
        save_path=str(save_path.parent),
        name_prefix=save_path.name + "_ckpt",
        save_vecnormalize=True,
    )
    model = PPO(
        policy=policy_cfg.get("type", "MlpPolicy"),
        env=env,
        learning_rate=float(config["learning_rate"]),
        n_steps=int(config["n_steps"]),
        batch_size=int(config["batch_size"]),
        n_epochs=int(config["n_epochs"]),
        gamma=float(config["gamma"]),
        gae_lambda=float(config["gae_lambda"]),
        clip_range=float(config["clip_range"]),
        ent_coef=float(config["ent_coef"]),
        vf_coef=float(config["vf_coef"]),
        max_grad_norm=float(config["max_grad_norm"]),
        policy_kwargs={
            "net_arch": net_arch,
            "activation_fn": activation,
            "log_std_init": float(policy_cfg.get("log_std_init", -1.0)),
        },
        device="cpu",
        seed=args.seed,
        verbose=1,
        tensorboard_log=str(ROOT / "results" / "tensorboard"),
    )
    model.learn(total_timesteps=timesteps, callback=checkpoint)
    model.save(str(save_path))
    env.save(str(vecnormalize_path(save_path)))
    env.close()
    print(f"Saved {save_path}.zip and {vecnormalize_path(save_path).name}", flush=True)


if __name__ == "__main__":
    main()
