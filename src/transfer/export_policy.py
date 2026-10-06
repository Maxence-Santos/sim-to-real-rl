"""Export the PPO actor mean to ONNX. Inference target is CPU."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from stable_baselines3 import PPO

from src.utils.torch_runtime import assert_cpu_pytorch


class _ActorMean(torch.nn.Module):
    """Deterministic action mean. Observation normalization is baked in when present."""

    def __init__(self, policy, obs_mean: torch.Tensor | None = None, obs_var: torch.Tensor | None = None):
        super().__init__()
        self.features_extractor = policy.features_extractor
        self.mlp_extractor = policy.mlp_extractor
        self.action_net = policy.action_net
        if obs_mean is None:
            obs_mean = torch.zeros(1, int(policy.observation_space.shape[0]))
            obs_var = torch.ones(1, int(policy.observation_space.shape[0]))
        self.register_buffer("obs_mean", obs_mean.float().view(1, -1))
        self.register_buffer("obs_var", obs_var.float().view(1, -1))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        obs = (observation - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8)
        obs = torch.clamp(obs, -10.0, 10.0)
        features = self.features_extractor(obs)
        latent = self.mlp_extractor.forward_actor(features)
        return self.action_net(latent)


def _normalization(model_path: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from src.envs.quadruped_env import QuadrupedEnv

    stats = Path(model_path)
    if stats.suffix == ".zip":
        stats = stats.with_suffix("")
    stats = stats.with_name(stats.name + "_vecnormalize.pkl")
    if not stats.is_file():
        return None, None
    vec = VecNormalize.load(str(stats), DummyVecEnv([lambda: QuadrupedEnv(dr_config=None)]))
    mean = torch.as_tensor(vec.obs_rms.mean)
    var = torch.as_tensor(vec.obs_rms.var)
    vec.close()
    print(f"Baking observation normalization from {stats.name}")
    return mean, var


def export_to_onnx(model_path: str, output_path: str) -> None:
    assert_cpu_pytorch()
    print(f"Loading {model_path} on CPU")
    model = PPO.load(model_path, device="cpu")
    mean, var = _normalization(model_path)
    actor = _ActorMean(model.policy, mean, var)
    actor.eval()
    dummy = torch.zeros(1, int(model.observation_space.shape[0]))
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # dynamo=False keeps the TorchScript exporter. The torch.export path is the
    # default from PyTorch 2.9 and is unnecessary for this MLP.
    torch.onnx.export(
        actor,
        dummy,
        destination,
        dynamo=False,
        opset_version=17,
        input_names=["observation"],
        output_names=["action"],
    )
    print(f"Wrote {destination}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/latest_policy.zip")
    parser.add_argument("--output", type=str, default="models/exported_policy.onnx")
    args = parser.parse_args()
    export_to_onnx(args.model, args.output)


if __name__ == "__main__":
    main()
