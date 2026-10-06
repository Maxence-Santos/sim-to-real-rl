"""Replay a saved policy in MuJoCo and write a GIF, or open the viewer.

The interactive window follows the trunk in real time. The GIF uses the same
tracking camera and does not need ffmpeg. Domain randomization stays on unless
``--no-dr`` is set, because the randomized policy only tracks speed when the
action delay it trained with is present.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from stable_baselines3 import PPO

from src.envs.quadruped_env import QuadrupedEnv
from src.evaluation.compare_dr import load_dr_config, load_observation_stats, normalize_observation
from src.training.train import ROOT
from src.utils.torch_runtime import assert_cpu_pytorch


def tracking_camera(model: mujoco.MjModel) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.distance = 1.8
    camera.azimuth = 140.0
    camera.elevation = -18.0
    return camera


def look_at_trunk(env: QuadrupedEnv, camera: mujoco.MjvCamera) -> None:
    camera.lookat[:] = env.data.xpos[env._trunk_id]


def run_live(env: QuadrupedEnv, model: PPO, mean: np.ndarray, var: np.ndarray, seed: int, seconds: float) -> None:
    import mujoco.viewer

    obs, _info = env.reset(seed=seed)
    print("Opening the MuJoCo window. Close it to stop the replay.", flush=True)
    viewer = mujoco.viewer.launch_passive(env.model, env.data)
    camera = tracking_camera(env.model)
    steps = int(seconds / env.control_dt)
    try:
        for _ in range(steps):
            if not viewer.is_running():
                break
            started = time.perf_counter()
            look_at_trunk(env, camera)
            viewer.cam.lookat[:] = camera.lookat
            viewer.cam.distance = camera.distance
            viewer.cam.azimuth = camera.azimuth
            viewer.cam.elevation = camera.elevation
            action, _state = model.predict(normalize_observation(obs, mean, var), deterministic=True)
            obs, _reward, terminated, truncated, _info = env.step(np.asarray(action, dtype=np.float64).reshape(-1))
            if terminated or truncated:
                obs, _info = env.reset(seed=seed)
            viewer.sync()
            remaining = env.control_dt - (time.perf_counter() - started)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        viewer.close()


def record_gif(
    env: QuadrupedEnv,
    model: PPO,
    mean: np.ndarray,
    var: np.ndarray,
    seed: int,
    seconds: float,
    out: Path,
    fps: int,
) -> dict:
    renderer = mujoco.Renderer(env.model, height=480, width=640)
    camera = tracking_camera(env.model)
    obs, info = env.reset(seed=seed)
    hold = max(1, int(round(1.0 / (fps * env.control_dt))))
    frames: list[Image.Image] = []
    vx: list[float] = []
    start_x = float(env.data.qpos[0])
    steps = int(seconds / env.control_dt)
    try:
        for index in range(steps):
            look_at_trunk(env, camera)
            if index % hold == 0:
                renderer.update_scene(env.data, camera=camera)
                frames.append(Image.fromarray(renderer.render()))
            action, _state = model.predict(normalize_observation(obs, mean, var), deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float64).reshape(-1))
            vx.append(float(info["linear_vel"][0]))
            if terminated or truncated:
                break
    finally:
        renderer.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=int(round(1000 * hold * env.control_dt)),
        loop=0,
    )
    return {
        "frames": len(frames),
        "command_vx": float(info["command"][0]),
        "mean_vx": float(np.mean(vx)) if vx else 0.0,
        "distance_m": float(env.data.qpos[0] - start_x),
        "fallen": bool(info.get("fallen", True)),
    }


def main() -> None:
    assert_cpu_pytorch()
    parser = argparse.ArgumentParser(description="Watch or record a MuJoCo walk.")
    parser.add_argument("--model", type=str, default="models/walk_dr.zip")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--out", type=str, default="results/walk_dr_mujoco.gif")
    parser.add_argument("--no-dr", action="store_true")
    parser.add_argument("--live", action="store_true", help="Open the MuJoCo window instead of writing a GIF.")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    dr_config = None if args.no_dr else load_dr_config()
    mean, var = load_observation_stats(model_path)
    policy = PPO.load(str(model_path), device="cpu")
    env = QuadrupedEnv(dr_config=dr_config)
    try:
        if args.live:
            run_live(env, policy, mean, var, args.seed, args.seconds)
            return
        summary = record_gif(env, policy, mean, var, args.seed, args.seconds, out, args.fps)
    finally:
        env.close()
    print(
        f"Wrote {out} ({summary['frames']} frames). "
        f"command {summary['command_vx']:.2f} m/s, mean vx {summary['mean_vx']:.2f} m/s, "
        f"forward {summary['distance_m']:.2f} m, fallen={summary['fallen']}"
    )


if __name__ == "__main__":
    main()
