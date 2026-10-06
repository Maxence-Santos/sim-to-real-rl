"""Side-by-side GIF of one policy in MuJoCo and in Gazebo.

The clip is a separate recording from the 20 s table in ``docs/gazebo_transfer.md``.
Both panels use ``walk_dr`` at a fixed 0.70 m/s command, with one control step
of delay. Frames are written with Pillow because ffmpeg is not installed.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.training.train import ROOT
from src.transfer.a1_description import (
    CONTROL_DT,
    DEMO_CAMERA_OFFSET,
    DEMO_CAMERA_PITCH,
    DEMO_CAMERA_YAW,
    TRUNK_HEIGHT,
    build_observation,
    gazebo_command,
)
from src.transfer.gazebo_bridge import (
    Actor,
    GazeboA1,
    fallen,
    hold_pose,
    start_gazebo,
    stop_gazebo,
)
from src.utils.torch_runtime import assert_cpu_pytorch

WIDTH = 480
HEIGHT = 360
FPS = 10
COMMAND = 0.70
DELAY_STEPS = 1
CAMERA_TOPIC = "/a1/demo_cam"


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    )
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _mujoco_frames(actor: Actor, seconds: float) -> tuple[list[Image.Image], dict]:
    from src.envs.quadruped_env import QuadrupedEnv

    env = QuadrupedEnv(dr_config={"action_latency": {"enabled": True, "range_ms": [20.0, 20.0]}})
    renderer = mujoco.Renderer(env.model, height=HEIGHT, width=WIDTH)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(env.model, camera)
    camera.distance = 1.7
    camera.azimuth = 140.0
    camera.elevation = -16.0
    frames: list[Image.Image] = []
    vx: list[float] = []
    try:
        env.reset(seed=1)
        env._command[:] = np.array([COMMAND, 0.0, 0.0])
        obs = env._observation()
        steps = int(seconds / env.control_dt)
        hold = max(1, int(round(1.0 / (FPS * env.control_dt))))
        did_fall = False
        for index in range(steps):
            camera.lookat[:] = env.data.xpos[env._trunk_id]
            if index % hold == 0:
                renderer.update_scene(env.data, camera=camera)
                frames.append(Image.fromarray(renderer.render()))
            action = np.clip(actor(obs), -1.0, 1.0)
            obs, _reward, terminated, truncated, info = env.step(action)
            vx.append(float(info["linear_vel"][0]))
            did_fall = bool(info["fallen"])
            if terminated or truncated:
                break
    finally:
        renderer.close()
        env.close()
    return frames, {
        "simulator": "mujoco",
        "frames": len(frames),
        "vx": float(np.mean(vx)) if vx else 0.0,
        "fallen": did_fall,
    }


def _image_from_msg(msg) -> Image.Image | None:
    if msg.width == 0 or msg.height == 0 or not msg.data:
        return None
    channels = {3: 3, 8: 3}.get(int(msg.pixel_format_type))
    # RGB_INT8 is 3, BGR_INT8 is 8.
    if channels is None or msg.step < msg.width * channels:
        return None
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    rows = msg.height
    expected = msg.step * rows
    if raw.size < expected:
        return None
    image = raw[:expected].reshape(rows, msg.step)[:, : msg.width * channels].reshape(rows, msg.width, channels)
    if int(msg.pixel_format_type) == 8:
        image = image[:, :, ::-1]
    return Image.fromarray(image.copy())


def _place_camera(node, boolean_type, position: np.ndarray) -> None:
    """Keep the recording camera level and a fixed offset from the trunk.

    The call is synchronous, so the recorder runs it off the control loop.
    """
    from gz.msgs.pose_pb2 import Pose as GzPose

    pitch = DEMO_CAMERA_PITCH
    yaw = DEMO_CAMERA_YAW
    half_pitch = pitch / 2.0
    half_yaw = yaw / 2.0
    cp, sp = float(np.cos(half_pitch)), float(np.sin(half_pitch))
    cy, sy = float(np.cos(half_yaw)), float(np.sin(half_yaw))
    ox, oy, oz = DEMO_CAMERA_OFFSET
    pose = GzPose()
    pose.name = "demo_view"
    pose.position.x = float(position[0] + ox)
    pose.position.y = float(position[1] + oy)
    pose.position.z = float(TRUNK_HEIGHT + oz)
    pose.orientation.w = cp * cy
    pose.orientation.x = -sp * sy
    pose.orientation.y = sp * cy
    pose.orientation.z = cp * sy
    node.request("/world/a1_flat/set_pose", pose, GzPose, boolean_type, 50)


def _gazebo_frames(actor: Actor, seconds: float) -> tuple[list[Image.Image], dict]:
    latest: dict[str, Image.Image] = {}
    seen_format: dict[str, int] = {}

    def on_image(msg) -> None:
        seen_format["pixel_format_type"] = int(msg.pixel_format_type)
        seen_format["width"] = int(msg.width)
        seen_format["step"] = int(msg.step)
        frame = _image_from_msg(msg)
        if frame is not None:
            latest["frame"] = frame
            latest["n"] = latest.get("n", 0) + 1

    proc = start_gazebo(ROOT / "results" / "gazebo_cam.log", with_camera=True)
    frames: list[Image.Image] = []
    vx: list[float] = []
    did_fall = False
    stop_camera = threading.Event()
    follower: threading.Thread | None = None
    try:
        robot = GazeboA1()
        from gz.msgs.image_pb2 import Image as GzImage

        robot.connect(timeout=40)
        robot.node.subscribe(GzImage, CAMERA_TOPIC, on_image)

        def follow_trunk() -> None:
            from gz.transport import Node

            node = Node()
            while not stop_camera.is_set():
                _place_camera(node, robot.Boolean, robot.position)
                stop_camera.wait(0.05)

        follower = threading.Thread(target=follow_trunk, daemon=True)
        follower.start()
        hold_pose(robot, 1.0)
        deadline = time.monotonic() + 15.0
        while "frame" not in latest and time.monotonic() < deadline:
            time.sleep(0.05)
        if "frame" not in latest:
            topics = []
            if hasattr(robot.node, "topic_list"):
                topics = list(robot.node.topic_list())
            raise TimeoutError(
                f"No camera frame on {CAMERA_TOPIC}. format={seen_format} topics={topics}"
            )
        command = np.array([COMMAND, 0.0, 0.0])
        history: deque[np.ndarray] = deque(np.zeros(12) for _ in range(DELAY_STEPS))
        last_action = np.zeros(12)
        start = robot.sim_time
        last = start - CONTROL_DT
        next_frame = start
        wall_deadline = time.monotonic() + seconds * 8.0 + 30.0
        while robot.sim_time < start + seconds and time.monotonic() < wall_deadline:
            if robot.sim_time - last < CONTROL_DT:
                time.sleep(0.001)
                continue
            last = robot.sim_time
            state = robot.state()
            did_fall = fallen(state)
            observation = build_observation(
                state["q"],
                state["qd"],
                state["linear"],
                state["angular"],
                state["gravity"],
                command,
                last_action,
            )
            action = np.clip(actor(observation), -1.0, 1.0)
            history.append(action.copy())
            applied = history.popleft()
            last_action = applied
            robot.send(gazebo_command(applied))
            vx.append(float(state["linear"][0]))
            if robot.sim_time >= next_frame and "frame" in latest:
                frames.append(latest["frame"].resize((WIDTH, HEIGHT)))
                next_frame += 1.0 / FPS
            if did_fall:
                break
    finally:
        stop_camera.set()
        if follower is not None:
            follower.join(timeout=1.0)
        stop_gazebo(proc)
    return frames, {
        "simulator": "gazebo",
        "frames": len(frames),
        "vx": float(np.mean(vx)) if vx else 0.0,
        "fallen": did_fall,
    }


def _label(frame: Image.Image, text: str) -> Image.Image:
    banner = Image.new("RGB", (frame.width, 28), (24, 26, 28))
    draw = ImageDraw.Draw(banner)
    draw.text((10, 4), text, fill=(245, 245, 245), font=_font(16))
    canvas = Image.new("RGB", (frame.width, frame.height + 28), (24, 26, 28))
    canvas.paste(banner, (0, 0))
    canvas.paste(frame, (0, 28))
    return canvas


def write_gif(mujoco_frames: list[Image.Image], gazebo_frames: list[Image.Image], out: Path) -> None:
    count = min(len(mujoco_frames), len(gazebo_frames))
    if count < 2:
        raise RuntimeError(f"Not enough frames: mujoco={len(mujoco_frames)} gazebo={len(gazebo_frames)}")
    panels = []
    for left, right in zip(mujoco_frames[:count], gazebo_frames[:count]):
        left_l = _label(left, "MuJoCo")
        right_l = _label(right, "Gazebo")
        panel = Image.new("RGB", (left_l.width + right_l.width + 4, left_l.height), (24, 26, 28))
        panel.paste(left_l, (0, 0))
        panel.paste(right_l, (left_l.width + 4, 0))
        panels.append(panel)
    out.parent.mkdir(parents=True, exist_ok=True)
    panels[0].save(
        out,
        save_all=True,
        append_images=panels[1:],
        duration=int(round(1000 / FPS)),
        loop=0,
        optimize=True,
    )


def main() -> None:
    assert_cpu_pytorch()
    seconds = 8.0
    actor = Actor(ROOT / "models" / "walk_dr.zip")
    print("Recording MuJoCo", flush=True)
    mujoco_frames, mujoco_info = _mujoco_frames(actor, seconds)
    print(mujoco_info, flush=True)
    print("Recording Gazebo", flush=True)
    gazebo_frames, gazebo_info = _gazebo_frames(actor, seconds)
    print(gazebo_info, flush=True)
    out = ROOT / "docs" / "figures" / "sim_vs_gazebo.gif"
    write_gif(mujoco_frames, gazebo_frames, out)
    print(f"Wrote {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
