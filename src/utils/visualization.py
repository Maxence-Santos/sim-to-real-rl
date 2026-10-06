"""Write the training, domain-randomization, and Gazebo figures.

The TensorBoard folders and the JSON rollouts live under ``results/``, which
is gitignored. This script turns those local files into the pictures under
``docs/figures/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.training.train import ROOT

FIGURES = ROOT / "docs" / "figures"
# PPO_4 is the 10M run without domain randomization (final return about 3172).
# PPO_5 is the 10M run with domain randomization (final return about 3093).
TRAINING_RUNS = {
    "No domain randomization": ROOT / "results" / "tensorboard" / "PPO_4",
    "Domain randomization": ROOT / "results" / "tensorboard" / "PPO_5",
}
LATENCY_CONDITIONS = (
    ("latency_20ms", "20 ms"),
    ("latency_40ms", "40 ms"),
    ("mass_given_latency", "mass"),
    ("friction_given_latency", "friction"),
    ("damping_given_latency", "damping"),
    ("noise_given_latency", "noise"),
    ("all", "all"),
)
BASELINE_COLOR = "#3c6e9a"
DR_COLOR = "#c46b2f"


def _scalars(event_dir: Path, tag: str) -> tuple[list[float], list[float]]:
    accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    events = accumulator.Scalars(tag)
    steps = [event.step / 1e6 for event in events]
    values = [event.value for event in events]
    return steps, values


def plot_training_return(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 4.4))
    for label, event_dir in TRAINING_RUNS.items():
        steps, values = _scalars(event_dir, "rollout/ep_rew_mean")
        color = BASELINE_COLOR if "No " in label else DR_COLOR
        axis.plot(steps, values, label=label, color=color, linewidth=1.6)
    axis.set_xlabel("Environment steps (millions)")
    axis.set_ylabel("Mean episode return")
    axis.set_title("PPO training return, 10 million steps, CPU")
    axis.legend(frameon=False)
    axis.set_xlim(0, 10)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def _condition_error(payload: dict, name: str) -> float:
    for row in payload["conditions"]:
        if row["condition"] == name:
            return float(row["vx_error"])
    raise KeyError(name)


def plot_dr_with_latency(path: Path) -> None:
    baseline_latency = json.loads(
        (ROOT / "results" / "dr_axes_walk_baseline_given_latency.json").read_text()
    )
    dr_latency = json.loads((ROOT / "results" / "dr_axes_walk_dr_given_latency.json").read_text())
    baseline_all = json.loads((ROOT / "results" / "dr_axes_walk_baseline.json").read_text())
    dr_all = json.loads((ROOT / "results" / "dr_axes_walk_dr.json").read_text())
    labels = []
    baseline = []
    randomized = []
    for key, label in LATENCY_CONDITIONS:
        labels.append(label)
        if key == "all":
            baseline.append(_condition_error(baseline_all, "all"))
            randomized.append(_condition_error(dr_all, "all"))
        else:
            baseline.append(_condition_error(baseline_latency, key))
            randomized.append(_condition_error(dr_latency, key))
    positions = list(range(len(labels)))
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.2, 4.4))
    axis.bar(
        [item - width / 2 for item in positions],
        baseline,
        width=width,
        label="No domain randomization",
        color=BASELINE_COLOR,
    )
    axis.bar(
        [item + width / 2 for item in positions],
        randomized,
        width=width,
        label="Domain randomization",
        color=DR_COLOR,
    )
    axis.axhline(0.3, color="#666666", linewidth=0.8, linestyle="--")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Mean |forward speed − command| (m/s)")
    axis.set_title("16 paired episodes, action delay kept in the training range")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def plot_gazebo_gap(path: Path) -> None:
    rows = json.loads((ROOT / "results" / "gazebo_transfer.json").read_text())
    commands = [0.4, 0.55, 0.7]
    series = (
        ("walk_baseline", "mujoco", "Baseline, MuJoCo", "#9bb7d0"),
        ("walk_baseline", "gazebo", "Baseline, Gazebo", BASELINE_COLOR),
        ("walk_dr", "mujoco", "DR, MuJoCo", "#e2b08a"),
        ("walk_dr", "gazebo", "DR, Gazebo", DR_COLOR),
    )
    figure, axis = plt.subplots(figsize=(8.2, 4.4))
    width = 0.18
    for index, (policy, simulator, label, color) in enumerate(series):
        values = []
        for command in commands:
            match = next(
                row
                for row in rows
                if row["policy"] == policy
                and row["simulator"] == simulator
                and abs(row["command_vx"] - command) < 1e-6
            )
            values.append(match["vx_error"])
        offset = (index - 1.5) * width
        axis.bar(
            [item + offset for item in range(len(commands))],
            values,
            width=width,
            label=label,
            color=color,
        )
    axis.axhline(0.3, color="#666666", linewidth=0.8, linestyle="--")
    axis.set_xticks(range(len(commands)), [f"{command:.2f} m/s" for command in commands])
    axis.set_ylabel("Mean |forward speed − command| (m/s)")
    axis.set_title("One 20 s episode per command")
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    plot_training_return(FIGURES / "training_return.png")
    plot_dr_with_latency(FIGURES / "dr_with_latency.png")
    plot_gazebo_gap(FIGURES / "gazebo_speed_error.png")
    print(f"Wrote figures in {FIGURES}")


if __name__ == "__main__":
    main()
