#!/bin/bash
# Export both actors and replay them in Gazebo and MuJoCo.
# Source ROS 2 first so the bridge can publish /a1/joint_states and /a1/odom.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# setup.bash reads optional variables such as AMENT_TRACE_SETUP_FILES.
# nounset aborts on those before Gazebo starts.
set +u
# shellcheck disable=SC1091
source /opt/ros/lyrical/setup.bash
set -u

echo "Exporting both actors to ONNX (CPU)."
"$ROOT/.venv/bin/python" -m src.transfer.export_policy --model models/walk_baseline.zip --output models/walk_baseline.onnx
"$ROOT/.venv/bin/python" -m src.transfer.export_policy --model models/walk_dr.zip --output models/walk_dr.onnx

echo "Zero-shot rollout in Gazebo and MuJoCo (20 s, commands 0.4, 0.55, 0.7)."
"$ROOT/.venv/bin/python" -u -m src.transfer.gazebo_bridge --seconds 20 --commands 0.4 0.55 0.7
