#!/bin/bash
# ROS 2 puts launch_testing on PYTHONPATH, and pytest then crashes while loading it.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
"$ROOT/.venv/bin/python" -m pytest tests -q "$@"
