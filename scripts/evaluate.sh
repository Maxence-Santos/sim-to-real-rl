#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
MODEL_PATH="${1:-models/latest_policy}"
"$ROOT/.venv/bin/python" -m src.evaluation.evaluate --model "$MODEL_PATH"
