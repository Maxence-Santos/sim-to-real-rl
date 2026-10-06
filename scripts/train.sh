#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing .venv. Run ./scripts/install_cpu.sh first." >&2
  exit 1
fi
"$PY" -c "import torch; assert torch.version.cuda is None and '+cpu' in torch.__version__, torch.__version__"
"$PY" -m src.training.train "$@"
