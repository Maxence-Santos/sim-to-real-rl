#!/bin/bash
# CPU-only install. Do not point pip at PyPI for torch: on Linux that wheel is the CUDA build.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install 'torch==2.14.1+cpu' --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install -e '.[dev]'
.venv/bin/python -c "import torch; assert torch.version.cuda is None and '+cpu' in torch.__version__, torch.__version__; print('torch', torch.__version__)"
