"""Runtime check that this project stays on the CPU build of PyTorch."""

from __future__ import annotations


def assert_cpu_pytorch() -> str:
    """Return the installed version, or raise if this is a CUDA build.

    A default ``pip install torch`` on Linux pulls the multi-gigabyte CUDA wheel.
    Training and export must keep using the CPU wheel from the PyTorch CPU index.
    """
    import torch

    version = torch.__version__
    cuda_build = torch.version.cuda is not None or torch.backends.cuda.is_built()
    if cuda_build or "+cpu" not in version:
        raise RuntimeError(
            "This project uses the CPU build of PyTorch only. "
            f"Found torch {version} (cuda={torch.version.cuda}). "
            "Install it with ./scripts/install_cpu.sh and do not reinstall torch from PyPI."
        )
    return version
