"""Deterministic execution controls and reproducibility environment configuration."""

from __future__ import annotations

import os
import random
import subprocess
import sys
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int = 1234) -> None:
    """Enforce complete determinism across Python, NumPy, PyTorch, and CUDA."""
    os.environ["PYTHONHASHSEED"] = "0"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        # Some platforms may not support all deterministic algorithms
        pass


def get_generator(seed: int, rank: int = 0, step: int = 0) -> torch.Generator:
    """Return a deterministic PyTorch Generator seeded for a specific rank and step."""
    gen = torch.Generator(device="cpu")
    # Formulate unique 64-bit integer seed
    combined_seed = (seed * 1_000_003 + rank * 1_009 + step) & 0x7FFFFFFF
    gen.manual_seed(combined_seed)
    return gen


def get_environment_metadata() -> dict[str, Any]:
    """Collect runtime environment metadata for audit manifests."""
    git_sha = "unknown"
    try:
        git_sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        pass

    return {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "cpu_count": os.cpu_count() or 1,
        "git_sha": git_sha,
    }
