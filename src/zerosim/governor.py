"""Resource governor enforcing concurrency and memory limits (50% capacity cap)."""

from __future__ import annotations

import contextlib
import math
import os
import threading
from typing import Any

import torch

from zerosim.config import SimConfig


def get_available_memory() -> int:
    """Return available host RAM in bytes."""
    try:
        import psutil

        return psutil.virtual_memory().available
    except Exception:
        pass

    # Fallback to /proc/meminfo on Linux
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    parts = line.split()
                    return int(parts[1]) * 1024  # MemAvailable in kB
    except Exception:
        pass

    # Default fallback: 8 GB
    return 8 * 1024 * 1024 * 1024


def get_cuda_memory() -> int | None:
    """Return total CUDA memory on device 0 in bytes, if available."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory
    return None


class ComputeScope:
    """Context manager acquiring and releasing a governor compute slot."""

    def __init__(self, semaphore: threading.BoundedSemaphore) -> None:
        self._semaphore = semaphore

    def __enter__(self) -> None:
        self._semaphore.acquire()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._semaphore.release()


class ResourceGovernor:
    """Enforces 50% concurrency and memory caps across virtual ranks."""

    def __init__(
        self,
        max_concurrent: int | None = None,
        memory_fraction: float = 0.5,
    ) -> None:
        self.memory_fraction = memory_fraction
        cpu_total = os.cpu_count() or 1
        if max_concurrent is not None:
            self.max_concurrent = max(1, max_concurrent)
        else:
            self.max_concurrent = max(1, math.floor(self.memory_fraction * cpu_total))

        self._semaphore = threading.BoundedSemaphore(self.max_concurrent)

        # Set CUDA memory fraction if CUDA is enabled
        if torch.cuda.is_available():
            try:
                torch.cuda.set_per_process_memory_fraction(self.memory_fraction, 0)
            except Exception:
                pass

    def compute_scope(self) -> ComputeScope:
        """Acquire compute slot. MUST NOT be held during collective communications."""
        return ComputeScope(self._semaphore)

    def plan_memory(self, cfg: SimConfig, approx_numel: int) -> dict[str, Any]:
        """Pre-flight memory budget verification."""
        bytes_per_elem = 8 if cfg.compute_dtype == "float64" else 4
        opt_factor = 3  # master, m, v
        full_bytes = approx_numel * bytes_per_elem

        # Model state per rank according to ZeRO stage
        if cfg.stage == 0:
            model_state_per_rank = (1 + 1 + opt_factor) * full_bytes
        elif cfg.stage == 1:
            model_state_per_rank = (1 + 1) * full_bytes + (opt_factor * full_bytes) // cfg.world_size
        elif cfg.stage == 2:
            model_state_per_rank = 1 * full_bytes + ((1 + opt_factor) * full_bytes) // cfg.world_size
        elif cfg.stage == 3:
            model_state_per_rank = ((1 + 1 + opt_factor) * full_bytes) // cfg.world_size
        else:
            model_state_per_rank = (1 + 1 + opt_factor) * full_bytes

        # Conservative activations estimate per rank
        act_estimate = (
            2
            * cfg.microbatch
            * cfg.model.seq_len
            * cfg.model.d_model
            * cfg.model.n_layer
            * bytes_per_elem
        )
        total_predicted = cfg.world_size * (model_state_per_rank + act_estimate)

        if cfg.device == "cuda" and torch.cuda.is_available():
            cuda_total = get_cuda_memory() or (16 * 1024**3)
            limit = int(cuda_total * self.memory_fraction)
            if total_predicted > limit:
                raise RuntimeError(
                    f"Governor rejected plan: Predicted {total_predicted / 1e6:.1f} MB exceeds "
                    f"50% CUDA memory limit ({limit / 1e6:.1f} MB). "
                    f"Please reduce world_size or use a smaller model config (e.g., 'tiny')."
                )
        else:
            ram_avail = get_available_memory()
            limit = int(ram_avail * self.memory_fraction)
            if total_predicted > limit:
                raise RuntimeError(
                    f"Governor rejected plan: Predicted {total_predicted / 1e6:.1f} MB exceeds "
                    f"50% available RAM limit ({limit / 1e6:.1f} MB). "
                    f"Please reduce world_size or use a smaller model config (e.g., 'tiny')."
                )

        return {
            "predicted_bytes": total_predicted,
            "limit_bytes": limit,
            "max_concurrent": self.max_concurrent,
        }
