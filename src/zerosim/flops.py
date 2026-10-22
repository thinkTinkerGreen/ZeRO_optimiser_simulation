"""FLOPs accounting for transformer forward, backward, recomputation, and AdamW."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from zerosim.config import ModelConfig


@dataclass(frozen=True)
class FlopBreakdown:
    fwd_flops: int
    bwd_flops: int
    z3_recompute_simulation_overhead_flops: int
    optimizer_flops: int
    total_flops: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "fwd_flops": self.fwd_flops,
            "bwd_flops": self.bwd_flops,
            "z3_recompute_simulation_overhead_flops": self.z3_recompute_simulation_overhead_flops,
            "optimizer_flops": self.optimizer_flops,
            "total_flops": self.total_flops,
        }


def compute_fwd_flops(cfg: ModelConfig, microbatch: int) -> int:
    """Compute forward floating point operations per rank per step.

    Formula: L * (2 * B * T * 12 * d^2 + 4 * B * T^2 * d) + 2 * B * T * d * V
    """
    B = microbatch
    T = cfg.seq_len
    d = cfg.d_model
    L = cfg.n_layer
    V = cfg.vocab

    block_flops = L * (2 * B * T * 12 * (d**2) + 4 * B * (T**2) * d)
    head_flops = 2 * B * T * d * V
    return int(block_flops + head_flops)


def compute_bwd_flops(fwd_flops: int) -> int:
    """Backward FLOPs = 2x forward matmul FLOPs."""
    return 2 * fwd_flops


def compute_optimizer_flops(
    updated_params: int,
    flops_per_param: int = 10,
) -> int:
    """Elementwise AdamW FLOPs per parameter updated by this rank."""
    return updated_params * flops_per_param


def compute_rank_flops(
    model_cfg: ModelConfig,
    microbatch: int,
    stage: int,
    updated_params: int,
    adam_flops_per_param: int = 10,
) -> FlopBreakdown:
    """Compute full rank FLOPs breakdown for one step."""
    fwd = compute_fwd_flops(model_cfg, microbatch)
    bwd = compute_bwd_flops(fwd)
    recompute = fwd if stage == 3 else 0
    opt = compute_optimizer_flops(updated_params, flops_per_param=adam_flops_per_param)
    total = fwd + bwd + recompute + opt

    return FlopBreakdown(
        fwd_flops=fwd,
        bwd_flops=bwd,
        z3_recompute_simulation_overhead_flops=recompute,
        optimizer_flops=opt,
        total_flops=total,
    )


def cross_check_flop_counter(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
) -> tuple[int | None, str]:
    """Cross-check analytic forward FLOPs against torch.utils.flop_counter.FlopCounterMode if available."""
    try:
        from torch.utils.flop_counter import FlopCounterMode

        with FlopCounterMode(display=False) as fcm:
            _ = model(sample_input)
        total_flops = fcm.get_total_flops()
        return total_flops, "FlopCounterMode measured"
    except Exception as e:
        return None, f"FlopCounterMode unavailable or raised: {e}"
