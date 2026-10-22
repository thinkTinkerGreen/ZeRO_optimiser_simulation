"""Latency-bandwidth (alpha-beta) cost model for step execution time estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from zerosim.comm.stats import CommStats
from zerosim.flops import FlopBreakdown


@dataclass(frozen=True)
class InterconnectPreset:
    name: str
    alpha_s: float
    bw_bytes_per_s: float


PRESET_NVLINK = InterconnectPreset(
    name="nvlink_like",
    alpha_s=5e-6,
    bw_bytes_per_s=300e9,
)

PRESET_IB = InterconnectPreset(
    name="ib_like",
    alpha_s=5e-6,
    bw_bytes_per_s=12.5e9,
)


@dataclass(frozen=True)
class StepCostEstimate:
    preset_name: str
    t_compute_s: float
    t_comm_s: float
    t_step_s: float
    simulator_wall_clock_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset_name": self.preset_name,
            "t_compute_s": round(self.t_compute_s, 6),
            "t_comm_s": round(self.t_comm_s, 6),
            "t_step_s": round(self.t_step_s, 6),
            "simulator_wall_clock_s": round(self.simulator_wall_clock_s, 6),
            "note": "Simulator time is host multi-threaded simulation time, not hardware time.",
        }


def estimate_step_time(
    flops: FlopBreakdown,
    comm_stats: CommStats,
    preset: InterconnectPreset,
    compute_flops_per_s: float = 2e13,
    simulator_wall_clock_s: float = 0.0,
) -> StepCostEstimate:
    """Model step time under explicit latency-bandwidth parameters."""
    t_compute = flops.total_flops / compute_flops_per_s
    t_comm = (comm_stats.total_calls * preset.alpha_s) + (
        comm_stats.total_ring_bytes / preset.bw_bytes_per_s
    )
    t_step = t_compute + t_comm

    return StepCostEstimate(
        preset_name=preset.name,
        t_compute_s=t_compute,
        t_comm_s=t_comm,
        t_step_s=t_step,
        simulator_wall_clock_s=simulator_wall_clock_s,
    )
