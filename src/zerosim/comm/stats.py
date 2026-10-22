"""Communication metrics and element accounting per rank and step."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class OpStats:
    calls: int = 0
    elems: int = 0
    logical_bytes: int = 0
    ring_bytes: int = 0


@dataclass
class CommStats:
    """Tracks communication metrics per rank."""

    rank: int
    world_size: int
    all_reduce: OpStats = field(default_factory=OpStats)
    reduce_scatter: OpStats = field(default_factory=OpStats)
    all_gather: OpStats = field(default_factory=OpStats)
    barrier_calls: int = 0
    notes: list[str] = field(default_factory=list)

    def record_all_reduce(self, numel: int, bytes_per_elem: int) -> None:
        """Paper convention: all-reduce counts 2 * numel elements."""
        self.all_reduce.calls += 1
        self.all_reduce.elems += 2 * numel
        self.all_reduce.logical_bytes += 2 * numel * bytes_per_elem
        if self.world_size > 1:
            ring = int(2 * ((self.world_size - 1) / self.world_size) * numel * bytes_per_elem)
        else:
            ring = 0
        self.all_reduce.ring_bytes += ring

    def record_reduce_scatter(self, full_numel: int, bytes_per_elem: int) -> None:
        """Paper convention: reduce-scatter counts 1 * full_numel elements."""
        self.reduce_scatter.calls += 1
        self.reduce_scatter.elems += full_numel
        self.reduce_scatter.logical_bytes += full_numel * bytes_per_elem
        if self.world_size > 1:
            ring = int(((self.world_size - 1) / self.world_size) * full_numel * bytes_per_elem)
        else:
            ring = 0
        self.reduce_scatter.ring_bytes += ring

    def record_all_gather(self, full_numel: int, bytes_per_elem: int) -> None:
        """Paper convention: all-gather counts 1 * full_numel elements."""
        self.all_gather.calls += 1
        self.all_gather.elems += full_numel
        self.all_gather.logical_bytes += full_numel * bytes_per_elem
        if self.world_size > 1:
            ring = int(((self.world_size - 1) / self.world_size) * full_numel * bytes_per_elem)
        else:
            ring = 0
        self.all_gather.ring_bytes += ring

    def record_barrier(self) -> None:
        self.barrier_calls += 1

    @property
    def total_calls(self) -> int:
        return self.all_reduce.calls + self.reduce_scatter.calls + self.all_gather.calls

    @property
    def total_elems(self) -> int:
        return self.all_reduce.elems + self.reduce_scatter.elems + self.all_gather.elems

    @property
    def total_logical_bytes(self) -> int:
        return (
            self.all_reduce.logical_bytes
            + self.reduce_scatter.logical_bytes
            + self.all_gather.logical_bytes
        )

    @property
    def total_ring_bytes(self) -> int:
        return (
            self.all_reduce.ring_bytes + self.reduce_scatter.ring_bytes + self.all_gather.ring_bytes
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "world_size": self.world_size,
            "total_calls": self.total_calls,
            "total_elems": self.total_elems,
            "total_logical_bytes": self.total_logical_bytes,
            "total_ring_bytes": self.total_ring_bytes,
            "all_reduce": {
                "calls": self.all_reduce.calls,
                "elems": self.all_reduce.elems,
                "logical_bytes": self.all_reduce.logical_bytes,
                "ring_bytes": self.all_reduce.ring_bytes,
            },
            "reduce_scatter": {
                "calls": self.reduce_scatter.calls,
                "elems": self.reduce_scatter.elems,
                "logical_bytes": self.reduce_scatter.logical_bytes,
                "ring_bytes": self.reduce_scatter.ring_bytes,
            },
            "all_gather": {
                "calls": self.all_gather.calls,
                "elems": self.all_gather.elems,
                "logical_bytes": self.all_gather.logical_bytes,
                "ring_bytes": self.all_gather.ring_bytes,
            },
            "barrier_calls": self.barrier_calls,
            "notes": list(self.notes),
        }
