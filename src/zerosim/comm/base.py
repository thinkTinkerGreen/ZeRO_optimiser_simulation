"""Abstract communication protocol for ZeRO simulation backends."""

from __future__ import annotations

from typing import Protocol

import torch

from zerosim.comm.stats import CommStats


class CollectiveMismatchError(RuntimeError):
    """Raised when ranks invoke desynchronized collective operations."""

    pass


class Comm(Protocol):
    """Protocol for distributed communication backends."""

    rank: int
    world_size: int
    stats: CommStats

    def all_reduce_mean(self, t: torch.Tensor) -> torch.Tensor:
        """All-reduce averaging across all ranks. Returns reduced tensor of same shape."""
        ...

    def reduce_scatter_mean(self, t: torch.Tensor) -> torch.Tensor:
        """Reduce-scatter averaging across all ranks. Returns this rank's shard."""
        ...

    def all_gather(self, shard: torch.Tensor) -> torch.Tensor:
        """All-gather concatenating shards from all ranks in ascending rank order."""
        ...

    def barrier(self) -> None:
        """Synchronize all ranks."""
        ...
