"""Distributed communication backend backed by torch.distributed (Gloo CPU)."""

from __future__ import annotations

import torch
import torch.distributed as dist

from zerosim.comm.base import Comm
from zerosim.comm.stats import CommStats


class DistComm(Comm):
    """Communicator backed by torch.distributed process groups."""

    def __init__(
        self,
        comm_bytes_per_elem: int = 2,
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError(
                "torch.distributed must be initialized before instantiating DistComm."
            )
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.comm_bytes = comm_bytes_per_elem
        self.stats = CommStats(rank=self.rank, world_size=self.world_size)

    @torch.no_grad()
    def all_reduce_mean(self, t: torch.Tensor) -> torch.Tensor:
        numel = t.numel()
        out = t.detach().clone().contiguous()
        dist.all_reduce(out, op=dist.ReduceOp.SUM)
        out.div_(float(self.world_size))
        self.stats.record_all_reduce(numel=numel, bytes_per_elem=self.comm_bytes)
        return out

    @torch.no_grad()
    def reduce_scatter_mean(self, t: torch.Tensor) -> torch.Tensor:
        full_t = t.detach().contiguous()
        full_numel = full_t.numel()
        shard_numel = full_numel // self.world_size
        reduced = full_t.clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced.div_(float(self.world_size))
        start = self.rank * shard_numel
        end = (self.rank + 1) * shard_numel
        shard = reduced[start:end].clone()
        self.stats.notes.append("Gloo fallback to all_reduce for reduce_scatter")
        self.stats.record_reduce_scatter(full_numel=full_numel, bytes_per_elem=self.comm_bytes)
        return shard

    @torch.no_grad()
    def all_gather(self, shard: torch.Tensor) -> torch.Tensor:
        s = shard.detach().contiguous()
        shard_numel = s.numel()
        full_numel = shard_numel * self.world_size
        tensor_list = [torch.empty_like(s) for _ in range(self.world_size)]
        dist.all_gather(tensor_list, s)
        gathered = torch.cat(tensor_list, dim=0)
        self.stats.record_all_gather(full_numel=full_numel, bytes_per_elem=self.comm_bytes)
        return gathered

    @torch.no_grad()
    def barrier(self) -> None:
        dist.barrier()
        self.stats.record_barrier()
