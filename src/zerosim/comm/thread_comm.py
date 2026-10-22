"""Thread-based SPMD communication coordinator and virtual rank communicator."""

from __future__ import annotations

import threading
from typing import Any

import torch

from zerosim.comm.base import CollectiveMismatchError, Comm
from zerosim.comm.stats import CommStats


class ThreadCoordinator:
    """Coordinates collective communication across virtual rank worker threads."""

    def __init__(self, world_size: int, timeout: float = 120.0) -> None:
        self.world_size = world_size
        self.timeout = timeout

        self.barrier_entry = threading.Barrier(world_size)
        self.barrier_exit = threading.Barrier(world_size)

        self.slots: list[torch.Tensor | None] = [None] * world_size
        self.signatures: list[tuple[str, int, int, torch.dtype] | None] = [None] * world_size
        self.outputs: list[torch.Tensor | None] = [None] * world_size
        self._error: Exception | None = None

    def abort(self, exc: Exception) -> None:
        """Abort coordinator barriers upon thread failure."""
        self._error = exc
        self.barrier_entry.abort()
        self.barrier_exit.abort()

    def sync_collective(
        self,
        rank: int,
        op_name: str,
        seq_id: int,
        input_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Execute synchronized collective across all virtual rank threads."""
        if self._error is not None:
            raise self._error

        sig = (op_name, seq_id, input_tensor.numel(), input_tensor.dtype)
        self.signatures[rank] = sig
        self.slots[rank] = input_tensor

        try:
            self.barrier_entry.wait(self.timeout)
        except threading.BrokenBarrierError as e:
            raise RuntimeError(f"ThreadComm collective aborted: {self._error or e}") from e

        # Rank 0 verifies signatures and computes output
        if rank == 0:
            try:
                base_sig = self.signatures[0]
                mismatches = []
                for r, s in enumerate(self.signatures):
                    if s != base_sig:
                        mismatches.append((r, s))
                if mismatches:
                    err_msg = (
                        f"Collective signature mismatch for op '{op_name}' (seq {seq_id}): "
                        f"Rank 0 expected {base_sig}, but mismatches occurred: {mismatches}"
                    )
                    raise CollectiveMismatchError(err_msg)

                # Execute collective computation
                if op_name == "all_reduce_mean":
                    # Sum in ascending rank order
                    stacked = torch.stack([self.slots[r] for r in range(self.world_size)], dim=0)  # type: ignore
                    reduced = torch.sum(stacked, dim=0) / float(self.world_size)
                    for r in range(self.world_size):
                        self.outputs[r] = reduced.clone()

                elif op_name == "reduce_scatter_mean":
                    full_len = input_tensor.numel()
                    if full_len % self.world_size != 0:
                        raise ValueError(
                            f"Tensor numel {full_len} must be divisible by world_size {self.world_size}"
                        )
                    shard_len = full_len // self.world_size
                    # Stack full inputs and reduce
                    stacked = torch.stack([self.slots[r] for r in range(self.world_size)], dim=0)  # type: ignore
                    reduced_full = torch.sum(stacked, dim=0) / float(self.world_size)
                    for r in range(self.world_size):
                        start = r * shard_len
                        end = (r + 1) * shard_len
                        self.outputs[r] = reduced_full[start:end].clone()

                elif op_name == "all_gather":
                    # Concatenate shards in ascending rank order
                    shards = [self.slots[r] for r in range(self.world_size)]  # type: ignore
                    gathered = torch.cat(shards, dim=0)
                    for r in range(self.world_size):
                        self.outputs[r] = gathered.clone()

                elif op_name == "barrier":
                    for r in range(self.world_size):
                        self.outputs[r] = input_tensor

                else:
                    raise ValueError(f"Unknown collective operation '{op_name}'")

            except Exception as exc:
                self.abort(exc)
                raise exc

        try:
            self.barrier_exit.wait(self.timeout)
        except threading.BrokenBarrierError as e:
            raise RuntimeError(f"ThreadComm collective aborted: {self._error or e}") from e

        out = self.outputs[rank]
        if out is None:
            raise RuntimeError(f"Output for rank {rank} is None after collective")
        return out


class ThreadComm(Comm):
    """Communicator bound to a single virtual rank backed by ThreadCoordinator."""

    def __init__(
        self,
        rank: int,
        coordinator: ThreadCoordinator,
        comm_bytes_per_elem: int = 2,
    ) -> None:
        self.rank = rank
        self.world_size = coordinator.world_size
        self.coordinator = coordinator
        self.comm_bytes = comm_bytes_per_elem
        self.stats = CommStats(rank=rank, world_size=self.world_size)
        self._seq = 0

    def all_reduce_mean(self, t: torch.Tensor) -> torch.Tensor:
        self._seq += 1
        numel = t.numel()
        res = self.coordinator.sync_collective(
            rank=self.rank,
            op_name="all_reduce_mean",
            seq_id=self._seq,
            input_tensor=t,
        )
        self.stats.record_all_reduce(numel=numel, bytes_per_elem=self.comm_bytes)
        return res

    def reduce_scatter_mean(self, t: torch.Tensor) -> torch.Tensor:
        self._seq += 1
        numel = t.numel()
        res = self.coordinator.sync_collective(
            rank=self.rank,
            op_name="reduce_scatter_mean",
            seq_id=self._seq,
            input_tensor=t,
        )
        self.stats.record_reduce_scatter(full_numel=numel, bytes_per_elem=self.comm_bytes)
        return res

    def all_gather(self, shard: torch.Tensor) -> torch.Tensor:
        self._seq += 1
        shard_numel = shard.numel()
        full_numel = shard_numel * self.world_size
        res = self.coordinator.sync_collective(
            rank=self.rank,
            op_name="all_gather",
            seq_id=self._seq,
            input_tensor=shard,
        )
        self.stats.record_all_gather(full_numel=full_numel, bytes_per_elem=self.comm_bytes)
        return res

    def barrier(self) -> None:
        self._seq += 1
        dummy = torch.empty(0, dtype=torch.uint8)
        self.coordinator.sync_collective(
            rank=self.rank,
            op_name="barrier",
            seq_id=self._seq,
            input_tensor=dummy,
        )
        self.stats.record_barrier()
