"""Threaded SPMD runner orchestrating N virtual GPU ranks."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

import torch

from zerosim.comm.stats import CommStats
from zerosim.comm.thread_comm import ThreadComm, ThreadCoordinator
from zerosim.config import SimConfig
from zerosim.data import generate_batch, get_rank_batch
from zerosim.governor import ResourceGovernor
from zerosim.ledger import MemoryLedger
from zerosim.model import count_params
from zerosim.stages.base import BaseStage
from zerosim.stages.stage0 import Stage0
from zerosim.stages.stage1 import Stage1
from zerosim.stages.stage2 import Stage2
from zerosim.stages.stage3 import Stage3


@dataclass
class SimulationResult:
    config: SimConfig
    losses_per_step: list[float]  # mean across ranks per step
    losses_per_rank: list[list[float]]
    ledgers: list[MemoryLedger]
    comm_stats: list[CommStats]
    final_parameters: torch.Tensor
    simulator_wall_clock_s: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "config_hash": self.config.config_hash,
            "losses_per_step": [round(l, 8) for l in self.losses_per_step],
            "simulator_wall_clock_s": round(self.simulator_wall_clock_s, 4),
            "rank0_peak_logical": self.ledgers[0].peak_logical,
            "rank0_peak_actual": self.ledgers[0].peak_actual,
            "rank0_peak_model_state_logical": self.ledgers[0].peak_model_state_logical,
            "rank0_peak_by_category": dict(self.ledgers[0].peak_by_category),
            "rank0_comm_stats": self.comm_stats[0].to_dict(),
        }


def get_stage_cls(stage_num: int) -> type[BaseStage]:
    if stage_num == 0:
        return Stage0
    elif stage_num == 1:
        return Stage1
    elif stage_num == 2:
        return Stage2
    elif stage_num == 3:
        return Stage3
    raise ValueError(f"Invalid ZeRO stage: {stage_num}")


def run_simulation(cfg: SimConfig) -> SimulationResult:
    """Launch N virtual rank threads executing the configured ZeRO stage."""
    approx_numel = count_params(cfg.model)
    governor = ResourceGovernor(max_concurrent=cfg.max_concurrent)
    governor.plan_memory(cfg, approx_numel=approx_numel)

    coordinator = ThreadCoordinator(world_size=cfg.world_size)
    stage_cls = get_stage_cls(cfg.stage)

    ledgers = [MemoryLedger(rank=r) for r in range(cfg.world_size)]
    comms = [
        ThreadComm(
            rank=r,
            coordinator=coordinator,
            comm_bytes_per_elem=cfg.policy.comm_param_bytes,
        )
        for r in range(cfg.world_size)
    ]

    losses_per_rank: list[list[float]] = [[] for _ in range(cfg.world_size)]
    final_params: list[torch.Tensor | None] = [None] * cfg.world_size
    thread_exceptions: list[Exception | None] = [None] * cfg.world_size
    final_comm_stats: list[CommStats | None] = [None] * cfg.world_size

    def rank_worker(rank: int) -> None:
        try:
            # Set single thread inside rank worker for numerical determinism
            torch.set_num_threads(1)
            stage_instance = stage_cls(
                cfg=cfg,
                rank=rank,
                comm=comms[rank],
                ledger=ledgers[rank],
                governor=governor,
            )

            for step_idx in range(cfg.steps):
                # Deterministic synthetic batch generated identically across ranks
                glob_inputs, glob_targets = generate_batch(cfg, step=step_idx, device=cfg.device)
                r_inp, r_tar = get_rank_batch(
                    glob_inputs,
                    glob_targets,
                    rank=rank,
                    microbatch=cfg.microbatch,
                )
                step_loss = stage_instance.step(step_idx, r_inp, r_tar)
                losses_per_rank[rank].append(step_loss)

            # Snapshot training communication stats before diagnostic parameter gathering
            import copy
            final_comm_stats[rank] = copy.deepcopy(stage_instance.comm.stats)

            # Extract full parameters at completion for numerical verification
            final_params[rank] = stage_instance.get_full_parameters()
        except Exception as exc:
            thread_exceptions[rank] = exc
            coordinator.abort(exc)

    start_time = time.perf_counter()
    threads = [
        threading.Thread(target=rank_worker, args=(r,), name=f"rank-{r}")
        for r in range(cfg.world_size)
    ]

    for th in threads:
        th.start()
    for th in threads:
        th.join()
    elapsed = time.perf_counter() - start_time

    # Propagate any worker thread exception
    for r, exc in enumerate(thread_exceptions):
        if exc is not None:
            raise RuntimeError(f"Virtual rank {r} failed with exception: {exc}") from exc

    # Compute mean loss per step across ranks
    mean_losses: list[float] = []
    for step_idx in range(cfg.steps):
        step_mean = sum(losses_per_rank[r][step_idx] for r in range(cfg.world_size)) / float(
            cfg.world_size
        )
        mean_losses.append(step_mean)

    rank0_params = final_params[0]
    if rank0_params is None:
        raise RuntimeError("Rank 0 failed to return final parameters")

    return SimulationResult(
        config=cfg,
        losses_per_step=mean_losses,
        losses_per_rank=losses_per_rank,
        ledgers=ledgers,
        comm_stats=[s for s in final_comm_stats if s is not None],
        final_parameters=rank0_params,
        simulator_wall_clock_s=elapsed,
    )
