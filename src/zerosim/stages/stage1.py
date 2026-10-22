"""Stage 1: Optimizer State Partitioning (Pos) with sharded optimizer states."""

from __future__ import annotations

import torch

from zerosim.comm.base import Comm
from zerosim.config import SimConfig
from zerosim.data import compute_loss
from zerosim.governor import ResourceGovernor
from zerosim.ledger import MemoryCategory, MemoryLedger
from zerosim.optim import FlatShardAdamW
from zerosim.sharding import (
    flatten_unit_tensors,
    get_shard_slice,
    unflatten_unit_tensor,
)
from zerosim.stages.base import BaseStage


class Stage1(BaseStage):
    """ZeRO Stage 1: shards optimizer states (M, m, v); parameters and grads remain full."""

    def __init__(
        self,
        cfg: SimConfig,
        rank: int,
        comm: Comm,
        ledger: MemoryLedger,
        governor: ResourceGovernor | None = None,
    ) -> None:
        super().__init__(cfg, rank, comm, ledger, governor)
        self.ledger.mark_phase(0, "init")

        self.unit_modules = self.model.get_unit_modules()
        self.optimizers: list[FlatShardAdamW] = []
        self.param_shards: list[torch.Tensor] = []
        self.flat_params: list[torch.Tensor] = []
        self.master_shards: list[torch.Tensor] | None = (
            [] if self.cfg.policy.master_bytes > 0 else None
        )

        for unit, (u_name, module) in zip(self.units, self.unit_modules):
            params = [p for _, p in module.named_parameters()]
            flat_w = flatten_unit_tensors(params, unit)
            self.flat_params.append(flat_w)

            # Full parameters in ledger
            self.ledger.alloc(
                f"param_{u_name}",
                MemoryCategory.PARAM,
                unit.padded_numel,
                self.cfg.policy.param_bytes,
            )

            # Extract rank's shard slice
            sl = get_shard_slice(unit, self.rank)
            shard_w = flat_w[sl]
            self.param_shards.append(shard_w)

            # Master shard in ledger (if policy dictates)
            if self.master_shards is not None:
                m_shard = shard_w.clone().to(
                    dtype=torch.float32 if self.dtype != torch.float64 else torch.float64
                )
                self.master_shards.append(m_shard)
                self.ledger.alloc(
                    f"master_shard_{u_name}",
                    MemoryCategory.MASTER,
                    unit.shard_numel,
                    self.cfg.policy.master_bytes,
                )

            # Sharded optimizer states m and v in ledger
            self.ledger.alloc(
                f"opt_m_shard_{u_name}",
                MemoryCategory.OPT_M,
                unit.shard_numel,
                self.cfg.policy.m_bytes,
            )
            self.ledger.alloc(
                f"opt_v_shard_{u_name}",
                MemoryCategory.OPT_V,
                unit.shard_numel,
                self.cfg.policy.v_bytes,
            )

        # Initialize flat AdamW optimizer per shard
        for i, shard_w in enumerate(self.param_shards):
            m_list = [self.master_shards[i]] if self.master_shards is not None else None
            opt = FlatShardAdamW(
                [shard_w],
                lr=self.cfg.lr,
                betas=self.cfg.betas,
                eps=self.cfg.eps,
                weight_decay=self.cfg.weight_decay,
                master_params=m_list,
            )
            self.optimizers.append(opt)

    def step(self, step_idx: int, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        self.ledger.mark_phase(step_idx, "forward")
        self.model.zero_grad(set_to_none=True)

        with self.compute_scope():
            with torch.autograd.graph.saved_tensors_hooks(
                self._pack_activation_hook, self._unpack_activation_hook
            ):
                logits = self.model(inputs)
                loss = compute_loss(logits, targets)
            loss_val = loss.item()

        self.ledger.mark_phase(step_idx, "backward")
        with self.compute_scope():
            loss.backward()

        self._free_saved_activations()

        # In Stage 1: Full gradients are retained across all units until end of backward
        for unit, (u_name, _) in zip(self.units, self.unit_modules):
            self.ledger.alloc(
                f"grad_{u_name}",
                MemoryCategory.GRAD,
                unit.padded_numel,
                self.cfg.policy.grad_bytes,
            )

        # Communication phase: reduce-scatter gradients unit by unit, then free full gradient
        self.ledger.mark_phase(step_idx, "comm")
        grad_shards: list[torch.Tensor] = []
        for unit, (u_name, module) in zip(self.units, self.unit_modules):
            raw_grads = [
                p.grad if p.grad is not None else torch.zeros_like(p)
                for _, p in module.named_parameters()
            ]
            flat_g = flatten_unit_tensors(raw_grads, unit)
            # Reduce-scatter returns this rank's shard
            g_shard = self.comm.reduce_scatter_mean(flat_g)
            grad_shards.append(g_shard)
            # Free full unit gradient buffer immediately after reduce-scatter
            self.ledger.free(f"grad_{u_name}")

        # Optimizer phase: update parameter shards
        self.ledger.mark_phase(step_idx, "optimizer")
        with self.compute_scope():
            for i, opt in enumerate(self.optimizers):
                opt.step([grad_shards[i]])

        # Gather updated parameters to restore full parameters for all ranks
        self.ledger.mark_phase(step_idx, "comm")
        for i, (unit, (u_name, module)) in enumerate(zip(self.units, self.unit_modules)):
            updated_full = self.comm.all_gather(self.param_shards[i])
            self.flat_params[i].copy_(updated_full)
            updated_params = unflatten_unit_tensor(self.flat_params[i], unit)
            with self.compute_scope():
                for p, u_p in zip(module.parameters(), updated_params):
                    p.data.copy_(u_p.data)

        self.ledger.mark_phase(step_idx, "post_step")
        return loss_val

    def get_full_parameters(self) -> torch.Tensor:
        all_tensors = [p.detach().reshape(-1) for p in self.model.parameters()]
        return torch.cat(all_tensors)


class open_noop:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None
