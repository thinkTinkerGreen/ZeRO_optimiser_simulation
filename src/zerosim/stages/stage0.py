"""Stage 0: Baseline Data Parallelism (DDP) with full replication of model states."""

from __future__ import annotations

import torch

from zerosim.comm.base import Comm
from zerosim.config import SimConfig
from zerosim.data import compute_loss
from zerosim.governor import ResourceGovernor
from zerosim.ledger import MemoryCategory, MemoryLedger
from zerosim.optim import FlatShardAdamW
from zerosim.sharding import flatten_unit_tensors, unflatten_unit_tensor
from zerosim.stages.base import BaseStage


class Stage0(BaseStage):
    """Standard DDP: replicates parameters, gradients, and optimizer states."""

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
        self.flat_params: list[torch.Tensor] = []
        self.master_params: list[torch.Tensor] | None = (
            [] if self.cfg.policy.master_bytes > 0 else None
        )

        # Allocate persistent model states in ledger
        for unit, (u_name, module) in zip(self.units, self.unit_modules):
            params = [p for _, p in module.named_parameters()]
            flat_w = flatten_unit_tensors(params, unit)
            self.flat_params.append(flat_w)

            # Record in ledger
            self.ledger.alloc(
                f"param_{u_name}",
                MemoryCategory.PARAM,
                unit.padded_numel,
                self.cfg.policy.param_bytes,
            )

            # Master parameters (if policy dictates)
            if self.master_params is not None:
                m_p = flat_w.clone().to(
                    dtype=torch.float32 if self.dtype != torch.float64 else torch.float64
                )
                self.master_params.append(m_p)
                self.ledger.alloc(
                    f"master_{u_name}",
                    MemoryCategory.MASTER,
                    unit.padded_numel,
                    self.cfg.policy.master_bytes,
                )

            # Optimizer states m and v
            self.ledger.alloc(
                f"opt_m_{u_name}",
                MemoryCategory.OPT_M,
                unit.padded_numel,
                self.cfg.policy.m_bytes,
            )
            self.ledger.alloc(
                f"opt_v_{u_name}",
                MemoryCategory.OPT_V,
                unit.padded_numel,
                self.cfg.policy.v_bytes,
            )

        # Initialize flat AdamW optimizer per unit
        for i, flat_w in enumerate(self.flat_params):
            m_list = [self.master_params[i]] if self.master_params is not None else None
            opt = FlatShardAdamW(
                [flat_w],
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

        # Free activation memory after backward completes
        self._free_saved_activations()

        # Allocate full gradients in ledger
        for unit, (u_name, _) in zip(self.units, self.unit_modules):
            self.ledger.alloc(
                f"grad_{u_name}",
                MemoryCategory.GRAD,
                unit.padded_numel,
                self.cfg.policy.grad_bytes,
            )

        # Communication phase: all-reduce gradients per unit
        self.ledger.mark_phase(step_idx, "comm")
        reduced_grads: list[torch.Tensor] = []
        for unit, (u_name, module) in zip(self.units, self.unit_modules):
            raw_grads = [
                p.grad if p.grad is not None else torch.zeros_like(p)
                for _, p in module.named_parameters()
            ]
            flat_g = flatten_unit_tensors(raw_grads, unit)
            # All-reduce averaging across all ranks
            g_reduced = self.comm.all_reduce_mean(flat_g)
            reduced_grads.append(g_reduced)

        # Optimizer step
        self.ledger.mark_phase(step_idx, "optimizer")
        with self.compute_scope():
            for i, (unit, (u_name, module)) in enumerate(zip(self.units, self.unit_modules)):
                self.optimizers[i].step([reduced_grads[i]])
                # Unflatten updated flat_w back into module parameters
                updated_params = unflatten_unit_tensor(self.flat_params[i], unit)
                for p, u_p in zip(module.parameters(), updated_params):
                    p.data.copy_(u_p.data)

        # Free gradients
        for unit, (u_name, _) in zip(self.units, self.unit_modules):
            self.ledger.free(f"grad_{u_name}")

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
