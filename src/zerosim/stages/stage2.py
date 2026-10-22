"""Stage 2: Optimizer State + Gradient Partitioning (Pos+g) with eager reduce-scatter."""

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


class Stage2(BaseStage):
    """ZeRO Stage 2: shards optimizer states and gradients; reduces gradients eagerly during backward."""

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
        self.grad_shards: list[torch.Tensor] = []
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

            # Gradient shard buffer
            g_shard = torch.zeros_like(shard_w)
            self.grad_shards.append(g_shard)
            self.ledger.alloc(
                f"grad_shard_{u_name}",
                MemoryCategory.GRAD,
                unit.shard_numel,
                self.cfg.policy.grad_bytes,
            )

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

        self._hook_handles = []
        self._register_gradient_hooks()

    def _register_gradient_hooks(self) -> None:
        """Register deterministic post-accumulate grad hooks for eager unit reduce-scatter."""
        self._param_counts: dict[str, int] = {}
        self._total_params_per_unit = {
            u_name: len(list(module.parameters())) for u_name, module in self.unit_modules
        }

        for i, (unit, (u_name, module)) in enumerate(zip(self.units, self.unit_modules)):
            for p in module.parameters():
                handle = p.register_post_accumulate_grad_hook(self._make_grad_hook(i, unit, u_name))
                self._hook_handles.append(handle)

    def _make_grad_hook(self, unit_idx: int, unit, u_name: str):
        def hook(param: torch.nn.Parameter) -> None:
            self._param_counts[u_name] = self._param_counts.get(u_name, 0) + 1
            if self._param_counts[u_name] == self._total_params_per_unit[u_name]:
                # All parameter gradients for this unit are accumulated!
                module = self.unit_modules[unit_idx][1]
                raw_grads = [
                    p.grad if p.grad is not None else torch.zeros_like(p)
                    for p in module.parameters()
                ]

                # Transient live unit gradient buffer in ledger
                live_name = f"live_grad_{u_name}"
                self.ledger.alloc(
                    live_name,
                    MemoryCategory.GRAD,
                    unit.padded_numel,
                    self.cfg.policy.grad_bytes,
                )

                flat_g = flatten_unit_tensors(raw_grads, unit)
                # Reduce-scatter immediately across ranks
                reduced_shard = self.comm.reduce_scatter_mean(flat_g)
                self.grad_shards[unit_idx].copy_(reduced_shard)

                # Free raw gradients on module parameters
                for p in module.parameters():
                    p.grad = None

                # Free transient live unit gradient buffer
                self.ledger.free(live_name)

        return hook

    def step(self, step_idx: int, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        self.ledger.mark_phase(step_idx, "forward")
        self.model.zero_grad(set_to_none=True)
        self._param_counts.clear()

        with self.compute_scope():
            with torch.autograd.graph.saved_tensors_hooks(
                self._pack_activation_hook, self._unpack_activation_hook
            ):
                logits = self.model(inputs)
                loss = compute_loss(logits, targets)
            loss_val = loss.item()

        self.ledger.mark_phase(step_idx, "backward")
        # In Stage 2, backward invokes collective reduce-scatter inside hooks.
        # Compute semaphore MUST NOT be held around backward to prevent deadlock across collectives.
        loss.backward()

        self._free_saved_activations()

        # Optimizer phase: update parameter shards
        self.ledger.mark_phase(step_idx, "optimizer")
        with self.compute_scope():
            for i, opt in enumerate(self.optimizers):
                opt.step([self.grad_shards[i]])

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
