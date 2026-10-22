"""Stage 3: Full Sharding (Pos+g+p) with layer-by-layer parameter all-gather and backward recomputation."""

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


class Stage3(BaseStage):
    """ZeRO Stage 3: Full sharding of parameters, gradients, and optimizer states."""

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
        self.grad_shards: list[torch.Tensor] = []
        self.master_shards: list[torch.Tensor] | None = (
            [] if self.cfg.policy.master_bytes > 0 else None
        )

        for unit, (u_name, module) in zip(self.units, self.unit_modules):
            params = [p for _, p in module.named_parameters()]
            flat_w = flatten_unit_tensors(params, unit)
            sl = get_shard_slice(unit, self.rank)
            shard_w = flat_w[sl].clone()
            self.param_shards.append(shard_w)

            # In Stage 3, ONLY the shard is held persistently
            self.ledger.alloc(
                f"param_shard_{u_name}",
                MemoryCategory.PARAM,
                unit.shard_numel,
                self.cfg.policy.param_bytes,
            )

            g_shard = torch.zeros_like(shard_w)
            self.grad_shards.append(g_shard)
            self.ledger.alloc(
                f"grad_shard_{u_name}",
                MemoryCategory.GRAD,
                unit.shard_numel,
                self.cfg.policy.grad_bytes,
            )

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

        unit_inputs: list[torch.Tensor] = []
        with torch.autograd.graph.saved_tensors_hooks(
            self._pack_activation_hook, self._unpack_activation_hook
        ):
            x = inputs
            for idx in range(len(self.units)):
                unit_inputs.append(x)
                unit = self.units[idx]
                u_name, module = self.unit_modules[idx]

                # Allocate transient gathered parameter in ledger
                live_param_name = f"live_param_{u_name}"
                self.ledger.alloc(
                    live_param_name,
                    MemoryCategory.PARAM,
                    unit.padded_numel,
                    self.cfg.policy.param_bytes,
                )

                # All-gather parameter shards from all ranks
                flat_w = self.comm.all_gather(self.param_shards[idx])
                params = unflatten_unit_tensor(flat_w, unit)

                # Load gathered parameters into module
                for p, w in zip(module.parameters(), params):
                    p.data.copy_(w.data)

                # Execute unit forward
                with self.compute_scope():
                    with torch.enable_grad():
                        out = module(x)
                    out = out.detach()

                # Free transient gathered parameter immediately after forward execution
                self.ledger.free(live_param_name)
                x = out

            logits = x.requires_grad_(True)
            with self.compute_scope():
                loss = compute_loss(logits, targets)
        loss_val = loss.item()

        self.ledger.mark_phase(step_idx, "backward")
        (grad_output,) = torch.autograd.grad(loss, logits)

        for idx in reversed(range(len(self.units))):
            unit = self.units[idx]
            u_name, module = self.unit_modules[idx]
            unit_in = unit_inputs[idx]

            # Transient live gathered parameters in ledger
            live_param_name = f"live_param_{u_name}"
            self.ledger.alloc(
                live_param_name,
                MemoryCategory.PARAM,
                unit.padded_numel,
                self.cfg.policy.param_bytes,
            )

            # All-gather parameter shards again for backward
            flat_w = self.comm.all_gather(self.param_shards[idx])
            params = unflatten_unit_tensor(flat_w, unit)

            for p, w in zip(module.parameters(), params):
                p.data.copy_(w.data)
                p.requires_grad = True

            # Recompute unit forward pass under enable_grad
            if unit_in.is_floating_point():
                u_in = unit_in.detach().requires_grad_(True)
            else:
                u_in = unit_in

            with self.compute_scope():
                with torch.enable_grad():
                    out = module(u_in)

                # Compute gradients with respect to parameters and unit input
                if u_in.requires_grad:
                    grads = torch.autograd.grad(
                        out,
                        list(module.parameters()) + [u_in],
                        grad_output,
                    )
                    param_grads = grads[:-1]
                    input_grad = grads[-1]
                else:
                    grads = torch.autograd.grad(
                        out,
                        list(module.parameters()),
                        grad_output,
                    )
                    param_grads = grads
                    input_grad = None

            # Transient live gradient buffer in ledger
            live_grad_name = f"live_grad_{u_name}"
            self.ledger.alloc(
                live_grad_name,
                MemoryCategory.GRAD,
                unit.padded_numel,
                self.cfg.policy.grad_bytes,
            )

            flat_g = flatten_unit_tensors(param_grads, unit)
            # Reduce-scatter parameter gradients immediately
            reduced_shard = self.comm.reduce_scatter_mean(flat_g)
            self.grad_shards[idx].copy_(reduced_shard)

            # Free transient live gradient and parameter buffers
            self.ledger.free(live_grad_name)
            self.ledger.free(live_param_name)

            grad_output = input_grad

        self._free_saved_activations()

        # Optimizer phase: update parameter shards directly (no post-step all-gather)
        self.ledger.mark_phase(step_idx, "optimizer")
        with self.compute_scope():
            for i, opt in enumerate(self.optimizers):
                opt.step([self.grad_shards[i]])

        self.ledger.mark_phase(step_idx, "post_step")
        return loss_val

    def get_full_parameters(self) -> torch.Tensor:
        """Gather parameter shards across all units and ranks."""
        full_unit_tensors = []
        for i, unit in enumerate(self.units):
            gathered = self.comm.all_gather(self.param_shards[i])
            # Trim padding if any
            if unit.padding_elements > 0:
                gathered = gathered[: unit.numel]
            full_unit_tensors.append(gathered)
        return torch.cat(full_unit_tensors)


class open_noop:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None
