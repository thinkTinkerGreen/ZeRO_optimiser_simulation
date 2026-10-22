"""Flat-shard AdamW optimizer with decoupled weight decay and master weights."""

from __future__ import annotations

import math

import torch


class FlatShardAdamW:
    """AdamW optimizer operating directly on contiguous 1-D shard buffers."""

    def __init__(
        self,
        params: list[torch.Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        master_params: list[torch.Tensor] | None = None,
    ) -> None:
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.master_params = master_params
        self.step_count = 0

        # Optimizer moment states
        self.exp_avg: list[torch.Tensor] = [
            torch.zeros_like(p, dtype=torch.float32 if p.dtype != torch.float64 else torch.float64)
            for p in self.params
        ]
        self.exp_avg_sq: list[torch.Tensor] = [
            torch.zeros_like(p, dtype=torch.float32 if p.dtype != torch.float64 else torch.float64)
            for p in self.params
        ]

    @torch.no_grad()
    def step(self, grads: list[torch.Tensor]) -> None:
        """Execute one AdamW optimization step on all parameter shards."""
        self.step_count += 1
        bias_correction1 = 1.0 - (self.beta1**self.step_count)
        bias_correction2 = 1.0 - (self.beta2**self.step_count)
        step_size = self.lr / bias_correction1

        for i, (p, g) in enumerate(zip(self.params, grads)):
            if g is None:
                continue

            target_p = self.master_params[i] if self.master_params is not None else p
            m = self.exp_avg[i]
            v = self.exp_avg_sq[i]

            # Perform decoupled weight decay
            if self.weight_decay != 0.0:
                target_p.mul_(1.0 - self.lr * self.weight_decay)

            # Decay the first and second moment running average coefficient
            m.mul_(self.beta1).add_(g, alpha=1.0 - self.beta1)
            v.mul_(self.beta2).addcmul_(g, g, value=1.0 - self.beta2)

            denom = (v.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
            target_p.addcdiv_(m, denom, value=-step_size)

            if self.master_params is not None:
                p.copy_(target_p)
