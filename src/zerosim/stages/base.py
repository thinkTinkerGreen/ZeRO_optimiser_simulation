"""Base class and common infrastructure for ZeRO stages."""

from __future__ import annotations

import abc
from typing import Any

import torch
import torch.nn as nn

from zerosim.comm.base import Comm
from zerosim.config import SimConfig
from zerosim.governor import ResourceGovernor
from zerosim.ledger import MemoryCategory, MemoryLedger
from zerosim.model import GPT
from zerosim.sharding import FlatUnit


class open_noop:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None


class BaseStage(abc.ABC):
    """Abstract base class for SPMD rank stage execution."""

    def __init__(
        self,
        cfg: SimConfig,
        rank: int,
        comm: Comm,
        ledger: MemoryLedger,
        governor: ResourceGovernor | None = None,
    ) -> None:
        self.cfg = cfg
        self.rank = rank
        self.comm = comm
        self.ledger = ledger
        self.governor = governor
        self.device = cfg.device
        self.dtype = torch.float64 if cfg.compute_dtype == "float64" else torch.float32

        # Build local model
        # Every rank creates model with same seed Generator to start with identical weights
        from zerosim.determinism import get_generator

        gen = get_generator(cfg.seed, rank=0, step=0)
        self.model = GPT(cfg.model, generator=gen, dtype=self.dtype).to(self.device)
        self.units = self.model.get_flat_units(world_size=cfg.world_size)

        # Hook to count saved activation tensors
        self._saved_activation_names: list[str] = []
        self._saved_act_bytes = 0

    def compute_scope(self):
        if self.governor is not None:
            return self.governor.compute_scope()
        return open_noop()

    def _pack_activation_hook(self, x: torch.Tensor) -> torch.Tensor:
        b = x.numel() * x.element_size()
        name = f"act_{len(self._saved_activation_names)}"
        self._saved_activation_names.append(name)
        self.ledger.alloc(
            name=name,
            category=MemoryCategory.ACTIVATION,
            tensor_or_numel=x,
            logical_bytes_per_elem=self.cfg.policy.param_bytes,
        )
        return x

    def _unpack_activation_hook(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def _free_saved_activations(self) -> None:
        for name in reversed(self._saved_activation_names):
            try:
                self.ledger.free(name)
            except KeyError:
                pass
        self._saved_activation_names.clear()

    @abc.abstractmethod
    def step(self, step_idx: int, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        """Execute one training step and return the step loss."""
        pass

    @abc.abstractmethod
    def get_full_parameters(self) -> torch.Tensor:
        """Return the complete gathered parameter vector of the model."""
        pass
