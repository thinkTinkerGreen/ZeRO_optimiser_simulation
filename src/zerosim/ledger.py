"""Memory ledger tracking logical policy allocations, actual device bytes, and timelines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch


class MemoryCategory(str, Enum):
    PARAM = "PARAM"
    GRAD = "GRAD"
    MASTER = "MASTER"
    OPT_M = "OPT_M"
    OPT_V = "OPT_V"
    ACTIVATION = "ACTIVATION"
    COMM_BUFFER = "COMM_BUFFER"
    TEMP = "TEMP"


MODEL_STATE_CATEGORIES = {
    MemoryCategory.PARAM,
    MemoryCategory.GRAD,
    MemoryCategory.MASTER,
    MemoryCategory.OPT_M,
    MemoryCategory.OPT_V,
}


@dataclass(frozen=True)
class TimelineEvent:
    step: int
    phase: str
    seq: int
    category: str
    logical_bytes_total: int
    actual_bytes_total: int
    delta_logical: int
    delta_actual: int
    name: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "phase": self.phase,
            "seq": self.seq,
            "category": self.category,
            "logical_bytes_total": self.logical_bytes_total,
            "actual_bytes_total": self.actual_bytes_total,
            "delta_logical": self.delta_logical,
            "delta_actual": self.delta_actual,
            "name": self.name,
        }


@dataclass
class AllocationRecord:
    name: str
    category: MemoryCategory
    numel: int
    logical_bytes: int
    actual_bytes: int


class MemoryLedger:
    """Per-rank ledger tracking memory allocations and phase timelines."""

    def __init__(self, rank: int = 0) -> None:
        self.rank = rank
        self.current_step = 0
        self.current_phase = "init"
        self._seq = 0

        self._active_allocations: dict[str, AllocationRecord] = {}
        self._timeline: list[TimelineEvent] = []

        # Current live bytes per category
        self._live_logical: dict[MemoryCategory, int] = dict.fromkeys(MemoryCategory, 0)
        self._live_actual: dict[MemoryCategory, int] = dict.fromkeys(MemoryCategory, 0)

        # Peak tracking
        self.peak_logical = 0
        self.peak_actual = 0
        self.peak_model_state_logical = 0
        self.peak_by_category: dict[str, int] = {cat.value: 0 for cat in MemoryCategory}

    def mark_phase(self, step: int, phase: str) -> None:
        """Mark step and phase transition (init, forward, backward, comm, optimizer, post_step)."""
        self.current_step = step
        self.current_phase = phase

    def alloc(
        self,
        name: str,
        category: MemoryCategory,
        tensor_or_numel: torch.Tensor | int,
        logical_bytes_per_elem: int,
        actual_bytes: int | None = None,
    ) -> None:
        """Allocate a named tensor entry in the ledger."""
        if name in self._active_allocations:
            raise ValueError(f"Allocation '{name}' already exists in rank {self.rank} ledger.")

        if isinstance(tensor_or_numel, int):
            numel = tensor_or_numel
            actual_b = numel * logical_bytes_per_elem if actual_bytes is None else actual_bytes
        else:
            numel = int(tensor_or_numel.numel())
            actual_b = (
                numel * tensor_or_numel.element_size() if actual_bytes is None else actual_bytes
            )

        logical_b = numel * logical_bytes_per_elem

        record = AllocationRecord(
            name=name,
            category=category,
            numel=numel,
            logical_bytes=logical_b,
            actual_bytes=actual_b,
        )
        self._active_allocations[name] = record

        self._live_logical[category] += logical_b
        self._live_actual[category] += actual_b

        current_tot_log = sum(self._live_logical.values())
        current_tot_act = sum(self._live_actual.values())

        # Update category peaks
        if self._live_logical[category] > self.peak_by_category[category.value]:
            self.peak_by_category[category.value] = self._live_logical[category]

        # Update overall peaks
        if current_tot_log > self.peak_logical:
            self.peak_logical = current_tot_log
        if current_tot_act > self.peak_actual:
            self.peak_actual = current_tot_act

        # Update model state peak
        model_state_log = sum(self._live_logical[cat] for cat in MODEL_STATE_CATEGORIES)
        if model_state_log > self.peak_model_state_logical:
            self.peak_model_state_logical = model_state_log

        self._seq += 1
        self._timeline.append(
            TimelineEvent(
                step=self.current_step,
                phase=self.current_phase,
                seq=self._seq,
                category=category.value,
                logical_bytes_total=current_tot_log,
                actual_bytes_total=current_tot_act,
                delta_logical=logical_b,
                delta_actual=actual_b,
                name=name,
            )
        )

    def free(self, name: str) -> None:
        """Free a previously allocated named entry."""
        if name not in self._active_allocations:
            raise KeyError(f"Attempted to free unknown allocation '{name}' on rank {self.rank}.")

        record = self._active_allocations.pop(name)
        self._live_logical[record.category] -= record.logical_bytes
        self._live_actual[record.category] -= record.actual_bytes

        current_tot_log = sum(self._live_logical.values())
        current_tot_act = sum(self._live_actual.values())

        self._seq += 1
        self._timeline.append(
            TimelineEvent(
                step=self.current_step,
                phase=self.current_phase,
                seq=self._seq,
                category=record.category.value,
                logical_bytes_total=current_tot_log,
                actual_bytes_total=current_tot_act,
                delta_logical=-record.logical_bytes,
                delta_actual=-record.actual_bytes,
                name=name,
            )
        )

    @property
    def timeline(self) -> list[TimelineEvent]:
        return list(self._timeline)

    def get_summary(self) -> dict[str, Any]:
        """Return ledger peaks and category summary."""
        return {
            "rank": self.rank,
            "peak_logical": self.peak_logical,
            "peak_actual": self.peak_actual,
            "peak_model_state_logical": self.peak_model_state_logical,
            "peak_by_category": dict(self.peak_by_category),
        }
