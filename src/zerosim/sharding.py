"""Flat unit-wise parameter sharding and contiguous slicing."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ParamSpec:
    name: str
    shape: tuple[int, ...]
    numel: int


@dataclass(frozen=True)
class FlatUnit:
    name: str
    param_specs: tuple[ParamSpec, ...]
    numel: int
    padded_numel: int
    shard_numel: int
    world_size: int

    @property
    def padding_elements(self) -> int:
        return self.padded_numel - self.numel


def create_flat_unit(
    name: str,
    params: Sequence[tuple[str, torch.Size | tuple[int, ...]]],
    world_size: int,
) -> FlatUnit:
    """Create a FlatUnit with padding to make total elements a multiple of world_size."""
    specs: list[ParamSpec] = []
    total_numel = 0
    for p_name, p_shape in params:
        shape_tuple = tuple(p_shape)
        count = math.prod(shape_tuple)
        specs.append(ParamSpec(name=p_name, shape=shape_tuple, numel=count))
        total_numel += count

    padded_numel = math.ceil(total_numel / world_size) * world_size
    shard_numel = padded_numel // world_size

    return FlatUnit(
        name=name,
        param_specs=tuple(specs),
        numel=total_numel,
        padded_numel=padded_numel,
        shard_numel=shard_numel,
        world_size=world_size,
    )


def flatten_unit_tensors(
    tensors: Sequence[torch.Tensor],
    unit: FlatUnit,
) -> torch.Tensor:
    """Flatten a sequence of parameter tensors into a single 1-D padded tensor."""
    if len(tensors) != len(unit.param_specs):
        raise ValueError(
            f"Expected {len(unit.param_specs)} tensors for unit '{unit.name}', got {len(tensors)}"
        )

    flat_list = [t.reshape(-1) for t in tensors]
    if flat_list:
        combined = torch.cat(flat_list)
    else:
        combined = torch.empty(0, dtype=torch.float32)

    if combined.numel() != unit.numel:
        raise ValueError(
            f"Tensors total numel {combined.numel()} does not match unit numel {unit.numel}"
        )

    pad_count = unit.padding_elements
    if pad_count > 0:
        padding = torch.zeros(pad_count, dtype=combined.dtype, device=combined.device)
        return torch.cat([combined, padding])
    return combined


def unflatten_unit_tensor(
    flat_tensor: torch.Tensor,
    unit: FlatUnit,
) -> list[torch.Tensor]:
    """Unflatten a 1-D flat tensor back into the sequence of original shaped tensors."""
    if flat_tensor.numel() < unit.numel:
        raise ValueError(
            f"Flat tensor has {flat_tensor.numel()} elements, expected at least {unit.numel}"
        )

    unflattened: list[torch.Tensor] = []
    offset = 0
    for spec in unit.param_specs:
        slice_1d = flat_tensor[offset : offset + spec.numel]
        unflattened.append(slice_1d.reshape(spec.shape).clone())
        offset += spec.numel

    return unflattened


def get_shard_slice(unit: FlatUnit, rank: int) -> slice:
    """Return slice for rank's partition within the padded flat buffer."""
    if not (0 <= rank < unit.world_size):
        raise IndexError(f"Rank {rank} out of bounds for world size {unit.world_size}")
    start = rank * unit.shard_numel
    end = (rank + 1) * unit.shard_numel
    return slice(start, end)


def compute_padding_overhead(units: Sequence[FlatUnit]) -> float:
    """Compute total padding overhead: sum(padded_numel) / sum(numel) - 1."""
    total_numel = sum(u.numel for u in units)
    total_padded = sum(u.padded_numel for u in units)
    if total_numel == 0:
        return 0.0
    return (total_padded / total_numel) - 1.0
