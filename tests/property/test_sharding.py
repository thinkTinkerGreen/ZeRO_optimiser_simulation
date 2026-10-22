"""Property tests for flat sharding and round-trip consistency (AC-04)."""

from typing import Any

import hypothesis.strategies as st
import torch
from hypothesis import given, settings

from zerosim.sharding import (
    compute_padding_overhead,
    create_flat_unit,
    flatten_unit_tensors,
    get_shard_slice,
    unflatten_unit_tensor,
)


@st.composite
def random_param_shapes(draw: Any):
    num_params = draw(st.integers(min_value=1, max_value=5))
    shapes = []
    for idx in range(num_params):
        dims = draw(st.lists(st.integers(min_value=1, max_value=8), min_size=1, max_size=3))
        shapes.append((f"p_{idx}", tuple(dims)))
    return shapes


@settings(max_examples=50, deadline=None)
@given(
    param_shapes=random_param_shapes(),  # type: ignore
    world_size=st.integers(min_value=1, max_value=64),
)
def test_sharding_roundtrip_and_tiling_properties(param_shapes, world_size):
    unit = create_flat_unit("test_unit", param_shapes, world_size=world_size)

    # 1. Check tiling of shard slices: [0, padded_numel) with no gaps or overlaps
    covered = set()
    for rank in range(world_size):
        sl = get_shard_slice(unit, rank)
        rank_indices = set(range(sl.start, sl.stop))
        assert len(rank_indices) == unit.shard_numel
        assert covered.isdisjoint(rank_indices), f"Overlap detected at rank {rank}"
        covered.update(rank_indices)

    assert covered == set(range(unit.padded_numel)), "Gap detected in tiling!"

    # 2. Check flatten / unflatten bit-exact round trip
    tensors = [torch.randn(s[1], dtype=torch.float64) for s in param_shapes]
    flat = flatten_unit_tensors(tensors, unit)

    assert flat.numel() == unit.padded_numel
    if unit.padding_elements > 0:
        padding_part = flat[unit.numel :]
        assert torch.all(padding_part == 0), "Padding region is non-zero!"

    unflattened = unflatten_unit_tensor(flat, unit)
    assert len(unflattened) == len(tensors)
    for orig, reconstructed in zip(tensors, unflattened):
        assert torch.equal(orig, reconstructed), "Reconstructed tensor does not match original!"


def test_padding_overhead_calculation():
    # Unit with 10 elements, N=4 -> padded to 12
    unit1 = create_flat_unit("u1", [("w1", (10,))], world_size=4)
    assert unit1.numel == 10
    assert unit1.padded_numel == 12
    assert unit1.padding_elements == 2

    # Unit with 8 elements, N=4 -> padded to 8 (0 overhead)
    unit2 = create_flat_unit("u2", [("w2", (8,))], world_size=4)
    assert unit2.numel == 8
    assert unit2.padded_numel == 8
    assert unit2.padding_elements == 0

    overhead = compute_padding_overhead([unit1, unit2])
    # Total padded = 20, total numel = 18 -> 20/18 - 1 = 2/18 = 1/9
    assert abs(overhead - (20 / 18 - 1)) < 1e-12
