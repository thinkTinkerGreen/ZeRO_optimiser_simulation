"""Unit tests for Milestone M1 components: config, determinism, ledger, governor."""

import pytest
import torch

from zerosim.config import (
    MODEL_DEMO,
    MODEL_TINY,
    POLICY_FP32,
    POLICY_MIXED_FP16,
    ModelConfig,
    SimConfig,
)
from zerosim.determinism import get_environment_metadata, get_generator, seed_everything
from zerosim.governor import ResourceGovernor
from zerosim.ledger import MemoryCategory, MemoryLedger


def test_config_hash_consistency():
    cfg1 = SimConfig(stage=1, model=MODEL_TINY, seed=42)
    cfg2 = SimConfig(stage=1, model=MODEL_TINY, seed=42)
    cfg3 = SimConfig(stage=2, model=MODEL_TINY, seed=42)

    assert cfg1.config_hash == cfg2.config_hash
    assert cfg1.config_hash != cfg3.config_hash
    assert len(cfg1.config_hash) == 16


def test_dtype_policy_k():
    assert POLICY_MIXED_FP16.K == 12
    assert POLICY_FP32.K == 8


def test_determinism_seed_everything():
    seed_everything(1234)
    t1 = torch.randn(10)
    seed_everything(1234)
    t2 = torch.randn(10)
    assert torch.equal(t1, t2)

    gen1 = get_generator(1234, rank=1, step=2)
    t3 = torch.randn(10, generator=gen1)
    gen2 = get_generator(1234, rank=1, step=2)
    t4 = torch.randn(10, generator=gen2)
    assert torch.equal(t3, t4)

    meta = get_environment_metadata()
    assert "torch_version" in meta
    assert meta["cpu_count"] >= 1


def test_ledger_alloc_free_symmetry():
    ledger = MemoryLedger(rank=0)
    ledger.mark_phase(0, "init")

    # Allocate 100 elements of PARAM
    tensor = torch.zeros(100, dtype=torch.float32)
    ledger.alloc("w_test", MemoryCategory.PARAM, tensor, logical_bytes_per_elem=2)

    assert ledger.peak_logical == 200
    assert ledger.peak_actual == 400
    assert ledger.peak_model_state_logical == 200
    assert ledger.peak_by_category[MemoryCategory.PARAM.value] == 200

    ledger.free("w_test")
    # Peak remains at 200, but live is 0
    assert ledger.peak_logical == 200
    assert ledger._live_logical[MemoryCategory.PARAM] == 0


def test_ledger_double_free_raises():
    ledger = MemoryLedger(rank=0)
    tensor = torch.zeros(10)
    ledger.alloc("t", MemoryCategory.TEMP, tensor, logical_bytes_per_elem=4)
    ledger.free("t")
    with pytest.raises(KeyError):
        ledger.free("t")


def test_ledger_double_alloc_raises():
    ledger = MemoryLedger(rank=0)
    tensor = torch.zeros(10)
    ledger.alloc("t", MemoryCategory.TEMP, tensor, logical_bytes_per_elem=4)
    with pytest.raises(ValueError):
        ledger.alloc("t", MemoryCategory.TEMP, tensor, logical_bytes_per_elem=4)


def test_ledger_timeline_monotonic_seq():
    ledger = MemoryLedger(rank=0)
    for i in range(5):
        ledger.alloc(f"t_{i}", MemoryCategory.TEMP, 10, logical_bytes_per_elem=4)
    for i in range(5):
        ledger.free(f"t_{i}")

    seqs = [ev.seq for ev in ledger.timeline]
    assert seqs == list(range(1, 11))
    assert all(seqs[i] < seqs[i + 1] for i in range(len(seqs) - 1))


def test_governor_concurrency_and_refusal():
    gov = ResourceGovernor(max_concurrent=2, memory_fraction=0.5)
    assert gov.max_concurrent == 2

    # Should succeed with small config
    small_cfg = SimConfig(world_size=4, model=MODEL_TINY, compute_dtype="float32")
    plan = gov.plan_memory(small_cfg, approx_numel=10_000)
    assert plan["predicted_bytes"] > 0
    assert plan["predicted_bytes"] <= plan["limit_bytes"]

    # Huge model config should raise refusal
    huge_cfg = SimConfig(
        world_size=64,
        model=ModelConfig("huge", 100, 8192, 64, 32000, 2048),
        compute_dtype="float64",
    )
    with pytest.raises(RuntimeError) as exc_info:
        gov.plan_memory(huge_cfg, approx_numel=10_000_000_000)
    assert "Governor rejected plan" in str(exc_info.value)


def test_governor_compute_scope():
    gov = ResourceGovernor(max_concurrent=1)
    with gov.compute_scope():
        val = 1 + 1
    assert val == 2
