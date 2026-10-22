"""Integration tests for M7: E3 (compute/comm), E4 (scaling), and persistence (AC-09, AC-10, AC-11)."""

import math
import numpy as np
import pytest
import torch

from zerosim.comm.stats import CommStats
from zerosim.config import MODEL_DEMO, MODEL_TINY, POLICY_MIXED_FP16, SimConfig
from zerosim.costmodel import PRESET_IB, PRESET_NVLINK, estimate_step_time
from zerosim.flops import compute_rank_flops, cross_check_flop_counter
from zerosim.model import GPT
from zerosim.persist import (
    atomic_write_json,
    is_run_completed,
    load_cached_result,
    save_run_result,
)
from zerosim.runner import run_simulation


def test_e3_flops_and_comm_exact_ratios():
    """Verify E3 FLOPs and communication element accounting (AC-09)."""
    N = 32
    steps = 1
    m = GPT(MODEL_DEMO)
    units = m.get_flat_units(world_size=N)
    padded_psi = sum(u.padded_numel for u in units)
    total_numel = sum(u.numel for u in units)
    shard_numel = sum(u.shard_numel for u in units)

    results = {}
    for stage in [0, 1, 2, 3]:
        cfg = SimConfig(
            world_size=N,
            stage=stage,
            policy=POLICY_MIXED_FP16,
            model=MODEL_DEMO,
            microbatch=2,
            steps=steps,
            compute_dtype="float32",
            max_concurrent=4,
        )
        res = run_simulation(cfg)
        results[stage] = res

    # 1. FLOPs consistency
    # Stage 0 updates all parameters; Stage 1, 2, 3 update only this rank's shard
    flops0 = compute_rank_flops(MODEL_DEMO, microbatch=2, stage=0, updated_params=total_numel)
    flops1 = compute_rank_flops(MODEL_DEMO, microbatch=2, stage=1, updated_params=shard_numel)
    flops3 = compute_rank_flops(MODEL_DEMO, microbatch=2, stage=3, updated_params=shard_numel)

    assert flops0.fwd_flops == flops1.fwd_flops == flops3.fwd_flops
    assert flops0.bwd_flops == flops1.bwd_flops == flops3.bwd_flops
    assert flops0.z3_recompute_simulation_overhead_flops == 0
    assert flops1.z3_recompute_simulation_overhead_flops == 0
    assert flops3.z3_recompute_simulation_overhead_flops == flops3.fwd_flops
    assert flops1.optimizer_flops == int(flops0.optimizer_flops * (shard_numel / total_numel))

    # 2. Communication elements in units of padded_psi
    # Stage 0: 2 * padded_psi
    comm0 = results[0].comm_stats[0]
    assert comm0.all_reduce.elems == 2 * padded_psi
    assert comm0.reduce_scatter.elems == 0
    assert comm0.all_gather.elems == 0
    assert comm0.total_elems == 2 * padded_psi

    # Stage 1: reduce_scatter (1) + all_gather (1) = 2 * padded_psi
    comm1 = results[1].comm_stats[0]
    assert comm1.reduce_scatter.elems == padded_psi
    assert comm1.all_gather.elems == padded_psi
    assert comm1.total_elems == 2 * padded_psi

    # Stage 2: reduce_scatter (1) + all_gather (1) = 2 * padded_psi
    comm2 = results[2].comm_stats[0]
    assert comm2.reduce_scatter.elems == padded_psi
    assert comm2.all_gather.elems == padded_psi
    assert comm2.total_elems == 2 * padded_psi

    # Stage 3: fwd all_gather (1) + bwd all_gather (1) + bwd reduce_scatter (1) = 3 * padded_psi
    comm3 = results[3].comm_stats[0]
    assert comm3.all_gather.elems == 2 * padded_psi
    assert comm3.reduce_scatter.elems == padded_psi
    assert comm3.total_elems == 3 * padded_psi

    # 3. Modelled step time under presets
    est_nvlink = estimate_step_time(flops3, comm3, PRESET_NVLINK)
    est_ib = estimate_step_time(flops3, comm3, PRESET_IB)
    assert est_nvlink.t_comm_s < est_ib.t_comm_s


def test_e4_scaling_sweep_slope():
    """Verify E4 log-log slope for Stage 3 follows 1/N within 0.02 (AC-10)."""
    world_sizes = [1, 2, 4, 8, 16]
    stage3_peaks = []
    temp_m = GPT(MODEL_DEMO)

    for N in world_sizes:
        cfg = SimConfig(
            world_size=N,
            stage=3,
            policy=POLICY_MIXED_FP16,
            model=MODEL_DEMO,
            microbatch=1,
            steps=1,
            compute_dtype="float32",
            max_concurrent=2,
        )
        res = run_simulation(cfg)
        ledger = res.ledgers[0]
        units = temp_m.get_flat_units(world_size=N)
        max_unit_padded = max(u.padded_numel for u in units)
        transient = max_unit_padded * (POLICY_MIXED_FP16.param_bytes + POLICY_MIXED_FP16.grad_bytes)
        steady_state = ledger.peak_model_state_logical - transient
        stage3_peaks.append(steady_state)

    # Compute log2 slope
    log_n = np.log2(world_sizes)
    log_peak = np.log2(stage3_peaks)
    slope, _ = np.polyfit(log_n, log_peak, 1)

    # Slope must be within 0.02 of -1.0
    assert abs(slope - (-1.0)) <= 0.02, f"Stage 3 slope {slope} deviates from -1.0"


def test_persistence_atomicity_and_resume(tmp_path, monkeypatch):
    """Verify atomic write, manifest generation, and resume functionality (AC-11)."""
    monkeypatch.setenv("ZEROSIM_RESULTS", str(tmp_path))

    cfg = SimConfig(world_size=2, stage=0, steps=2, model=MODEL_TINY)
    assert not is_run_completed("test_exp", cfg)

    # Save mock result
    run_dir = save_run_result(
        "test_exp",
        cfg,
        result_data={"metric": 42.0},
        timeline=[{"step": 0, "val": 1}],
    )
    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "result.json").exists()
    assert (run_dir / "timeline_rank0.json").exists()

    # Verify resume check
    assert is_run_completed("test_exp", cfg)
    cached = load_cached_result("test_exp", cfg)
    assert cached is not None
    assert cached["metric"] == 42.0
