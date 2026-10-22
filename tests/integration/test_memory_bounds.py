"""Integration tests for E1: Memory bounds, uniformity, and activation parity (AC-08)."""

import pytest

from zerosim.config import MODEL_DEMO, POLICY_FP32, POLICY_MIXED_FP16, SimConfig
from zerosim.model import GPT
from zerosim.runner import run_simulation


@pytest.mark.parametrize("policy", [POLICY_MIXED_FP16, POLICY_FP32])
@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_e1_memory_bounds_and_uniformity(policy, stage):
    N = 32
    cfg = SimConfig(
        world_size=N,
        stage=stage,
        policy=policy,
        model=MODEL_DEMO,
        microbatch=2,
        steps=3,
        compute_dtype="float32",
        seed=1234,
        max_concurrent=4,
    )

    # Compute padded unit parameters and max transient unit
    temp_model = GPT(MODEL_DEMO)
    units = temp_model.get_flat_units(world_size=N)
    padded_psi = sum(u.padded_numel for u in units)
    max_unit_padded = max(u.padded_numel for u in units)

    # Calculate theoretical steady-state model state memory under padded sizes
    K = policy.K
    param_b = policy.param_bytes
    grad_b = policy.grad_bytes

    if stage == 0:
        f_padded = (param_b + grad_b + K) * padded_psi
        transient_bound = 0
    elif stage == 1:
        f_padded = (param_b + grad_b) * padded_psi + (K * padded_psi) // N
        transient_bound = 0
    elif stage == 2:
        f_padded = param_b * padded_psi + ((grad_b + K) * padded_psi) // N
        transient_bound = max_unit_padded * grad_b
    elif stage == 3:
        f_padded = ((param_b + grad_b + K) * padded_psi) // N
        transient_bound = max_unit_padded * (param_b + grad_b)
    else:
        raise ValueError(f"Invalid stage {stage}")

    res = run_simulation(cfg)

    # Verify every rank satisfies bounds
    rank0_peaks = res.ledgers[0].peak_by_category
    for r in range(N):
        ledger = res.ledgers[r]
        peak = ledger.peak_model_state_logical
        assert f_padded <= peak <= f_padded + transient_bound, (
            f"Rank {r} peak {peak} out of bounds [{f_padded}, {f_padded + transient_bound}]"
        )

        # Check uniformity: identical peak_by_category across all 32 ranks
        assert ledger.peak_by_category == rank0_peaks, (
            f"Rank {r} category peaks differ from Rank 0!"
        )


def test_e1_activation_peak_identical_across_stages():
    """Verify that activation peak is identical (+-0) across all stages 0-3."""
    act_peaks = []
    for stage in [0, 1, 2, 3]:
        cfg = SimConfig(
            world_size=4,
            stage=stage,
            policy=POLICY_MIXED_FP16,
            model=MODEL_DEMO,
            microbatch=2,
            steps=2,
            compute_dtype="float32",
            seed=42,
            max_concurrent=2,
        )
        res = run_simulation(cfg)
        act_peaks.append(res.ledgers[0].peak_by_category["ACTIVATION"])

    assert len(set(act_peaks)) == 1, f"Activation peaks vary across stages: {act_peaks}"
    assert act_peaks[0] > 0
