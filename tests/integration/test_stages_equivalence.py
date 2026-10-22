"""Integration test for E2: Numerical equivalence of Stages 0-3 against single-process reference (AC-07)."""

import pytest

from tests.integration.reference import run_reference_trainer
from zerosim.config import MODEL_TINY, POLICY_FP32, SimConfig
from zerosim.runner import run_simulation


@pytest.mark.parametrize("world_size", [1, 2, 3, 5])
@pytest.mark.parametrize("stage", [0, 1, 2, 3])
def test_stage_equivalence_arbitrary_ranks(world_size: int, stage: int):
    """Test equivalence for N in {1, 2, 3, 5} for 5 steps in float64."""
    steps = 5
    cfg = SimConfig(
        world_size=world_size,
        stage=stage,
        policy=POLICY_FP32,
        model=MODEL_TINY,
        microbatch=2,
        steps=steps,
        compute_dtype="float64",
        seed=42,
    )

    # Run reference single-process trainer
    ref_losses, ref_params = run_reference_trainer(cfg, steps=steps)

    # Run simulation
    sim_res = run_simulation(cfg)

    # 1. Compare per-step losses: max |loss_stage - loss_ref| <= 1e-10
    for s in range(steps):
        diff = abs(sim_res.losses_per_step[s] - ref_losses[s])
        assert diff <= 1e-10, (
            f"Stage {stage} N={world_size} step {s} loss diff {diff} exceeds 1e-10"
        )

    # 2. Compare final parameter vector: max abs diff <= 1e-10
    param_diff = (sim_res.final_parameters.cpu() - ref_params.cpu()).abs().max().item()
    assert param_diff <= 1e-10, (
        f"Stage {stage} N={world_size} param diff {param_diff} exceeds 1e-10"
    )


def test_stage_equivalence_32_ranks():
    """Test equivalence at N=32 for 5 steps across all stages (AC-07)."""
    steps = 5
    ref_cfg = SimConfig(
        world_size=32,
        stage=0,
        policy=POLICY_FP32,
        model=MODEL_TINY,
        microbatch=1,
        steps=steps,
        compute_dtype="float64",
        seed=1234,
    )
    ref_losses, ref_params = run_reference_trainer(ref_cfg, steps=steps)

    for stage in [0, 1, 2, 3]:
        cfg = SimConfig(
            world_size=32,
            stage=stage,
            policy=POLICY_FP32,
            model=MODEL_TINY,
            microbatch=1,
            steps=steps,
            compute_dtype="float64",
            seed=1234,
        )
        sim_res = run_simulation(cfg)

        for s in range(steps):
            diff = abs(sim_res.losses_per_step[s] - ref_losses[s])
            assert diff <= 1e-10, f"Stage {stage} N=32 step {s} loss diff {diff} exceeds 1e-10"

        param_diff = (sim_res.final_parameters.cpu() - ref_params.cpu()).abs().max().item()
        assert param_diff <= 1e-10, f"Stage {stage} N=32 param diff {param_diff} exceeds 1e-10"
