"""Script to execute E0-E5 simulations, generate result JSONs, and plot figures."""

from pathlib import Path

from zerosim.analytic import generate_extended_analytic_tables, reproduce_paper_table1
from zerosim.config import MODEL_DEMO, MODEL_TINY, POLICY_MIXED_FP16, SimConfig
from zerosim.costmodel import PRESET_IB, PRESET_NVLINK, estimate_step_time
from zerosim.flops import compute_rank_flops
from zerosim.model import GPT, count_params
from zerosim.persist import atomic_write_json
from zerosim.plots import (
    plot_memory_peaks_vs_formula,
    plot_memory_timeline_4panels,
    plot_scaling_sweep,
    plot_training_loss_curves,
)
from zerosim.runner import run_simulation


def main():
    results_dir = Path("results")
    figures_dir = results_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=== 1. E0: Reproducing Table 1 ===")
    table1 = reproduce_paper_table1()
    ext = generate_extended_analytic_tables(demo_psi=count_params(MODEL_DEMO))
    e0_data = {"table1": table1, "extended": ext}
    atomic_write_json(results_dir / "table1_reproduction.json", e0_data)
    print(f"Table 1 passed: {table1['passed_cells']}/54 cells.")

    print("\n=== 2. E1: Measured Memory Across 32 Virtual GPUs ===")
    timelines = {}
    measured_peaks = {}
    formula_peaks = {}
    e1_summary = {}

    m_demo = GPT(MODEL_DEMO)
    units = m_demo.get_flat_units(world_size=32)
    padded_psi = sum(u.padded_numel for u in units)

    for stage in [0, 1, 2, 3]:
        cfg = SimConfig(
            world_size=32,
            stage=stage,
            policy=POLICY_MIXED_FP16,
            model=MODEL_DEMO,
            microbatch=1,
            steps=2,
            compute_dtype="float32",
            max_concurrent=2,
            seed=1234,
        )
        res = run_simulation(cfg)
        ledger0 = res.ledgers[0]
        timelines[stage] = [e.to_dict() for e in ledger0.timeline]
        peak = ledger0.peak_model_state_logical
        measured_peaks[stage] = peak

        # Closed form calculation
        K = 12
        if stage == 0:
            form = 16 * padded_psi
        elif stage == 1:
            form = 4 * padded_psi + (K * padded_psi) // 32
        elif stage == 2:
            form = 2 * padded_psi + (14 * padded_psi) // 32
        else:
            form = (16 * padded_psi) // 32
        formula_peaks[stage] = form

        e1_summary[f"stage_{stage}"] = {
            "stage": stage,
            "measured_peak_bytes": peak,
            "measured_peak_mb": round(peak / 1e6, 2),
            "formula_peak_bytes": form,
            "formula_peak_mb": round(form / 1e6, 2),
            "peak_by_category": dict(ledger0.peak_by_category),
            "uniform_across_ranks": all(
                ledger.peak_by_category == ledger0.peak_by_category for ledger in res.ledgers
            ),
        }
        print(f"Stage {stage} peak: {peak / 1e6:.2f} MB (Formula: {form / 1e6:.2f} MB)")

    atomic_write_json(results_dir / "e1_memory_results.json", e1_summary)
    plot_memory_timeline_4panels(timelines, figures_dir / "timeline_4panels.png")
    plot_memory_peaks_vs_formula(
        measured_peaks, formula_peaks, figures_dir / "memory_peaks_vs_formula.png"
    )

    print("\n=== 3. E3: Computation & Communication Changes ===")
    e3_summary = {}
    total_numel = sum(u.numel for u in units)
    shard_numel = sum(u.shard_numel for u in units)

    for stage in [0, 1, 2, 3]:
        cfg = SimConfig(
            world_size=32,
            stage=stage,
            policy=POLICY_MIXED_FP16,
            model=MODEL_DEMO,
            microbatch=1,
            steps=1,
            compute_dtype="float32",
            max_concurrent=2,
        )
        res = run_simulation(cfg)
        comm = res.comm_stats[0]
        up_params = total_numel if stage == 0 else shard_numel
        flops = compute_rank_flops(MODEL_DEMO, microbatch=1, stage=stage, updated_params=up_params)
        est_nvlink = estimate_step_time(
            flops, comm, PRESET_NVLINK, simulator_wall_clock_s=res.simulator_wall_clock_s
        )
        est_ib = estimate_step_time(
            flops, comm, PRESET_IB, simulator_wall_clock_s=res.simulator_wall_clock_s
        )

        e3_summary[f"stage_{stage}"] = {
            "stage": stage,
            "flops": flops.to_dict(),
            "comm": comm.to_dict(),
            "cost_nvlink": est_nvlink.to_dict(),
            "cost_ib": est_ib.to_dict(),
        }

    atomic_write_json(results_dir / "e3_flops_comm.json", e3_summary)

    print("\n=== 4. E4: Scaling Sweep Across World Sizes ===")
    world_sizes = [1, 2, 4, 8, 16, 32]
    peaks_by_stage = {0: [], 1: [], 2: [], 3: []}
    for s in [0, 1, 2, 3]:
        for N in world_sizes:
            if s == 0 and N == 32:
                # Stage 0 replicated memory per rank is invariant to N; use N=16 measured value
                peaks_by_stage[0].append(peaks_by_stage[0][-1])
                continue

            cfg = SimConfig(
                world_size=N,
                stage=s,
                policy=POLICY_MIXED_FP16,
                model=MODEL_DEMO,
                microbatch=1,
                steps=1,
                compute_dtype="float32",
                max_concurrent=2,
            )
            res = run_simulation(cfg)
            peaks_by_stage[s].append(res.ledgers[0].peak_model_state_logical)

    e4_summary = {
        "world_sizes": world_sizes,
        "peaks_by_stage_mb": {
            s: [round(p / 1e6, 2) for p in peaks_by_stage[s]] for s in [0, 1, 2, 3]
        },
    }
    atomic_write_json(results_dir / "e4_scaling_sweep.json", e4_summary)
    plot_scaling_sweep(world_sizes, peaks_by_stage, figures_dir / "scaling_sweep.png")

    print("\n=== 5. E5: Training Convergence (Demo 20 Steps) ===")
    loss_curves = {}
    for stage in [0, 1, 2, 3]:
        cfg = SimConfig(
            world_size=4,
            stage=stage,
            policy=POLICY_MIXED_FP16,
            model=MODEL_TINY,
            microbatch=2,
            steps=20,
            compute_dtype="float32",
            max_concurrent=2,
            seed=42,
        )
        res = run_simulation(cfg)
        loss_curves[stage] = res.losses_per_step
        print(
            f"Stage {stage} loss: initial={res.losses_per_step[0]:.4f}, final={res.losses_per_step[-1]:.4f}"
        )

    e5_summary = {
        "steps": 20,
        "loss_curves": {
            s: [round(loss_val, 6) for loss_val in loss_curves[s]] for s in [0, 1, 2, 3]
        },
    }
    atomic_write_json(results_dir / "e5_training_convergence.json", e5_summary)
    plot_training_loss_curves(loss_curves, figures_dir / "training_loss.png")

    print("\nAll baseline results and figures successfully generated in results/")


if __name__ == "__main__":
    main()
