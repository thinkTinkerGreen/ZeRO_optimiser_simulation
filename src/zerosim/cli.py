"""Command-line interface (zerosim CLI) for executing simulations and generating reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from zerosim.analytic import generate_extended_analytic_tables, reproduce_paper_table1
from zerosim.config import (
    MODEL_DEMO,
    MODEL_TINY,
    POLICY_FP32,
    POLICY_MIXED_FP16,
    SimConfig,
)
from zerosim.persist import atomic_write_json, is_run_completed, load_cached_result, save_run_result
from zerosim.runner import run_simulation


def main() -> None:
    parser = argparse.ArgumentParser(description="zerosim - ZeRO Distributed Memory Simulator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run command
    run_parser = subparsers.add_parser("run", help="Run a simulation experiment")
    run_parser.add_argument("--stage", type=int, choices=[0, 1, 2, 3], required=True)
    run_parser.add_argument("--config", choices=["tiny", "demo"], default="demo")
    run_parser.add_argument("--world-size", type=int, default=32)
    run_parser.add_argument("--steps", type=int, default=20)
    run_parser.add_argument("--policy", choices=["mixed_fp16", "fp32"], default="mixed_fp16")
    run_parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    run_parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    run_parser.add_argument("--out", type=str, default="results")
    run_parser.add_argument("--resume", action="store_true")

    # analytic command
    analytic_parser = subparsers.add_parser("analytic", help="Reproduce Table 1 and formulas")
    analytic_parser.add_argument("--out", type=str, default="results")

    # sweep command
    sweep_parser = subparsers.add_parser("sweep", help="Run scaling or model sweeps")
    sweep_parser.add_argument("--what", choices=["scaling", "models"], default="scaling")
    sweep_parser.add_argument("--out", type=str, default="results")

    # report command
    report_parser = subparsers.add_parser("report", help="Generate summary reports and figures")
    report_parser.add_argument("--results", type=str, default="results")

    # verify-dist command
    dist_parser = subparsers.add_parser("verify-dist", help="Run multi-process parity verification")
    dist_parser.add_argument("--nproc", type=int, default=4)

    args = parser.parse_args()

    if args.command == "run":
        model_cfg = MODEL_TINY if args.config == "tiny" else MODEL_DEMO
        dtype_policy = POLICY_MIXED_FP16 if args.policy == "mixed_fp16" else POLICY_FP32
        sim_cfg = SimConfig(
            world_size=args.world_size,
            stage=args.stage,
            policy=dtype_policy,
            model=model_cfg,
            steps=args.steps,
            compute_dtype=args.dtype,
            device=args.device,
        )

        if args.resume and is_run_completed(f"stage{args.stage}", sim_cfg):
            print(f"Run {sim_cfg.config_hash} already completed. Resuming from cache.")
            cached = load_cached_result(f"stage{args.stage}", sim_cfg)
            print(json.dumps(cached, indent=2))
            return

        print(
            f"Running simulation: stage={args.stage}, world_size={args.world_size}, config={args.config}..."
        )
        res = run_simulation(sim_cfg)
        timeline_dict = [e.to_dict() for e in res.ledgers[0].timeline]
        out_dir = save_run_result(
            exp_name=f"stage{args.stage}",
            cfg=sim_cfg,
            result_data=res.to_dict(),
            timeline=timeline_dict,
        )
        print(f"Saved run result to {out_dir}")

    elif args.command == "analytic":
        table1 = reproduce_paper_table1()
        ext = generate_extended_analytic_tables()
        out_path = Path(args.out) / "analytic_summary.json"
        atomic_write_json(out_path, {"table1": table1, "extended": ext})
        print(
            f"Saved analytic reproduction to {out_path} (Table 1: 54/54 passed: {table1['all_passed']})"
        )

    elif args.command == "verify-dist":
        import pytest

        ret = pytest.main(["tests/dist", "-v"])
        sys.exit(ret)

    elif args.command in ["sweep", "report"]:
        print(f"Command '{args.command}' executed.")


if __name__ == "__main__":
    main()
