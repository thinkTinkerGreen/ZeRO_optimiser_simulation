"""Plotting utilities for memory timelines, cost breakdowns, and scaling sweeps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def plot_memory_timeline_4panels(
    timelines_per_stage: dict[int, list[dict[str, Any]]],
    out_path: str | Path,
) -> None:
    """Generate 4-panel stacked area memory timeline across stages 0-3 for rank 0."""
    fig, axes = plt.subplots(4, 1, figsize=(10, 14), sharex=False)

    for stage_idx, ax in enumerate(axes):
        timeline = timelines_per_stage.get(stage_idx, [])
        if not timeline:
            ax.text(0.5, 0.5, f"Stage {stage_idx} data missing", ha="center")
            continue

        seqs = [e["seq"] for e in timeline]
        # Track logical bytes
        tot_bytes = [e["logical_bytes_total"] / 1e6 for e in timeline]
        ax.plot(seqs, tot_bytes, label="Total Logical (MB)", color="black", lw=1.5)
        ax.set_title(
            f"ZeRO Stage {stage_idx} Memory Timeline (Rank 0)", fontsize=11, fontweight="bold"
        )
        ax.set_ylabel("Memory (MB)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].set_xlabel("Event Sequence Number")
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_memory_peaks_vs_formula(
    measured_peaks: dict[int, float],
    formula_peaks: dict[int, float],
    out_path: str | Path,
) -> None:
    """Bar chart comparing measured model state peaks vs paper formulas."""
    stages = [0, 1, 2, 3]
    labels = ["Stage 0 (DDP)", "Stage 1 (Pos)", "Stage 2 (Pos+g)", "Stage 3 (Pos+g+p)"]
    x = np.arange(len(stages))
    width = 0.35

    meas = [measured_peaks.get(s, 0.0) / 1e6 for s in stages]
    form = [formula_peaks.get(s, 0.0) / 1e6 for s in stages]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, meas, width, label="Measured Peak (MB)", color="#3498db")
    ax.bar(x + width / 2, form, width, label="Paper Formula (MB)", color="#2ecc71")

    ax.set_ylabel("Model State Peak Memory (MB)", fontsize=11)
    ax.set_title(
        "Peak Model State Memory vs. Theoretical Closed Form (N=32)", fontsize=12, fontweight="bold"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_scaling_sweep(
    world_sizes: list[int],
    peaks_by_stage: dict[int, list[float]],
    out_path: str | Path,
) -> None:
    """Log-log scaling curve across world sizes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {0: "#e74c3c", 1: "#e67e22", 2: "#f1c40f", 3: "#2ecc71"}
    labels = {0: "Stage 0 (DDP)", 1: "Stage 1 (Pos)", 2: "Stage 2 (Pos+g)", 3: "Stage 3 (Pos+g+p)"}

    for s in [0, 1, 2, 3]:
        pts = [p / 1e6 for p in peaks_by_stage.get(s, [])]
        if pts:
            ax.plot(world_sizes[: len(pts)], pts, "o-", label=labels[s], color=colors[s], lw=2)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xlabel("World Size (N ranks)", fontsize=11)
    ax.set_ylabel("Peak Model State Memory (MB)", fontsize=11)
    ax.set_title("ZeRO Memory Scaling with World Size N (Log-Log)", fontsize=12, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_training_loss_curves(
    loss_curves: dict[int, list[float]],
    out_path: str | Path,
) -> None:
    """Plot training loss curves across stages."""
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {0: "#2c3e50", 1: "#2980b9", 2: "#8e44ad", 3: "#27ae60"}
    labels = {0: "Stage 0 (DDP)", 1: "Stage 1 (Pos)", 2: "Stage 2 (Pos+g)", 3: "Stage 3 (Pos+g+p)"}

    for s in [0, 1, 2, 3]:
        curve = loss_curves.get(s, [])
        if curve:
            ax.plot(curve, label=labels[s], color=colors[s], alpha=0.8, lw=1.5)

    ax.set_xlabel("Training Step", fontsize=11)
    ax.set_ylabel("Cross Entropy Loss", fontsize=11)
    ax.set_title("Training Loss Convergence Across ZeRO Stages", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
