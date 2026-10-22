"""Unit tests for plotting functions."""

from pathlib import Path

from zerosim.plots import (
    plot_memory_peaks_vs_formula,
    plot_memory_timeline_4panels,
    plot_scaling_sweep,
    plot_training_loss_curves,
)


def test_plots_generation(tmp_path: Path):
    p1 = tmp_path / "timeline.png"
    timelines = {
        0: [{"seq": 1, "logical_bytes_total": 1000}],
        1: [{"seq": 1, "logical_bytes_total": 1000}],
        2: [{"seq": 1, "logical_bytes_total": 1000}],
        3: [{"seq": 1, "logical_bytes_total": 1000}],
    }
    plot_memory_timeline_4panels(timelines, p1)
    assert p1.exists()

    p2 = tmp_path / "peaks.png"
    measured = {0: 80.0, 1: 25.0, 2: 12.0, 3: 2.5}
    formula = {0: 80.0, 1: 25.0, 2: 12.0, 3: 2.5}
    plot_memory_peaks_vs_formula(measured, formula, p2)
    assert p2.exists()

    p3 = tmp_path / "scaling.png"
    world_sizes = [1, 2, 4, 8]
    peaks_by_stage = {
        0: [80.0, 80.0, 80.0, 80.0],
        1: [80.0, 45.0, 25.0, 15.0],
        2: [80.0, 42.0, 22.0, 12.0],
        3: [80.0, 40.0, 20.0, 10.0],
    }
    plot_scaling_sweep(world_sizes, peaks_by_stage, p3)
    assert p3.exists()

    p4 = tmp_path / "loss.png"
    loss_curves = {
        0: [10.0, 9.0, 8.0],
        1: [10.0, 9.0, 8.0],
        2: [10.0, 9.0, 8.0],
        3: [10.0, 9.0, 8.0],
    }
    plot_training_loss_curves(loss_curves, p4)
    assert p4.exists()
