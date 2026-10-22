"""Analytical model-state memory formulas and Table 1 reproduction from the ZeRO paper."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class AnalyticMemoryResult:
    psi: float
    N: int
    K: int
    baseline_bytes: float
    pos_bytes: float
    pos_g_bytes: float
    pos_g_p_bytes: float

    @property
    def baseline_gb(self) -> float:
        return self.baseline_bytes / 1e9

    @property
    def pos_gb(self) -> float:
        return self.pos_bytes / 1e9

    @property
    def pos_g_gb(self) -> float:
        return self.pos_g_bytes / 1e9

    @property
    def pos_g_p_gb(self) -> float:
        return self.pos_g_p_bytes / 1e9

    def to_dict(self) -> dict[str, Any]:
        return {
            "psi": self.psi,
            "N": self.N,
            "K": self.K,
            "baseline_gb": round(self.baseline_gb, 4),
            "pos_gb": round(self.pos_gb, 4),
            "pos_g_gb": round(self.pos_g_gb, 4),
            "pos_g_p_gb": round(self.pos_g_p_gb, 4),
        }


def compute_analytic_memory(
    psi: float,
    N: int,
    K: int = 12,
    param_bytes: int = 2,
    grad_bytes: int = 2,
) -> AnalyticMemoryResult:
    """Compute per-device model-state memory in bytes according to ZeRO paper formulas.

    Formulas from Rajbhandari et al., 2020:
    - Baseline DP: (2 + 2 + K) * psi
    - Pos: 2*param_bytes*psi + K*psi/N = 4*psi + K*psi/N
    - Pos+g: param_bytes*psi + (grad_bytes + K)*psi/N = 2*psi + (2 + K)*psi/N
    - Pos+g+p: (param_bytes + grad_bytes + K)*psi/N = 16*psi/N (for K=12)
    """
    baseline_bytes = (param_bytes + grad_bytes + K) * psi
    pos_bytes = (param_bytes + grad_bytes) * psi + (K * psi) / N
    pos_g_bytes = param_bytes * psi + ((grad_bytes + K) * psi) / N
    pos_g_p_bytes = ((param_bytes + grad_bytes + K) * psi) / N

    return AnalyticMemoryResult(
        psi=psi,
        N=N,
        K=K,
        baseline_bytes=baseline_bytes,
        pos_bytes=pos_bytes,
        pos_g_bytes=pos_g_bytes,
        pos_g_p_bytes=pos_g_p_bytes,
    )


def verify_table1_cell(computed: float, printed: float) -> bool:
    """Check if computed matches printed within the paper comparison rule.

    The paper rounds in some cells (e.g. 41.25 -> 41.3) and truncates/floors in others
    (e.g. 257.75 -> 257, 2218.75 -> 2218). We verify that abs(computed - printed) <= 0.51 * unit,
    or that the paper's integer printed value matches floor(computed) or round(computed).
    """
    import math

    s = str(printed)
    if "." in s and not s.endswith(".0"):
        decimals = len(s.split(".")[1])
        unit = 10 ** (-decimals)
    else:
        unit = 1.0

    tolerance = 0.51 * unit
    if abs(computed - printed) <= tolerance:
        return True

    if unit == 1.0:
        int_printed = int(printed)
        return (math.floor(computed) == int_printed) or (round(computed) == int_printed)

    return False


def reproduce_paper_table1(fixture_path: str | Path | None = None) -> dict[str, Any]:
    """Reproduce all 54 cells of paper Table 1 and verify against golden fixture."""
    if fixture_path is None:
        fixture_path = (
            Path(__file__).resolve().parent.parent.parent / "tests/fixtures/paper_table1.json"
        )

    with open(fixture_path, encoding="utf-8") as f:
        fixture = json.load(f)

    K = fixture.get("K", 12)
    models = fixture["models"]
    results: dict[str, Any] = {}
    total_cells = 0
    passed_cells = 0

    for model_name, model_info in models.items():
        psi = model_info["psi"]
        rows = model_info["rows"]
        model_results = []

        for row in rows:
            N = row["N"]
            res = compute_analytic_memory(psi=psi, N=N, K=K)
            pos_pass = verify_table1_cell(res.pos_gb, row["Pos"])
            pos_g_pass = verify_table1_cell(res.pos_g_gb, row["Pos_g"])
            pos_g_p_pass = verify_table1_cell(res.pos_g_p_gb, row["Pos_g_p"])

            total_cells += 3
            if pos_pass:
                passed_cells += 1
            if pos_g_pass:
                passed_cells += 1
            if pos_g_p_pass:
                passed_cells += 1

            model_results.append(
                {
                    "N": N,
                    "Pos_printed": row["Pos"],
                    "Pos_computed": round(res.pos_gb, 2),
                    "Pos_pass": pos_pass,
                    "Pos_g_printed": row["Pos_g"],
                    "Pos_g_computed": round(res.pos_g_gb, 2),
                    "Pos_g_pass": pos_g_pass,
                    "Pos_g_p_printed": row["Pos_g_p"],
                    "Pos_g_p_computed": round(res.pos_g_p_gb, 2),
                    "Pos_g_p_pass": pos_g_p_pass,
                }
            )

        results[model_name] = model_results

    return {
        "total_cells": total_cells,
        "passed_cells": passed_cells,
        "all_passed": (total_cells == passed_cells == 54),
        "models": results,
    }


def generate_extended_analytic_tables(demo_psi: float = 4.88e6) -> dict[str, Any]:
    """Generate extended analytic predictions for GPT-2 XL (1.5B) and demo model."""
    extended = {}
    # 1.5B model across N in [1, 4, 16, 64, 256, 1024]
    gpt2_xl_rows = []
    for N in [1, 4, 16, 64, 256, 1024]:
        res = compute_analytic_memory(psi=1.5e9, N=N, K=12)
        gpt2_xl_rows.append(res.to_dict())
    extended["1.5B_GPT2_XL"] = gpt2_xl_rows

    # Demo model at N=32
    demo_res = compute_analytic_memory(psi=demo_psi, N=32, K=12)
    extended["demo_N32"] = {
        "psi": demo_psi,
        "N": 32,
        "baseline_mb": round(demo_res.baseline_bytes / 1e6, 4),
        "pos_mb": round(demo_res.pos_bytes / 1e6, 4),
        "pos_g_mb": round(demo_res.pos_g_bytes / 1e6, 4),
        "pos_g_p_mb": round(demo_res.pos_g_p_bytes / 1e6, 4),
    }
    return extended
