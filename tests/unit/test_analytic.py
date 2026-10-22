"""Unit tests for Milestone M2: analytic formulas and Table 1 reproduction (AC-03)."""

from zerosim.analytic import generate_extended_analytic_tables, reproduce_paper_table1


def test_reproduce_paper_table1():
    summary = reproduce_paper_table1()
    assert summary["total_cells"] == 54
    assert summary["passed_cells"] == 54
    assert summary["all_passed"] is True


def test_extended_analytic_tables():
    ext = generate_extended_analytic_tables(demo_psi=5_000_000)
    assert "1.5B_GPT2_XL" in ext
    assert len(ext["1.5B_GPT2_XL"]) == 6
    assert "demo_N32" in ext
    assert ext["demo_N32"]["N"] == 32
    assert ext["demo_N32"]["pos_g_p_mb"] < ext["demo_N32"]["baseline_mb"]
