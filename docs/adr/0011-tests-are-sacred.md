# ADR-0011: Tests are Sacred — No Deletion, Loosening, or Skipping

## Context
In scientific and high-performance computing simulations, loosening test tolerances or skipping failing tests masks subtle bugs and invalidates numerical claims.

## Decision
No test may ever be deleted, skipped, loosened, or marked with `xfail` simply to make CI green. If a numerical tolerance must change due to hardware or precision differences, an ADR must be documented with theoretical justification and mathematical error bounds.
