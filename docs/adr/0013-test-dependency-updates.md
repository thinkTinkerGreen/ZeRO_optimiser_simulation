# ADR-0013: Rigorous Testing of Dependency Updates

## Context
Breaking changes across minor or major library releases (PyTorch, NumPy, Ruff) can silently alter numerical results or API contracts.

## Decision
Whenever any dependency updates:
1. Run all unit, property, and integration tests across Python 3.11, 3.12, and 3.13.
2. Verify all benchmarks, notebooks, and numerical tolerances pass without regression before bumping dependencies.
