# ADR-0014: Preservation of Tests Directory

## Context
Accidental deletion or fragmentation of tests reduces code reliability and regression safety.

## Decision
The `tests/` directory and its contents must never be deleted. If any test becomes legacy, it is moved to `tests/legacy/` with an explanatory comment detailing the architectural transition.
