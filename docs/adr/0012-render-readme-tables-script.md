# ADR-0012: README Generation via tools/render_readme_tables.py

## Context
Hand-typed numbers and static markdown tables frequently drift from real experimental results, leading to stale or fabricated performance metrics.

## Decision
All numerical tables and benchmark summaries in `README.md` must be dynamically rendered between `<!-- BEGIN:generated:NAME -->` and `<!-- END:generated:NAME -->` comment tags using `tools/render_readme_tables.py`. CI validates that no drift exists between committed results JSON files and README contents via `render_readme_tables.py --check`.
