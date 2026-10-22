.PHONY: setup lint fmt type test test-dist nb-smoke results site all clean

PYTHON := .venv/bin/python
PYTEST := .venv/bin/pytest
RUFF := .venv/bin/ruff
PYRIGHT := .venv/bin/pyright

setup:
	uv venv --python 3.11
	uv pip install -e ".[dev,torch]"

lint:
	$(RUFF) check .
	$(RUFF) format --check .

fmt:
	$(RUFF) check --fix .
	$(RUFF) format .

type:
	$(PYRIGHT)

test:
	PYTHONHASHSEED=0 $(PYTEST) tests/unit tests/property tests/integration --cov=src/zerosim --cov-report=term-missing --cov-fail-under=90

test-dist:
	PYTHONHASHSEED=0 $(PYTEST) tests/dist -v

nb-smoke:
	ZEROSIM_SMOKE=1 $(PYTEST) --nbmake notebooks/ZeRO_optimiser_sim.ipynb
	$(PYTHON) tools/check_notebook.py --smoke-check

results:
	$(PYTHON) tools/render_readme_tables.py

site:
	$(PYTHON) tools/build_site.py

all: lint type test test-dist results site
