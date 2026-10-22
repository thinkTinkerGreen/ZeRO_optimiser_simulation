# Contributing Guidelines

Thank you for contributing to `ZeRO_optimiser_simulation`!

## Development Workflow

1. Install prerequisites: Python 3.10+, `uv`, and `git`.
2. Set up the development virtual environment:
   ```bash
   uv venv --python 3.11
   source .venv/bin/activate
   uv pip install -e ".[dev,torch]"
   ```
3. Run tests and linters:
   ```bash
   make lint
   make test
   ```
4. Adhere to Conventional Commits format (`feat:`, `fix:`, `docs:`, `test:`, `refactor:`).
5. All tests must be deterministic, non-flaky, and satisfy numerical tolerances.
