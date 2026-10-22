# ADR-0009: src Layout and Optional PyTorch Extra

## Context
Google Colab ships with its own pre-installed, CUDA-optimized PyTorch build. Reinstalling PyTorch on Colab often downgrades or breaks CUDA runtime bindings. In CI, we want lean, CPU-only PyTorch wheels.

## Decision
- Use standard `src/` layout: `src/zerosim/`.
- Declare base dependencies in `pyproject.toml` (`numpy`, `matplotlib`).
- Provide PyTorch as an optional extra: `torch = ["torch>=2.4.0"]`.
- On Colab, execute `pip install -e .` without reinstalling PyTorch.
- In CI and local development, install `.[dev,torch]`.
