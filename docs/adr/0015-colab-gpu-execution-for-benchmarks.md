# ADR-0015: Colab GPU Execution for Benchmarks

## Context
Running 32-rank deep learning simulations on headless CPU-only machines (such as OCI aarch64 without GPU) has no access to CUDA hardware features, limiting benchmarking fidelity. Google Colab provides free-tier NVIDIA T4 GPUs with 15–16 GB VRAM.

## Decision
All full-scale benchmarks and CUDA validation tests (E7) are designed to run in a Google Colab GPU environment. The local environment runs determinism, unit, property, and CPU integration tests.
