# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-22

### Added
- Core ZeRO simulator implementation for Stage 0 (DDP), Stage 1 ($P_{os}$), Stage 2 ($P_{os+g}$), and Stage 3 ($P_{os+g+p}$).
- `ThreadComm` 32-virtual-GPU SPMD communication engine with collective signature verification.
- `DistComm` backend wrapping `torch.distributed` (Gloo) for multi-process parity verification.
- Flat unit-wise parameter sharding with hypothesis property tests for arbitrary world sizes.
- Exact `MemoryLedger` tracking logical vs. actual allocations, autograd saved activation hooks, and timeline generation.
- GPT transformer model with modular architectural units (`embed`, `block_0`..`block_{L-1}`, `head`).
- FLOPs accounting and $\alpha$-$\beta$ latency-bandwidth cost modeling (NVLink and InfiniBand presets).
- Resource governor capping CPU threads, memory allocations, and guarding against OOM.
- Google Drive resumable atomic persistence layer.
- Comprehensive test suite covering E0 through E8 with strict numerical validation.
