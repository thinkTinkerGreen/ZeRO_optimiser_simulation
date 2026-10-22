# ADR-0001: One Process, 32 Threads (ThreadComm) Virtual GPUs

## Context
Simulating 32 virtual GPUs requires coordinating 32 SPMD rank programs. Running 32 distinct OS processes on a single host (such as Google Colab free tier with ~12.7 GB RAM or CPU containers) incurs massive memory overhead. A single idle `import torch` process consumes ~250–300 MB RSS on Linux. Spawning 32 separate processes would demand ~8–10 GB RAM just for baseline Python/PyTorch runtime initialization, risking host OOM before model state allocation begins.

## Decision
We implement virtual ranks as 32 concurrent worker threads (`ThreadComm`) within a single process. Each rank maintains private tensors, a private `MemoryLedger`, and an independent step execution lifecycle, interacting with other ranks strictly through collective communication operations (`all_reduce_mean`, `reduce_scatter_mean`, `all_gather`, `barrier`).

## Rationale & Measured Evidence
- Threading shares the Python runtime, PyTorch binary text, and CUDA/CPU context, keeping base runtime memory footprint below 400 MB total.
- Synchronization is deterministic: thread barriers synchronize collective calls, and rank 0 executes order-deterministic reductions across rank slots.
- Eliminates IPC / socket serialization overhead, providing fast simulation speed.

## Consequences
- Requires careful handling of shared state: each rank MUST hold private tensors and never mutate shared buffers.
- Python GIL is released during PyTorch C++ tensor operations. A bounded semaphore (`ResourceGovernor`) caps concurrent compute segments to prevent CPU thrashing.
