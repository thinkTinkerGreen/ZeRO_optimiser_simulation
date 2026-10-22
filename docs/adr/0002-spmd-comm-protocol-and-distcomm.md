# ADR-0002: SPMD Program and Comm Protocol with ThreadComm and DistComm

## Context
To prove that our virtualized multi-thread simulation is faithful to real distributed execution, we need identical stage algorithms to run seamlessly across both simulated virtual ranks and real multi-process environments.

## Decision
We define an abstract `Comm` protocol (in `zerosim.comm.base`) specifying:
- `all_reduce_mean(t: Tensor) -> Tensor`
- `reduce_scatter_mean(t: Tensor) -> Tensor`
- `all_gather(shard: Tensor) -> Tensor`
- `barrier() -> None`
- `stats: CommStats`

We provide two implementations:
1. `ThreadComm`: in-process multi-threaded communication using double barriers and rank slots.
2. `DistComm`: real multi-process communication backed by `torch.distributed` (Gloo backend on CPU). For `reduce_scatter_mean`, if Gloo does not natively support tensor reduction across unequal or specific topologies, it falls back to `all_reduce` followed by rank-sharding, recording the fallback in `CommStats.notes` while preserving logical accounting.

## Rationale
Running the exact same stage code against both backends guarantees that `ThreadComm` is mathematically and algorithmically identical to real multi-process execution (validated in experiment E6).
