# ADR-0004: Unit-Wise Flat Sharding with Padding

## Context
In deep neural networks, individual layers and modules have parameters with varying tensor shapes and dimensions. Splitting arbitrary-shaped tensors across $N$ ranks can lead to non-divisible dimensions and complex indexing.

## Decision
We partition the model into structural architectural units: `[embed, block_0, ..., block_{L-1}, head]`. For each unit:
1. All parameter tensors are flattened into a contiguous 1-D vector.
2. The vector is zero-padded to `padded_numel = ceil(numel / N) * N`.
3. Contiguous slices of size `shard_numel = padded_numel / N` are assigned to rank $r$.

## Rationale
- Enables simple, bit-exact round-tripping (`unflatten(flatten(x)) == x`).
- Guarantees uniform shard sizes across all $N$ ranks.
- Shard padding region gradients remain zero; hence optimizer updates produce zero, preserving zero-padded boundaries.
- Padding overhead $\frac{\sum \text{padded\_numel}}{\sum \text{numel}} - 1$ is explicitly tracked and reported.
