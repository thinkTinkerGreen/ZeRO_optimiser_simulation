# ADR-0007: Flat-Shard AdamW Optimizer

## Context
Standard PyTorch optimizers (`torch.optim.AdamW`) operate over lists of parameter tensors. In ZeRO stages 1, 2, and 3, ranks maintain contiguous 1-D shard buffers for master parameters, first moments $m$, and second moments $v$.

## Decision
We implement a dedicated `FlatShardAdamW` optimizer (`zerosim.optim`) operating on contiguous 1-D tensors with:
- Decoupled weight decay ($w \leftarrow w \cdot (1 - \text{lr} \cdot \lambda)$)
- Bias-corrected first and second moment updates ($\beta_1, \beta_2$)
- Numerical validation against `torch.optim.AdamW` in float64 with tolerance $\le 10^{-12}$.

## Rationale
Enables direct memory shard indexing without slicing overhead or unnecessary Python object creations, while guaranteeing mathematical equivalence with PyTorch's reference AdamW.
