# ZeRO Optimization: Comprehensive Study Guide

> **Agent-written reference — not part of the understanding attestation**  
> *This document is written by the coding agent as a comprehensive educational and mathematical guide to assist the human author in studying, understanding, and mastering ZeRO concepts from first principles.*

---

## 1. Executive Summary & Why ZeRO Exists

In standard Data Parallelism (DDP), every GPU holds an identical replica of:
1. Model parameters
2. Parameter gradients
3. Optimizer states (e.g. Adam master parameters, momentum, and variance)

As models scale to billions or trillions of parameters, the memory required to store these states exceeds the VRAM of a single accelerator (e.g. 16 GB on a T4, 80 GB on an A100/H100), even when batch size per GPU is 1.

The **Zero Redundancy Optimizer (ZeRO)** (Rajbhandari et al., SC20) eliminates memory redundancy across data-parallel ranks by partitioning model states across the $N$ data-parallel devices instead of replicating them, while retaining the computational granularity of data parallelism.

---

## 2. Derivation of the 16Ψ Byte Rule for Mixed-Precision Adam

Let $\Psi$ denote the total number of model parameters.

In standard mixed-precision training:
1. **Model Parameters ($W$)**: Stored in 16-bit floating point (`fp16` or `bf16`) for forward and backward passes.
   $$\text{Bytes} = 2 \times \Psi$$
2. **Gradients ($G$)**: Computed and stored in 16-bit floating point (`fp16` or `bf16`).
   $$\text{Bytes} = 2 \times \Psi$$
3. **Optimizer States ($O$)**: Standard Adam maintains:
   - **FP32 Master Parameters**: Required to prevent numerical underflow when small weight updates are added ($4 \times \Psi$ bytes).
   - **FP32 First Moment (Momentum $m$)**: Moving average of gradients ($4 \times \Psi$ bytes).
   - **FP32 Second Moment (Variance $v$)**: Moving average of squared gradients ($4 \times \Psi$ bytes).
   $$\text{Optimizer State Bytes} = (4 + 4 + 4) \times \Psi = 12\Psi \quad (K = 12)$$

### Total Model-State Memory per Device in Standard DDP:
$$\text{Total} = 2\Psi + 2\Psi + 12\Psi = 16\Psi \text{ bytes}$$

For example, a 7.5 billion parameter model ($\Psi = 7.5\times 10^9$):
$$\text{Memory} = 16 \times 7.5 \times 10^9 \text{ bytes} = 120 \text{ GB}$$
This exceeds the capacity of an 80 GB A100 GPU before allocating a single activation tensor!

---

## 3. The Three ZeRO-DP Stages

### Stage 1: Optimizer State Partitioning ($P_{os}$)
- **What is partitioned**: Adam states ($M, m, v$, $K=12$ bytes/param) are split evenly across $N$ ranks.
- **What is replicated**: Working parameters ($2\Psi$) and gradients ($2\Psi$) remain fully replicated on every device.
- **Memory footprint**:
  $$\text{Mem}_{P_{os}} = 2\Psi + 2\Psi + \frac{K\Psi}{N} = 4\Psi + \frac{12\Psi}{N} \text{ bytes}$$
- **Communication volume**: Identical to standard DDP ($2\Psi$ elements communicated per step via reduce-scatter + all-gather).

### Stage 2: Optimizer State + Gradient Partitioning ($P_{os+g}$)
- **What is partitioned**: Both Adam states ($12\Psi / N$) and gradients ($2\Psi / N$).
- **What is replicated**: Working parameters ($2\Psi$) remain replicated.
- **Key mechanism**: During the backward pass, as each layer finishes computing gradients, its gradients are immediately **reduce-scattered** to the rank responsible for updating that shard, and the full layer gradient buffer is immediately freed.
- **Memory footprint**:
  $$\text{Mem}_{P_{os+g}} = 2\Psi + \frac{2\Psi}{N} + \frac{K\Psi}{N} = 2\Psi + \frac{14\Psi}{N} \text{ bytes}$$
- **Communication volume**: Still $2\Psi$ elements communicated per step (only the timing changes: gradients are reduce-scattered eagerly during backward rather than all-reduced after backward).

### Stage 3: Full Sharding ($P_{os+g+p}$)
- **What is partitioned**: Optimizer states, gradients, **and** working model parameters.
- **What is replicated**: Nothing! Each rank stores only a $1/N$ shard of parameters, gradients, and optimizer states.
- **Key mechanism**:
  - In **forward**: When a layer begins, its parameters are **all-gathered** on-the-fly from all ranks. The layer executes forward, and the unsharded parameters are immediately discarded.
  - In **backward**: Parameters for that layer are **all-gathered** again. Gradients are computed, immediately **reduce-scattered** to shards, and both unsharded parameters and unsharded gradients are discarded.
- **Memory footprint**:
  $$\text{Mem}_{P_{os+g+p}} = \frac{2\Psi + 2\Psi + K\Psi}{N} = \frac{16\Psi}{N} \text{ bytes}$$
- **Communication volume**: $3\Psi$ elements per step ($1.5\times$ baseline DDP):
  1. Forward parameter all-gather: $\Psi$
  2. Backward parameter all-gather: $\Psi$
  3. Backward gradient reduce-scatter: $\Psi$

---

## 4. Worked Numeric Example ($\Psi = 8$ parameters, $N = 4$ ranks)

Suppose we have an 8-parameter model ($\Psi = 8$) partitioned across $N = 4$ virtual GPUs, using mixed precision ($K = 12$ bytes/param for optimizer states).

Each rank $r \in \{0, 1, 2, 3\}$ owns a shard of $8 / 4 = 2$ parameters:
- Rank 0 owns indices $\{0, 1\}$
- Rank 1 owns indices $\{2, 3\}$
- Rank 2 owns indices $\{4, 5\}$
- Rank 3 owns indices $\{6, 7\}$

### Comparison Table for Rank 0:

| Component | Baseline DDP | Stage 1 ($P_{os}$) | Stage 2 ($P_{os+g}$) | Stage 3 ($P_{os+g+p}$) |
|---|---|---|---|---|
| **Parameters $W$** | 8 params (16 B) | 8 params (16 B) | 8 params (16 B) | **2 params (4 B)** |
| **Gradients $G$** | 8 grads (16 B) | 8 grads (16 B) | **2 grads (4 B)** (+ transient live unit) | **2 grads (4 B)** (+ transient live unit) |
| **Master $M$** | 8 floats (32 B) | **2 floats (8 B)** | **2 floats (8 B)** | **2 floats (8 B)** |
| **Momentum $m$** | 8 floats (32 B) | **2 floats (8 B)** | **2 floats (8 B)** | **2 floats (8 B)** |
| **Variance $v$** | 8 floats (32 B) | **2 floats (8 B)** | **2 floats (8 B)** | **2 floats (8 B)** |
| **Total Model State** | **128 Bytes** | **56 Bytes** | **38 Bytes** | **32 Bytes** |

- Paper formula check:
  - Baseline: $16 \times 8 = 128\text{ B}$.
  - Stage 1: $4\times 8 + (12\times 8)/4 = 32 + 24 = 56\text{ B}$.
  - Stage 2: $2\times 8 + (14\times 8)/4 = 16 + 28 = 44\text{ B}$ (with steady shard grads) / steady state $2\times 8 + 2\times 2 + 12\times 2 = 44\text{ B}$.
  - Stage 3: $16 \times 8 / 4 = 32\text{ B}$.

---

## 5. Communication: Why Stage 1 is $2\Psi$ and Stage 3 is $3\Psi$

Recall ring-based collective communication fundamentals:
- **All-Reduce** is mathematically and operationally composed of two distinct phases:
  1. **Reduce-Scatter**: Each rank receives a reduced shard of the sum ($1\times \Psi$ volume).
  2. **All-Gather**: Each rank broadcasts its reduced shard to all other ranks to reconstruct the full sum ($1\times \Psi$ volume).
  $$\text{All-Reduce}(\Psi) = \text{Reduce-Scatter}(\Psi) + \text{All-Gather}(\Psi) = 2\Psi$$

### In Stage 1:
- Gradients are reduced using Reduce-Scatter $\to \Psi$ elements communicated.
- Optimizer updates the shards.
- Updated parameters are gathered using All-Gather $\to \Psi$ elements communicated.
- Total communication = $\Psi + \Psi = 2\Psi$ (identical to standard DDP).

### In Stage 3:
- Forward pass: All-Gather parameters layer-by-layer $\to \Psi$ elements.
- Backward pass: All-Gather parameters layer-by-layer $\to \Psi$ elements.
- Backward pass: Reduce-Scatter gradients layer-by-layer $\to \Psi$ elements.
- Total communication = $\Psi + \Psi + \Psi = 3\Psi$ ($1.5\times$ communication overhead).

---

## 6. Real-World Framework Mapping

| Feature | DeepSpeed | PyTorch Distributed | Our Simulator |
|---|---|---|---|
| **ZeRO-1** | `"zero_optimization": {"stage": 1}` | `torch.distributed.optim.ZeroRedundancyOptimizer` | `zerosim.stages.Stage1` |
| **ZeRO-2** | `"zero_optimization": {"stage": 2}` | Custom hooks or legacy FSDP1 | `zerosim.stages.Stage2` |
| **ZeRO-3** | `"zero_optimization": {"stage": 3}` | `torch.distributed.fsdp.fully_shard` (FSDP2) | `zerosim.stages.Stage3` |
| **Activation Sharding** | ZeRO-R / Activation Partitioning | Activation Checkpointing + DTensor | Measured via autograd hooks |
| **Offload** | ZeRO-Offload (CPU/NVMe) | FSDP CPU Offload | N/A (in-core simulation) |

---

## 7. Glossary

- **SPMD (Single Program, Multiple Data)**: Distributed computing paradigm where every worker rank runs an identical copy of the algorithm on a distinct partition of data.
- **DTensor (Distributed Tensor)**: PyTorch's native multi-dimensional tensor abstraction representing a global tensor partitioned across a DeviceMesh.
- **Reduce-Scatter**: Collective operation where input arrays from all ranks are elementwise summed and the resulting array is split into shards, with each rank receiving one shard.
- **All-Gather**: Collective operation where each rank contributes a local shard, and all shards are concatenated and distributed to every rank.
- **Residual Memory**: Non-model-state memory consumed during training, predominantly activations, temporary workspace buffers, and allocator fragmentation.
- **Padding Overhead**: Extra zero-elements appended to flat parameter vectors to ensure exact divisibility by world size $N$.
