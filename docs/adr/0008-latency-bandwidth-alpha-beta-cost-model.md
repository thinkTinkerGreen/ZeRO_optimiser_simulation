# ADR-0008: Latency-Bandwidth (α-β) Cost Model

## Context
Accurately benchmarking multi-GPU hardware wall-clock time on a single CPU machine or single Colab GPU is impossible because virtual ranks share the same bus and memory controllers.

## Decision
We implement an analytical $\alpha$-$\beta$ communication-computation cost model (`zerosim.costmodel`):
$$t_{\text{step}} = t_{\text{compute}} + t_{\text{comm}}$$
where:
$$t_{\text{compute}} = \frac{\text{FLOPs}}{\text{TFLOPS}_{\text{device}}}$$
$$t_{\text{comm}} = \sum_{\text{collectives}} \left(\alpha + \frac{\text{ring\_bytes}}{\text{Bandwidth}}\right)$$
We provide two standard presets:
1. `nvlink_like`: $\alpha = 5\ \mu\text{s}$, $\text{BW} = 300\ \text{GB/s}$
2. `ib_like`: $\alpha = 5\ \mu\text{s}$, $\text{BW} = 12.5\ \text{GB/s}$

Simulator wall-clock time is tracked separately and clearly labeled as "simulator time, not hardware time".
