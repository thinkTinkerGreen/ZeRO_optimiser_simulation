# ADR-0003: Two Byte Counters per Tensor (Logical vs. Actual Bytes)

## Context
The ZeRO paper (Rajbhandari et al., 2020) assumes a mixed-precision policy where parameters and gradients are stored in FP16 (2 bytes) and master parameters, momentum, and variance are stored in FP32 (4 bytes each, totaling $K=12$ bytes/param). However, running numerical verification on CPU or float64 requires actual memory representations that differ from the theoretical mixed-precision specification.

## Decision
`MemoryLedger` tracks two byte counters for every allocated tensor:
1. `logical_bytes`: Computed strictly according to the active `DtypePolicy` (e.g. `MIXED_FP16`: 2B param, 2B grad, 4B master, 4B m, 4B v).
2. `actual_bytes`: Computed from actual runtime memory (`numel * tensor.element_size()`).

## Rationale
- `logical_bytes` allows direct, bit-for-bit comparison against paper closed-form formulas and Table 1.
- `actual_bytes` tracks real device consumption, ensuring the simulator never breaches host/GPU limits.
