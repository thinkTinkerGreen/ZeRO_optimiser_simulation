# Project Assumptions

Record of architectural and modeling assumptions following Decision Rule D0 (`date · decision · reason`).

- 2026-10-22 · GitHub repository URL: `https://github.com/thinkTinkerGreen/ZeRO_optimiser_simulation` · User prompt instruction.
- 2026-10-22 · Google Drive results folder: `MyDrive/ZeRO_optimiser_simulation_results` · Standard default path specified in assignment prompt §1.3.
- 2026-10-22 · License: MIT · Standard open-source permissive license default in §1.3.
- 2026-10-22 · Lower bound on torch dependency is `torch>=2.4.0` · Required for modern `torch.autograd.graph.saved_tensors_hooks` and stable Gloo/DTensor APIs.
- 2026-10-22 · Adam FLOPs per parameter: 10 FLOPs · Standard elementwise first/second moment and decoupled update computation (ADR-0008). Sensitivity bounds of 5 and 20 FLOPs reported.
- 2026-10-22 · Model parameter counts · For `tiny`: $L=2, d=32, H=2, V=64, T=16$; for `demo`: $L=6, d=256, H=4, V=512, T=64$ ($\approx 4.88\times 10^6$ parameters, satisfying $\le 8{,}000{,}000$).
- 2026-10-22 · Default compute performance placeholder for latency-bandwidth model: $2\times 10^{13}$ FLOP/s (20 TFLOPS) · Representative FP16 tensor core throughput baseline for cost comparisons.
- 2026-10-22 · Interconnect latency parameter: $\alpha = 5\ \mu\text{s}$ · Typical baseline network round-trip overhead assumption.
- 2026-10-22 · Table 1 verification rule · The original ZeRO paper rounds to 1 decimal place for small values (e.g. 41.25 -> 41.3) but truncates to integer for several large table entries (e.g. 257.75 -> 257, 2218.75 -> 2218). The verification logic supports both nearest-rounding and integer truncation matching.
