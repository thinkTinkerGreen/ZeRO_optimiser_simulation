# Known Issues and Limitations

Record of known limitations, platform edge cases, and simulation boundaries.

## Limitations of Simulation
1. **No Overlap of Compute and Communication**: The cost model calculates $t_{\text{total}} = t_{\text{compute}} + t_{\text{comm}}$ without pipelined overlapping. Modern systems (such as DeepSpeed and FSDP2) pipeline backward gradient reduction with upstream backward layers.
2. **Simplified Memory Allocator Dynamics**: The simulator tracks exact tensor allocations (`actual_bytes` and `logical_bytes`) via `MemoryLedger` and saved tensor hooks, but does not simulate CUDA caching allocator memory fragmentation or page table overhead.
3. **Stage 3 Simulation Recomputation Overhead**: In our Python-level simulation, Stage 3 backward recomputes each unit's forward to avoid autograd retaining unsharded weights in memory (ADR-0006). These additional FLOPs are tracked separately under `z3_recompute_simulation_overhead`. Production C++ implementations use backward pre-hooks and do not recompute forward activations.
4. **Platform Availability**: CUDA validation (E7) requires a physical GPU (such as Google Colab free tier NVIDIA T4). On headless CPU environments, E7 is safely skipped with a logged reason.
