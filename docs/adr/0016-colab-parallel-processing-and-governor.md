# ADR-0016: Colab Parallel Processing and Resource Governor

## Context
Running 32 virtual ranks concurrently on a shared GPU or CPU on Google Colab free tier can cause Out Of Memory (OOM) crashes or session disconnections if resources are unconstrained.

## Decision
We enforce a strict `ResourceGovernor` (`zerosim.governor`):
1. **Concurrency Cap**: Thread compute concurrency is capped at $\le 50\%$ of available CPU cores via a `BoundedSemaphore`.
2. **CUDA Memory Cap**: In CUDA environments, GPU memory fraction is set to 0.5 (`torch.cuda.set_per_process_memory_fraction(0.5)`).
3. **Pre-flight Planning**: Before launching an experiment, `plan_memory` checks aggregate peak predicted memory against available RAM/VRAM and refuses over-budget configurations.
4. **Deadlock Prevention**: The compute semaphore is held only during local compute operations and is explicitly released before entering collective communication barriers.
