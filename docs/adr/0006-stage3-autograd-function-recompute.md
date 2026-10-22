# ADR-0006: Stage 3 Autograd Function and Unit Forward Recomputation

## Context
In ZeRO-3 ($P_{os+g+p}$), parameters are sharded across ranks. In PyTorch standard autograd, any parameter used in forward that requires gradients is retained by autograd references until backward completes. If unsharded parameters are retained by autograd graph nodes, parameter memory would remain full throughout the forward and backward passes.

## Decision
For Stage 3, each architectural unit executes via a custom `torch.autograd.Function` (`ZeroUnitFn`):
1. **Forward**: All-gather parameter shards for unit $u$, unflatten into parameters, execute unit forward under `torch.no_grad()`, save only the unit input activation, and immediately free the gathered parameters.
2. **Backward**: All-gather parameter shards for unit $u$ again, rebuild parameters with `requires_grad=True`, recompute the unit forward pass under `torch.enable_grad()`, compute gradients with `torch.autograd.grad`, flatten and reduce-scatter gradients to shards, and immediately free full parameter and gradient buffers.
3. FLOPs from this backward recomputation are measured and explicitly reported under the distinct label `z3_recompute_simulation_overhead`.

## Rationale
This guarantees that gathered parameters are never retained in autograd graphs. Production frameworks like DeepSpeed or FSDP use custom C++/backward pre-hooks without recomputation; categorizing the recompute FLOPs as a simulation artifact maintains full honesty and transparent accounting.
