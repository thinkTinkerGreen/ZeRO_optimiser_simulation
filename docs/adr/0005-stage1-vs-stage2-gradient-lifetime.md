# ADR-0005: Stage 1 vs. Stage 2 Differentiation by Gradient Lifetime

## Context
In theoretical analyses, Stage 1 ($P_{os}$) and Stage 2 ($P_{os+g}$) both communicate $2\Psi$ elements per step. A common misconception is that their memory footprints only differ by $2\Psi / N$. In reality, the key architectural difference is *when* gradients are reduced and freed.

## Decision
- **Stage 1 ($P_{os}$)**: Full gradient buffers ($2\Psi$) are retained across the entire model throughout the backward pass. Only after backward completes does each rank reduce-scatter gradients and free the full gradient buffers before the optimizer step.
- **Stage 2 ($P_{os+g}$)**: Gradients are reduced eagerly during backward. We register post-accumulate gradient hooks on unit parameters. As soon as all gradients for unit $u$ are accumulated, they are flattened, reduce-scattered, and the full unit gradient buffer is freed immediately.

## Rationale
This matches the ZeRO paper's definition of $P_{os+g}$ (bucketed reduce-scatter during backward). Stage 2 peak memory is significantly lower because only one unit's gradient buffer is live at any moment during backward.
