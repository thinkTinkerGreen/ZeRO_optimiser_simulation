# ADR-0010: Maintain Comprehensive docs/STUDY_GUIDE.md

## Context
Rule R1 explicitly prohibits the AI agent from writing, drafting, or ghostwriting the human author's personal understanding in `docs/UNDERSTANDING.md`. However, the human author needs a clear, technically rigorous reference to learn and deeply understand the concepts.

## Decision
We maintain `docs/STUDY_GUIDE.md`, clearly labeled as "Agent-written reference — not part of the understanding attestation". It provides:
- Mathematical derivations for mixed-precision memory breakdowns ($16\Psi$).
- Step-by-step communication volume analysis ($2\Psi$ vs $3\Psi$).
- Worked numeric example for $\Psi=8, N=4$.
- Differences between ZeRO stages and commercial frameworks (FSDP2, DeepSpeed).
