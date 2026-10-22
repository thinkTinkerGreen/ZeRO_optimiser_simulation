"""Deterministic synthetic dataset and batch partitioning across ranks."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from zerosim.config import SimConfig


def generate_batch(
    cfg: SimConfig,
    step: int,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate global batch of sequences: inputs (B_glob, T) and targets (B_glob, T).

    Sequence rule: x[t+1] = (31 * x[t] + 17) % V
    """
    total_batch = cfg.world_size * cfg.microbatch
    seq_len = cfg.model.seq_len
    vocab = cfg.model.vocab

    # Generate start tokens deterministically using Generator(seed + step)
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed + step)
    start_tokens = torch.randint(0, vocab, (total_batch, 1), generator=gen, dtype=torch.long)

    # Generate sequence using deterministic recurrence
    full_seq = [start_tokens]
    curr = start_tokens
    for _ in range(seq_len):
        curr = (31 * curr + 17) % vocab
        full_seq.append(curr)

    seq_tensor = torch.cat(full_seq, dim=1).to(device)
    # Inputs: [0:T], Targets: [1:T+1]
    inputs = seq_tensor[:, :seq_len]
    targets = seq_tensor[:, 1 : seq_len + 1]

    return inputs, targets


def get_rank_batch(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    rank: int,
    microbatch: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract rank's microbatch partition from global batch."""
    start = rank * microbatch
    end = (rank + 1) * microbatch
    return inputs[start:end], targets[start:end]


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Mean cross-entropy loss over tokens."""
    # logits: (B, T, V), targets: (B, T)
    b, t, v = logits.shape
    return F.cross_entropy(logits.reshape(-1, v), targets.reshape(-1))
