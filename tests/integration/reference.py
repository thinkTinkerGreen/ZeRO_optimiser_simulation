"""Single-process reference trainer using standard PyTorch AdamW and full global batch."""

from __future__ import annotations

import torch

from zerosim.config import SimConfig
from zerosim.data import compute_loss, generate_batch
from zerosim.determinism import get_generator
from zerosim.model import GPT


def run_reference_trainer(
    cfg: SimConfig,
    steps: int | None = None,
) -> tuple[list[float], torch.Tensor]:
    """Train single-process reference model on full global batch in float64.

    Returns:
        losses: list of per-step loss values
        flat_params: flattened tensor of all model parameters after training
    """
    num_steps = steps if steps is not None else cfg.steps
    gen = get_generator(cfg.seed, rank=0, step=0)
    model = GPT(cfg.model, generator=gen, dtype=torch.float64)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=cfg.betas,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )

    losses: list[float] = []

    for step in range(num_steps):
        inputs, targets = generate_batch(cfg, step=step, device="cpu")
        optimizer.zero_grad()
        logits = model(inputs)
        loss = compute_loss(logits, targets)
        losses.append(loss.item())

        loss.backward()
        optimizer.step()

    # Collect flattened parameter vector
    flat_params = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
    return losses, flat_params
