"""Unit tests for model, data, and flat AdamW parity with PyTorch AdamW (AC-06)."""

import torch

from zerosim.config import MODEL_DEMO, MODEL_TINY, SimConfig
from zerosim.data import generate_batch, get_rank_batch
from zerosim.determinism import get_generator
from zerosim.model import GPT, count_params
from zerosim.optim import FlatShardAdamW


def test_model_param_counts():
    gen = get_generator(42)
    m_tiny = GPT(MODEL_TINY, generator=gen)
    actual_tiny = sum(p.numel() for p in m_tiny.parameters())
    assert count_params(MODEL_TINY) == actual_tiny

    m_demo = GPT(MODEL_DEMO, generator=gen)
    actual_demo = sum(p.numel() for p in m_demo.parameters())
    assert count_params(MODEL_DEMO) == actual_demo
    assert actual_demo <= 8_000_000


def test_data_generation_and_partitioning():
    cfg = SimConfig(world_size=4, microbatch=2, model=MODEL_TINY)
    inputs, targets = generate_batch(cfg, step=0)
    assert inputs.shape == (8, MODEL_TINY.seq_len)
    assert targets.shape == (8, MODEL_TINY.seq_len)

    r_inp, r_tar = get_rank_batch(inputs, targets, rank=2, microbatch=2)
    assert r_inp.shape == (2, MODEL_TINY.seq_len)
    assert torch.equal(r_inp, inputs[4:6])


def test_flat_shard_adamw_vs_pytorch_parity():
    """Verify FlatShardAdamW produces identical updates to torch.optim.AdamW within 1e-12."""
    torch.manual_seed(1234)
    # Generate test parameters and gradients in float64
    p_torch = [torch.randn(10, 10, dtype=torch.float64, requires_grad=True) for _ in range(3)]
    # Flat copies for our optimizer
    flat_params = [p.detach().clone().reshape(-1) for p in p_torch]

    opt_torch = torch.optim.AdamW(p_torch, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
    opt_custom = FlatShardAdamW(
        flat_params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2
    )

    for step in range(5):
        grads = [torch.randn_like(p) for p in p_torch]
        for p, g in zip(p_torch, grads):
            p.grad = g.clone()
        opt_torch.step()

        flat_grads = [g.reshape(-1) for g in grads]
        opt_custom.step(flat_grads)

        for p_t, p_c in zip(p_torch, flat_params):
            max_diff = (p_t.detach().reshape(-1) - p_c).abs().max().item()
            assert max_diff <= 1e-12, f"Discrepancy {max_diff} at step {step}"
