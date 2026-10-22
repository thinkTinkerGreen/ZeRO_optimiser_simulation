"""GPT model with explicit architectural units, causal attention, and parameter formulas."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from zerosim.config import ModelConfig
from zerosim.sharding import FlatUnit, create_flat_unit


def count_params(cfg: ModelConfig) -> int:
    """Exact closed-form parameter count for our GPT model."""
    d = cfg.d_model
    L = cfg.n_layer
    V = cfg.vocab
    T = cfg.seq_len

    embed_params = (V + T) * d
    block_params = L * (12 * (d**2) + 13 * d)
    head_params = (V + 2) * d + V
    return embed_params + block_params + head_params


class EmbedUnit(nn.Module):
    """Token and position embeddings unit."""

    def __init__(self, cfg: ModelConfig, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.wte = nn.Embedding(cfg.vocab, cfg.d_model, dtype=dtype)
        self.wpe = nn.Embedding(cfg.seq_len, cfg.d_model, dtype=dtype)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)
        return self.wte(idx) + self.wpe(pos)


class CausalSelfAttention(nn.Module):
    """Causal multi-head self-attention with explicit matmul and masked softmax."""

    def __init__(self, cfg: ModelConfig, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.d_model = cfg.d_model
        self.head_dim = cfg.d_model // cfg.n_head

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=True, dtype=dtype)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        # qkv: (b, t, 3*c)
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to (b, n_head, t, head_dim)
        q = q.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.head_dim).transpose(1, 2)

        # Explicit causal attention matmul + softmax
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.proj(out)


class MLP(nn.Module):
    """Feedforward network with GELU activation and 4x expansion."""

    def __init__(self, cfg: ModelConfig, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, 4 * cfg.d_model, bias=True, dtype=dtype)
        self.proj = nn.Linear(4 * cfg.d_model, cfg.d_model, bias=True, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(x)))


class BlockUnit(nn.Module):
    """Transformer block with pre-LayerNorm."""

    def __init__(self, cfg: ModelConfig, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.d_model, dtype=dtype)
        self.attn = CausalSelfAttention(cfg, dtype=dtype)
        self.ln_2 = nn.LayerNorm(cfg.d_model, dtype=dtype)
        self.mlp = MLP(cfg, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class HeadUnit(nn.Module):
    """Final LayerNorm and untied language model head."""

    def __init__(self, cfg: ModelConfig, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        self.ln_f = nn.LayerNorm(cfg.d_model, dtype=dtype)
        self.head = nn.Linear(cfg.d_model, cfg.vocab, bias=True, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.ln_f(x))


class GPT(nn.Module):
    """Modular GPT Transformer model organized into discrete FlatUnits."""

    def __init__(
        self,
        cfg: ModelConfig,
        generator: torch.Generator | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = EmbedUnit(cfg, dtype=dtype)
        self.blocks = nn.ModuleList([BlockUnit(cfg, dtype=dtype) for _ in range(cfg.n_layer)])
        self.head = HeadUnit(cfg, dtype=dtype)

        self._init_weights(generator)

    def _init_weights(self, generator: torch.Generator | None) -> None:
        """Initialize all weights from N(0, 0.02) using explicit Generator."""
        for p in self.parameters():
            if p.dim() >= 2:
                if generator is not None:
                    p.data.normal_(mean=0.0, std=0.02, generator=generator)
                else:
                    p.data.normal_(mean=0.0, std=0.02)
            else:
                # Biases and 1D parameters initialized to zero
                p.data.zero_()

    def get_unit_modules(self) -> list[tuple[str, nn.Module]]:
        """Return list of (unit_name, module) pairs in execution order."""
        units: list[tuple[str, nn.Module]] = [("embed", self.embed)]
        for i, b in enumerate(self.blocks):
            units.append((f"block_{i}", b))
        units.append(("head", self.head))
        return units

    def get_flat_units(self, world_size: int) -> list[FlatUnit]:
        """Construct FlatUnit descriptors for all model units."""
        flat_units: list[FlatUnit] = []
        for name, module in self.get_unit_modules():
            params = [(p_name, p.shape) for p_name, p in module.named_parameters()]
            flat_units.append(create_flat_unit(name, params, world_size=world_size))
        return flat_units

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.embed(idx)
        for b in self.blocks:
            x = b(x)
        return self.head(x)
