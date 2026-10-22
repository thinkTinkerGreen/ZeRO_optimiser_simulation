"""Configuration data classes and policies for ZeRO simulator."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ModelConfig:
    name: str
    n_layer: int
    d_model: int
    n_head: int
    vocab: int
    seq_len: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Predefined standard model configs
MODEL_TINY = ModelConfig(name="tiny", n_layer=2, d_model=32, n_head=2, vocab=64, seq_len=16)
MODEL_DEMO = ModelConfig(name="demo", n_layer=6, d_model=256, n_head=4, vocab=512, seq_len=64)


@dataclass(frozen=True)
class DtypePolicy:
    name: str
    param_bytes: int
    grad_bytes: int
    master_bytes: int
    m_bytes: int
    v_bytes: int
    comm_param_bytes: int
    comm_grad_bytes: int

    @property
    def K(self) -> int:
        """Optimizer state bytes per parameter (master + momentum + variance)."""
        return self.master_bytes + self.m_bytes + self.v_bytes

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Predefined precision policies
POLICY_MIXED_FP16 = DtypePolicy(
    name="mixed_fp16",
    param_bytes=2,
    grad_bytes=2,
    master_bytes=4,
    m_bytes=4,
    v_bytes=4,
    comm_param_bytes=2,
    comm_grad_bytes=2,
)

POLICY_FP32 = DtypePolicy(
    name="fp32",
    param_bytes=4,
    grad_bytes=4,
    master_bytes=0,
    m_bytes=4,
    v_bytes=4,
    comm_param_bytes=4,
    comm_grad_bytes=4,
)


@dataclass(frozen=True)
class SimConfig:
    world_size: int = 32
    stage: int = 0
    policy: DtypePolicy = POLICY_MIXED_FP16
    model: ModelConfig = MODEL_DEMO
    microbatch: int = 2
    steps: int = 20
    seed: int = 1234
    compute_dtype: str = "float32"
    device: str = "cpu"
    lr: float = 1e-3
    betas: tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    weight_decay: float = 0.0
    max_concurrent: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["policy"] = self.policy.to_dict()
        d["model"] = self.model.to_dict()
        return d

    @property
    def config_hash(self) -> str:
        """Deterministic 16-hex sha256 of canonical JSON."""
        canonical = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
