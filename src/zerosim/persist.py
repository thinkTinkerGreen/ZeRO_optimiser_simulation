"""Google Drive and filesystem atomic persistence and resumable experiment runs."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from zerosim.config import SimConfig
from zerosim.determinism import get_environment_metadata
from zerosim.governor import get_available_memory

SCHEMA_VERSION = "1.0.0"


def get_results_root() -> Path:
    """Return the active results directory root from ZEROSIM_RESULTS env var or default."""
    root_str = os.environ.get("ZEROSIM_RESULTS", "./results")
    path = Path(root_str).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomically write JSON data using temporary file and fsync."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def is_run_completed(exp_name: str, cfg: SimConfig) -> bool:
    """Check if result.json already exists for this experiment and config hash."""
    run_id = f"{exp_name}-{cfg.config_hash}"
    result_path = get_results_root() / run_id / "result.json"
    if not result_path.exists():
        return False
    try:
        with open(result_path, encoding="utf-8") as f:
            data = json.load(f)
        return (
            data.get("schema_version") == SCHEMA_VERSION
            and data.get("config_hash") == cfg.config_hash
        )
    except Exception:
        return False


def load_cached_result(exp_name: str, cfg: SimConfig) -> dict[str, Any] | None:
    """Load cached result dictionary if valid."""
    if not is_run_completed(exp_name, cfg):
        return None
    run_id = f"{exp_name}-{cfg.config_hash}"
    result_path = get_results_root() / run_id / "result.json"
    with open(result_path, encoding="utf-8") as f:
        return json.load(f)


def save_run_result(
    exp_name: str,
    cfg: SimConfig,
    result_data: dict[str, Any],
    timeline: list[dict[str, Any]] | None = None,
    started_utc: str | None = None,
) -> Path:
    """Save experiment result atomically with complete environment manifest."""
    root = get_results_root()
    run_id = f"{exp_name}-{cfg.config_hash}"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    finished_utc = datetime.now(timezone.utc).isoformat()
    if started_utc is None:
        started_utc = finished_utc

    env_meta = get_environment_metadata()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "exp_name": exp_name,
        "run_id": run_id,
        "config_hash": cfg.config_hash,
        "git_sha": env_meta["git_sha"],
        "python": sys.version,
        "torch": env_meta["torch_version"],
        "device_name": env_meta["device_name"],
        "cpu_count": env_meta["cpu_count"],
        "ram_bytes": get_available_memory(),
        "seed": cfg.seed,
        "config": cfg.to_dict(),
        "started_utc": started_utc,
        "finished_utc": finished_utc,
    }

    result_payload = {
        "schema_version": SCHEMA_VERSION,
        "exp_name": exp_name,
        "run_id": run_id,
        "config_hash": cfg.config_hash,
        **result_data,
    }

    atomic_write_json(run_dir / "manifest.json", manifest)
    atomic_write_json(run_dir / "result.json", result_payload)

    if timeline is not None:
        atomic_write_json(run_dir / "timeline_rank0.json", {"timeline": timeline})

    return run_dir
