"""`runs/pipeline_status.json` writer for the notebook STATUS cells (HANDOFF.md section 10)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scd.config import Config


def status_path(cfg: Config) -> Path:
    return cfg.runs_dir / "pipeline_status.json"


def read(cfg: Config) -> dict[str, Any]:
    p = status_path(cfg)
    if not p.exists():
        return {"stages": {}}
    return dict(json.loads(p.read_text()))


def write(cfg: Config, stage: str, outputs: list[str] | None = None, **extra: Any) -> dict[str, Any]:
    """Record that `stage` finished now, with its outputs and the config hash."""
    doc = read(cfg)
    doc.setdefault("stages", {})[stage] = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "outputs": outputs or [],
        "config_hash": cfg.hash(),
        **extra,
    }
    p = status_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2, default=str))
    return doc


def is_current(cfg: Config, stage: str) -> bool:
    """True if `stage` was completed under the current config hash (idempotent notebook cells)."""
    entry = read(cfg).get("stages", {}).get(stage)
    return bool(entry) and entry.get("config_hash") == cfg.hash()
