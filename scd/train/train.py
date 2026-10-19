"""Training entry points.

HANDOFF.md section 7. Milestone M6/M9. finetune is an M9 stub.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scd.config import Config


def run(cfg: Config, run_name: str, chips_version: int = 1, resume: bool = False) -> Path:
    """Train; returns the run dir."""
    raise NotImplementedError("scd.train.train.run: built at M6/M9; see HANDOFF.md section 7")


def finetune(cfg: Config, run_name: str) -> Path:
    """Stage-III fine-tune with the 50/50 sampler and forgetting check."""
    raise NotImplementedError("scd.train.train.finetune: built at M6/M9; see HANDOFF.md section 7")
