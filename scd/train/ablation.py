"""Ablation runner (HANDOFF.md 7.5): runs/abl_<id>_<seed> and a row in runs/ablations.csv.

Milestone M8 (stub in this pass). scripts/14_ablation.py --dry-run will append a placeholder row
once this is built.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def run(
    ablation_id: str, seed: int = 42, chips_version: int = 1, overrides: Sequence[str] = (), dry_run: bool = False
) -> dict[str, Any]:
    """Train configs/ablations/<id>.yaml as abl_<id>_<seed>; append proxy, IoUs, robustness, wall-clock."""
    raise NotImplementedError("scd.train.ablation.run: built at M8; see HANDOFF.md section 7.5")
