"""Evaluation of a run on a split: per-class / per-stratum tables, confusion matrix, robustness, worst chips.

HANDOFF.md 7.1 and 11; scripts/13_evaluate.py. Milestone M8 (stub in this pass; the metric
machinery it will call lives in scd.train.metrics and scd.train.robustness).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def run(cfg: Any, run_name: str, split: str = "val", chips_version: int = 1, baseline: bool = False) -> Path:
    """Write runs/<run>/metrics/<split>.json (+ confusion.csv, worst_chips.csv); returns the directory."""
    raise NotImplementedError("scd.eval.evaluate.run: built at M8; see HANDOFF.md sections 7.1, 11 and 13")


def show(cfg: Any, run_name: str) -> None:
    """Notebook section 12: tables, confusion matrix, worst-10 gallery."""
    raise NotImplementedError("scd.eval.evaluate.show: built at M8")
