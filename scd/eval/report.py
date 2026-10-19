"""Methodology report (report.md) for the Approach score.

HANDOFF.md section 10 section 16. Milestone M8.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scd.config import Config


def write_report(cfg: Config, run_name: str) -> Path:
    """runs/<run>/report.md with tables and figures."""
    raise NotImplementedError("scd.eval.report.write_report: built at M8; see HANDOFF.md section 10 section 16")
