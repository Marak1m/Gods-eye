"""QC figures.

HANDOFF.md section 10. Milestone M8.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


def quicklook_panel(t1: Any, t2: Any, mask: Any | None, path: Path) -> Path:
    """T1 | T2 | mask panel."""
    raise NotImplementedError("scd.eval.figures.quicklook_panel: built at M8; see HANDOFF.md section 10")


def confusion_figure(cm: Any, path: Path) -> Path:
    """confusion_figure"""
    raise NotImplementedError("scd.eval.figures.confusion_figure: built at M8; see HANDOFF.md section 10")


def worst_chips_gallery(rows: list[dict[str, Any]], path: Path) -> Path:
    """worst_chips_gallery"""
    raise NotImplementedError("scd.eval.figures.worst_chips_gallery: built at M8; see HANDOFF.md section 10")
