"""Post-training calibration of tau and the class biases.

HANDOFF.md section 7.4. Milestone M7.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scd.config import Config


class Calibration:
    """tau, bias[4], weighted_iou, per_class."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("scd.train.calibrate.Calibration: built at M7; see HANDOFF.md section 7.4")


def calibrate(run_dir: Path, val_loader: Any, cfg: Config, stub: bool = False) -> Any:
    """Coordinate ascent; writes calibration.json."""
    raise NotImplementedError("scd.train.calibrate.calibrate: built at M7; see HANDOFF.md section 7.4")


def load_calibration(path: Path) -> Any:
    """Read calibration.json."""
    raise NotImplementedError("scd.train.calibrate.load_calibration: built at M7; see HANDOFF.md section 7.4")
