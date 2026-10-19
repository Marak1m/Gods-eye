"""Two GeoTIFFs -> change.tif, valid.tif, shapefile, report.json, quicklook.png.

HANDOFF.md section 8. Milestone M7.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scd.config import Config


class Result:
    """Paths and summary numbers."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("scd.infer.pipeline.Result: built at M7; see HANDOFF.md section 8")


def run(t1_path: Path, t2_path: Path, out_dir: Path, cfg: Config, calib: Any, net: Any | None = None) -> Result:
    """HANDOFF 8 steps 1-8."""
    raise NotImplementedError("scd.infer.pipeline.run: built at M7; see HANDOFF.md section 8")
