"""GeoTIFF, shapefile, quicklook and report writers.

HANDOFF.md section 8 steps 6-8. Milestone M7.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


def write_geotiff(mask: Any, grid: Any, path: Path, tags: dict[str, str], invalid_value: int = 0) -> Path:
    """write_geotiff"""
    raise NotImplementedError("scd.infer.export.write_geotiff: built at M7; see HANDOFF.md section 8 steps 6-8")


def write_shapefile(mask: Any, grid: Any, path: Path, conf: Any | None = None, simplify_m: float = 2.5) -> Path:
    """write_shapefile"""
    raise NotImplementedError("scd.infer.export.write_shapefile: built at M7; see HANDOFF.md section 8 steps 6-8")


def write_quicklook(t1: Any, t2: Any, mask: Any, path: Path) -> Path:
    """write_quicklook"""
    raise NotImplementedError("scd.infer.export.write_quicklook: built at M7; see HANDOFF.md section 8 steps 6-8")


def write_report(path: Path, report: dict[str, Any]) -> Path:
    """write_report"""
    raise NotImplementedError("scd.infer.export.write_report: built at M7; see HANDOFF.md section 8 steps 6-8")
