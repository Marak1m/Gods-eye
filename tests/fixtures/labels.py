"""Reviewed-GeoPackage fixtures (HANDOFF 5.7 schema) for tests and the smoke data."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import geopandas as gpd
from shapely.geometry.base import BaseGeometry

from scd.labels.rasterize import REVIEWED_LAYER


def reviewed_rows(
    polys: list[tuple[BaseGeometry, int]], present_t1: bool = False, present_t2: bool = True, **extra: Any
) -> list[dict[str, Any]]:
    rows = []
    for geom, cid in polys:
        rows.append(
            {
                "geometry": geom,
                "class_id": int(cid),
                "confidence": "corrected",
                "annotator": "fixture",
                "timestamp": "2026-10-18T00:00:00Z",
                "present_t1": present_t1,
                "present_t2": present_t2,
                "source": "fixture",
                "source_ref": "",
                "road_class": "access",
                "notes": "",
                **extra,
            }
        )
    return rows


def write_reviewed_gpkg(path: Path, rows: list[dict[str, Any]], epsg: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=f"EPSG:{epsg}")
    if path.exists():
        path.unlink()
    gdf.to_file(path, layer=REVIEWED_LAYER, driver="GPKG")
    return path
