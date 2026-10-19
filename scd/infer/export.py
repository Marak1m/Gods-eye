"""Output writers (HANDOFF.md 8 steps 6-8; DEVIATIONS U3).

change.tif: uint8 values 0-4, invalid pixels written as `infer.invalid_value` (default 0; 255 only
when configured, then also the nodata tag), DEFLATE, tags with model hash, calibration, offsets and
band order. valid.tif is always written beside it. The shapefile has one feature per 4-connected
polygon with fields class (int 1-4), area_m2 (float) and conf (float, mean sigmoid(c)).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fiona
import numpy as np
from rasterio import features
from shapely.geometry import mapping, shape
from skimage import measure

from scd.constants import BAND_ORDER, IGNORE
from scd.geo.grid import Grid
from scd.geo.io import write_mask
from scd.geo.vector import require_projected


def write_geotiff(
    mask: np.ndarray,
    grid: Grid,
    path: Path,
    tags: dict[str, str],
    invalid_value: int = 0,
    valid: np.ndarray | None = None,
) -> Path:
    out = mask.astype(np.uint8).copy()
    out[out == IGNORE] = invalid_value
    if valid is not None:
        out[valid == 0] = invalid_value
    nodata = IGNORE if invalid_value == IGNORE else None
    full_tags = {"band_order": ",".join(BAND_ORDER), "invalid_value": str(invalid_value), **tags}
    return write_mask(Path(path), out, grid, nodata=nodata, tags=full_tags)


def write_valid(valid: np.ndarray, grid: Grid, path: Path) -> Path:
    return write_mask(Path(path), valid.astype(np.uint8), grid, nodata=None, tags={"meaning": "1 valid, 0 invalid"})


SHP_SCHEMA = {"geometry": "Polygon", "properties": {"class": "int", "area_m2": "float", "conf": "float"}}


def write_shapefile(
    mask: np.ndarray, grid: Grid, path: Path, conf: np.ndarray | None = None, simplify_m: float = 2.5
) -> Path:
    """One polygon per 4-connected component of each class; ESRI Shapefile with .prj."""
    require_projected(grid.crs)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    m = mask.astype(np.uint8)
    change = (m > 0) & (m != IGNORE)
    records: list[dict[str, Any]] = []
    for geom_json, value in features.shapes(m, mask=change, transform=grid.transform, connectivity=4):
        geom = shape(geom_json)
        if simplify_m > 0:
            geom = geom.simplify(simplify_m, preserve_topology=True)
        if geom.is_empty:
            continue
        c = float("nan")
        if conf is not None:
            pix = (
                features.rasterize([(shape(geom_json), 1)], out_shape=m.shape, transform=grid.transform, dtype="uint8")
                == 1
            )
            c = float(conf[pix].mean()) if pix.any() else float("nan")
        records.append(
            {
                "geometry": mapping(geom),
                "properties": {"class": int(value), "area_m2": float(geom.area), "conf": c},
            }
        )
    for ext in (".shp", ".shx", ".dbf", ".prj", ".cpg"):
        p = path.with_suffix(ext)
        if p.exists():
            p.unlink()
    with fiona.open(path, "w", driver="ESRI Shapefile", crs=grid.crs.to_wkt(), schema=SHP_SCHEMA) as dst:
        for rec in records:
            dst.write(rec)
    return path


def component_count(mask: np.ndarray) -> int:
    """Number of 4-connected components over all change classes (equals the shapefile feature count)."""
    m = mask.astype(np.uint8)
    n = 0
    for k in range(1, 5):
        n += int(measure.label(m == k, connectivity=1).max())
    return n


def write_quicklook(t1: np.ndarray, t2: np.ndarray, mask: np.ndarray, path: Path) -> Path:
    from scd.imagery.preprocess import write_quicklook as _ql

    return _ql(t1, t2, np.ones(mask.shape, dtype=np.uint8), Path(path), mask=mask)


def write_report(path: Path, report: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, default=str))
    return path
