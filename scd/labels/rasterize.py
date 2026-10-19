"""Reviewed polygons -> change.tif, sem_t1.tif, sem_t2.tif (HANDOFF.md 5.8).

All geometry work happens in the pair's UTM CRS (areas in m2). Undersized objects are dropped,
roads are burned as centreline buffers, classes are burned lowest-priority-first so the highest
priority wins on overlap, `fix_t1` is applied to the T1 semantic mask, and invalid pixels become
255 in all three rasters.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from rasterio import features
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

from scd.config import Config
from scd.constants import CHANGE_CLASSES, CLASS_IDS, CLASS_NAMES, IGNORE
from scd.geo.grid import Grid
from scd.geo.io import read_raster, write_mask
from scd.geo.vector import require_projected
from scd.imagery.preprocess import PairProduct
from scd.labels.conventions import Conventions
from scd.labels.fix_t1 import fix_t1
from scd.log import get_logger

log = get_logger(__name__)

REVIEWED_LAYER = "reviewed"
ID_TO_NAME = {v: k for k, v in CLASS_IDS.items()}


@dataclass(frozen=True)
class LabelProduct:
    dir: Path

    @property
    def change(self) -> Path:
        return self.dir / "change.tif"

    @property
    def sem_t1(self) -> Path:
        return self.dir / "sem_t1.tif"

    @property
    def sem_t2(self) -> Path:
        return self.dir / "sem_t2.tif"

    @property
    def report(self) -> Path:
        return self.dir / "rasterize_report.json"

    def exists(self) -> bool:
        return self.change.exists() and self.sem_t1.exists() and self.sem_t2.exists()

    def read(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c, _ = read_raster(self.change)
        a, _ = read_raster(self.sem_t1)
        b, _ = read_raster(self.sem_t2)
        return c[0], a[0], b[0]


def load_reviewed(gpkg: Path, layer: str | None = None) -> gpd.GeoDataFrame:
    layers: list[str] | None = None
    try:
        import pyogrio

        layers = [str(x[0]) for x in pyogrio.list_layers(gpkg)]
    except Exception:  # pragma: no cover - fiona fallback
        pass
    use = layer or (REVIEWED_LAYER if layers and REVIEWED_LAYER in layers else None)
    gdf = gpd.read_file(gpkg, layer=use) if use else gpd.read_file(gpkg)
    if "class_id" not in gdf:
        raise ValueError(f"{gpkg}: missing class_id column")
    gdf["class_id"] = gdf["class_id"].astype(int)
    for col in ("present_t1", "present_t2"):
        if col in gdf:
            gdf[col] = gdf[col].astype(bool)
    return gdf


def _road_geometry(geom: BaseGeometry, road_class: Any, conv: Conventions) -> BaseGeometry:
    """Centreline -> buffer at the configured width; polygons pass through."""
    if geom.geom_type in {"LineString", "MultiLineString"}:
        key = str(road_class) if isinstance(road_class, str) and road_class in conv.road_width_m else "access"
        return geom.buffer(conv.road_width_m[key] / 2.0, cap_style="flat")
    return geom


def _road_length_m(geom: BaseGeometry) -> float:
    if geom.geom_type in {"LineString", "MultiLineString"}:
        return float(geom.length)
    rect = geom.minimum_rotated_rectangle
    if isinstance(rect, Polygon) and not rect.is_empty:
        xs, ys = rect.exterior.coords.xy
        sides = [float(np.hypot(xs[i + 1] - xs[i], ys[i + 1] - ys[i])) for i in range(2)]
        return max(sides)
    return float(np.sqrt(geom.area))


def filter_by_rules(gdf: gpd.GeoDataFrame, conv: Conventions) -> tuple[gpd.GeoDataFrame, dict[str, Any]]:
    """Apply the minimum-size rules in a projected CRS; returns kept rows and a report."""
    require_projected(gdf.crs)
    keep: list[bool] = []
    reasons: list[str] = []
    road_id = conv.class_ids["road"]
    for _, row in gdf.iterrows():
        cid = int(row["class_id"])
        geom = row.geometry
        if geom is None or geom.is_empty or not geom.is_valid:
            keep.append(False)
            reasons.append("invalid geometry")
            continue
        if cid == road_id:
            length = _road_length_m(geom)
            ok = length >= conv.min_length_m["road"]
            keep.append(ok)
            reasons.append("" if ok else f"road {length:.0f} m < {conv.min_length_m['road']:.0f} m")
        else:
            min_area = conv.min_area_for(cid)
            ok = float(geom.area) >= min_area
            keep.append(ok)
            reasons.append("" if ok else f"{ID_TO_NAME.get(cid, cid)} {geom.area:.0f} m2 < {min_area:.0f} m2")
    report = {
        "n_in": int(len(gdf)),
        "n_kept": int(sum(keep)),
        "dropped": [{"index": int(i), "reason": r} for i, (k, r) in enumerate(zip(keep, reasons)) if not k],
    }
    out = gdf[np.array(keep, dtype=bool)].copy() if len(gdf) else gdf.copy()
    return out, report


def _burn(shapes: Sequence[tuple[BaseGeometry, int]], grid: Grid, out: np.ndarray) -> np.ndarray:
    if not shapes:
        return out
    return features.rasterize(shapes, out=out, transform=grid.transform, all_touched=False)


def rasterize_pair(
    reviewed_gpkg: Path, pair_product: PairProduct, conventions: Conventions, out_dir: Path
) -> LabelProduct:
    """HANDOFF 5.8. Writes change.tif, sem_t1.tif, sem_t2.tif (uint8, 255 ignore) + a report."""
    grid = pair_product.grid()
    gdf = load_reviewed(reviewed_gpkg)
    if gdf.crs is None:
        raise ValueError(f"{reviewed_gpkg}: no CRS")
    if gdf.crs.to_epsg() != grid.epsg:
        gdf = gdf.to_crs(epsg=grid.epsg)
    road_id = conventions.class_ids["road"]
    if len(gdf):
        gdf["geometry"] = [
            _road_geometry(g, row.get("road_class"), conventions) if int(row["class_id"]) == road_id else g
            for g, (_, row) in zip(gdf.geometry, gdf.iterrows())
        ]
    kept, report = filter_by_rules(gdf, conventions)
    has_presence = "present_t1" in kept and "present_t2" in kept
    if has_presence:
        p1 = kept["present_t1"].to_numpy(bool)
        p2 = kept["present_t2"].to_numpy(bool)
        is_change = p1 != p2
        if not conventions.removals_labeled:
            is_change &= p2
    else:
        p1 = np.zeros(len(kept), dtype=bool)
        p2 = np.ones(len(kept), dtype=bool)
        is_change = np.ones(len(kept), dtype=bool)

    change = np.zeros(grid.shape, dtype=np.uint8)
    sem_t1 = np.zeros(grid.shape, dtype=np.uint8)
    sem_t2 = np.zeros(grid.shape, dtype=np.uint8)
    cls = kept["class_id"].to_numpy(int) if len(kept) else np.zeros(0, dtype=int)
    geoms = list(kept.geometry)
    for cid in conventions.burn_order():  # lowest priority first; last burned wins
        sel = cls == cid
        change = _burn([(geoms[i], cid) for i in np.flatnonzero(sel & is_change)], grid, change)
        sem_t1 = _burn([(geoms[i], cid) for i in np.flatnonzero(sel & p1)], grid, sem_t1)
        sem_t2 = _burn([(geoms[i], cid) for i in np.flatnonzero(sel & p2)], grid, sem_t2)
    if not has_presence:
        sem_t1[:] = IGNORE  # no per-date presence: the T1 semantic state is unknown
    sem_t1 = fix_t1(sem_t1, sem_t2)

    _, _, valid = pair_product.read()
    invalid = valid == 0
    for arr in (change, sem_t1, sem_t2):
        arr[invalid] = IGNORE

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tags = {"class_ids": json.dumps(CLASS_IDS), "source": str(reviewed_gpkg)}
    write_mask(out_dir / "change.tif", change, grid, nodata=IGNORE, tags=tags)
    write_mask(out_dir / "sem_t1.tif", sem_t1, grid, nodata=IGNORE, tags=tags)
    write_mask(out_dir / "sem_t2.tif", sem_t2, grid, nodata=IGNORE, tags=tags)
    report.update(
        change_px={ID_TO_NAME[c]: int((change == c).sum()) for c in range(1, 5)},
        has_presence=has_presence,
        removals_labeled=conventions.removals_labeled,
    )
    (out_dir / "rasterize_report.json").write_text(json.dumps(report, indent=2))
    return LabelProduct(out_dir)


def run_all(cfg: Config, aoi_ids: Sequence[str] | None = None, force: bool = False) -> list[LabelProduct]:
    from scd.labels import conventions as conv_mod

    conv = conv_mod.load(cfg.labeling_guide_path())
    out: list[LabelProduct] = []
    for gpkg in sorted(cfg.reviewed_dir.glob("*/*.gpkg")):
        aoi, pair = gpkg.parent.name, gpkg.stem
        if aoi_ids and aoi not in set(aoi_ids):
            continue
        prod = PairProduct(cfg.pairs_dir / aoi / pair)
        if not prod.exists():
            log.warning("%s/%s: reviewed labels without a preprocessed pair; skipped", aoi, pair)
            continue
        lab = LabelProduct(cfg.labels_dir / aoi / pair)
        if lab.exists() and not force:
            out.append(lab)
            continue
        out.append(rasterize_pair(gpkg, prod, conv, lab.dir))
        log.info("%s/%s: labels written", aoi, pair)
    return out


def area_table(cfg: Config) -> pd.DataFrame:
    """Per-stratum change area (ha) per class from the label rasters (notebook section 6)."""
    rows = []
    for change_path in sorted(cfg.labels_dir.glob("*/*/change.tif")):
        arr, profile = read_raster(change_path)
        res = abs(profile["transform"].a)
        aoi = change_path.parent.parent.name
        row: dict[str, Any] = {"aoi_id": aoi, "pair_id": change_path.parent.name, "stratum": aoi[0]}
        for cid in range(1, 5):
            row[CLASS_NAMES[cid]] = float((arr[0] == cid).sum()) * res * res / 10_000.0
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["stratum", *CHANGE_CLASSES])
    df = pd.DataFrame(rows)
    return df.groupby("stratum")[list(CHANGE_CLASSES)].sum()
