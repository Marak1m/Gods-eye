"""Candidate mining (HANDOFF.md 5.6): run every `Source` for a pair, write candidates/<aoi>/<pair>.gpkg.

Candidates are leads for human review. They are never used as labels without passing through
`verification`, unless the chip builder explicitly admits `confidence == 'auto'` (loss weight 0.5).
The concrete sources (`scd.labels.sources.*`) need external data and are stubs until Phase 1.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import geopandas as gpd
import pandas as pd

from scd.config import Config
from scd.constants import CHANGE_CLASSES, CLASS_IDS, STRATA
from scd.geo.vector import require_projected
from scd.imagery.preprocess import PairProduct
from scd.log import get_logger

log = get_logger(__name__)

CANDIDATE_COLUMNS = ("geometry", "class_id", "source", "source_ref", "t_event", "confidence", "notes")
PAIR_LAYER = "pair"
SOURCE_MODULES = {
    "gem": "GemSource",
    "kruitwagen": "KruitwagenSource",
    "cbra": "CbraSource",
    "obt": "ObtSource",
    "io_lulc": "IoLulcSource",
    "dynamic_world": "DynamicWorldSource",
    "osm_history": "OsmHistorySource",
}
ID_TO_NAME = {v: k for k, v in CLASS_IDS.items()}


@runtime_checkable
class Source(Protocol):
    name: str

    def fetch(self, aoi: Any, cache: Path) -> Path: ...

    def candidates(self, aoi: Any, pair: Any) -> gpd.GeoDataFrame: ...


def make_sources(names: Sequence[str]) -> list[Source]:
    out: list[Source] = []
    for n in names:
        if n not in SOURCE_MODULES:
            raise ValueError(f"unknown source {n!r}; known: {sorted(SOURCE_MODULES)}")
        mod = importlib.import_module(f"scd.labels.sources.{n}")
        out.append(getattr(mod, SOURCE_MODULES[n])())
    return out


def validate_candidates(gdf: gpd.GeoDataFrame, source_name: str) -> gpd.GeoDataFrame:
    missing = [c for c in CANDIDATE_COLUMNS if c not in gdf.columns]
    if missing:
        raise ValueError(f"source {source_name}: candidates missing columns {missing}")
    if len(gdf) == 0:
        return gdf
    require_projected(gdf.crs)
    bad = ~gdf["class_id"].astype(int).isin(range(1, 5))
    if bad.any():
        raise ValueError(f"source {source_name}: class_id outside 1..4 in {int(bad.sum())} rows")
    out = gdf.copy()
    out["class_id"] = out["class_id"].astype(int)
    out["confidence"] = "auto"
    out["source"] = source_name
    return out


def run(aoi: Any, pair: PairProduct, sources: Sequence[Source], out_path: Path) -> Path:
    """One GeoPackage: a layer per change class with candidates, plus a `pair` layer."""
    grid = pair.grid()
    meta = pair.meta()
    frames: list[gpd.GeoDataFrame] = []
    for src in sources:
        try:
            gdf = src.candidates(aoi, pair)
        except NotImplementedError as e:
            log.warning("source %s skipped: %s", src.name, e)
            continue
        gdf = validate_candidates(gdf, src.name)
        if len(gdf):
            if gdf.crs.to_epsg() != grid.epsg:
                gdf = gdf.to_crs(epsg=grid.epsg)
            frames.append(gdf[list(CANDIDATE_COLUMNS)])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()
    pair_layer = gpd.GeoDataFrame(
        {
            "aoi_id": [meta.get("aoi_id")],
            "pair_id": [meta.get("pair_id")],
            "t1_date": [str(meta.get("t1", {}).get("date"))],
            "t2_date": [str(meta.get("t2", {}).get("date"))],
            "grid": [json.dumps(grid.to_dict())],
        },
        geometry=[grid.polygon],
        crs=f"EPSG:{grid.epsg}",
    )
    pair_layer.to_file(out_path, layer=PAIR_LAYER, driver="GPKG")
    if frames:
        allc = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=f"EPSG:{grid.epsg}")
        for cid in range(1, 5):
            sel = allc[allc["class_id"] == cid]
            if len(sel):
                sel.to_file(out_path, layer=ID_TO_NAME[cid], driver="GPKG")
    return out_path


def run_all(cfg: Config, source_names: Sequence[str], aoi_ids: Sequence[str] | None = None) -> list[Path]:
    from scd.imagery import registry
    from scd.imagery.preprocess import list_products

    sources = make_sources(source_names)
    aois = registry.load(registry.registry_path(cfg), strict=False).set_index("aoi_id")
    out: list[Path] = []
    for prod in list_products(cfg):
        if aoi_ids and prod.aoi_id not in set(aoi_ids):
            continue
        aoi = (
            aois.loc[prod.aoi_id].to_dict() | {"aoi_id": prod.aoi_id}
            if prod.aoi_id in aois.index
            else {"aoi_id": prod.aoi_id}
        )
        out.append(run(aoi, prod, sources, cfg.candidates_dir / prod.aoi_id / f"{prod.pair_id}.gpkg"))
    return out


def candidate_counts(cfg: Config) -> pd.DataFrame:
    """Stratum x class candidate counts across candidates/<aoi>/<pair>.gpkg."""
    import pyogrio

    counts = pd.DataFrame(0, index=list(STRATA), columns=list(CHANGE_CLASSES))
    for gpkg in sorted(cfg.candidates_dir.glob("*/*.gpkg")):
        stratum = gpkg.parent.name[0]
        for layer, _geom in pyogrio.list_layers(gpkg):
            if layer in CHANGE_CLASSES:
                counts.loc[stratum, layer] += len(gpd.read_file(gpkg, layer=layer))
    return counts
