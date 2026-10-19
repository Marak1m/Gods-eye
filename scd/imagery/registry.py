"""AOI registry: `registry/aois.parquet` (+ .csv mirror) with the HANDOFF.md 5.1 schema and checks.

The registry and the configs are the only source of truth; everything under data/ is rebuilt from
them. `load()` refuses an invalid registry.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from shapely import wkb as shp_wkb
from shapely.geometry import Polygon
from sklearn.cluster import DBSCAN

from scd.constants import LEAD_SOURCES, ROLES, STRATA, TARGET_TYPES
from scd.geo.grid import utm_epsg

COLUMNS: tuple[str, ...] = (
    "aoi_id",
    "stratum",
    "target_type",
    "role",
    "centre_lat",
    "centre_lon",
    "size_km",
    "utm_epsg",
    "geometry",
    "lead_source",
    "lead_ref",
    "event_year",
    "t1_window",
    "t2_window",
    "notes",
)
MIN_SPACING_KM = 10.0
CLUSTER_EPS_KM = 25.0
CLUSTER_CAP = 8
EARTH_RADIUS_KM = 6371.0088


def registry_path(cfg: Any) -> Path:
    return Path(cfg.registry_dir) / "aois.parquet"


def aoi_polygon(lat: float, lon: float, size_km: float) -> Polygon:
    """Square AOI in WGS84 centred on (lat, lon)."""
    half_lat = size_km / 2.0 / 111.32
    half_lon = size_km / 2.0 / (111.32 * max(math.cos(math.radians(lat)), 1e-6))
    return Polygon(
        [
            (lon - half_lon, lat - half_lat),
            (lon + half_lon, lat - half_lat),
            (lon + half_lon, lat + half_lat),
            (lon - half_lon, lat + half_lat),
        ]
    )


def _haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = p2 - p1
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in COLUMNS:
        if col not in df:
            df[col] = None
    df["utm_epsg"] = [utm_epsg(float(lo), float(la)) for la, lo in zip(df["centre_lat"], df["centre_lon"])]
    geoms = []
    for g, la, lo, s in zip(df["geometry"], df["centre_lat"], df["centre_lon"], df["size_km"]):
        if isinstance(g, Polygon):
            geoms.append(g)
        elif isinstance(g, bytes | bytearray):
            geoms.append(shp_wkb.loads(bytes(g)))
        else:
            geoms.append(aoi_polygon(float(la), float(lo), float(s) if pd.notna(s) else 10.0))
    df["geometry"] = geoms
    df["size_km"] = df["size_km"].fillna(10.0).astype(float)
    df["event_year"] = pd.array([int(v) if pd.notna(v) else None for v in df["event_year"]], dtype="Int64")
    for col in ("t1_window", "t2_window", "notes", "lead_ref"):
        df[col] = df[col].fillna("").astype(str)
    return df[list(COLUMNS)].reset_index(drop=True)


def init_from_targets(csv_path: Path, size_km: float = 10.0) -> pd.DataFrame:
    """The 42 PS-10 targets (configs/targets.csv) as `role=targets` rows."""
    t = pd.read_csv(csv_path)
    df = pd.DataFrame(
        {
            "aoi_id": [f"{s}_target_{int(n):02d}" for s, n in zip(t["stratum"], t["sample"])],
            "stratum": t["stratum"],
            "target_type": "unknown",
            "role": "targets",
            "centre_lat": t["lat"].astype(float),
            "centre_lon": t["lon"].astype(float),
            "size_km": size_km,
            "lead_source": "manual",
            "lead_ref": [f"annexure1:{int(n)}" for n in t["sample"]],
            "event_year": None,
            "t1_window": "",
            "t2_window": "",
            "notes": t["setting"],
        }
    )
    return _finish(df)


def append_from_csv(df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    """Append AOIs from a CSV with the registry columns (geometry optional)."""
    new = pd.read_csv(csv_path)
    missing = {"aoi_id", "stratum", "target_type", "role", "centre_lat", "centre_lon"} - set(new.columns)
    if missing:
        raise ValueError(f"{csv_path}: missing columns {sorted(missing)}")
    return _finish(pd.concat([df, new], ignore_index=True))


def validate(df: pd.DataFrame) -> list[str]:
    """HANDOFF 5.1 checks. Returns a list of error strings (empty when valid)."""
    errors: list[str] = []
    if df["aoi_id"].duplicated().any():
        errors.append(f"duplicate aoi_id: {sorted(df.loc[df['aoi_id'].duplicated(), 'aoi_id'])}")
    bad = df.loc[~df["stratum"].isin(STRATA), "aoi_id"].tolist()
    if bad:
        errors.append(f"stratum not in {STRATA}: {bad}")
    bad = df.loc[~df["role"].isin(ROLES), "aoi_id"].tolist()
    if bad:
        errors.append(f"role not in {ROLES}: {bad}")
    bad = df.loc[~df["target_type"].isin((*TARGET_TYPES, "unknown")), "aoi_id"].tolist()
    if bad:
        errors.append(f"target_type not in {TARGET_TYPES}: {bad}")
    bad = df.loc[~df["lead_source"].isin(LEAD_SOURCES), "aoi_id"].tolist()
    if bad:
        errors.append(f"lead_source not in {LEAD_SOURCES}: {bad}")
    mism = [a for a, s in zip(df["aoi_id"], df["stratum"]) if not str(a).startswith(f"{s}_")]
    if mism:
        errors.append(f"aoi_id must start with '<stratum>_': {mism}")
    tv = df[df["role"].isin(["train", "val"])].reset_index(drop=True)
    if len(tv) > 1:
        lat = tv["centre_lat"].to_numpy(float)
        lon = tv["centre_lon"].to_numpy(float)
        d = _haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
        np.fill_diagonal(d, np.inf)
        close = np.argwhere(np.triu(d < MIN_SPACING_KM))
        for i, j in close:
            errors.append(
                f"train/val AOIs {tv.aoi_id[i]} and {tv.aoi_id[j]} are {d[i, j]:.1f} km apart (< {MIN_SPACING_KM})"
            )
        labels = DBSCAN(eps=CLUSTER_EPS_KM / EARTH_RADIUS_KM, min_samples=1, metric="haversine").fit_predict(
            np.radians(np.c_[lat, lon])
        )
        for lab, n in zip(*np.unique(labels, return_counts=True)):
            if n > CLUSTER_CAP:
                sample = sorted(tv.aoi_id[labels == lab])[:5]
                errors.append(
                    f"cluster of {n} train/val AOIs within {CLUSTER_EPS_KM} km (cap {CLUSTER_CAP}): {sample}..."
                )
    targets = df[df["role"] == "targets"]
    if len(tv) and len(targets):
        d = _haversine_km(
            tv["centre_lat"].to_numpy(float)[:, None],
            tv["centre_lon"].to_numpy(float)[:, None],
            targets["centre_lat"].to_numpy(float)[None, :],
            targets["centre_lon"].to_numpy(float)[None, :],
        )
        for i, j in np.argwhere(d < MIN_SPACING_KM):
            errors.append(
                f"train/val AOI {tv.aoi_id.iloc[i]} overlaps target {targets.aoi_id.iloc[j]} ({d[i, j]:.1f} km)"
            )
    return errors


def save(df: pd.DataFrame, path: Path) -> Path:
    df = _finish(df)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    out["geometry"] = [g.wkb for g in df["geometry"]]
    out.to_parquet(path, index=False)
    csv = df.copy()
    csv["geometry"] = [g.wkt for g in df["geometry"]]
    csv.to_csv(path.with_suffix(".csv"), index=False)
    return path


def load(path: Path, strict: bool = True) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = _finish(df)
    if strict:
        errors = validate(df)
        if errors:
            raise ValueError("invalid registry:\n" + "\n".join(errors))
    return df


def subset(df: pd.DataFrame, aoi_ids: Sequence[str] | None) -> pd.DataFrame:
    return df if not aoi_ids else df[df["aoi_id"].isin(list(aoi_ids))].reset_index(drop=True)
