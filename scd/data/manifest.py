"""manifest.parquet schema, IO and summaries (HANDOFF.md 5.10).

Only CHANGE_POS / NOCHANGE_HARD / NOCHANGE_PLAIN are stored; IDENTICAL, NEAR_IDENTICAL,
SYNTHETIC and STAR are synthesised at draw time (DEVIATIONS C15).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from scd.constants import CHANGE_CLASSES, LABEL_CONFIDENCE, SPLITS, STORED_TILE_TYPES, STRATA

MANIFEST_COLUMNS: tuple[str, ...] = (
    "chip_id",
    "aoi_id",
    "pair_id",
    "stratum",
    "target_type",
    "t1_date",
    "t2_date",
    "same_orbit",
    "season_match",
    "tile_type",
    "classes_present",
    "change_px_1",
    "change_px_2",
    "change_px_3",
    "change_px_4",
    "valid_fraction",
    "label_confidence",
    "split",
    "row",
    "col",
    "window",
)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest missing columns {missing}")
    if df["chip_id"].duplicated().any():
        raise ValueError("manifest has duplicate chip_id")
    bad = set(df["tile_type"]) - set(STORED_TILE_TYPES)
    if bad:
        raise ValueError(f"manifest tile_type values {bad} are not stored types {STORED_TILE_TYPES}")
    bad = set(df["split"]) - set(SPLITS)
    if bad:
        raise ValueError(f"manifest split values {bad} not in {SPLITS}")
    bad = set(df["label_confidence"]) - set(LABEL_CONFIDENCE)
    if bad:
        raise ValueError(f"manifest label_confidence values {bad} not in {LABEL_CONFIDENCE}")
    bad = set(df["stratum"]) - set(STRATA)
    if bad:
        raise ValueError(f"manifest stratum values {bad} not in {STRATA}")
    return df


def load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["classes_present"] = df["classes_present"].map(lambda v: [int(x) for x in (v if v is not None else [])])
    return validate(df)


def save(df: pd.DataFrame, path: Path, cfg_hash: str) -> Path:
    validate(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[list(MANIFEST_COLUMNS)].copy()
    out["classes_present"] = out["classes_present"].map(lambda v: [int(x) for x in v])
    table = pa.Table.from_pandas(out, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[b"scd_config_hash"] = cfg_hash.encode()
    meta[b"scd_columns"] = json.dumps(list(MANIFEST_COLUMNS)).encode()
    pq.write_table(table.replace_schema_metadata(meta), path)
    out.assign(classes_present=out["classes_present"].map(json.dumps)).to_csv(path.with_suffix(".csv"), index=False)
    return path


def read_hash(path: Path) -> str | None:
    meta = pq.read_schema(path).metadata or {}
    v = meta.get(b"scd_config_hash")
    return v.decode() if v else None


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Tile types x strata per split, plus change pixels per class (notebook section 7)."""
    if len(df) == 0:
        return pd.DataFrame()
    counts = df.pivot_table(index=["split", "stratum"], columns="tile_type", values="chip_id", aggfunc="count", fill_value=0)
    px = df.groupby(["split", "stratum"])[[f"change_px_{k}" for k in range(1, 5)]].sum()
    px.columns = [f"px_{c}" for c in CHANGE_CLASSES]
    out = counts.join(px)
    out["chips"] = df.groupby(["split", "stratum"]).size()
    return out


def class_weights(df: pd.DataFrame, power: float = 0.5) -> dict[str, float]:
    """Inverse frequency^power over change pixels in the training split, renormalised to mean 1."""
    train = df[df["split"] == "train"] if (df["split"] == "train").any() else df
    freq = np.array([float(train[f"change_px_{k}"].sum()) for k in range(1, 5)])
    if freq.sum() <= 0:
        return {c: 1.0 for c in CHANGE_CLASSES}
    freq = np.where(freq > 0, freq, freq[freq > 0].min())
    w = (1.0 / freq) ** power
    w = w / w.mean()
    return {c: float(v) for c, v in zip(CHANGE_CLASSES, w)}


def chip_paths(chips_dir: Path, chip_id: str) -> dict[str, Path]:
    return {
        "t1": chips_dir / "images" / f"{chip_id}_t1.tif",
        "t2": chips_dir / "images" / f"{chip_id}_t2.tif",
        "valid": chips_dir / "images" / f"{chip_id}_valid.tif",
        "change": chips_dir / "labels" / f"{chip_id}_change.tif",
        "sem_t1": chips_dir / "labels" / f"{chip_id}_sem_t1.tif",
        "sem_t2": chips_dir / "labels" / f"{chip_id}_sem_t2.tif",
    }


def row_dict(df: pd.DataFrame, chip_id: str) -> dict[str, Any]:
    r = df.loc[df["chip_id"] == chip_id]
    if r.empty:
        raise KeyError(chip_id)
    return dict(r.iloc[0])
