"""Chip builder (HANDOFF.md 5.10): 1024 x 1024 windows on a 768 stride where valid >= 50 %.

Pairs with reviewed labels get their label rasters; pairs flagged NOCHANGE_HARD (or whose AOI is
a `negative`) without reviewed labels become auto negatives: all-zero change with the
image-change gate applied (255 where the difference is unusually large) and
`label_confidence = auto`. Everything else is skipped.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scd.config import Config
from scd.constants import IGNORE, TileType
from scd.data import manifest as man_mod
from scd.data.splits import assign_splits
from scd.geo.io import write_imagery, write_mask
from scd.imagery import registry
from scd.imagery.preprocess import PairProduct, list_products
from scd.labels.negatives import apply_gate
from scd.labels.rasterize import LabelProduct, load_reviewed
from scd.log import get_logger

log = get_logger(__name__)


def assign_tile_type(change: np.ndarray, is_contrast_pair: bool, min_change_px: int) -> str:
    counts = [int((change == k).sum()) for k in range(1, 5)]
    if max(counts) >= min_change_px:
        return TileType.CHANGE_POS.value
    return TileType.NOCHANGE_HARD.value if is_contrast_pair else TileType.NOCHANGE_PLAIN.value


def chip_windows(height: int, width: int, chip_px: int, stride_px: int) -> list[tuple[int, int]]:
    """Top-left (row, col) positions covering the raster; the last window is aligned to the edge."""
    if height < chip_px or width < chip_px:
        return []

    def positions(n: int) -> list[int]:
        pos = list(range(0, n - chip_px + 1, stride_px))
        if pos[-1] != n - chip_px:
            pos.append(n - chip_px)
        return pos

    return [(r, c) for r in positions(height) for c in positions(width)]


def _confidence(reviewed_gpkg: Path) -> str:
    try:
        conf = set(load_reviewed(reviewed_gpkg)["confidence"].astype(str))
    except Exception:
        return "corrected"
    if conf and conf <= {"double_checked"}:
        return "double_checked"
    return "corrected"


def build_chips(cfg: Config, version: int, pairs: Sequence[PairProduct] | None = None, force: bool = False) -> Path:
    chips_dir = cfg.chips_dir(version)
    man_path = chips_dir / "manifest.parquet"
    if man_path.exists() and not force and man_mod.read_hash(man_path) == cfg.hash():
        log.info("chips v%d are current for this config; skipping", version)
        return chips_dir
    reg = registry.load(registry.registry_path(cfg), strict=False)
    splits = assign_splits(reg, seed=cfg.train.seed)
    reg_idx = reg.set_index("aoi_id")
    products = list(pairs) if pairs is not None else list_products(cfg)
    rows: list[dict[str, Any]] = []
    (chips_dir / "images").mkdir(parents=True, exist_ok=True)
    (chips_dir / "labels").mkdir(parents=True, exist_ok=True)
    chip_px, stride = cfg.grid.chip_px, cfg.chips.stride_px
    for prod in products:
        meta = prod.meta()
        aoi_id, pair_id = prod.aoi_id, prod.pair_id
        aoi = reg_idx.loc[aoi_id] if aoi_id in reg_idx.index else None
        stratum = str(aoi["stratum"]) if aoi is not None else aoi_id[0]
        target_type = str(aoi["target_type"]) if aoi is not None else "unknown"
        is_contrast = meta.get("tile_type_hint") == TileType.NOCHANGE_HARD.value or target_type == "negative"
        t1, t2, valid = prod.read()
        lab = LabelProduct(cfg.labels_dir / aoi_id / pair_id)
        reviewed = cfg.reviewed_dir / aoi_id / f"{pair_id}.gpkg"
        if lab.exists():
            change, sem1, sem2 = lab.read()
            confidence = _confidence(reviewed) if reviewed.exists() else "corrected"
        elif is_contrast:
            change = np.zeros(valid.shape, dtype=np.uint8)
            change[valid == 0] = IGNORE
            change = apply_gate(change, t1, t2, valid)
            sem1 = np.full(valid.shape, IGNORE, dtype=np.uint8)
            sem2 = sem1.copy()
            confidence = "auto"
        else:
            log.warning("%s/%s: no labels and not a negative pair; skipped", aoi_id, pair_id)
            continue
        grid = prod.grid()
        for r, c in chip_windows(grid.height, grid.width, chip_px, stride):
            sl = (slice(r, r + chip_px), slice(c, c + chip_px))
            v = valid[sl]
            vf = float(v.mean())
            if vf < cfg.chips.min_valid_fraction:
                continue
            ch = change[sl]
            chip_id = f"{aoi_id}_{pair_id}_r{r}_c{c}"
            g = grid.window(r, c, chip_px, chip_px)
            write_imagery(chips_dir / "images" / f"{chip_id}_t1.tif", t1[(slice(None), *sl)], g)
            write_imagery(chips_dir / "images" / f"{chip_id}_t2.tif", t2[(slice(None), *sl)], g)
            write_mask(chips_dir / "images" / f"{chip_id}_valid.tif", v, g, nodata=None)
            write_mask(chips_dir / "labels" / f"{chip_id}_change.tif", ch, g)
            write_mask(chips_dir / "labels" / f"{chip_id}_sem_t1.tif", sem1[sl], g)
            write_mask(chips_dir / "labels" / f"{chip_id}_sem_t2.tif", sem2[sl], g)
            counts = {k: int((ch == k).sum()) for k in range(1, 5)}
            rows.append(
                {
                    "chip_id": chip_id,
                    "aoi_id": aoi_id,
                    "pair_id": pair_id,
                    "stratum": stratum,
                    "target_type": target_type,
                    "t1_date": str(meta.get("t1_date") or meta.get("t1", {}).get("date")),
                    "t2_date": str(meta.get("t2_date") or meta.get("t2", {}).get("date")),
                    "same_orbit": bool(meta.get("same_orbit", True)),
                    "season_match": bool(meta.get("season_match", True)),
                    "tile_type": assign_tile_type(ch, is_contrast, cfg.chips.min_change_px),
                    "classes_present": [k for k, n in counts.items() if n > 0],
                    **{f"change_px_{k}": n for k, n in counts.items()},
                    "valid_fraction": vf,
                    "label_confidence": confidence,
                    "split": splits.get(aoi_id, "train"),
                    "row": r,
                    "col": c,
                    "window": json.dumps([r, c, chip_px, chip_px]),
                }
            )
    df = pd.DataFrame(rows, columns=list(man_mod.MANIFEST_COLUMNS))
    man_mod.save(df, man_path, cfg.hash())
    log.info("chips v%d: %d chips from %d pairs -> %s", version, len(df), len(products), chips_dir)
    return chips_dir
