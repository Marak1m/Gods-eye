"""preprocess_pair: two scenes -> t1.tif, t2.tif, valid.tif, meta.json, quicklook.png (HANDOFF 5.4).

Order invariance by construction: the two scenes are put in canonical order by content key, the
grid comes from the footprint *set*, every per-scene step is a pure function, the co-registration
is a half-shift, and the validity mask is the AND of both dates. The caller's order only decides
which array is written as t1.tif and which as t2.tif. Pixel arrays are therefore identical for
`preprocess_pair(a, b)` and `preprocess_pair(b, a)` (swapped names); only order-dependent
metadata differs (DEVIATIONS C3, C5, C6).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from affine import Affine

from scd.config import Config
from scd.constants import BAND_INDEX, BAND_ORDER, RGBN_BANDS, SWIR_BANDS
from scd.geo.coreg import half_shift, half_shift_mask, shift_array
from scd.geo.grid import Grid, canonical_grid
from scd.geo.io import read_raster, write_imagery, write_mask
from scd.geo.reproject import reproject_to_grid, resample
from scd.geo.vector import canonical_pair
from scd.imagery import cloud as cloudmod
from scd.imagery import sr as srmod
from scd.imagery.scenes import Scene, open_scene
from scd.log import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class PairProduct:
    """pairs/<aoi>/<pair>/ with t1.tif, t2.tif (6-band uint16), valid.tif (uint8), meta.json."""

    dir: Path

    @property
    def t1(self) -> Path:
        return self.dir / "t1.tif"

    @property
    def t2(self) -> Path:
        return self.dir / "t2.tif"

    @property
    def valid(self) -> Path:
        return self.dir / "valid.tif"

    @property
    def meta_path(self) -> Path:
        return self.dir / "meta.json"

    @property
    def quicklook(self) -> Path:
        return self.dir / "quicklook.png"

    def exists(self) -> bool:
        return all(p.exists() for p in (self.t1, self.t2, self.valid, self.meta_path))

    def meta(self) -> dict[str, Any]:
        return dict(json.loads(self.meta_path.read_text()))

    def grid(self) -> Grid:
        return Grid.from_dict(self.meta()["grid"])

    def read(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        t1, _ = read_raster(self.t1)
        t2, _ = read_raster(self.t2)
        v, _ = read_raster(self.valid)
        return t1, t2, v[0]

    @property
    def aoi_id(self) -> str:
        return self.dir.parent.name

    @property
    def pair_id(self) -> str:
        return self.dir.name


def _device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # pragma: no cover
        return "cpu"


def _scaled_transform(t: Affine, factor: float) -> Affine:
    return Affine(t.a / factor, t.b, t.c, t.d, t.e / factor, t.f)


def scene_to_canonical(
    scene: Scene, grid: Grid, cfg: Config, sr_model: Any | None = None
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """One scene -> (uint16 (6,H,W) imagery, uint8 validity, info) on `grid`. Pure per-scene."""
    target_res = cfg.grid.res_m
    rgbn = [scene.read_band(b) for b in RGBN_BANDS]
    t10, epsg10 = rgbn[0].transform, rgbn[0].epsg
    for bd in rgbn[1:]:
        if bd.transform != t10 or bd.array.shape != rgbn[0].array.shape or bd.epsg != epsg10:
            raise ValueError(f"{scene.product_id}: 10 m bands differ in grid")
    rgbn_arr = np.stack([b.array for b in rgbn]).astype(np.float32)
    hole10 = (rgbn_arr == 0).any(axis=0)
    res10 = rgbn[0].res_m

    if cfg.sr.method == "bicubic":
        rgbn_hi = srmod.bicubic_to_5m(rgbn_arr, res10, target_res)
        factor = res10 / target_res
    elif cfg.sr.method == "sen2sr_lite_rgbn":
        if sr_model is None:
            sr_model = srmod.load_sen2sr(cfg.sr.model_dir, _device(), cfg.sr.method)
        rgbn_hi = srmod.super_resolve_rgbn(rgbn_arr, sr_model, pool_to_5m=cfg.sr.pool_to_5m)
        factor = rgbn_hi.shape[1] / rgbn_arr.shape[1]
    else:
        raise NotImplementedError("sr.method=sen2sr_lite_main (ablation A1b) is not built; see docs/DEVIATIONS.md")
    rgbn_t = _scaled_transform(t10, factor)
    rgbn_g = reproject_to_grid(rgbn_hi, rgbn_t, epsg10, grid, resampling="bilinear", dst_dtype=np.float32)

    avail = scene.available_bands()
    if all(b in avail for b in SWIR_BANDS):
        swir = [scene.read_band(b) for b in SWIR_BANDS]
        swir_arr = np.stack([b.array for b in swir]).astype(np.float32)
        f20 = swir[0].res_m / target_res
        swir_hi = resample(swir_arr, f20, "bilinear")
        swir_g = reproject_to_grid(
            swir_hi,
            _scaled_transform(swir[0].transform, f20),
            swir[0].epsg,
            grid,
            resampling="bilinear",
            dst_dtype=np.float32,
        )
        swir_filled = False
    else:  # 4-band input: zero reflectance after scaling (DEVIATIONS U5)
        log.warning("%s: no SWIR bands; filling B11/B12 with 0 (as band_dropout)", scene.product_id)
        swir_g = np.zeros((2, *grid.shape), dtype=np.float32)
        swir_filled = True

    img = np.concatenate([rgbn_g, swir_g], axis=0)
    img = np.clip(np.rint(img), 0, cfg.bands.clip_max * cfg.bands.scale).astype(np.uint16)

    inside = reproject_to_grid(np.ones(hole10.shape, dtype=np.uint8), t10, epsg10, grid, "nearest", nodata=0) == 1
    holes = reproject_to_grid(hole10.astype(np.uint8), t10, epsg10, grid, "nearest", nodata=1) == 1
    cloud_native = cloudmod.invalid_native(scene, cfg.cloud.method, _device())
    cloud_valid = cloudmod.validity_on_grid(cloud_native, grid, cfg.cloud.dilate_px) == 1
    valid = (inside & ~holes & cloud_valid).astype(np.uint8)
    info = dict(scene.meta())
    info.update(
        cloud_fraction=float(1.0 - cloud_valid[inside].mean()) if inside.any() else 1.0,
        hole_fraction=float(holes[inside].mean()) if inside.any() else 1.0,
        coverage=float(inside.mean()),
        swir_filled=swir_filled,
    )
    return img, valid, info


def preprocess_pair(
    t1: Path | Scene,
    t2: Path | Scene,
    aoi: Any | None,
    cfg: Config,
    out_dir: Path,
    pair_row: dict[str, Any] | None = None,
    sr_model: Any | None = None,
    scene_kwargs: dict[str, Any] | None = None,
) -> PairProduct:
    """HANDOFF 5.4 steps 1-6. `t1`/`t2` may be paths (SAFE zip, directory, GeoTIFF) or Scenes."""
    kw = scene_kwargs or {}
    s1, s2 = open_scene(t1, **kw), open_scene(t2, **kw)
    k1, k2 = s1.content_key(), s2.content_key()
    identical = k1 == k2
    swapped, first, second = canonical_pair(s1, s2, k1, k2)

    grid = canonical_grid([first.footprint_wgs84(), second.footprint_wgs84()], cfg.grid.res_m, cfg.grid.snap_m)
    if sr_model is None and cfg.sr.method.startswith("sen2sr"):
        sr_model = srmod.load_sen2sr(cfg.sr.model_dir, _device(), cfg.sr.method)

    img_a, valid_a, info_a = scene_to_canonical(first, grid, cfg, sr_model)
    if identical:
        img_b, valid_b, info_b = img_a.copy(), valid_a.copy(), dict(info_a)
    else:
        img_b, valid_b, info_b = scene_to_canonical(second, grid, cfg, sr_model)

    dx = dy = 0.0
    applied = False
    reason = "identical inputs" if identical else f"coreg method {cfg.coreg.method}"
    if not identical and cfg.coreg.method != "none":
        if cfg.coreg.method == "arosics_local_half":
            raise NotImplementedError("coreg.method=arosics_local_half needs third_party/ard (see its README)")
        both = (valid_a.astype(bool) & valid_b.astype(bool)).astype(np.uint8)
        nir = BAND_INDEX["B08"]
        res = half_shift(
            img_a[nir].astype(np.float32),
            img_b[nir].astype(np.float32),
            valid=both,
            max_shift_px=cfg.coreg.max_shift_px,
        )
        dx, dy, applied, reason = res.dx, res.dy, res.applied, res.reason or "phase correlation"
        if applied:
            img_a = shift_array(img_a, -dx / 2.0, -dy / 2.0, order=1)
            img_b = shift_array(img_b, dx / 2.0, dy / 2.0, order=1)
            valid_a = half_shift_mask(valid_a, dx, dy, "a")
            valid_b = half_shift_mask(valid_b, dx, dy, "b")
    valid = (valid_a.astype(bool) & valid_b.astype(bool)).astype(np.uint8)

    img_t1, img_t2 = (img_b, img_a) if swapped else (img_a, img_b)
    info_t1, info_t2 = (info_b, info_a) if swapped else (info_a, info_b)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    common = {"config_hash": cfg.hash(), "scale": str(cfg.bands.scale), "clip_max": str(cfg.bands.clip_max)}
    write_imagery(out_dir / "t1.tif", img_t1, grid, tags={**common, "product_id": str(info_t1["product_id"])})
    write_imagery(out_dir / "t2.tif", img_t2, grid, tags={**common, "product_id": str(info_t2["product_id"])})
    write_mask(out_dir / "valid.tif", valid, grid, nodata=None, tags=common)

    meta: dict[str, Any] = {
        "aoi_id": str(aoi["aoi_id"]) if aoi is not None else out_dir.parent.name,
        "pair_id": out_dir.name,
        "grid": grid.to_dict(),
        "band_order": list(BAND_ORDER),
        "scale": cfg.bands.scale,
        "clip_max": cfg.bands.clip_max,
        "t1": {**info_t1, "sha256": k2 if swapped else k1},
        "t2": {**info_t2, "sha256": k1 if swapped else k2},
        "identical_inputs": identical,
        "canonical_first": str(first.product_id),
        "coreg": {
            "method": cfg.coreg.method,
            "dx": dx,
            "dy": dy,
            "applied": applied,
            "reason": reason,
            "note": "dx,dy = shift of canonical-second relative to canonical-first (px)",
        },
        "valid_fraction": float(valid.mean()),
        "sr_method": cfg.sr.method,
        "cloud_method": cfg.cloud.method,
        "config_hash": cfg.hash(),
    }
    if pair_row:
        for k in ("same_orbit", "season_match", "tile_type_hint", "t1_date", "t2_date"):
            if k in pair_row and pair_row[k] is not None:
                meta[k] = pair_row[k] if not hasattr(pair_row[k], "item") else pair_row[k].item()
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    try:
        write_quicklook(img_t1, img_t2, valid, out_dir / "quicklook.png")
    except Exception as e:  # pragma: no cover - plotting is best effort
        log.warning("quicklook failed: %s", e)
    return PairProduct(out_dir)


def _stretch_rgb(img: np.ndarray) -> np.ndarray:
    rgb = np.stack([img[BAND_INDEX[b]] for b in ("B04", "B03", "B02")]).astype(np.float32)
    out = np.zeros_like(rgb)
    for i in range(3):
        band = rgb[i]
        nz = band[band > 0]
        lo, hi = (np.percentile(nz, 2), np.percentile(nz, 98)) if nz.size else (0.0, 1.0)
        out[i] = np.clip((band - lo) / max(hi - lo, 1e-6), 0, 1)
    return np.transpose(out, (1, 2, 0))


def write_quicklook(
    t1: np.ndarray, t2: np.ndarray, valid: np.ndarray, path: Path, mask: np.ndarray | None = None
) -> Path:
    """T1 | T2 | valid (or mask) panel as PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = [("T1", _stretch_rgb(t1)), ("T2", _stretch_rgb(t2))]
    if mask is not None:
        panels.append(("mask", mask))
    else:
        panels.append(("valid", valid))
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for ax, (title, arr) in zip(np.atleast_1d(axes), panels):
        if arr.ndim == 2:
            ax.imshow(
                arr,
                cmap="tab10" if title == "mask" else "gray",
                vmin=0,
                vmax=9 if title == "mask" else 1,
                interpolation="nearest",
            )
        else:
            ax.imshow(arr)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def run_all(
    cfg: Config,
    aoi_ids: Sequence[str] | None = None,
    pair_ids: Sequence[str] | None = None,
    force: bool = False,
) -> list[PairProduct]:
    """Preprocess every selected pair in registry/pairs.parquet; skip pairs current for this config."""
    from scd.imagery import registry

    pairs_path = cfg.registry_dir / "pairs.parquet"
    scenes_path = cfg.registry_dir / "scenes.parquet"
    pairs = pd.read_parquet(pairs_path)
    scenes = pd.read_parquet(scenes_path)
    aois = registry.load(registry.registry_path(cfg), strict=False).set_index("aoi_id")
    local = dict(zip(scenes["product_id"], scenes["local_path"]))
    sr_model = None
    out: list[PairProduct] = []
    for idx, row in pairs.iterrows():
        if aoi_ids and row["aoi_id"] not in set(aoi_ids):
            continue
        if pair_ids and row["pair_id"] not in set(pair_ids):
            continue
        if row.get("status", "selected") not in {"selected", "preprocessed"}:
            continue
        out_dir = cfg.pairs_dir / str(row["aoi_id"]) / str(row["pair_id"])
        prod = PairProduct(out_dir)
        if prod.exists() and not force and prod.meta().get("config_hash") == cfg.hash():
            out.append(prod)
            continue
        p1, p2 = local.get(row["t1_product"], ""), local.get(row["t2_product"], "")
        if not p1 or not p2 or not Path(p1).exists() or not Path(p2).exists():
            log.warning("%s/%s: products not on disk; run 01_fetch_scenes.py --download", row["aoi_id"], row["pair_id"])
            continue
        aoi = aois.loc[row["aoi_id"]].to_dict() | {"aoi_id": row["aoi_id"]} if row["aoi_id"] in aois.index else None
        if sr_model is None and cfg.sr.method.startswith("sen2sr"):
            sr_model = srmod.load_sen2sr(cfg.sr.model_dir, _device(), cfg.sr.method)
        pair_row = {str(k): v for k, v in row.items()}
        prod = preprocess_pair(Path(p1), Path(p2), aoi, cfg, out_dir, pair_row=pair_row, sr_model=sr_model)
        pairs.loc[idx, "valid_fraction"] = prod.meta()["valid_fraction"]
        pairs.loc[idx, "status"] = "preprocessed"
        out.append(prod)
        log.info("%s/%s: valid %.1f%%", row["aoi_id"], row["pair_id"], 100 * prod.meta()["valid_fraction"])
    pairs.to_parquet(pairs_path, index=False)
    pairs.to_csv(pairs_path.with_suffix(".csv"), index=False)
    return out


def list_products(cfg: Config) -> list[PairProduct]:
    return sorted((PairProduct(p.parent) for p in cfg.pairs_dir.glob("*/*/meta.json")), key=lambda p: str(p.dir))
