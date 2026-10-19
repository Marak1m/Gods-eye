"""GeoTIFF IO with the HANDOFF.md 3.2 profiles.

Imagery: uint16 x10000, tiled 512, uncompressed (the v3 read-speed lesson). Masks and outputs:
uint8, 255 = ignore/nodata, DEFLATE. One dataset handle per call: rasterio handles are not
thread-safe (section 14).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.windows import Window

from scd.constants import BAND_ORDER, GEOTIFF_BLOCK
from scd.geo.grid import Grid


def _tiling(grid: Grid) -> dict[str, Any]:
    if grid.width >= GEOTIFF_BLOCK and grid.height >= GEOTIFF_BLOCK:
        return {"tiled": True, "blockxsize": GEOTIFF_BLOCK, "blockysize": GEOTIFF_BLOCK}
    return {"tiled": False}


def imagery_profile(grid: Grid, count: int = len(BAND_ORDER)) -> dict[str, Any]:
    return {
        "driver": "GTiff",
        "dtype": "uint16",
        "count": count,
        "nodata": None,
        "compress": None,
        "BIGTIFF": "IF_SAFER",
        **grid.rasterio_profile(),
        **_tiling(grid),
    }


def mask_profile(grid: Grid, nodata: int | None = 255) -> dict[str, Any]:
    return {
        "driver": "GTiff",
        "dtype": "uint8",
        "count": 1,
        "nodata": nodata,
        "compress": "DEFLATE",
        "predictor": 2,
        **grid.rasterio_profile(),
        **_tiling(grid),
    }


def write_imagery(
    path: Path,
    arr: np.ndarray,
    grid: Grid,
    tags: dict[str, str] | None = None,
    band_names: Sequence[str] = BAND_ORDER,
) -> Path:
    """Write (bands, H, W) uint16 imagery with band descriptions."""
    if arr.ndim != 3 or arr.shape[1:] != grid.shape:
        raise ValueError(f"imagery shape {arr.shape} does not match grid {grid.shape}")
    if arr.dtype != np.uint16:
        raise ValueError(f"imagery must be uint16, got {arr.dtype}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **imagery_profile(grid, arr.shape[0])) as ds:
        ds.write(arr)
        for i, name in enumerate(band_names[: arr.shape[0]]):
            ds.set_band_description(i + 1, name)
        ds.update_tags(band_order=",".join(band_names[: arr.shape[0]]), **(tags or {}))
    return path


def write_mask(
    path: Path,
    arr: np.ndarray,
    grid: Grid,
    nodata: int | None = 255,
    tags: dict[str, str] | None = None,
) -> Path:
    """Write a (H, W) uint8 mask."""
    if arr.ndim != 2 or arr.shape != grid.shape:
        raise ValueError(f"mask shape {arr.shape} does not match grid {grid.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **mask_profile(grid, nodata)) as ds:
        ds.write(arr.astype(np.uint8), 1)
        if tags:
            ds.update_tags(**tags)
    return path


def read_raster(
    path: Path, bands: Sequence[int] | None = None, window: Window | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Read (bands, H, W) and the profile (plus 'tags' and 'descriptions')."""
    with rasterio.open(path) as ds:
        arr = ds.read(list(bands) if bands else None, window=window)
        profile = dict(ds.profile)
        profile["tags"] = ds.tags()
        profile["descriptions"] = list(ds.descriptions)
        if window is not None:
            profile["transform"] = ds.window_transform(window)
            profile["height"], profile["width"] = arr.shape[1], arr.shape[2]
    return arr, profile


def grid_from_profile(profile: dict[str, Any]) -> Grid:
    crs = profile["crs"]
    epsg = crs.to_epsg() if hasattr(crs, "to_epsg") else None
    if epsg is None:
        raise ValueError("raster CRS has no EPSG code")
    return Grid(int(epsg), profile["transform"], int(profile["width"]), int(profile["height"]))


def read_grid(path: Path) -> Grid:
    with rasterio.open(path) as ds:
        if ds.crs is None or ds.crs.to_epsg() is None:
            raise ValueError(f"{path}: CRS missing or without EPSG code")
        return Grid(int(ds.crs.to_epsg()), ds.transform, ds.width, ds.height)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
