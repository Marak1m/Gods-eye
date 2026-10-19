"""Array resampling and reprojection onto a `Grid` (HANDOFF.md section 5.4 step 2).

Per-image operations only; symmetry comes from calling them identically for both dates.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from affine import Affine
from rasterio.crs import CRS as RioCRS
from rasterio.enums import Resampling
from rasterio.warp import reproject as rio_reproject
from skimage.transform import resize

from scd.geo.grid import Grid

RESAMPLING: dict[str, Resampling] = {
    "nearest": Resampling.nearest,
    "bilinear": Resampling.bilinear,
    "cubic": Resampling.cubic,
    "average": Resampling.average,
}
_ORDER = {"nearest": 0, "bilinear": 1, "bicubic": 3}


def reproject_to_grid(
    arr: np.ndarray,
    src_transform: Affine,
    src_epsg: int,
    grid: Grid,
    resampling: str = "bilinear",
    nodata: float | None = None,
    dst_dtype: Any | None = None,
) -> np.ndarray:
    """Warp a (bands, H, W) or (H, W) array onto `grid`."""
    squeeze = arr.ndim == 2
    src = arr[None] if squeeze else arr
    dtype = np.dtype(dst_dtype) if dst_dtype is not None else src.dtype
    out = np.zeros((src.shape[0], grid.height, grid.width), dtype=dtype)
    if nodata is not None:
        out[...] = nodata
    rio_reproject(
        source=np.ascontiguousarray(src),
        destination=out,
        src_transform=src_transform,
        src_crs=RioCRS.from_epsg(src_epsg),
        dst_transform=grid.transform,
        dst_crs=RioCRS.from_epsg(grid.epsg),
        resampling=RESAMPLING[resampling],
        src_nodata=nodata,
        dst_nodata=nodata,
        num_threads=1,
    )
    return out[0] if squeeze else out


def resample(arr: np.ndarray, factor: float, method: str) -> np.ndarray:
    """Resample by `factor` (>1 upsamples). Methods: nearest, bilinear, bicubic, mean.

    `mean` is an exact block mean and requires 1/factor to be an integer (the 2x2 pool to 5 m).
    """
    if factor == 1.0:
        return arr.copy()
    squeeze = arr.ndim == 2
    src = arr[None] if squeeze else arr
    if method == "mean":
        inv = 1.0 / factor
        k = int(round(inv))
        if abs(inv - k) > 1e-9 or k < 1:
            raise ValueError(f"mean pooling needs an integer block size, got factor {factor}")
        b, h, w = src.shape
        if h % k or w % k:
            raise ValueError(f"shape {src.shape} not divisible by block {k}")
        out = src.reshape(b, h // k, k, w // k, k).astype(np.float64).mean(axis=(2, 4))
    else:
        order = _ORDER[method]
        h = int(round(src.shape[1] * factor))
        w = int(round(src.shape[2] * factor))
        out = np.stack(
            [
                resize(
                    band.astype(np.float64),
                    (h, w),
                    order=order,
                    mode="reflect",
                    anti_aliasing=False,
                    preserve_range=True,
                )
                for band in src
            ]
        )
    out = out.astype(np.float32)
    return out[0] if squeeze else out
