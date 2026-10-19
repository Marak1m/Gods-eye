"""Cloud and shadow masking (HANDOFF.md 5.4 step 3): OmniCloudMask, SCL fallback.

OmniCloudMask path UNVERIFIED on this machine (DEVIATIONS C18): fed DN after the BOA offset in
Red, Green, NIR order, per the API notes in HANDOFF.md 2.1. Classes {1,2,3} (thick, thin,
shadow) are invalid; SCL fallback uses {0,1,3,8,9,10}. Snow stays valid in both paths.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from scd.constants import OMNICLOUD_INVALID, OMNICLOUD_ORDER, SCL_INVALID
from scd.geo.grid import Grid
from scd.geo.reproject import reproject_to_grid
from scd.imagery.scenes import BandData, Scene
from scd.log import get_logger

log = get_logger(__name__)


def scl_invalid(scl: np.ndarray) -> np.ndarray:
    """Boolean mask of SCL classes treated as invalid."""
    return np.isin(scl.astype(np.int32), list(SCL_INVALID))


def omnicloud_mask(
    rgn_dn: np.ndarray, device: str = "cuda", batch_size: int = 1, compile_models: bool = False
) -> np.ndarray:
    """(3, H, W) DN in Red, Green, NIR -> uint8 classes (0 clear, 1 thick, 2 thin, 3 shadow)."""
    from omnicloudmask import predict_from_array

    kwargs: dict[str, Any] = {"inference_device": device, "batch_size": batch_size}
    if device.startswith("cuda"):
        kwargs["inference_dtype"] = "bf16"
    if compile_models:
        kwargs["compile_models"] = True
    pred = predict_from_array(np.ascontiguousarray(rgn_dn.astype(np.float32)), **kwargs)
    return np.asarray(pred)[0].astype(np.uint8)


def invalid_native(scene: Scene, method: str, device: str = "cuda") -> BandData | None:
    """Invalid (1) / valid (0) uint8 at the native 10 or 20 m grid, before dilation; None = no source."""
    if method == "scl":
        scl = scene.read_scl()
        if scl is None:
            log.warning("%s: cloud.method=scl but no SCL available; all pixels valid", scene.product_id)
            return None
        return BandData(scl_invalid(scl.array).astype(np.uint8), scl.transform, scl.epsg)
    if method == "omnicloudmask":
        bands = [scene.read_band(b) for b in OMNICLOUD_ORDER]
        rgn = np.stack([b.array for b in bands])
        classes = omnicloud_mask(rgn, device=device)
        inv = np.isin(classes, list(OMNICLOUD_INVALID)).astype(np.uint8)
        return BandData(inv, bands[0].transform, bands[0].epsg)
    raise ValueError(f"unknown cloud method {method!r}")


def validity_on_grid(native: BandData | None, grid: Grid, dilate_px: int) -> np.ndarray:
    """Dilate the invalid mask at its native resolution, resample nearest to `grid`; 1 = valid."""
    if native is None:
        return np.ones(grid.shape, dtype=np.uint8)
    inv = native.array.astype(bool)
    if dilate_px > 0 and inv.any():
        inv = ndimage.binary_dilation(inv, iterations=dilate_px)
    inv_grid = reproject_to_grid(
        inv.astype(np.uint8), native.transform, native.epsg, grid, resampling="nearest", nodata=1
    )
    return (inv_grid == 0).astype(np.uint8)
