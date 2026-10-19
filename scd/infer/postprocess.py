"""Per-class post-processing (HANDOFF.md 8 step 5, plan doc 10 step 9).

Building: drop components under min_area_m2, fill holes under hole_fill_m2. Solar: drop under
min_area_m2, closing with a disc. Buildup: drop under min_area_m2, fill holes. Road: drop
components whose skeleton is shorter than min_length_m; no closing (the v2 lesson). Water rule:
building or solar components whose 20 m surround is water (NDWI > threshold) at both dates are
removed (ships, floating structures); buildup over former water is kept (reclamation).
Operations that add pixels only ever fill background (0) pixels.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import measure, morphology

from scd.config import Config
from scd.constants import BAND_INDEX, CLASS_IDS, IGNORE


def ndwi(img: np.ndarray) -> np.ndarray:
    g = img[BAND_INDEX["B03"]].astype(np.float32)
    n = img[BAND_INDEX["B08"]].astype(np.float32)
    return (g - n) / np.maximum(g + n, 1.0)


def _remove_small(binary: np.ndarray, min_px: float) -> np.ndarray:
    if min_px <= 1:
        return binary
    # skimage >= 0.26: max_size removes objects smaller than or equal to the value
    return morphology.remove_small_objects(binary, max_size=int(np.ceil(min_px)) - 1, connectivity=2)


def _fill_small_holes(binary: np.ndarray, max_px: float) -> np.ndarray:
    if max_px <= 0:
        return binary
    return morphology.remove_small_holes(binary, max_size=int(np.ceil(max_px)) - 1, connectivity=1)


def _road_filter(binary: np.ndarray, min_length_m: float, res_m: float) -> np.ndarray:
    out = binary.copy()
    lab = measure.label(binary, connectivity=2)
    for region in measure.regionprops(lab):
        comp = lab == region.label
        skel = morphology.skeletonize(comp)
        length_m = float(skel.sum()) * res_m
        if length_m < min_length_m:
            out[comp] = False
    return out


def _water_rule(binary: np.ndarray, water_both: np.ndarray, surround_px: int) -> np.ndarray:
    out = binary.copy()
    lab = measure.label(binary, connectivity=2)
    for region in measure.regionprops(lab):
        comp = lab == region.label
        ring = ndimage.binary_dilation(comp, iterations=max(1, surround_px)) & ~comp
        if ring.any() and float(water_both[ring].mean()) > 0.5:
            out[comp] = False
    return out


def apply(mask: np.ndarray, cfg: Config, t1: np.ndarray, t2: np.ndarray, valid: np.ndarray, res_m: float) -> np.ndarray:
    pp = cfg.infer.postprocess
    px_area = res_m * res_m
    out = mask.copy().astype(np.uint8)
    out[out == IGNORE] = 0
    out[valid == 0] = 0
    water_both = (ndwi(t1) > cfg.infer.water_ndwi) & (ndwi(t2) > cfg.infer.water_ndwi)
    surround_px = max(1, int(round(20.0 / res_m)))
    result = np.zeros_like(out)
    background = out == 0
    for name in ("buildup", "building", "solar"):
        cid = CLASS_IDS[name]
        binary = out == cid
        binary = _remove_small(binary, pp.min_area_m2.get(name, 0.0) / px_area)
        if name == "solar" and pp.solar_close_px > 0:
            closed = morphology.closing(binary, morphology.disk(pp.solar_close_px)).astype(bool)
            binary = binary | (closed & background)
        if name in pp.hole_fill_m2:
            filled = _fill_small_holes(binary, pp.hole_fill_m2[name] / px_area)
            binary = binary | (filled & background)
        if name in ("building", "solar"):
            binary = _water_rule(binary, water_both, surround_px)
        result[binary & (result == 0)] = cid
    road = out == CLASS_IDS["road"]
    road = _road_filter(road, pp.min_length_m.get("road", 0.0), res_m)
    result[road & (result == 0)] = CLASS_IDS["road"]
    result[valid == 0] = 0
    return result
