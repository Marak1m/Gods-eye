"""Negatives (HANDOFF.md 5.9): stable-site selection and the image-change gate.

`image_change_gate` is the companion project's v3 gate: pixels whose mean absolute band
difference exceeds the chip median + 1 std (3x3 smoothed) carry no negative target (255) in
auto-only negatives. `stable_sites` needs CBRA / OBT / IO-LULC stability and OSM history and is a
stub until Phase 1 (external data).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from scd.constants import IGNORE, UINT16_SCALE


def image_change_gate(t1: np.ndarray, t2: np.ndarray, valid: np.ndarray, smooth_px: int = 3) -> np.ndarray:
    """Boolean mask of pixels where the image difference is unusually large (valid pixels only)."""
    x1 = t1.astype(np.float32) / UINT16_SCALE
    x2 = t2.astype(np.float32) / UINT16_SCALE
    d = np.abs(x2 - x1).mean(axis=0)
    if smooth_px > 1:
        d = ndimage.uniform_filter(d, size=smooth_px, mode="reflect")
    v = valid.astype(bool)
    if not v.any():
        return np.zeros(v.shape, dtype=bool)
    vals = d[v]
    thr = float(np.median(vals) + vals.std())
    return (d > thr) & v


def apply_gate(change: np.ndarray, t1: np.ndarray, t2: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """For auto-only negatives: set gated pixels to 255 so they carry no negative target."""
    out = change.copy()
    out[image_change_gate(t1, t2, valid)] = IGNORE
    return out


def stable_sites(stratum: str, n: int, cfg: Any) -> list[Any]:
    """AOIs with no built change 2017-2024 (CBRA/OBT/IO-LULC) and no OSM edits in the window."""
    raise NotImplementedError(
        "scd.labels.negatives.stable_sites: needs external stability products (Phase 1); "
        "see HANDOFF.md section 5.9 and the sources/ stubs"
    )
