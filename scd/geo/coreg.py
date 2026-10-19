"""Symmetric half-shift co-registration (HANDOFF.md section 5.4 step 4).

The shift is estimated by windowed cross-correlation (Hann window, upsampled-DFT sub-pixel peak,
two refinement passes) on the NIR band over valid pixels; `a` moves by (-dx/2, -dy/2) and `b` by
(+dx/2, +dy/2). The pair is first put in canonical order by content key so that
`half_shift(b, a)` is the exact mirror of `half_shift(a, b)` (DEVIATIONS C3). Shifts over
`max_shift_px` are refused: logged and left uncorrected. Validity masks move with their images
and the exposed strip becomes invalid (DEVIATIONS C6).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from skimage.registration import phase_cross_correlation

from scd.geo.vector import canonical_pair, content_key
from scd.log import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class ShiftResult:
    """`dx, dy` is the shift (cols, rows) that aligns `b` to `a`; both outputs moved half-way."""

    a: np.ndarray
    b: np.ndarray
    dx: float
    dy: float
    applied: bool
    reason: str = ""
    residual_before: float = float("nan")
    residual_after: float = float("nan")


def _prepare(arr: np.ndarray, valid: np.ndarray | None) -> np.ndarray:
    """Normalised, masked, Hann-windowed float64 copy of the (first band of the) array."""
    x = arr.astype(np.float64)
    if x.ndim == 3:
        x = x[0]
    m = np.ones_like(x, dtype=bool) if valid is None else valid.astype(bool)
    if m.sum() < 16:
        return np.zeros_like(x)
    mu = x[m].mean()
    sd = x[m].std()
    out = (x - mu) / (sd if sd > 0 else 1.0)
    out[~m] = 0.0
    window = np.hanning(x.shape[0])[:, None] * np.hanning(x.shape[1])[None, :]
    return out * window


def shift_array(arr: np.ndarray, dx: float, dy: float, order: int = 1, fill: float = 0.0) -> np.ndarray:
    """Shift a (H, W) or (bands, H, W) array by (dx cols, dy rows); exposed border gets `fill`."""
    if dx == 0.0 and dy == 0.0:
        return arr.copy()
    shift = (dy, dx) if arr.ndim == 2 else (0.0, dy, dx)
    out = ndimage.shift(arr.astype(np.float64), shift, order=order, mode="constant", cval=fill, prefilter=order > 1)
    if np.issubdtype(arr.dtype, np.integer):
        info = np.iinfo(arr.dtype)
        out = np.clip(np.rint(out), info.min, info.max)
    return out.astype(arr.dtype)


def estimate_shift(
    a: np.ndarray,
    b: np.ndarray,
    valid: np.ndarray | None = None,
    max_shift_px: float = 8.0,
    upsample: int = 20,
    iterations: int = 2,
) -> tuple[float, float]:
    """Sub-pixel shift (dx, dy) to apply to `b` so that it aligns with `a`.

    Windowed cross-correlation with an upsampled-DFT peak, refined by re-estimating on the
    shifted `b` so that the edge-induced bias of the first pass is largely removed.
    """
    ra = _prepare(a, valid)
    if not ra.any():
        return 0.0, 0.0
    total_dx = total_dy = 0.0
    b2 = b.astype(np.float64)
    if b2.ndim == 3:
        b2 = b2[0]
    for _ in range(max(1, iterations)):
        moving = shift_array(b2, total_dx, total_dy, order=1) if (total_dx or total_dy) else b2
        rb = _prepare(moving, valid)
        if not rb.any():
            break
        shift, _err, _phase = phase_cross_correlation(ra, rb, upsample_factor=upsample, normalization=None)
        ddy, ddx = float(shift[0]), float(shift[1])
        total_dx += ddx
        total_dy += ddy
        if math.hypot(ddx, ddy) < 0.02 or math.hypot(total_dx, total_dy) > max_shift_px:
            break
    return total_dx, total_dy


def _residual(a: np.ndarray, b: np.ndarray, valid: np.ndarray | None) -> float:
    x, y = a.astype(np.float64), b.astype(np.float64)
    if x.ndim == 3:
        x, y = x[0], y[0]
    m = np.ones_like(x, dtype=bool) if valid is None else valid.astype(bool)
    return float(np.abs(x - y)[m].mean()) if m.any() else float("nan")


def half_shift_mask(valid: np.ndarray, dx: float, dy: float, which: str) -> np.ndarray:
    """Move a validity mask with its image (nearest) and mark the exposed strip invalid.

    `which` is 'a' (moved by -d/2) or 'b' (moved by +d/2), with (dx, dy) as reported by `half_shift`.
    """
    s = -0.5 if which == "a" else 0.5
    return shift_array(valid.astype(np.uint8), s * dx, s * dy, order=0, fill=0).astype(np.uint8)


def half_shift(
    a: np.ndarray,
    b: np.ndarray,
    valid: np.ndarray | None = None,
    max_shift_px: float = 8.0,
    order: int = 1,
) -> ShiftResult:
    """Half-shift both arrays towards each other. Exact mirror under argument swap."""
    ka, kb = content_key(a), content_key(b)
    if ka == kb:
        return ShiftResult(a.copy(), b.copy(), 0.0, 0.0, False, "identical inputs", 0.0, 0.0)
    swapped, first, second = canonical_pair(a, b, ka, kb)
    dx, dy = estimate_shift(first, second, valid, max_shift_px)
    before = _residual(first, second, valid)
    sign = -1.0 if swapped else 1.0  # report the shift of b relative to a
    if math.hypot(dx, dy) > max_shift_px:
        log.warning("coreg: shift (%.2f, %.2f) px exceeds max %.1f; left uncorrected", dx, dy, max_shift_px)
        return ShiftResult(a.copy(), b.copy(), sign * dx, sign * dy, False, "exceeds max_shift_px", before, before)
    first_s = shift_array(first, -dx / 2.0, -dy / 2.0, order=order)
    second_s = shift_array(second, dx / 2.0, dy / 2.0, order=order)
    ones = np.ones(first.shape[-2:], dtype=np.uint8) if valid is None else valid.astype(np.uint8)
    valid_after = half_shift_mask(ones, dx, dy, "a") & half_shift_mask(ones, dx, dy, "b")
    after = _residual(first_s, second_s, valid_after)
    if not after <= before:
        log.warning("coreg: residual did not improve (%.4f -> %.4f); shift kept", before, after)
    if swapped:
        return ShiftResult(second_s, first_s, sign * dx, sign * dy, True, "", before, after)
    return ShiftResult(first_s, second_s, dx, dy, True, "", before, after)
