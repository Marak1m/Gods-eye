"""M1: `half_shift(a, b)` is the exact mirror of `half_shift(b, a)` (HANDOFF 9, `test_coreg_symmetry`)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import ndimage

from scd.geo.coreg import estimate_shift, half_shift, half_shift_mask, shift_array

N = 192


def _pair(seed: int, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    big = ndimage.gaussian_filter(rng.normal(size=(N + 64, N + 64)), 2.5)
    big = (big - big.min()) / (big.max() - big.min())
    a = big[32 : 32 + N, 32 : 32 + N]
    moved = ndimage.shift(big, (dy, dx), order=3, mode="reflect")
    b = moved[32 : 32 + N, 32 : 32 + N]
    return a.astype(np.float32), b.astype(np.float32)


def test_estimate_recovers_subpixel_shift() -> None:
    a, b = _pair(0, 1.6, -0.8)
    dx, dy = estimate_shift(a, b)  # shift to apply to b to align with a
    assert abs(dx + 1.6) < 0.25 and abs(dy - 0.8) < 0.25


@pytest.mark.parametrize("seed,dx,dy", [(1, 1.6, -0.8), (2, -2.3, 0.4), (3, 0.0, 3.1), (4, 0.7, 0.7)])
def test_half_shift_mirror(seed: int, dx: float, dy: float) -> None:
    a, b = _pair(seed, dx, dy)
    r1 = half_shift(a, b)
    r2 = half_shift(b, a)
    assert r1.applied and r2.applied
    assert np.array_equal(r1.a, r2.b)
    assert np.array_equal(r1.b, r2.a)
    assert r1.dx == -r2.dx and r1.dy == -r2.dy  # exact negation, not approximate
    assert r1.residual_after < r1.residual_before


def test_half_shift_identical_inputs() -> None:
    a, _ = _pair(5, 0.0, 0.0)
    r = half_shift(a, a.copy())
    assert not r.applied and r.dx == 0.0 and r.dy == 0.0
    assert np.array_equal(r.a, a) and np.array_equal(r.b, a)


def test_half_shift_refuses_large_shift() -> None:
    a, b = _pair(6, 11.0, 0.0)
    r = half_shift(a, b, max_shift_px=8.0)
    assert not r.applied and "max_shift" in r.reason
    assert np.array_equal(r.a, a) and np.array_equal(r.b, b)
    assert abs(abs(r.dx) - 11.0) < 0.5


def test_half_shift_with_validity_mask_is_still_mirror() -> None:
    a, b = _pair(7, 1.2, -1.4)
    valid = np.ones_like(a, dtype=np.uint8)
    valid[40:90, 60:120] = 0
    r1 = half_shift(a, b, valid)
    r2 = half_shift(b, a, valid)
    assert np.array_equal(r1.a, r2.b) and np.array_equal(r1.b, r2.a)
    assert r1.dx == -r2.dx and r1.dy == -r2.dy


def test_mask_moves_with_image_and_exposed_strip_is_invalid() -> None:
    valid = np.ones((32, 32), dtype=np.uint8)
    mb = half_shift_mask(valid, dx=4.0, dy=0.0, which="b")  # b moves by +2 px in x
    ma = half_shift_mask(valid, dx=4.0, dy=0.0, which="a")  # a moves by -2 px in x
    assert mb[:, :2].sum() == 0 and mb[:, 2:].all()
    assert ma[:, -2:].sum() == 0 and ma[:, :-2].all()


def test_shift_array_keeps_dtype_and_handles_bands() -> None:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 12000, size=(6, 16, 16), dtype=np.uint16)
    out = shift_array(arr, 1.0, 0.0, order=1)
    assert out.dtype == np.uint16 and out.shape == arr.shape
    assert np.array_equal(out[:, :, 1:], arr[:, :, :-1])
