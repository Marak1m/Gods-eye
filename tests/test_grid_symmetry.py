"""M1: `canonical_grid([a, b]) == canonical_grid([b, a])` for random footprint pairs (HANDOFF 9)."""

from __future__ import annotations

import math
import random

import pytest
from affine import Affine
from shapely.geometry import box

from scd.geo.grid import Grid, canonical_grid, utm_epsg

RES, SNAP = 5.0, 5.0


def _random_pair(rng: random.Random):
    lon = rng.uniform(-170.0, 170.0)
    lat = rng.uniform(-70.0, 70.0)
    size = rng.uniform(0.03, 0.12)
    a = box(lon, lat, lon + size, lat + size)
    dx = rng.uniform(-0.4, 0.4) * size
    dy = rng.uniform(-0.4, 0.4) * size
    sb = size * rng.uniform(0.8, 1.2)
    b = box(lon + dx, lat + dy, lon + dx + sb, lat + dy + sb)
    return a, b


def _on_multiple(v: float, m: float) -> bool:
    return math.isclose(v / m, round(v / m), abs_tol=1e-9)


@pytest.mark.parametrize("seed", range(50))
def test_grid_symmetry(seed: int) -> None:
    rng = random.Random(seed)
    a, b = _random_pair(rng)
    g1 = canonical_grid([a, b], RES, SNAP)
    g2 = canonical_grid([b, a], RES, SNAP)
    assert g1 == g2  # bit-identical transform, epsg, shape
    assert _on_multiple(g1.transform.c, SNAP) and _on_multiple(g1.transform.f, SNAP)
    assert g1.res_m == RES and g1.width > 0 and g1.height > 0
    lat = a.union(b).centroid.y
    assert (32601 <= g1.epsg <= 32660) if lat >= 0 else (32701 <= g1.epsg <= 32760)


def test_grid_identity_and_single() -> None:
    a = box(89.0, 29.0, 89.1, 29.1)
    assert canonical_grid([a, a], RES, SNAP) == canonical_grid([a], RES, SNAP)


def test_grid_requires_overlap() -> None:
    a = box(89.0, 29.0, 89.1, 29.1)
    b = box(90.0, 30.0, 90.1, 30.1)
    with pytest.raises(ValueError, match="overlap"):
        canonical_grid([a, b], RES, SNAP)


def test_grid_extent_covers_intersection() -> None:
    a = box(89.0, 29.0, 89.1, 29.1)
    b = box(89.05, 29.05, 89.15, 29.15)
    g = canonical_grid([a, b], RES, SNAP)
    left, bottom, right, top = g.bounds
    assert right - left <= 0.06 * 111_000 and top - bottom <= 0.06 * 111_000  # ~5 km, not 10


def test_utm_epsg() -> None:
    assert utm_epsg(89.3, 29.35) == 32645
    assert utm_epsg(-70.0, -33.0) == 32719
    assert utm_epsg(0.0, 0.0) == 32631
    assert utm_epsg(179.9, 10.0) == 32660


def test_grid_window_and_roundtrip() -> None:
    g = Grid(32645, Affine(5.0, 0.0, 500000.0, 0.0, -5.0, 3300000.0), 1024, 1024)
    w = g.window(768, 256, 512, 512)
    assert w.width == 512 and w.transform.c == 500000.0 + 256 * 5 and w.transform.f == 3300000.0 - 768 * 5
    assert Grid.from_dict(g.to_dict()) == g
    assert g.pixel_area_m2 == 25.0
    with pytest.raises(ValueError):
        Grid(32645, Affine(5.0, 0.0, 0.0, 0.0, 5.0, 0.0), 10, 10)  # not north-up
