"""M1: raster profiles (HANDOFF 3.2), reprojection helpers and vector guards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from affine import Affine

from scd.constants import BAND_ORDER
from scd.geo import io as gio
from scd.geo.grid import Grid
from scd.geo.reproject import reproject_to_grid, resample
from scd.geo.vector import GeographicCRSError, content_key, require_projected

GRID = Grid(32645, Affine(5.0, 0.0, 500000.0, 0.0, -5.0, 3300000.0), 64, 48)


def test_imagery_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 12000, size=(6, 48, 64), dtype=np.uint16)
    p = gio.write_imagery(tmp_path / "t1.tif", arr, GRID, tags={"product": "x"})
    back, profile = gio.read_raster(p)
    assert np.array_equal(back, arr)
    assert profile["dtype"] == "uint16" and profile.get("compress") is None
    assert profile["descriptions"] == list(BAND_ORDER)
    assert profile["tags"]["band_order"] == ",".join(BAND_ORDER) and profile["tags"]["product"] == "x"
    assert gio.read_grid(p) == GRID


def test_mask_roundtrip(tmp_path: Path) -> None:
    m = np.zeros((48, 64), dtype=np.uint8)
    m[10:20, 10:20] = 2
    m[0, 0] = 255
    p = gio.write_mask(tmp_path / "m.tif", m, GRID)
    back, profile = gio.read_raster(p)
    assert np.array_equal(back[0], m)
    assert profile["nodata"] == 255 and profile["compress"].lower() == "deflate"


def test_tiling_rule() -> None:
    big = Grid(32645, GRID.transform, 1024, 1024)
    assert gio.imagery_profile(big)["tiled"] is True and gio.imagery_profile(big)["blockxsize"] == 512
    assert gio.imagery_profile(GRID)["tiled"] is False


def test_write_rejects_bad_input(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        gio.write_imagery(tmp_path / "x.tif", np.zeros((6, 48, 64), dtype=np.float32), GRID)
    with pytest.raises(ValueError):
        gio.write_mask(tmp_path / "x.tif", np.zeros((10, 10), dtype=np.uint8), GRID)


def test_sha256_and_content_key(tmp_path: Path) -> None:
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc")
    assert gio.sha256_file(p) == content_key(p) == content_key(b"abc")
    a = np.arange(10, dtype=np.float32)
    assert content_key(a) == content_key(a.copy()) and content_key(a) != content_key(a + 1)
    assert content_key(a) != content_key(a.astype(np.float64))


def test_require_projected() -> None:
    require_projected(32645)
    with pytest.raises(GeographicCRSError):
        require_projected(4326)


def test_resample_mean_and_bicubic() -> None:
    arr = np.arange(6 * 8 * 8, dtype=np.float32).reshape(6, 8, 8)
    pooled = resample(arr, 0.5, "mean")
    assert pooled.shape == (6, 4, 4)
    assert pooled[0, 0, 0] == arr[0, :2, :2].mean()
    up = resample(arr, 2.0, "bicubic")
    assert up.shape == (6, 16, 16)
    with pytest.raises(ValueError):
        resample(arr, 0.3, "mean")


def test_reproject_identity() -> None:
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 10000, size=(2, 48, 64), dtype=np.uint16)
    out = reproject_to_grid(arr, GRID.transform, GRID.epsg, GRID, resampling="nearest")
    assert np.array_equal(out, arr)
    out2 = reproject_to_grid(arr[0], GRID.transform, GRID.epsg, GRID, resampling="nearest")
    assert out2.shape == (48, 64)
