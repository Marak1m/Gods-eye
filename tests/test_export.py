"""M7: GeoTIFF/shapefile export contracts (HANDOFF 9 `test_export`; DEVIATIONS U3)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine

from scd.constants import IGNORE
from scd.geo.grid import Grid
from scd.infer import export

GRID = Grid(32645, Affine(5.0, 0.0, 500000.0, 0.0, -5.0, 3300000.0), 64, 48)


def _mask() -> tuple[np.ndarray, np.ndarray]:
    m = np.zeros((48, 64), dtype=np.uint8)
    m[2:10, 2:12] = 2
    m[20:30, 20:40] = 4
    m[20:22, 50:60] = 3
    m[35:45, 5:15] = 1
    m[40:44, 50:54] = 1  # second buildup component
    valid = np.ones((48, 64), dtype=np.uint8)
    valid[:, 60:] = 0
    m[0, 0] = IGNORE
    return m, valid


def test_geotiff_values_and_georeferencing(tmp_path: Path) -> None:
    m, valid = _mask()
    p = export.write_geotiff(m, GRID, tmp_path / "change.tif", {"tau": "0.5"}, invalid_value=0, valid=valid)
    with rasterio.open(p) as ds:
        arr = ds.read(1)
        assert set(np.unique(arr)) <= {0, 1, 2, 3, 4}  # invalid written as 0 by default (U3)
        assert ds.nodata is None and ds.crs.to_epsg() == 32645 and ds.transform == GRID.transform
        assert ds.tags()["tau"] == "0.5" and ds.tags()["invalid_value"] == "0"
        assert (arr[:, 60:] == 0).all()
    p2 = export.write_geotiff(m, GRID, tmp_path / "change255.tif", {}, invalid_value=255, valid=valid)
    with rasterio.open(p2) as ds:
        arr = ds.read(1)
        assert set(np.unique(arr)) <= {0, 1, 2, 3, 4, 255} and ds.nodata == 255
        assert (arr[:, 60:] == 255).all()
    v = export.write_valid(valid, GRID, tmp_path / "valid.tif")
    with rasterio.open(v) as ds:
        assert np.array_equal(ds.read(1), valid)


def test_shapefile_fields_and_component_count(tmp_path: Path) -> None:
    m, valid = _mask()
    m[m == IGNORE] = 0
    conf = np.full(m.shape, 0.7, dtype=np.float32)
    shp = export.write_shapefile(m, GRID, tmp_path / "change.shp", conf=conf, simplify_m=2.5)
    assert shp.with_suffix(".prj").exists() and shp.with_suffix(".dbf").exists()
    gdf = gpd.read_file(shp)
    assert len(gdf) == export.component_count(m) == 5
    assert set(gdf["class"]) == {1, 2, 3, 4} and gdf["class"].between(1, 4).all()
    assert (gdf["area_m2"] > 0).all() and np.allclose(gdf["conf"], 0.7)
    assert gdf.crs.to_epsg() == 32645
    building = gdf[gdf["class"] == 2].iloc[0]
    assert abs(building["area_m2"] - 8 * 10 * 25) < 1e-6  # 8 x 10 px of 25 m2


def test_empty_shapefile(tmp_path: Path) -> None:
    shp = export.write_shapefile(np.zeros((48, 64), dtype=np.uint8), GRID, tmp_path / "empty.shp")
    assert shp.exists() and len(gpd.read_file(shp)) == 0
    assert export.component_count(np.zeros((4, 4), dtype=np.uint8)) == 0
