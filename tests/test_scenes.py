"""M2: scene readers and pair-selection rules (HANDOFF 5.2, 5.4 step 2)."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio

from scd.constants import BAND_ORDER, UINT16_SCALE
from scd.imagery.scenes import GeoTiffScene, PairRules, SimpleScene, open_scene, select_pairs
from tests.fixtures.synth import make_pair


def test_simple_scene_applies_boa_offset(tmp_path: Path) -> None:
    d1, _, _ = make_pair(tmp_path, "s", seed=1, identical=True)
    sc = open_scene(d1)
    assert isinstance(sc, SimpleScene)
    b = sc.read_band("B08")
    with rasterio.open(d1 / "B08.tif") as ds:
        stored = ds.read(1).astype(np.float32)
    assert np.allclose(b.array, stored - 1000.0)  # stored = true + 1000; reader adds offset -1000
    assert sc.available_bands() == BAND_ORDER
    assert sc.read_scl() is not None and sc.footprint_wgs84().is_valid
    assert sc.content_key() == open_scene(d1).content_key()


def _write_geotiff(
    path: Path, arr: np.ndarray, descriptions: list[str] | None, tags: dict | None, src_dir: Path
) -> None:
    with rasterio.open(src_dir / "B02.tif") as ref:
        profile = ref.profile
    profile.update(count=arr.shape[0], dtype=str(arr.dtype))
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(arr)
        if descriptions:
            for i, d in enumerate(descriptions):
                ds.set_band_description(i + 1, d)
        if tags:
            ds.update_tags(**tags)


def test_geotiff_scene_detects_dn_and_reflectance(tmp_path: Path) -> None:
    d1, _, _ = make_pair(tmp_path, "g", seed=2, identical=True)
    sc = SimpleScene(d1)
    bands = np.stack([sc.read_band(b).array for b in ("B02", "B03", "B04", "B08")])
    dn = tmp_path / "dn.tif"
    _write_geotiff(dn, bands.astype(np.uint16), ["B02", "B03", "B04", "B08"], None, d1)
    g = GeoTiffScene(dn)
    assert g.available_bands() == ("B02", "B03", "B04", "B08") and g.scale_to_dn == 1.0
    refl = tmp_path / "refl.tif"
    _write_geotiff(refl, (bands / UINT16_SCALE).astype(np.float32), None, {"band_order": "B02,B03,B04,B08"}, d1)
    g2 = GeoTiffScene(refl)
    assert g2.scale_to_dn == UINT16_SCALE
    assert np.allclose(g2.read_band("B08").array, bands[3], atol=1.0)
    amb = tmp_path / "amb.tif"
    _write_geotiff(amb, (bands / 100.0).astype(np.float32), None, {"band_order": "B02,B03,B04,B08"}, d1)
    with pytest.raises(ValueError, match="ambiguous"):
        GeoTiffScene(amb)
    assert GeoTiffScene(amb, assume="dn").scale_to_dn == 1.0


def _scenes(aoi: str, rows: list[tuple[str, int, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "aoi_id": aoi,
                "product_id": pid,
                "tile": "T45RVL",
                "relative_orbit": orbit,
                "date": d,
                "cloud_cover": cc,
                "baseline": 5.0,
                "local_path": "",
                "status": "queried",
            }
            for pid, orbit, d, cc in rows
        ]
    )


def test_select_pairs_positive_prefers_same_orbit_and_season() -> None:
    aoi = pd.Series(
        {
            "aoi_id": "A_x_01",
            "target_type": "solar",
            "t1_window": "2019-05-01/2019-10-31",
            "t2_window": "2023-05-01/2023-10-31",
        }
    )
    df = _scenes(
        "A_x_01",
        [
            ("P1", 5, "2019-07-10", 3.0),
            ("P1b", 48, "2019-07-12", 1.0),  # cross orbit, cleaner
            ("P2", 5, "2023-07-14", 4.0),
            ("P2w", 5, "2023-10-30", 2.0),  # out of season vs P1
            ("P3", 5, "2023-06-01", 40.0),  # too cloudy
        ],
    )
    specs = select_pairs(aoi, df, PairRules(max_pairs=3))
    assert specs and specs[0].t1_product == "P1" and specs[0].t2_product == "P2"
    assert specs[0].same_orbit and specs[0].season_match and specs[0].pair_id == "p01"
    assert all(s.same_orbit for s in specs)  # cross-orbit excluded while same-orbit pairs exist
    assert any(not s.season_match for s in specs)
    assert "P3" not in {s.t2_product for s in specs}


def test_select_pairs_negative_maximises_seasonal_contrast() -> None:
    aoi = pd.Series({"aoi_id": "F_neg_01", "target_type": "negative"})
    df = _scenes(
        "F_neg_01",
        [
            ("W1", 5, "2021-02-10", 2.0),
            ("W2", 5, "2021-03-05", 1.0),
            ("S1", 5, "2021-08-20", 3.0),
            ("S2", 7, "2021-09-01", 1.0),
        ],
    )
    specs = select_pairs(aoi, df, PairRules(contrast_seasons=[[2, 3, 4], [8, 9]], max_pairs=2))
    assert len(specs) == 2 and all(s.tile_type_hint == "NOCHANGE_HARD" for s in specs)
    assert specs[0].same_orbit and specs[0].t1_product == "W1" and specs[0].t2_product == "S1"


def test_select_pairs_allowed_months_and_no_self_pair() -> None:
    aoi = pd.Series(
        {
            "aoi_id": "E_x_01",
            "target_type": "reclamation",
            "t1_window": "2019-01-01/2019-12-31",
            "t2_window": "2019-01-01/2019-12-31",
        }
    )
    df = _scenes("E_x_01", [("D1", 5, "2019-01-10", 2.0), ("M1", 5, "2019-07-10", 2.0), ("D2", 5, "2019-03-10", 2.0)])
    specs = select_pairs(aoi, df, PairRules(allowed_months=[11, 12, 1, 2, 3, 4], max_pairs=4))
    prods = {s.t1_product for s in specs} | {s.t2_product for s in specs}
    assert "M1" not in prods
    assert all(s.t1_product != s.t2_product for s in specs)
    assert specs[0].t1_date == date(2019, 1, 10)
