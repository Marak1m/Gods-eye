"""M3: reviewed import validation, double-annotation agreement, candidate mining round trip."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pyogrio
import pytest
from shapely.affinity import translate
from shapely.geometry import box

from scd.imagery.preprocess import preprocess_pair
from scd.labels import mining, verification
from tests.fixtures.labels import reviewed_rows, write_reviewed_gpkg
from tests.fixtures.synth import make_pair

EPSG = 32645
X0, Y0 = 500000.0, 3300000.0


def _good_rows():
    return reviewed_rows(
        [
            (box(X0, Y0, X0 + 60, Y0 + 40), 2),
            (box(X0 + 200, Y0, X0 + 300, Y0 + 100), 4),
            (box(X0 + 400, Y0, X0 + 460, Y0 + 60), 1),
        ]
    )


def test_import_accepts_valid_and_rejects_bad_rows(conventions, tmp_path: Path) -> None:
    rows = _good_rows()
    rows += reviewed_rows([(box(X0, Y0 + 500, X0 + 60, Y0 + 540), 7)])  # bad class
    rows += reviewed_rows([(box(X0, Y0 + 600, X0 + 60, Y0 + 640), 2)], confidence="auto")  # not reviewed
    rows += reviewed_rows([(box(X0, Y0 + 700, X0 + 10, Y0 + 710), 2)])  # 100 m2 < 1800
    rows += reviewed_rows([(box(X0, Y0 + 800, X0 + 60, Y0 + 840), 2)], annotator="")
    gpkg = write_reviewed_gpkg(tmp_path / "inbox" / "p.gpkg", rows, EPSG)
    rep = verification.import_reviewed(gpkg, conventions, tmp_path / "reviewed" / "p.gpkg", expect_epsg=EPSG)
    assert rep.ok and rep.accepted == 3 and len(rep.rejected) == 4
    reasons = " ".join(r["reason"] for r in rep.rejected)
    assert "class_id 7" in reasons and "confidence" in reasons and "m2 <" in reasons and "annotator" in reasons
    assert rep.out_path is not None and rep.out_path.exists() and rep.out_path.with_suffix(".rejected.csv").exists()
    assert len(gpd.read_file(rep.out_path)) == 3


def test_import_rejects_geographic_crs_and_wrong_zone(conventions, tmp_path: Path) -> None:
    rows = reviewed_rows([(box(89.0, 29.0, 89.001, 29.001), 2)])
    gpkg = write_reviewed_gpkg(tmp_path / "geo.gpkg", rows, 4326)
    rep = verification.import_reviewed(gpkg, conventions, tmp_path / "out.gpkg")
    assert not rep.ok and any("projected" in e for e in rep.errors)
    gpkg2 = write_reviewed_gpkg(tmp_path / "zone.gpkg", _good_rows(), 32644)
    rep2 = verification.import_reviewed(gpkg2, conventions, tmp_path / "out2.gpkg", expect_epsg=32645)
    assert not rep2.ok and any("EPSG:32644" in e for e in rep2.errors)


def test_agreement(tmp_path: Path) -> None:
    a = write_reviewed_gpkg(tmp_path / "a.gpkg", _good_rows(), EPSG)
    assert all(v == 1.0 for v in verification.agreement(a, a).values())
    shifted = [dict(r, geometry=translate(r["geometry"], 20, 0)) for r in _good_rows()]
    b = write_reviewed_gpkg(tmp_path / "b.gpkg", shifted, EPSG)
    iou = verification.agreement(a, b)
    assert set(iou) == {1, 2, 4} and all(0 < v < 1 for v in iou.values())


class FakeSource:
    name = "fake"

    def fetch(self, aoi, cache: Path) -> Path:
        return cache

    def candidates(self, aoi, pair) -> gpd.GeoDataFrame:
        g = pair.grid()
        left, bottom, right, top = g.bounds
        return gpd.GeoDataFrame(
            {
                "class_id": [2, 4],
                "source": ["fake", "fake"],
                "source_ref": ["a", "b"],
                "t_event": ["2021", "2022"],
                "confidence": ["auto", "auto"],
                "notes": ["", ""],
            },
            geometry=[
                box(left + 50, bottom + 50, left + 120, bottom + 100),
                box(left + 200, bottom + 200, left + 320, bottom + 300),
            ],
            crs=f"EPSG:{g.epsg}",
        )


def test_mining_run_writes_layers(data_root: Path, tmp_path: Path) -> None:
    from scd.config import load

    cfg = load(
        Path(__file__).resolve().parents[1] / "configs" / "default.yaml", ["sr.method=bicubic", "cloud.method=scl"]
    )
    d1, d2, _ = make_pair(tmp_path / "scenes", "mine", seed=4, objects_t2=[])
    prod = preprocess_pair(d1, d2, None, cfg, cfg.pairs_dir / "A_mine_01" / "p01")
    gem = mining.make_sources(["gem"])[0]
    out = mining.run({"aoi_id": "A_mine_01"}, prod, [FakeSource(), gem], cfg.candidates_dir / "A_mine_01" / "p01.gpkg")
    layers = {name for name, _ in pyogrio.list_layers(out)}
    assert layers == {"pair", "building", "solar"}
    counts = mining.candidate_counts(cfg)
    assert counts.loc["A", "building"] == 1 and counts.loc["A", "solar"] == 1
    pend = verification.pending(cfg)
    assert len(pend) == 1 and pend.iloc[0]["pair_id"] == "p01"
    with pytest.raises(ValueError, match="unknown source"):
        mining.make_sources(["nope"])
