"""M7: pipeline.run(A, A) -> all-zero mask, shapefile with 0 features (HANDOFF 9 `test_identity_guard`)."""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio

from scd.infer import pipeline
from scd.model.net import build_model
from tests.fixtures.synth import make_pair

REPO = Path(__file__).resolve().parents[1]
INFER_OVERRIDES = [
    "sr.method=bicubic",
    "cloud.method=scl",
    "model.encoder_init=random",
    "infer.window_px=128",
    "infer.stride_px=96",
]


@pytest.fixture
def infer_cfg(data_root: Path):
    from scd.config import load

    return load(REPO / "configs" / "default.yaml", INFER_OVERRIDES)


def test_identity_guard_on_scene_dirs(infer_cfg, tmp_path: Path) -> None:
    d1, _, _ = make_pair(tmp_path / "scenes", "idg", seed=1, identical=True)
    net = build_model(infer_cfg).eval()
    res = pipeline.run(d1, d1, tmp_path / "out", infer_cfg, None, net=net)
    with rasterio.open(res.change_tif) as ds:
        arr = ds.read(1)
    assert arr.max() == 0 and arr.shape == (128, 128)
    assert len(gpd.read_file(res.shapefile)) == 0
    assert res.identity_short_circuit and res.n_change_px == 0
    report = json.loads(res.report.read_text())
    assert report["identity_short_circuit"] is True and report["n_components"] == 0
    assert res.valid_tif.exists() and res.quicklook.exists()


def test_identity_guard_on_delivered_geotiffs(infer_cfg, tmp_path: Path) -> None:
    """A preprocessed t1.tif fed twice as a 6-band GeoTIFF (the organisers' delivery format)."""
    d1, d2, _ = make_pair(tmp_path / "scenes", "idg2", seed=2)
    net = build_model(infer_cfg).eval()
    first = pipeline.run(d1, d2, tmp_path / "first", infer_cfg, None, net=net)
    t1_tif = first.out_dir / "pre" / "t1.tif"
    res = pipeline.run(t1_tif, t1_tif, tmp_path / "out", infer_cfg, None, net=net)
    with rasterio.open(res.change_tif) as ds:
        assert ds.read(1).max() == 0
    assert len(gpd.read_file(res.shapefile)) == 0
    # and a genuinely different pair does run the network
    assert not first.identity_short_circuit
    with rasterio.open(first.change_tif) as ds:
        vals = set(np.unique(ds.read(1)).tolist())
    assert vals <= {0, 1, 2, 3, 4}
