"""M7: run(A, B) == run(B, A) on raster and vectors (plan doc 10; DEVIATIONS C16, U2)."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import torch

from scd.imagery.preprocess import preprocess_pair
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


def _read(p: Path) -> np.ndarray:
    with rasterio.open(p) as ds:
        return ds.read(1)


def _same_vectors(a: Path, b: Path) -> bool:
    ga = gpd.read_file(a).sort_values(["class", "area_m2"]).reset_index(drop=True)
    gb = gpd.read_file(b).sort_values(["class", "area_m2"]).reset_index(drop=True)
    if len(ga) != len(gb) or not ga["class"].equals(gb["class"]):
        return False
    return all(x.equals(y) for x, y in zip(ga.geometry, gb.geometry))


def test_pipeline_swap(infer_cfg, tmp_path: Path) -> None:
    d1, d2, _ = make_pair(tmp_path / "scenes", "swp", seed=13, clouds_t2=[(40, 40, 50, 52)])
    torch.manual_seed(0)
    net = build_model(infer_cfg).eval()
    r_ab = pipeline.run(d1, d2, tmp_path / "ab", infer_cfg, None, net=net)
    r_ba = pipeline.run(d2, d1, tmp_path / "ba", infer_cfg, None, net=net)
    assert np.array_equal(_read(r_ab.change_tif), _read(r_ba.change_tif))
    assert np.array_equal(_read(r_ab.valid_tif), _read(r_ba.valid_tif))
    assert _same_vectors(r_ab.shapefile, r_ba.shapefile)
    assert r_ab.class_px == r_ba.class_px and not r_ab.identity_short_circuit


def test_temporal_tta_is_redundant(infer_cfg, data_root: Path, tmp_path: Path) -> None:
    """With exact symmetry, tta_temporal=True yields the identical mask (DEVIATIONS U2)."""
    from scd.config import load

    cfg_t = load(REPO / "configs" / "default.yaml", [*INFER_OVERRIDES, "infer.tta_temporal=true"])
    d1, d2, _ = make_pair(tmp_path / "scenes", "tta", seed=14)
    torch.manual_seed(0)
    net = build_model(infer_cfg).eval()
    r0 = pipeline.run(d1, d2, tmp_path / "no_temporal", infer_cfg, None, net=net)
    r1 = pipeline.run(d1, d2, tmp_path / "temporal", cfg_t, None, net=net)
    assert np.array_equal(_read(r0.change_tif), _read(r1.change_tif))


def test_invariance_checks_helper(infer_cfg, tmp_path: Path) -> None:
    d1, d2, _ = make_pair(tmp_path / "scenes", "inv", seed=15)
    prod = preprocess_pair(d1, d2, None, infer_cfg, tmp_path / "pair")
    torch.manual_seed(0)
    checks = pipeline.invariance_checks(infer_cfg, None, prod.dir, tmp_path / "checks")
    assert checks == {"swap_raster_identical": True, "swap_vectors_identical": True, "identity_all_zero": True}
