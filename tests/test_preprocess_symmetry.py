"""M2: preprocess_pair(t1, t2) vs (t2, t1): swapped pixel arrays identical, valid.tif identical.

The CPU test runs on synthetic SimpleScenes with the bicubic + SCL path (DEVIATIONS C17). The
real-pair SEN2SR/OmniCloudMask test is gpu-marked and needs a cached pair.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from scd.imagery.preprocess import PairProduct, preprocess_pair
from tests.fixtures.synth import Obj, default_objects, make_pair

SMOKE_OVERRIDES = ["sr.method=bicubic", "cloud.method=scl"]


@pytest.fixture
def cpu_cfg(data_root: Path):
    from scd.config import load

    return load(Path(__file__).resolve().parents[1] / "configs" / "default.yaml", SMOKE_OVERRIDES)


def _run_both(cfg, d1: Path, d2: Path, tmp: Path) -> tuple[PairProduct, PairProduct]:
    p12 = preprocess_pair(d1, d2, None, cfg, tmp / "p12")
    p21 = preprocess_pair(d2, d1, None, cfg, tmp / "p21")
    return p12, p21


def test_preprocess_symmetry_synthetic(cpu_cfg, tmp_path: Path) -> None:
    d1, d2, _ = make_pair(tmp_path / "scenes", "sym", seed=11, clouds_t2=[(40, 40, 52, 56)])
    p12, p21 = _run_both(cpu_cfg, d1, d2, tmp_path)
    a1, a2, v = p12.read()
    b1, b2, w = p21.read()
    assert np.array_equal(a1, b2) and np.array_equal(a2, b1)  # swapped names, identical pixels
    assert np.array_equal(v, w)
    assert p12.grid() == p21.grid()
    m12, m21 = p12.meta(), p21.meta()
    assert m12["coreg"]["applied"] and m12["coreg"] == m21["coreg"]  # canonical-order metadata equal
    assert m12["t1"]["product_id"] == m21["t2"]["product_id"]
    assert a1.dtype == np.uint16 and a1.shape[0] == 6 and v.dtype == np.uint8
    assert 0.5 < v.mean() < 1.0  # clouds removed some validity
    assert m12["grid"]["epsg"] == 32645 and abs(m12["grid"]["transform"][0] - 5.0) < 1e-9


def test_preprocess_identity(cpu_cfg, tmp_path: Path) -> None:
    d1, _, _ = make_pair(tmp_path / "scenes", "idt", seed=3, identical=True)
    p = preprocess_pair(d1, d1, None, cfg=cpu_cfg, out_dir=tmp_path / "pid")
    t1, t2, v = p.read()
    assert np.array_equal(t1, t2)
    assert p.meta()["identical_inputs"] is True and p.meta()["coreg"]["applied"] is False
    assert v.min() == 1  # no clouds, no holes


def test_preprocess_coreg_reduces_offset_and_masks_strip(cpu_cfg, tmp_path: Path) -> None:
    d1, d2, _ = make_pair(tmp_path / "scenes", "shift", seed=5, shift=(2.0, 0.0), objects_t2=[])
    p = preprocess_pair(d1, d2, None, cpu_cfg, tmp_path / "ps")
    meta = p.meta()
    assert meta["coreg"]["applied"]
    # 2 px at 10 m = 4 px at 5 m
    assert abs(abs(meta["coreg"]["dx"]) - 4.0) < 0.6 and abs(meta["coreg"]["dy"]) < 0.6
    _, _, v = p.read()
    assert v[:, :2].sum() == 0 or v[:, -2:].sum() == 0  # exposed strip invalid on one side


def test_preprocess_meta_tracks_config_hash(cpu_cfg, tmp_path: Path) -> None:
    d1, d2, _ = make_pair(tmp_path / "scenes", "hash", seed=7, objects_t2=[Obj(2, 10, 10, 15, 18)])
    p = preprocess_pair(d1, d2, None, cpu_cfg, tmp_path / "ph")
    assert p.meta()["config_hash"] == cpu_cfg.hash()
    assert p.quicklook.exists()
    assert len(default_objects()) == 4


@pytest.mark.gpu
def test_preprocess_symmetry_real_pair(cfg, tmp_path: Path) -> None:
    """Real SAFE pair through SEN2SR + OmniCloudMask; needs SCD_TEST_PAIR=<zip1>,<zip2>."""
    pair = os.environ.get("SCD_TEST_PAIR")
    if not pair:
        pytest.skip("set SCD_TEST_PAIR=<t1.zip>,<t2.zip> to run on a cached real pair")
    z1, z2 = (Path(p) for p in pair.split(","))
    p12, p21 = _run_both(cfg, z1, z2, tmp_path)
    a1, a2, v = p12.read()
    b1, b2, w = p21.read()
    assert np.array_equal(a1, b2) and np.array_equal(a2, b1) and np.array_equal(v, w)
