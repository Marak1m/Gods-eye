"""M7: two inference runs are byte-identical (HANDOFF 9 `test_determinism`, CPU form per DEVIATIONS C17)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

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


def test_determinism(infer_cfg, tmp_path: Path) -> None:
    d1, d2, _ = make_pair(tmp_path / "scenes", "det", seed=9)
    torch.manual_seed(0)
    net = build_model(infer_cfg).eval()
    r1 = pipeline.run(d1, d2, tmp_path / "run1", infer_cfg, None, net=net)
    r2 = pipeline.run(d1, d2, tmp_path / "run2", infer_cfg, None, net=net)
    for name in ("change.tif", "valid.tif", "change.shp", "change.dbf", "change.shx"):
        assert (r1.out_dir / name).read_bytes() == (r2.out_dir / name).read_bytes(), name
    rep1 = json.loads(r1.report.read_text())
    rep2 = json.loads(r2.report.read_text())
    for key in ("class_px", "n_components", "coreg", "grid", "tags"):
        assert rep1[key] == rep2[key], key


@pytest.mark.gpu
def test_determinism_gpu(infer_cfg, tmp_path: Path) -> None:
    """Same check on CUDA with torch.use_deterministic_algorithms(True)."""
    d1, d2, _ = make_pair(tmp_path / "scenes", "detg", seed=9)
    torch.manual_seed(0)
    net = build_model(infer_cfg).eval()
    r1 = pipeline.run(d1, d2, tmp_path / "g1", infer_cfg, None, net=net, device="cuda")
    r2 = pipeline.run(d1, d2, tmp_path / "g2", infer_cfg, None, net=net, device="cuda")
    assert r1.change_tif.read_bytes() == r2.change_tif.read_bytes()
