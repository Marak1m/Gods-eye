"""Shared fixtures. All tests run on CPU with synthetic data unless marked gpu/network."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """gpu tests self-skip without CUDA; network tests self-skip unless SCD_RUN_NETWORK=1."""
    try:
        import torch

        has_cuda = torch.cuda.is_available()
    except Exception:  # pragma: no cover
        has_cuda = False
    run_network = os.environ.get("SCD_RUN_NETWORK") == "1"
    for item in items:
        if "gpu" in item.keywords and not has_cuda:
            item.add_marker(pytest.mark.skip(reason="needs a CUDA device / real data (HANDOFF 9)"))
        if "network" in item.keywords and not run_network:
            item.add_marker(pytest.mark.skip(reason="needs network/credentials; set SCD_RUN_NETWORK=1"))


@pytest.fixture
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "scd_data"
    root.mkdir()
    monkeypatch.setenv("SCD_DATA", str(root))
    monkeypatch.setenv("SCD_RUNS", str(root / "runs"))
    return root


@pytest.fixture
def cfg(data_root: Path):
    from scd.config import load

    return load(CONFIGS / "default.yaml")


@pytest.fixture
def smoke_cfg(data_root: Path):
    from scd.config import load

    return load(CONFIGS / "smoke.yaml")


@pytest.fixture
def conventions():
    from scd.labels import conventions as conv

    return conv.load(REPO / "tests" / "fixtures" / "labeling_guide_smoke.yaml")
