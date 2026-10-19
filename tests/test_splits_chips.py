"""M4: splits by AOI within stratum; end-to-end chip building with manifest, auto negatives, dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from scd.constants import IGNORE, TileType
from scd.data import chips
from scd.data import manifest as man_mod
from scd.data.dataset import DrawRequest, PairChipDataset, collate, full_chip_requests
from scd.data.splits import assign_splits
from scd.imagery import registry
from scd.imagery.preprocess import preprocess_pair
from scd.labels.rasterize import rasterize_pair
from tests.fixtures.labels import reviewed_rows, write_reviewed_gpkg
from tests.fixtures.synth import Obj, make_pair, objects_to_polygons

REPO = Path(__file__).resolve().parents[1]


def _registry_rows() -> list[dict]:
    rows = []
    for s, lat, lon in (("A", 30.0, 90.0), ("B", 40.0, 100.0)):
        for i in range(10):
            rows.append(
                {
                    "aoi_id": f"{s}_site_{i:02d}",
                    "stratum": s,
                    "target_type": "compound",
                    "role": "val" if i == 0 else "train",
                    "centre_lat": lat + 0.5 * i,
                    "centre_lon": lon,
                    "size_km": 10.0,
                    "lead_source": "manual",
                    "lead_ref": "",
                    "event_year": None,
                    "t1_window": "",
                    "t2_window": "",
                    "notes": "",
                }
            )
    rows.append({**rows[0], "aoi_id": "A_target_01", "role": "targets", "centre_lat": 20.0})
    rows.append({**rows[0], "aoi_id": "A_stage2_01", "role": "stage2", "centre_lat": 21.0})
    return rows


def test_assign_splits() -> None:
    reg = registry._finish(pd.DataFrame(_registry_rows()))
    sp = assign_splits(reg, seed=42, val_fraction=0.2)
    assert sp["A_target_01"] == "targets" and sp["A_stage2_01"] == "stage2"
    for s in ("A", "B"):
        ids = [a for a in sp if a.startswith(f"{s}_site")]
        assert sum(sp[a] == "val" for a in ids) == 2 and sp[f"{s}_site_00"] == "val"
    assert assign_splits(reg, seed=42) == sp and assign_splits(reg, seed=43) != sp


@pytest.fixture
def smoke_like(data_root: Path, tmp_path: Path, conventions):
    """Two preprocessed pairs (one labelled, one auto negative) plus a registry under data_root."""
    from scd.config import load

    cfg = load(
        REPO / "configs" / "default.yaml",
        ["sr.method=bicubic", "cloud.method=scl", "grid.chip_px=64", "grid.tile_px=32", "chips.stride_px=48"],
    )
    reg = registry._finish(
        pd.DataFrame(
            [
                {
                    "aoi_id": "A_lab_01",
                    "stratum": "A",
                    "target_type": "compound",
                    "role": "train",
                    "centre_lat": 29.35,
                    "centre_lon": 89.31,
                    "size_km": 1.0,
                    "lead_source": "manual",
                    "lead_ref": "",
                    "event_year": None,
                    "t1_window": "",
                    "t2_window": "",
                    "notes": "",
                },
                {
                    "aoi_id": "C_neg_01",
                    "stratum": "C",
                    "target_type": "negative",
                    "role": "train",
                    "centre_lat": 24.07,
                    "centre_lon": 89.05,
                    "size_km": 1.0,
                    "lead_source": "manual",
                    "lead_ref": "",
                    "event_year": None,
                    "t1_window": "",
                    "t2_window": "",
                    "notes": "",
                },
            ]
        )
    )
    registry.save(reg, registry.registry_path(cfg))
    objs = [Obj(2, 8, 8, 12, 15), Obj(4, 32, 8, 40, 18), Obj(1, 40, 32, 47, 40)]
    d1, d2, g10 = make_pair(tmp_path / "s1", "lab", seed=41, objects_t2=objs, shift=(0.0, 0.0))
    prod = preprocess_pair(
        d1,
        d2,
        {"aoi_id": "A_lab_01"},
        cfg,
        cfg.pairs_dir / "A_lab_01" / "p01",
        pair_row={"tile_type_hint": "CHANGE_POS"},
    )
    gpkg = write_reviewed_gpkg(
        cfg.reviewed_dir / "A_lab_01" / "p01.gpkg", reviewed_rows(objects_to_polygons(objs, g10)), prod.grid().epsg
    )
    rasterize_pair(gpkg, prod, conventions, cfg.labels_dir / "A_lab_01" / "p01")
    n1, n2, _ = make_pair(tmp_path / "s2", "neg", seed=42, objects_t2=[], gain_t2=1.1, shift=(0.6, 0.3))
    preprocess_pair(
        n1,
        n2,
        {"aoi_id": "C_neg_01"},
        cfg,
        cfg.pairs_dir / "C_neg_01" / "p01",
        pair_row={"tile_type_hint": TileType.NOCHANGE_HARD.value},
    )
    return cfg


def test_build_chips_manifest_and_dataset(smoke_like) -> None:
    cfg = smoke_like
    out = chips.build_chips(cfg, version=1)
    man_path = out / "manifest.parquet"
    df = man_mod.load(man_path)
    assert man_mod.read_hash(man_path) == cfg.hash()
    # 128 px pair, 64 px chips on a 48 stride -> positions 0, 48, 64 -> 3x3 = 9 windows per pair
    assert len(df) == 18 and set(df["aoi_id"]) == {"A_lab_01", "C_neg_01"}
    lab = df[df["aoi_id"] == "A_lab_01"]
    neg = df[df["aoi_id"] == "C_neg_01"]
    assert (lab["label_confidence"] == "corrected").all() and (neg["label_confidence"] == "auto").all()
    assert set(lab["tile_type"]) >= {"CHANGE_POS", "NOCHANGE_PLAIN"}
    assert set(neg["tile_type"]) == {"NOCHANGE_HARD"}
    assert (df["split"] == "train").all()
    for cid in lab["chip_id"]:
        for p in man_mod.chip_paths(out, cid).values():
            assert p.exists()
    # the auto negative carries the image-change gate (some 255) and no positive labels
    ds = PairChipDataset(out, df, cfg, tile_px=32, augment=False)
    ch = ds.read_chip(str(neg.iloc[0]["chip_id"]))[3]
    assert set(np.unique(ch)) <= {0, IGNORE} and (ch == IGNORE).any()
    weights = man_mod.class_weights(df, 0.5)
    assert (
        set(weights) == {"buildup", "building", "road", "solar"} and abs(np.mean(list(weights.values())) - 1.0) < 1e-6
    )
    summary = man_mod.summarise(df)
    assert "chips" in summary.columns
    # idempotent: a second call with the same config hash returns immediately
    assert chips.build_chips(cfg, version=1) == out

    # dataset draws: plain, identical, near-identical, synthetic-without-objects, swap
    pos_id = str(lab[lab["tile_type"] == "CHANGE_POS"].iloc[0]["chip_id"])
    plain = ds[DrawRequest(pos_id, "CHANGE_POS", seed=1, centre_class=2)]
    assert (
        plain["x1"].shape == (6, 32, 32) and plain["change"].dtype == torch.int64 and plain["valid"].dtype == torch.bool
    )
    assert plain["x1"].max() <= cfg.bands.clip_max and plain["weight"].item() == 1.0
    ident = ds[DrawRequest(pos_id, "IDENTICAL", seed=2)]
    assert torch.equal(ident["x1"], ident["x2"]) and set(torch.unique(ident["change"]).tolist()) <= {0, IGNORE}
    near = ds[DrawRequest(pos_id, "NEAR_IDENTICAL", seed=3)]
    assert not torch.equal(near["x1"], near["x2"]) and (near["change"][near["valid"]] == 0).all()
    neg_id = str(neg.iloc[0]["chip_id"])
    auto = ds[DrawRequest(neg_id, "NOCHANGE_HARD", seed=4)]
    assert auto["weight"].item() == cfg.sampler.auto_label_weight
    aug = PairChipDataset(out, df, cfg, tile_px=32, augment=True)
    a1 = aug[DrawRequest(pos_id, "CHANGE_POS", seed=5, swap=True, centre_class=2)]
    a2 = aug[DrawRequest(pos_id, "CHANGE_POS", seed=5, swap=True, centre_class=2)]
    assert torch.equal(a1["x1"], a2["x1"]) and a1["meta"]["swap"] is True  # deterministic given seed
    batch = collate([plain, ident])
    assert batch["x1"].shape == (2, 6, 32, 32) and len(batch["meta"]) == 2
    full = PairChipDataset(out, df, cfg, tile_px=cfg.grid.chip_px, augment=False)
    reqs = full_chip_requests(df, "train")
    assert len(reqs) == len(df) and full[reqs[0]]["x1"].shape == (6, 64, 64)
