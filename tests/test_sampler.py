"""M4: sampler shares within +-1.5 pts over 100k draws; deterministic with seed (HANDOFF 9)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from scd.constants import CLASS_IDS, STRATA, TileType
from scd.data import manifest as man_mod
from scd.data.sampler import TwoStageSampler, report


def _manifest(strata=STRATA, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for s in strata:
        for i in range(24):
            px = {k: int(rng.integers(60, 5000)) if rng.random() < 0.6 else 0 for k in range(1, 5)}
            if max(px.values()) == 0:
                px[1 + i % 4] = 500
            rows.append(("CHANGE_POS", px))
        rows += [("NOCHANGE_HARD", {k: 0 for k in range(1, 5)})] * 10
        rows += [("NOCHANGE_PLAIN", {k: 0 for k in range(1, 5)})] * 5
        for j, (tt, px) in enumerate(rows[-39:]):
            rows[-39 + j] = (s, tt, px)
    out = []
    for n, (s, tt, px) in enumerate(rows):
        out.append(
            {
                "chip_id": f"{s}_x_01_p01_r0_c{n}",
                "aoi_id": f"{s}_x_01",
                "pair_id": "p01",
                "stratum": s,
                "target_type": "compound",
                "t1_date": "2019-07-10",
                "t2_date": "2023-07-14",
                "same_orbit": True,
                "season_match": True,
                "tile_type": tt,
                "classes_present": [k for k, v in px.items() if v > 0],
                **{f"change_px_{k}": v for k, v in px.items()},
                "valid_fraction": 0.95,
                "label_confidence": "corrected",
                "split": "train",
                "row": 0,
                "col": 0,
                "window": json.dumps([0, 0, 1024, 1024]),
            }
        )
    return man_mod.validate(pd.DataFrame(out, columns=list(man_mod.MANIFEST_COLUMNS)))


@pytest.mark.slow
def test_sampler_shares_within_tolerance(cfg) -> None:
    df = _manifest()
    n = 100_000
    reqs = list(iter(TwoStageSampler(df, cfg, n, seed=1)))
    assert len(reqs) == n
    strat = df.set_index("chip_id")["stratum"]
    s = pd.Series([strat[r.chip_id] for r in reqs])
    tt = pd.Series([r.tile_type for r in reqs])
    for st, share in cfg.strata.shares.items():
        assert abs(float((s == st).mean()) - share) < 0.015, st
    for st in STRATA:
        sel = tt[s == st]
        for t, share in cfg.sampler.shares_for(st).items():
            assert abs(float((sel == t).mean()) - share) < 0.015, (st, t)
    pos = [r for r in reqs if r.tile_type == TileType.CHANGE_POS.value]
    cc = pd.Series([r.centre_class for r in pos])
    for cname, share in cfg.sampler.class_shares.items():
        assert abs(float((cc == CLASS_IDS[cname]).mean()) - share) < 0.015, cname
    assert all(r.synthetic_objs == () for r in reqs if r.tile_type == "SYNTHETIC")  # no library given


def test_sampler_deterministic_and_epoch_dependent(cfg) -> None:
    df = _manifest()
    a = list(iter(TwoStageSampler(df, cfg, 500, seed=7, epoch=0)))
    b = list(iter(TwoStageSampler(df, cfg, 500, seed=7, epoch=0)))
    assert a == b
    c = TwoStageSampler(df, cfg, 500, seed=7, epoch=0)
    c.set_epoch(1)
    assert list(iter(c)) != a


def test_sampler_density_weighting(cfg) -> None:
    df = _manifest()
    reqs = list(iter(TwoStageSampler(df, cfg, 30_000, seed=3)))
    pos = [r for r in reqs if r.tile_type == "CHANGE_POS" and r.centre_class == 1]
    counts = pd.Series([r.chip_id for r in pos]).value_counts()
    px = df.set_index("chip_id")["change_px_1"]
    big = px[px > 2500].index
    small = px[(px > 0) & (px < 600)].index
    assert counts.reindex(big).fillna(0).mean() > counts.reindex(small).fillna(0).mean()


def test_sampler_fallbacks_when_strata_missing(cfg) -> None:
    df = _manifest(strata=("A", "B"))
    reqs = list(iter(TwoStageSampler(df, cfg, 2000, seed=0)))
    chips = set(df["chip_id"])
    assert all(r.chip_id in chips for r in reqs)
    assert {r.tile_type for r in reqs} >= {"CHANGE_POS", "NOCHANGE_HARD", "IDENTICAL"}


def test_report_table(cfg) -> None:
    rep = report(cfg, _manifest(), n=5000, seed=0)
    assert set(rep["kind"]) == {"stratum", "tile_type", "class|CHANGE_POS"}
    assert rep["diff_pts"].abs().max() < 5.0
