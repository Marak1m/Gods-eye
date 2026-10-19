"""M3/M4: synthetic object extraction, paste and remove (HANDOFF 5.11)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scd.imagery.preprocess import preprocess_pair
from scd.labels import synthetic
from scd.labels.rasterize import rasterize_pair
from tests.fixtures.labels import reviewed_rows, write_reviewed_gpkg
from tests.fixtures.synth import Obj, make_pair, objects_to_polygons


@pytest.fixture
def labelled(data_root: Path, tmp_path: Path, conventions):
    from scd.config import load

    cfg = load(
        Path(__file__).resolve().parents[1] / "configs" / "default.yaml", ["sr.method=bicubic", "cloud.method=scl"]
    )
    objs = [Obj(2, 8, 8, 12, 15), Obj(4, 32, 8, 40, 18), Obj(3, 16, 24, 17, 54), Obj(1, 40, 32, 47, 40)]
    d1, d2, g10 = make_pair(tmp_path / "scenes", "syn", seed=31, objects_t2=objs, shift=(0.0, 0.0))
    prod = preprocess_pair(d1, d2, None, cfg, tmp_path / "pair")
    gpkg = write_reviewed_gpkg(tmp_path / "rev.gpkg", reviewed_rows(objects_to_polygons(objs, g10)), prod.grid().epsg)
    lab = rasterize_pair(gpkg, prod, conventions, tmp_path / "labels")
    t1, t2, valid = prod.read()
    change, s1, s2 = lab.read()
    return cfg, t1, t2, valid, change, s1, s2


def test_extract_objects(labelled) -> None:
    cfg, t1, t2, valid, change, s1, s2 = labelled
    objs = synthetic.extract_objects(t1, t2, change, s1, s2, valid, 5.0, "chip", "A")
    classes = sorted(o.class_id for o in objs)
    assert classes == [1, 2, 3, 4]
    for o in objs:
        assert o.img_present.shape[0] == 6 and o.img_present.shape[1:] == o.mask.shape
        assert o.meta["present_date"] == "t2" and o.size_px > 0


def test_paste_and_library_roundtrip(labelled, tmp_path: Path) -> None:
    cfg, t1, t2, valid, change, s1, s2 = labelled
    objs = synthetic.extract_objects(t1, t2, change, s1, s2, valid, 5.0, "chip", "A")
    for k, o in enumerate(objs):
        o.save(tmp_path / "lib" / o.meta["class"] / f"chip_{k:02d}.npz")
    lib = synthetic.ObjectLibrary.load(tmp_path / "lib")
    assert lib.counts() == {"buildup": 1, "building": 1, "road": 1, "solar": 1} and len(lib) == 4

    host_t1 = t1.copy()
    host_t2 = t1.copy()  # identical host: every difference must come from the paste
    host_change = np.zeros_like(change)
    host_valid = valid.copy()
    host_valid[:, :20] = 0  # invalid strip: nothing may be pasted there
    rng = np.random.default_rng(0)
    picks = lib.sample_mix(rng, cfg.sampler.class_shares, 3)
    a, b, c, recs = synthetic.paste(host_t1, host_t2, host_change, host_valid, picks, np.random.default_rng(1), cfg)
    assert 1 <= len(recs) <= 3
    union = np.zeros(c.shape, dtype=bool)
    for r in recs:
        assert (c[r.mask] == r.class_id).all()
        assert not (host_valid == 0)[r.mask].any()
        changed_1 = (a != host_t1).any(axis=0)[r.mask].any()
        changed_2 = (b != host_t2).any(axis=0)[r.mask].any()
        assert changed_1 != changed_2  # exactly one date carries the object
        union |= r.mask
    assert (c[~union] == 0).all()
    from scipy import ndimage

    outside = ~ndimage.binary_dilation(union, iterations=3)  # shadow strips may touch 1-2 px outside
    assert (a[:, outside] == host_t1[:, outside]).all() and (b[:, outside] == host_t2[:, outside]).all()
    # deterministic given the rng seed
    a2, b2, c2, _ = synthetic.paste(host_t1, host_t2, host_change, host_valid, picks, np.random.default_rng(1), cfg)
    assert np.array_equal(a, a2) and np.array_equal(b, b2) and np.array_equal(c, c2)


def test_remove(labelled) -> None:
    cfg, t1, t2, valid, change, s1, s2 = labelled
    mask = change == 2
    a, b, c = synthetic.remove(
        t2, t2.copy(), mask, np.random.default_rng(3), class_id=2, host_valid=valid, host_change=change
    )
    assert (c[mask] == 2).all() and (c[~mask] == 0).all()
    changed_1 = (a != t2).any(axis=0)[mask].any()
    changed_2 = (b != t2).any(axis=0)[mask].any()
    assert changed_1 != changed_2
    assert (a[:, ~mask] == t2[:, ~mask]).all() and (b[:, ~mask] == t2[:, ~mask]).all()
