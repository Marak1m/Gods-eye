"""M3: rasterisation rules (HANDOFF 9 `test_rasterize`): priority, m2 filters, fix_t1, validity -> 255."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from shapely.geometry import LineString, box

from scd.constants import IGNORE
from scd.imagery.preprocess import preprocess_pair
from scd.labels.fix_t1 import fix_t1
from scd.labels.rasterize import rasterize_pair
from tests.fixtures.labels import reviewed_rows, write_reviewed_gpkg
from tests.fixtures.synth import Obj, make_pair, objects_to_polygons


@pytest.fixture
def pair(data_root: Path, tmp_path: Path):
    from scd.config import load

    cfg = load(
        Path(__file__).resolve().parents[1] / "configs" / "default.yaml", ["sr.method=bicubic", "cloud.method=scl"]
    )
    objs = [Obj(2, 8, 8, 12, 15), Obj(4, 32, 8, 40, 18), Obj(3, 16, 24, 17, 54), Obj(1, 40, 32, 47, 40)]
    d1, d2, g10 = make_pair(
        tmp_path / "scenes", "ras", seed=21, objects_t2=objs, shift=(0.0, 0.0), clouds_t2=[(56, 56, 64, 64)]
    )
    prod = preprocess_pair(d1, d2, None, cfg, tmp_path / "pair")
    return cfg, prod, g10, objs


def test_rasterize_rules(pair, conventions, tmp_path: Path) -> None:
    cfg, prod, g10, objs = pair
    grid = prod.grid()
    t = g10.transform
    polys = objects_to_polygons([o for o in objs if o.class_id != 3], g10)
    rows = reviewed_rows(polys)  # appear at T2
    # road as a 300 m centreline (access width 6 m)
    x0, y0 = t @ (24, 16.5)
    x1, _ = t @ (54, 16.5)
    rows += reviewed_rows([(LineString([(x0, y0), (x1, y0)]), 3)])
    # building inside a buildup polygon -> building must win (priority building > buildup)
    bx0, by0 = t @ (40, 40)
    bx1, by1 = t @ (47, 47)
    rows += reviewed_rows([(box(bx0 + 10, by1 + 10, bx0 + 60, by1 + 50), 2)])
    # undersized building (10 x 10 m) and short road (50 m) -> dropped
    rows += reviewed_rows(
        [(box(x0, y0 + 100, x0 + 10, y0 + 110), 2), (LineString([(x0, y0 + 200), (x0 + 50, y0 + 200)]), 3)]
    )
    # removal: present at T1 only -> change (removals_labeled: true in the smoke guide)
    rx0, ry0 = t @ (2, 50)
    rows += reviewed_rows([(box(rx0, ry0 - 60, rx0 + 60, ry0), 2)], present_t1=True, present_t2=False)
    # stable object at both dates -> not change, but in both semantic masks
    sx0, sy0 = t @ (50, 2)
    rows += reviewed_rows([(box(sx0, sy0 - 50, sx0 + 60, sy0), 2)], present_t1=True, present_t2=True)
    gpkg = write_reviewed_gpkg(tmp_path / "reviewed" / "p.gpkg", rows, grid.epsg)

    lab = rasterize_pair(gpkg, prod, conventions, tmp_path / "labels")
    change, sem1, sem2 = lab.read()
    assert set(np.unique(change)) <= {0, 1, 2, 3, 4, IGNORE}
    for cid in (1, 2, 3, 4):
        assert (change == cid).sum() > 0, cid
    # priority: the small building inside the buildup area is burned as 2
    c, r = (~grid.transform) @ (bx0 + 35, by1 + 30)
    assert change[int(r), int(c)] == 2
    # road buffered to ~6 m: 1-2 px wide at 5 m, ~60 px long
    road = change == 3
    assert 50 <= road.sum() <= 130
    # undersized dropped
    import json

    report = json.loads(lab.report.read_text())
    assert len(report["dropped"]) == 2
    # removal counted as change, stable object not
    rc, rr = (~grid.transform) @ (rx0 + 30, ry0 - 30)
    assert change[int(rr), int(rc)] == 2 and sem1[int(rr), int(rc)] == 2 and sem2[int(rr), int(rc)] == 0
    sc, sr = (~grid.transform) @ (sx0 + 30, sy0 - 25)
    assert change[int(sr), int(sc)] == 0 and sem1[int(sr), int(sc)] == 2 and sem2[int(sr), int(sc)] == 2
    # fix_t1: background at T1 where positive at T2 -> 255 at T1
    appear = (change > 0) & (change != IGNORE) & (sem2 > 0) & (sem2 != IGNORE) & (sem1 != 2)
    assert (sem1[appear] == IGNORE).all()
    # validity -> 255 in all three
    _, _, valid = prod.read()
    inv = valid == 0
    assert inv.any()
    assert (change[inv] == IGNORE).all() and (sem1[inv] == IGNORE).all() and (sem2[inv] == IGNORE).all()


def test_fix_t1_rule() -> None:
    s1 = np.array([[0, 0, 2], [1, 0, 255]], dtype=np.uint8)
    s2 = np.array([[0, 3, 2], [1, 255, 4]], dtype=np.uint8)
    out = fix_t1(s1, s2)
    assert out.tolist() == [[0, 255, 2], [1, 0, 255]]


def test_rasterize_without_presence_columns(pair, conventions, tmp_path: Path) -> None:
    cfg, prod, g10, objs = pair
    polys = objects_to_polygons([o for o in objs if o.class_id in (1, 2, 4)], g10)
    rows = reviewed_rows(polys)
    for r in rows:
        r.pop("present_t1")
        r.pop("present_t2")
    gpkg = write_reviewed_gpkg(tmp_path / "reviewed" / "np.gpkg", rows, prod.grid().epsg)
    lab = rasterize_pair(gpkg, prod, conventions, tmp_path / "labels_np")
    change, sem1, sem2 = lab.read()
    assert (change > 0).any()
    assert set(np.unique(sem1)) == {IGNORE}  # T1 state unknown
    assert ((sem2 == change) | (change == IGNORE)).all()
