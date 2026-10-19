#!/usr/bin/env python
"""Build the smoke dataset under a data root (DEVIATIONS C12/C13).

Three AOIs, four pairs of synthetic SimpleScenes, a registry, scenes/pairs tables and reviewed
labels:
  A_smoke_01/p01  change pair (train)        A_smoke_01/p02  identical pair (train)
  B_smoke_02/p01  change pair + clouds (val) C_smoke_03/p01  NOCHANGE_HARD auto negative (train)
Run: python tests/fixtures/make_smoke_data.py --data-root /tmp/ps10-scd-smoke
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import typer
from affine import Affine
from pyproj import Transformer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scd.constants import TileType  # noqa: E402
from scd.geo.grid import Grid, utm_epsg  # noqa: E402
from scd.imagery import registry  # noqa: E402
from scd.imagery.scenes import PAIRS_COLUMNS, SCENES_COLUMNS  # noqa: E402
from tests.fixtures.labels import reviewed_rows, write_reviewed_gpkg  # noqa: E402
from tests.fixtures.synth import Obj, default_objects, make_pair, objects_to_polygons  # noqa: E402

app = typer.Typer(add_completion=False)

AOIS = [
    # aoi_id, stratum, target_type, role, lat, lon
    ("A_smoke_01", "A", "compound", "train", 29.3517, 89.3110),
    ("B_smoke_02", "B", "solar", "val", 40.3962, 99.7771),
    ("C_smoke_03", "C", "negative", "train", 24.0661, 89.0468),
]


def _grid10(lat: float, lon: float, size10: int) -> Grid:
    epsg = utm_epsg(lon, lat)
    x, y = Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    x0 = round(x / 10.0) * 10.0 - size10 * 5.0
    y0 = round(y / 10.0) * 10.0 + size10 * 5.0
    return Grid(epsg, Affine(10.0, 0.0, x0, 0.0, -10.0, y0), size10, size10)


@app.command()
def main(
    data_root: Path = typer.Option(..., "--data-root", help="SCD_DATA for the smoke run."),
    size10: int = typer.Option(192, "--size10", help="Scene size in 10 m pixels (192 -> 384 px at 5 m)."),
) -> None:
    data_root = Path(data_root)
    raw = data_root / "raw" / "simple"
    reg_rows = []
    for aoi_id, stratum, ttype, role, lat, lon in AOIS:
        reg_rows.append(
            {
                "aoi_id": aoi_id,
                "stratum": stratum,
                "target_type": ttype,
                "role": role,
                "centre_lat": lat,
                "centre_lon": lon,
                "size_km": size10 * 10 / 1000.0,
                "lead_source": "manual",
                "lead_ref": "smoke",
                "event_year": 2021,
                "t1_window": "2019-05-01/2019-10-31",
                "t2_window": "2023-05-01/2023-10-31",
                "notes": "smoke fixture",
            }
        )
    reg = registry._finish(pd.DataFrame(reg_rows))
    errors = registry.validate(reg)
    if errors:
        raise RuntimeError("\n".join(errors))
    registry.save(reg, registry.registry_path(type("C", (), {"registry_dir": data_root / "registry"})()))

    scenes: list[dict] = []
    pairs: list[dict] = []

    def add_scene(aoi_id: str, d: Path, date: str, orbit: int = 5) -> None:
        scenes.append(
            {
                "aoi_id": aoi_id,
                "product_id": d.name,
                "tile": "T45RVL",
                "relative_orbit": orbit,
                "date": date,
                "cloud_cover": 1.0,
                "baseline": 5.0,
                "local_path": str(d),
                "status": "downloaded",
            }
        )

    def add_pair(aoi_id: str, pair_id: str, d1: Path, d2: Path, hint: str) -> None:
        pairs.append(
            {
                "aoi_id": aoi_id,
                "pair_id": pair_id,
                "t1_product": d1.name,
                "t2_product": d2.name,
                "t1_date": "2019-07-10",
                "t2_date": "2023-07-14",
                "same_orbit": True,
                "season_match": True,
                "valid_fraction": float("nan"),
                "tile_type_hint": hint,
                "status": "selected",
            }
        )

    q = size10 // 8
    objs_a = default_objects(size10) + [Obj(2, 6 * q, 6 * q, 6 * q + 5, 6 * q + 8), Obj(1, q, 5 * q, q + 9, 5 * q + 9)]
    objs_b = [
        Obj(4, q, q, q + 12, q + 14),
        Obj(2, 5 * q, 5 * q, 5 * q + 5, 5 * q + 8),
        Obj(3, 3 * q, q, 3 * q + 1, q + 40),
    ]

    # A: change pair + identical pair
    g = _grid10(AOIS[0][4], AOIS[0][5], size10)
    a1, a2, _ = make_pair(raw / "A_smoke_01", "A01", seed=101, size10=size10, objects_t2=objs_a, grid10=g)
    add_scene("A_smoke_01", a1, "2019-07-10")
    add_scene("A_smoke_01", a2, "2023-07-14")
    add_pair("A_smoke_01", "p01", a1, a2, TileType.CHANGE_POS.value)
    add_pair("A_smoke_01", "p02", a1, a1, TileType.NOCHANGE_PLAIN.value)
    write_reviewed_gpkg(
        data_root / "reviewed" / "A_smoke_01" / "p01.gpkg", reviewed_rows(objects_to_polygons(objs_a, g)), g.epsg
    )
    # B: change pair with clouds (val)
    g = _grid10(AOIS[1][4], AOIS[1][5], size10)
    b1, b2, _ = make_pair(
        raw / "B_smoke_02",
        "B02",
        seed=202,
        size10=size10,
        objects_t2=objs_b,
        clouds_t2=[(size10 - 40, size10 - 48, size10 - 8, size10 - 8)],
        grid10=g,
    )
    add_scene("B_smoke_02", b1, "2019-07-10")
    add_scene("B_smoke_02", b2, "2023-07-14")
    add_pair("B_smoke_02", "p01", b1, b2, TileType.CHANGE_POS.value)
    write_reviewed_gpkg(
        data_root / "reviewed" / "B_smoke_02" / "p01.gpkg", reviewed_rows(objects_to_polygons(objs_b, g)), g.epsg
    )
    # C: seasonal-contrast negative, no labels -> auto negative with the image-change gate
    g = _grid10(AOIS[2][4], AOIS[2][5], size10)
    c1, c2, _ = make_pair(
        raw / "C_smoke_03", "C03", seed=303, size10=size10, objects_t2=[], gain_t2=1.10, shift=(0.8, 0.4), grid10=g
    )
    add_scene("C_smoke_03", c1, "2021-02-10")
    add_scene("C_smoke_03", c2, "2021-08-20")
    add_pair("C_smoke_03", "p01", c1, c2, TileType.NOCHANGE_HARD.value)

    reg_dir = data_root / "registry"
    sdf = pd.DataFrame(scenes, columns=list(SCENES_COLUMNS)).drop_duplicates(["aoi_id", "product_id"])
    pdf = pd.DataFrame(pairs, columns=list(PAIRS_COLUMNS))
    sdf.to_parquet(reg_dir / "scenes.parquet", index=False)
    sdf.to_csv(reg_dir / "scenes.csv", index=False)
    pdf.to_parquet(reg_dir / "pairs.parquet", index=False)
    pdf.to_csv(reg_dir / "pairs.csv", index=False)
    typer.echo(f"smoke data: {len(reg)} AOIs, {len(sdf)} scenes, {len(pdf)} pairs under {data_root}")


if __name__ == "__main__":
    app()
