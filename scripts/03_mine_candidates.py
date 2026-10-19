#!/usr/bin/env python
"""03_mine_candidates.py — mine candidate change polygons from dated open products into candidates/<aoi>/<pair>.gpkg.

HANDOFF.md section 5.6. Thin wrapper over `scd.labels.mining`; no logic lives here.
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", help="Config file (may use _base_)."),
    set_: list[str] = typer.Option([], "--set", help="key=value override; repeatable."),
    sources: str = typer.Option("gem,cbra,obt,io_lulc,osm_history", "--sources", help="Comma-separated source names."),
    aoi: list[str] = typer.Option([], "--aoi", help="Restrict to AOI id(s)."),
) -> None:
    from scd.config import load
    from scd.labels import mining

    cfg = load(config, set_)
    out = mining.run_all(cfg, [s.strip() for s in sources.split(",") if s.strip()], aoi_ids=aoi or None)
    typer.echo(f"{len(out)} candidate GeoPackages written under {cfg.candidates_dir}")


if __name__ == "__main__":
    app()
