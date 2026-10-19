#!/usr/bin/env python
"""01_fetch_scenes.py — query CDSE for Sentinel-2 L2A products per AOI, select pairs, optionally download.

HANDOFF.md section 5.2, 5.3. Thin wrapper over `scd.imagery.scenes / scd.imagery.cdse`; no logic lives here.
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", help="Config file (may use _base_)."),
    set_: list[str] = typer.Option([], "--set", help="key=value override; repeatable."),
    aoi: list[str] = typer.Option([], "--aoi", help="Restrict to AOI id(s)."),
    query_only: bool = typer.Option(False, "--query-only", help="Query and select pairs; no download."),
    download: bool = typer.Option(False, "--download", help="Download the selected products."),
) -> None:
    from scd.config import load
    from scd.imagery import cdse, scenes

    cfg = load(config, set_)
    scenes_df, pairs_df = scenes.query_and_select(cfg, aoi_ids=aoi or None)
    typer.echo(f"{len(scenes_df)} products considered, {len(pairs_df)} pairs selected")
    if download and not query_only:
        ids = sorted(set(pairs_df["t1_product"]) | set(pairs_df["t2_product"]))
        paths = cdse.fetch(ids, cfg.raw_dir)
        typer.echo(f"{len(paths)} products on disk under {cfg.raw_dir}")


if __name__ == "__main__":
    app()
