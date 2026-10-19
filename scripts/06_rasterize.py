#!/usr/bin/env python
"""06_rasterize.py — burn reviewed polygons into labels/<aoi>/<pair>/{change,sem_t1,sem_t2}.tif.

HANDOFF.md section 5.8. Thin wrapper over `scd.labels.rasterize`; no logic lives here.
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
    force: bool = typer.Option(False, "--force", help="Rebuild existing label rasters."),
) -> None:
    from scd.config import load
    from scd.labels import rasterize

    cfg = load(config, set_)
    out = rasterize.run_all(cfg, aoi_ids=aoi or None, force=force)
    typer.echo(f"{len(out)} label sets written under {cfg.labels_dir}")


if __name__ == "__main__":
    app()
