#!/usr/bin/env python
"""02_preprocess_pairs.py — build pairs/<aoi>/<pair>/{t1,t2,valid}.tif + meta.json for every selected pair.

HANDOFF.md section 5.4. Thin wrapper over `scd.imagery.preprocess`; no logic lives here.
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
    pair: list[str] = typer.Option([], "--pair", help="Restrict to pair id(s)."),
    force: bool = typer.Option(False, "--force", help="Rebuild even when meta.json matches the config hash."),
) -> None:
    from scd.config import load
    from scd.imagery import preprocess

    cfg = load(config, set_)
    products = preprocess.run_all(cfg, aoi_ids=aoi or None, pair_ids=pair or None, force=force)
    typer.echo(f"{len(products)} pairs ready under {cfg.pairs_dir}")


if __name__ == "__main__":
    app()
