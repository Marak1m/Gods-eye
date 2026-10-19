#!/usr/bin/env python
"""04_export_for_review.py — write QGIS projects (+GeoPackage) for human correction; list what is pending.

HANDOFF.md section 5.7. Thin wrapper over `scd.labels.verification`; no logic lives here.
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
    pending: bool = typer.Option(False, "--pending", help="Only list pairs awaiting review."),
) -> None:
    from scd.config import load
    from scd.labels import verification

    cfg = load(config, set_)
    if pending:
        typer.echo(verification.pending(cfg).to_string())
        return
    out = verification.export_all(cfg, aoi_ids=aoi or None)
    typer.echo(f"{len(out)} review projects written")


if __name__ == "__main__":
    app()
