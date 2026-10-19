#!/usr/bin/env python
"""05_import_reviewed.py — validate corrected GeoPackages into reviewed/<aoi>/<pair>.gpkg; double-annotation agreement.

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
    gpkg: list[Path] = typer.Option(
        [], "--gpkg", help="Reviewed GeoPackage(s); default: scan <data_root>/review_inbox."
    ),
    agreement: list[Path] = typer.Option([], "--agreement", help="Two GeoPackages to compare (per-class IoU)."),
) -> None:
    from scd.config import load
    from scd.labels import conventions, verification

    cfg = load(config, set_)
    conv = conventions.load(cfg.labeling_guide_path())
    if len(agreement) == 2:
        for cid, iou in verification.agreement(agreement[0], agreement[1]).items():
            typer.echo(f"class {cid}: IoU {iou:.3f}")
        return
    reports = verification.import_all(cfg, conv, gpkgs=gpkg or None)
    for r in reports:
        typer.echo(r.summary())


if __name__ == "__main__":
    app()
