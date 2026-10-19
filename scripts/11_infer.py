#!/usr/bin/env python
"""11_infer.py — two GeoTIFFs -> change.tif, valid.tif, change.shp, report.json, quicklook.png.

HANDOFF.md section 8. Thin wrapper over `scd.infer.pipeline`; no logic lives here.
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", help="Config file (may use _base_)."),
    set_: list[str] = typer.Option([], "--set", help="key=value override; repeatable."),
    t1: Path = typer.Option(..., "--t1", help="First image (GeoTIFF or .SAFE zip)."),
    t2: Path = typer.Option(..., "--t2", help="Second image."),
    out: Path = typer.Option(..., "--out", help="Output directory outputs/<job>."),
    run: str | None = typer.Option(None, "--run", help="Run name: uses its best checkpoint and calibration.json."),
    checkpoint: Path | None = typer.Option(None, "--checkpoint", help="Explicit checkpoint path."),
    calibration: Path | None = typer.Option(None, "--calibration", help="Explicit calibration.json."),
    bands: str | None = typer.Option(
        None, "--bands", help="Comma-separated band names of the inputs if tags are missing."
    ),
    assume_dn: bool = typer.Option(False, "--assume-dn", help="Inputs are uint16 DN x10000."),
    assume_reflectance: bool = typer.Option(False, "--assume-reflectance", help="Inputs are float reflectance."),
) -> None:
    from scd.config import load
    from scd.infer import pipeline

    cfg = load(config, set_)
    net, calib = pipeline.load_model_and_calibration(cfg, run=run, checkpoint=checkpoint, calibration=calibration)
    result = pipeline.run(
        t1,
        t2,
        out,
        cfg,
        calib,
        net=net,
        bands=bands.split(",") if bands else None,
        assume="dn" if assume_dn else ("reflectance" if assume_reflectance else None),
    )
    typer.echo(result.summary())


if __name__ == "__main__":
    app()
