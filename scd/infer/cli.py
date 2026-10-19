"""`scd-infer` console entry point (HANDOFF.md 8): the same thin wrapper as scripts/11_infer.py."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Two Sentinel-2 images -> change.tif, valid.tif, change.shp, report.json")


@app.command()
def main(
    t1: Path = typer.Option(..., "--t1", help="First image (GeoTIFF or .SAFE zip)."),
    t2: Path = typer.Option(..., "--t2", help="Second image."),
    out: Path = typer.Option(..., "--out", help="Output directory."),
    config: Path = typer.Option(Path("configs/default.yaml"), "--config"),
    set_: list[str] = typer.Option([], "--set", help="key=value override; repeatable."),
    run: str | None = typer.Option(None, "--run", help="Run name (best checkpoint + calibration.json)."),
    checkpoint: Path | None = typer.Option(None, "--checkpoint"),
    calibration: Path | None = typer.Option(None, "--calibration"),
    bands: str | None = typer.Option(None, "--bands", help="Comma-separated band names if tags are missing."),
    assume_dn: bool = typer.Option(False, "--assume-dn"),
    assume_reflectance: bool = typer.Option(False, "--assume-reflectance"),
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
