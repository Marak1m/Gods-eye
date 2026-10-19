#!/usr/bin/env python
"""10_calibrate.py — calibrate tau and the class biases on the val split; write runs/<run>/calibration.json.

HANDOFF.md section 7.4. Thin wrapper over `scd.train.calibrate`; no logic lives here.
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", help="Config file (may use _base_)."),
    set_: list[str] = typer.Option([], "--set", help="key=value override; repeatable."),
    run: str = typer.Option(..., "--run", help="Run name."),
    chips_version: int = typer.Option(1, "--chips-version", help="Chips version holding the val split."),
    stub: bool = typer.Option(False, "--stub", help="Write tau=0.5, bias=0 without evaluating (smoke)."),
) -> None:
    from scd.config import load
    from scd.train import calibrate

    cfg = load(config, set_)
    calib = calibrate.run(cfg, run, chips_version=chips_version, stub=stub)
    typer.echo(f"tau={calib.tau:.2f} bias={calib.bias} weighted_iou={calib.weighted_iou:.4f}")


if __name__ == "__main__":
    app()
