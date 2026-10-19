#!/usr/bin/env python
"""13_evaluate.py — per-class / per-stratum metrics, confusion matrix, robustness table and worst chips for a run.

HANDOFF.md section 7.1, 11. Thin wrapper over `scd.eval.evaluate`; no logic lives here.
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
    split: str = typer.Option("val", "--split", help="val | stage2 | targets."),
    chips_version: int = typer.Option(1, "--chips-version", help="Chips version."),
    baseline: bool = typer.Option(False, "--baseline", help="Also evaluate the PCCD baseline."),
) -> None:
    from scd.config import load
    from scd.eval import evaluate

    cfg = load(config, set_)
    out = evaluate.run(cfg, run, split=split, chips_version=chips_version, baseline=baseline)
    typer.echo(f"metrics written to {out}")


if __name__ == "__main__":
    app()
