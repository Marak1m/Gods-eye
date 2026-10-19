#!/usr/bin/env python
"""09_train.py — train the Siamese change network.

HANDOFF.md section 7. Thin wrapper over `scd.train.train`; no logic lives here.
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", help="Config file (may use _base_)."),
    set_: list[str] = typer.Option([], "--set", help="key=value override; repeatable."),
    run: str = typer.Option(..., "--run", help="Run name -> runs/<run>/."),
    chips_version: int = typer.Option(1, "--chips-version", help="Chips version to train on."),
    resume: bool = typer.Option(False, "--resume", help="Resume from runs/<run>/checkpoints/last.ckpt."),
) -> None:
    from scd.config import load
    from scd.train import train

    cfg = load(config, set_)
    out = train.run(cfg, run, chips_version=chips_version, resume=resume)
    typer.echo(f"run dir: {out}")


if __name__ == "__main__":
    app()
