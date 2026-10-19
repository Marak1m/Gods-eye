#!/usr/bin/env python
"""15_finetune.py — Stage-III fine-tune from a calibrated base run with the 50/50 sampler and forgetting check.

HANDOFF.md plan doc 12. Thin wrapper over `scd.train.train`; no logic lives here.
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", help="Config file (may use _base_)."),
    set_: list[str] = typer.Option([], "--set", help="key=value override; repeatable."),
    run: str = typer.Option(..., "--run", help="Fine-tune run name."),
) -> None:
    from scd.config import load
    from scd.train import train

    cfg = load(config, set_)
    out = train.finetune(cfg, run)
    typer.echo(f"fine-tune run dir: {out}")


if __name__ == "__main__":
    app()
