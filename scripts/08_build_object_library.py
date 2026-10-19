#!/usr/bin/env python
"""08_build_object_library.py — cut the synthetic object library objects/v<N>/<class>/*.npz from verified positives.

HANDOFF.md section 5.11. Thin wrapper over `scd.labels.synthetic`; no logic lives here.
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", help="Config file (may use _base_)."),
    set_: list[str] = typer.Option([], "--set", help="key=value override; repeatable."),
    version: int = typer.Option(1, "--version", help="Chips/objects version N."),
) -> None:
    from scd.config import load
    from scd.labels import synthetic

    cfg = load(config, set_)
    counts = synthetic.build_object_library(cfg, version)
    typer.echo(f"object library v{version}: {counts}")


if __name__ == "__main__":
    app()
