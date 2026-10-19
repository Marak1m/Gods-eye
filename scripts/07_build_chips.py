#!/usr/bin/env python
"""07_build_chips.py — cut 1024 px chips from pairs + labels and write chips/v<N>/manifest.parquet.

HANDOFF.md section 5.10, 5.12. Thin wrapper over `scd.data.chips`; no logic lives here.
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", help="Config file (may use _base_)."),
    set_: list[str] = typer.Option([], "--set", help="key=value override; repeatable."),
    version: int = typer.Option(1, "--version", help="Chips version N -> chips/v<N>."),
    sampler_report: bool = typer.Option(False, "--sampler-report", help="Print sampler share statistics."),
) -> None:
    from scd.config import load
    from scd.data import chips, manifest, sampler

    cfg = load(config, set_)
    out = chips.build_chips(cfg, version)
    df = manifest.load(out / "manifest.parquet")
    typer.echo(manifest.summarise(df).to_string())
    if sampler_report:
        typer.echo(sampler.report(cfg, df).to_string())


if __name__ == "__main__":
    app()
