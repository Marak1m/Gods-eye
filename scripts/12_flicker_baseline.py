#!/usr/bin/env python
"""12_flicker_baseline.py — PCCD flicker rate on NOCHANGE_HARD pairs per stratum and rubric IoU on val.

HANDOFF.md section 8, 11. Thin wrapper over `scd.eval.flicker`; no logic lives here.
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", help="Config file (may use _base_)."),
    set_: list[str] = typer.Option([], "--set", help="key=value override; repeatable."),
    strata: str = typer.Option("A,B,C,D,E,F", "--strata", help="Comma-separated strata."),
    n_pairs: int = typer.Option(10, "--n-pairs", help="NOCHANGE_HARD pairs per stratum."),
    run: str | None = typer.Option(None, "--run", help="Also score this run's model for comparison."),
) -> None:
    from scd.config import load
    from scd.eval import flicker

    cfg = load(config, set_)
    table = flicker.run(cfg, [s.strip() for s in strata.split(",")], n_pairs, run)
    typer.echo(table.to_string())


if __name__ == "__main__":
    app()
