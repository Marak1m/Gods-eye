#!/usr/bin/env python
"""16_report.py — write runs/<run>/report.md (methodology tables and figures) for the Approach score.

HANDOFF.md section 10 section 16. Thin wrapper over `scd.eval.report`; no logic lives here.
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
) -> None:
    from scd.config import load
    from scd.eval import report

    cfg = load(config, set_)
    out = report.write_report(cfg, run)
    typer.echo(f"report: {out}")


if __name__ == "__main__":
    app()
