#!/usr/bin/env python
"""14_ablation.py — run one ablation config as runs/abl_<id>_<seed> and append runs/ablations.csv.

HANDOFF.md section 7.5. Thin wrapper over `scd.train.ablation`; no logic lives here.
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", help="Config file (may use _base_)."),
    set_: list[str] = typer.Option([], "--set", help="key=value override; repeatable."),
    id_: str = typer.Option(..., "--id", help="Ablation id, e.g. A5_softmax5 (configs/ablations/<id>.yaml)."),
    seed: int = typer.Option(42, "--seed", help="Seed (final recipe: 42, 43, 44)."),
    chips_version: int = typer.Option(1, "--chips-version", help="Chips version."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Append a placeholder row without training."),
) -> None:

    from scd.train import ablation

    row = ablation.run(id_, seed=seed, chips_version=chips_version, overrides=set_, dry_run=dry_run)
    typer.echo(row)


if __name__ == "__main__":
    app()
