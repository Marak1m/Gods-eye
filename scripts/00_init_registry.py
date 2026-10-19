#!/usr/bin/env python
"""00_init_registry.py — create or extend registry/aois.parquet from configs/targets.csv and optional CSVs.

HANDOFF.md section 5.1, 11.2. Thin wrapper over `scd.imagery.registry`; no logic lives here.
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help=__doc__)


@app.command()
def main(
    config: Path = typer.Option(Path("configs/default.yaml"), "--config", help="Config file (may use _base_)."),
    set_: list[str] = typer.Option([], "--set", help="key=value override; repeatable."),
    targets_csv: Path = typer.Option(
        Path("configs/targets.csv"), "--targets-csv", help="The 42 targets (role=targets)."
    ),
    append_csv: Path | None = typer.Option(None, "--append-csv", help="CSV with registry columns to append."),
    size_km: float = typer.Option(10.0, "--size-km", help="AOI box size for the targets."),
) -> None:
    from scd.config import load
    from scd.imagery import registry

    cfg = load(config, set_)
    path = registry.registry_path(cfg)
    if path.exists():
        df = registry.load(path)
    else:
        df = registry.init_from_targets(targets_csv, size_km=size_km)
    if append_csv is not None:
        df = registry.append_from_csv(df, append_csv)
    errors = registry.validate(df)
    if errors:
        raise typer.Exit(code=_report(errors))
    registry.save(df, path)
    typer.echo(f"{len(df)} AOIs -> {path}")


def _report(errors: list[str]) -> int:
    for e in errors:
        typer.echo(f"registry error: {e}", err=True)
    return 1


if __name__ == "__main__":
    app()
