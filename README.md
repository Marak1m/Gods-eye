# ps10-scd — Sentinel-2 semantic change detection for PS-10 Stage III

Siamese ConvNeXt-tiny U-Net: two Sentinel-2 L2A images → GeoTIFF mask (0 none, 1 buildup,
2 building, 3 road, 4 solar) + shapefile. Order-invariant and identity-consistent by construction.

- Contracts: [HANDOFF.md](HANDOFF.md). Rationale: [the plan doc](claude-docs/PS-10%20Stage%20III%20Change%20Detection%20Build%20Plan.md).
- Session rules: [CLAUDE.md](CLAUDE.md). Deviations from the handoff: [docs/DEVIATIONS.md](docs/DEVIATIONS.md).

## Quickstart (Linux / WSL)

```bash
export SCD_DATA=~/scd_data SCD_RUNS=~/scd_data/runs   # data root; see WSL notes below
make setup            # uv venv (Python 3.11) + deps; creates the data -> $SCD_DATA symlink
make test             # CPU unit tests (gpu/network tests skipped)
make smoke-data       # fixtures -> pairs -> labels -> chips -> object library   (M4 gate)
make smoke            # + 2 training steps, stub calibration, inference, invariance tests (M7 gate)
make lint             # ruff + mypy
make notebook STAGE=0 # papermill run of notebooks/00_pipeline.ipynb
uv run scd-infer --t1 A.tif --t2 B.tif --out outputs/job1 --run v1_seed42
```

Every script in `scripts/` takes `--config configs/default.yaml` and `--set key=value` overrides.

## WSL notes

- Put the venv on the Linux filesystem: `export UV_PROJECT_ENVIRONMENT=~/.venvs/ps10-scd` before `make setup`.
  A venv under `/mnt/c` imports torch in tens of seconds.
- Put `SCD_DATA` on the Linux filesystem too (`~/scd_data`), never under `/mnt/c`: chips read through the
  9p mount bottleneck the DataLoader. `make setup` warns if it sees `/mnt/`.
