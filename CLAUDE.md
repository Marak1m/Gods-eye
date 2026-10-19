# ps10-scd — rules for Claude Code

Purpose: Sentinel-2 bitemporal change detection for PS-10 Stage III (classes:
1 buildup/clearance, 2 building ≥60×30 m, 3 road, 4 solar). Spec: HANDOFF.md.
Rationale: the plan doc (link in README).

Hard rules
- Order invariance and identity consistency are invariants, not features.
  Any change to scd/geo, scd/imagery, scd/model or scd/infer must keep
  tests/test_*symmetry*.py and tests/test_identity_guard.py green.
- Band order is B02,B03,B04,B08,B11,B12 everywhere. Never reorder silently.
- Logic lives in scd/. scripts/ and notebooks/ only call it.
- Never commit anything under data/ or runs/. Registry + configs rebuild everything.
- Never change a contract in HANDOFF.md to make a test pass; report the conflict.
- Metrics come from accumulated confusion matrices over valid pixels; never
  a mean of per-chip IoUs; never a subset of classes.
- Checkpoint selection monitors val/leaderboard_proxy; save_top_k=5 + milestones.
- Third-party code (third_party/) is wrapped, never edited.

Workflow
- `make setup` → `make test` → `make smoke` before claiming a milestone.
- Run `ruff` and `mypy scd` before each commit; commit messages name the
  milestone (M0..M9) and the tests touched.
- When a data source or API differs from HANDOFF.md section 2.1, record the
  difference in docs/DEVIATIONS.md with date and evidence, then proceed.

Conventions
- Python 3.11, type hints everywhere, pydantic configs, typer CLIs, rich logging.
- Rasters: GeoTIFF, uint16 ×10000 imagery, uint8 masks with 255 ignore.
- Seeds: 42 default; samplers and augmentations deterministic given (seed, epoch, index).
