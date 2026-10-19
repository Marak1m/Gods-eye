# Deviations from HANDOFF.md

Per CLAUDE.md: when a source, API or contract differs from HANDOFF.md, record it here with date
and evidence, then proceed. "H" = handoff section, "P" = plan doc
(`claude-docs/PS-10 Stage III Change Detection Build Plan.md`).

## User amendments (2026-10-18, approved with the build plan)

| # | amends | decision |
|---|---|---|
| U1 | H7.1 proxy | `val/leaderboard_proxy = 0.7*weighted_iou + 0.2*mean(robust_identity, robust_synthetic, robust_near)`. `val/robust_swap` is logged and asserted `== 1.0` as a regression guard, never averaged: a constant term dilutes the three that vary. |
| U2 | H4 `infer.tta_temporal: true` | Default **false**. Content-hash canonical ordering plus exact fusion symmetry make the (B,A) pass bit-identical to (A,B), so the second order only doubles cost. The flag stays; `tests/test_pipeline_swap.py` is the guard. |
| U3 | H8 steps 4/6, H3.2 nodata 255 | New key `infer.invalid_value` (default 0). Every invalid pixel (cloud, shadow, DN-0 holes) writes 0 in `change.tif`; 255 only when `invalid_value=255`. `valid.tif` is always written beside `change.tif`. Reason: the organisers' scorer expects values 0–4 and its treatment of 255 is unknown. |
| U4 | H6.4 encoder | Per-date (unbatched) encoder calls only in eval mode; in training mode both dates are stacked into one call (2x throughput). Exactness is only required at inference; `test_net_symmetry` runs in eval mode. |
| U5 | H11.2 Phase 4 "4-band input" | `GeoTiffScene` zero-fills B11/B12 **after** scaling (0 reflectance), identical to `augment.band_dropout`. `augment.band_dropout_p` is a validated `Config` field. If organiser chips turn out to be 4-band, retrain with `band_dropout_p >= 0.2` or a 4-band variant. |
| U6 | H2.2 | `SCD_DATA` must be on the Linux filesystem (never `/mnt/c` under WSL); `make setup` warns. Smoke data root is `${SMOKE_DATA:-/tmp/ps10-scd-smoke}`. |

## Contract conflicts found before M0 (2026-10-18, audit of H vs P)

| # | sections | conflict | resolution |
|---|---|---|---|
| C1 | H6.4 vs H6.2 | "out(x,x) change logit is spatially constant" cannot hold with `absdiff_sum` fusion (the f1+f2 term varies) nor with zero-padded convs at borders. | Assertion dropped. Tests check swap-equality (`torch.equal`) for every fusion mode and that the abs-diff half of the fused features is zero for identical inputs. Identity exactness comes from the H8 guard (`test_identity_guard`). |
| C2 | H0.3, H5.4, H6.2, H6.4, H9, P4.1 | Tolerances disagree (bit-identical / `torch.equal` / 1e-5). | Exact equality everywhere, by canonical ordering and per-date encoder calls in eval mode. |
| C3 | H5.4 steps 1 and 4 | Phase correlation (subpixel) and shapely union/intersection are not bitwise symmetric in floating point. | Every pairwise function sorts its inputs by a content key (sha256 of bytes / WKB), computes once, maps back. |
| C4 | H7.4 vs definitions | "Re-run robustness and assert unchanged" after calibration cannot hold for `robust_synthetic`/`robust_near`, which depend on tau and b; on the raw network `robust_identity` also depends on tau (it is the identity *guard* in the pipeline that is exact). | `calibrate` asserts `robust_swap == 1.0` before and after and logs identity/synthetic/near before and after. Training-time validation uses tau=0.5, b=0. |
| C5 | H8 step 1 vs H5.4 | `pipeline.run(A.tif, B.tif)` and `test_identity_guard` cannot call `preprocess_pair(ScenePath, ..., AoiRow)` as written (no MTD, no AOI row, maybe no SCL). | `Scene` reader protocol (`SafeZipScene`, `SimpleScene`, `GeoTiffScene`); AOI row optional at inference; `cloud.method=scl` with no SCL means all valid. |
| C6 | H5.4 steps 4–5 | Validity masks are not shifted with the half-shifted images. | Masks shifted nearest-neighbour; exposed strip set to 0. |
| C7 | H6.4 | "about 45 M parameters"; the H6.3 wiring gives about 38 M. | Test asserts 30–50 M. |
| C8 | H4 vs H12 / P11 | Ablations A2, A5, A12 and the P10 post-processing numbers have no config keys; the 50-px / 128-px / 768-stride chip constants are hard-coded. | Added: `bands.normalise`, `model.head_mode`, `model.encoder_init=random`, `train.lr_min`, `train.pretrain`, `train.max_steps`, `train.num_workers`, `chips.*`, `sampler.crop_centre_radius_px`, `infer.postprocess.*`, `infer.invalid_value`, `finetune.*`, `labeling_guide`. |
| C9 | H4/H12 vs P1/P7.3/P7.5 | SYNTHETIC share 0.18 vs 0.15. | 0.18 (handoff). Share-sweep ablation files carry full tables; the delta goes to CHANGE_POS (or NOCHANGE_HARD for stratum F). |
| C10 | H5.8 vs P10 step 9 | Road minimum: 200 m (labels) vs 150 m skeleton (inference). | `infer.postprocess.min_length_m.road: 150`; the label rule stays 200 m. |
| C11 | H3.2/H8 vs P10 | DEFLATE vs LZW; OmniCloudMask then SCL vs s2cloudless. | Handoff governs: DEFLATE; no s2cloudless. |
| C12 | H0.6, H9, H13 | Smoke is defined in section 9 (not 13); the M4 gate requires smoke, but smoke needs M5–M7 modules. | `make smoke-data` gates M4; full `make smoke` gates M7. |
| C13 | H9 smoke vs H5.5 | Smoke rasterises, but `conventions.load()` must fail while `labeling_guide.yaml` has `fill`; 1024-px fixtures are too large to commit. | `configs/smoke.yaml` points at `tests/fixtures/labeling_guide_smoke.yaml`; fixtures generated at run time at `chip_px=256`. |
| C14 | H5.13 | Date-asymmetric augmentations on IDENTICAL draws destroy identity. | Skipped for IDENTICAL draws. |
| C15 | H5.10 | Manifest `tile_type` enum lists types that are never stored; no sampler fallback for a stratum lacking a type/class. | Stored enum = {CHANGE_POS, NOCHANGE_HARD, NOCHANGE_PLAIN}; the 7-type enum lives on `DrawRequest`; fallback to the next available type/class is logged. |
| C16 | P10/P13 vs H9/H13 | No end-to-end run(A,B) == run(B,A) test in the handoff. | `tests/test_pipeline_swap.py` added (CPU, random-init net, raster and vectors). |
| C17 | H9 | `test_determinism` marked gpu though it needs none; ckpt-dependent tests have no CPU form. | CPU `test_determinism`; CPU oracle-net harness versions of the synthetic/near-identical tests; trained-ckpt versions stay `gpu`. |
| C18 | H2.1/H5.4 | OmniCloudMask DN scaling defers to the companion `ard/core/cloud.py`, not available here. | Fed DN after the BOA offset. Re-check when `third_party/ard` arrives (`test_cloud_parity`). |
| C19 | H13 M5/M6 | FP16-under-8-GB and 2-epoch-on-chips-v1 are GPU/real-data criteria. | CPU equivalents: fp32 forward/backward on 2x6x256x256; 2-step run on smoke chips. GPU criteria left for the H100 box. |
| C20 | H5.10, H4 under A3 | Pixel constants (50 px, `dilate_px`, `max_shift_px`, 128-px crop radius) change meaning at 2.5 m; `sr.pool_to_5m` must be false. | `configs/ablations/A3_2p5m.yaml` sets `pool_to_5m: false` and doubles the px constants. |
| C21 | H12 vs P7.7/P14 | A8 means the NOCHANGE_HARD share in H12/P11 but auto-label down-weighting in P7.7/P14; A1b missing from the H12 order. | H12 governs; `A1b_sen2sr_main.yaml` follows A1. |
| C22 | H3.1 vs H11.1/H15 | `configs/targets.csv` and `docs/DEVIATIONS.md` are referenced but absent from the layout; `models/` absent from the data layout. | All three added. |

## Additions made while implementing (2026-10-19)

| what | where | why |
|---|---|---|
| `scd/status.py` | package | `runs/pipeline_status.json` writer for the notebook STATUS cells (H10); logic must not live in the notebook. |
| `scd/geo/vector.content_key`, `canonical_pair` | package | the canonical-ordering primitive behind every exact-symmetry guarantee (C3). |
| `Scene` protocol with `SimpleScene`, `SafeZipScene`, `GeoTiffScene` | `scd/imagery/scenes.py` | C5; fixtures and delivered GeoTIFFs share the SAFE code path. |
| `scd/eval/evaluate.py`, `scd/train/ablation.py` | package (stubs, M8) | targets of `scripts/13_evaluate.py` and `scripts/14_ablation.py`; H3.1 had no module for them. |
| `tests/fixtures/{synth,labels,oracle,make_smoke_data}.py` | tests | synthetic scenes, reviewed GeoPackages, an exactly symmetric oracle network and the smoke dataset (C13, C17). |
| `make smoke` trains one full epoch of 2 steps instead of `max_steps=2` | `configs/smoke.yaml` | Lightning runs validation and checkpoints at epoch end; a `max_steps` stop mid-epoch would skip both. |
| `train.num_workers`, `train.max_steps`, `TrainCfg.precision` accepts `32` or `"32"` | config | smoke/debug knobs; `--set train.precision=32` parses as an int. |
| H5.2 rule 2 | `scenes.select_pairs` | the SCL-based AOI cloud fraction exists only after download; the query filter uses the tile-level percentage and `preprocess_pair` records the per-AOI fraction in `meta.json`. |
| `registry.target_type` accepts `unknown` | `scd/imagery/registry.py` | the 42 targets enter the registry before the Phase 0 survey assigns a type (H11.2). |

## Unverified code (written from the H2.1 API notes, not executed on this machine)

| module | why unverified | how to verify |
|---|---|---|
| `scd/imagery/sr.py` (SEN2SR-Lite) | no model download / GPU here | `pytest -m gpu tests/test_preprocess_symmetry.py` on a cached real pair |
| `scd/imagery/cloud.py` (OmniCloudMask) | no GPU / real scene | `pytest -m gpu tests/test_cloud_parity.py` once `third_party/ard` exists |
| `scd/imagery/cdse.py` (cdsetool) | no CDSE credentials | `scripts/01_fetch_scenes.py --download` on one product |
| `scd/imagery/scenes.py::SafeZipScene` | no `.SAFE` product on disk | first real pair in M2 |
