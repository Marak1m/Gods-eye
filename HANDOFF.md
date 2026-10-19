# HANDOFF — PS-10 Stage III semantic change detection (Sentinel-2)

Build specification for Claude Code. Read the whole file before writing code. The
companion document "PS-10 Stage III: Change Detection Build Plan" (the plan doc)
holds the rationale; this file holds the contracts: directory layout, module
signatures, data schemas, configs, tests, the end-to-end notebook, the data
curation runbook, and the build order with acceptance criteria.

Project owner: ai@yuvrik.com. Date: 2026-10-18.

---

## 0. How Claude Code should use this document

1. Build in the order of section 13 (milestones M0–M9). Do not start a later
   milestone until the earlier milestone's acceptance tests pass.
2. Every module has a contract in this file (inputs, outputs, invariants). Keep
   the names. If a contract is impossible as written, stop and report the
   conflict instead of silently changing the interface.
3. Two invariants are non-negotiable and have tests from M1 onward:
   - **Order invariance**: every pairwise operation must give bit-identical
     results when the two dates are swapped.
   - **Identity consistency**: two identical inputs must produce an all-zero
     change mask.
4. Scripts are thin wrappers over the `scd` package; logic lives in the
   package, never in scripts or the notebook.
5. Nothing under `data/` is ever committed. The AOI registry and configs are the
   only source of truth; everything else must be rebuildable from them.
6. Run `pytest -q` before reporting any milestone complete. Run `make smoke`
   (section 13) before M4 and after every change to preprocessing or the model.
7. Write `CLAUDE.md` first (section 15 has its content) so later sessions inherit
   the rules.

---

## 1. What we are building (one paragraph + the fixed decisions)

A Siamese ConvNeXt-tiny U-Net that takes two Sentinel-2 L2A images of the same
area (6 bands each, resampled to a 5 m canonical grid) and outputs a GeoTIFF
with values 0 (no change), 1 (clearance/built-up area), 2 (stand-alone building
≥ 60 × 30 m), 3 (road), 4 (solar farm), plus a shapefile. Change is undirected
(appearance or disappearance). The network decides *where* change happened
(binary head) and *what* it is (4-way head trained only on change pixels), with
an auxiliary single-date semantic decoder during training.

| Decision | Value |
|---|---|
| Sensor (this repo) | Sentinel-2 L2A only; LISS-4 is a later, separate model warm-started from this one |
| Bands | B02, B03, B04, B08, B11, B12 — in that order everywhere |
| Working grid | 5 m, UTM zone of the chip centroid, origin snapped to 5 m, extent = intersection of footprints |
| Resampling | RGBN: SEN2SR-Lite ×4 then 2×2 mean (ablation A1: bicubic); SWIR: bilinear 20→5 m |
| Scaling | reflectance = DN / 10000, clipped to [0, 1.2]; no per-scene stretch (ablation A2) |
| Fusion | per level: concat(|f1−f2|, f1+f2) → 1×1 conv → LayerNorm → GELU |
| Heads | change (1 logit, sigmoid), class (4 logits, softmax on change pixels), aux semantic (5 logits per date, training only) |
| Loss | BCE(pos_weight 2.0)+Dice on change; weighted CE + per-class Dice on class (masked to GT change); 0.3 × (CE+Dice) on aux semantics; all masked by validity |
| Output rule | M = 1[σ(c) > τ] · (1 + argmax_k(z_k + b_k)), τ and b_k from calibration |
| Training tiles | 512 × 512 at 5 m, random crops from 1024 × 1024 chips, batch 16, FP16 |
| Selection | proxy = 0.7 × rubric-weighted IoU (0.40/0.30/0.20/0.10) + 0.2 × robustness, on accumulated confusion matrices over deployment-style validation pairs |
| Class weights (rubric) | buildup 0.40, building 0.30, road 0.20, solar 0.10 |
| Robustness tests | swapped order → identical mask; identical input → zeros; pasted objects recovered; near-identical → < 0.1% change; deterministic |

---

## 2. Environment

Target machine: Linux, one NVIDIA H100 (80 GB) or equivalent, ≥ 24 CPU cores,
≥ 128 GB RAM, ≥ 2 TB NVMe for `data/`. Python 3.11.

### 2.1 `pyproject.toml` dependencies (pin minor versions at install time)

```
torch>=2.4, torchvision, lightning>=2.4, timm>=1.0, torchmetrics>=1.4
numpy, scipy, scikit-image, scikit-learn, pandas, pyarrow
rasterio>=1.4, geopandas>=1.0, shapely>=2.0, pyproj, affine, fiona
pystac-client, planetary-computer, cdsetool, sen2sr, mlstac
omnicloudmask, arosics (optional; extras "coreg")
omegaconf, pydantic>=2, typer, rich, tqdm
jupyter, ipywidgets, papermill, matplotlib
pytest, pytest-xdist, ruff, mypy
```

Verified APIs (as of 2026-10-18):

- `sen2sr`: `pip install sen2sr mlstac`. Load with
  `mlstac.download(file="https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/NonReference_RGBN_x4/mlm.json", output_dir=...)`
  then `mlstac.load(dir).compiled_model(device=...)`. Input `(1, 4, H, W)` float32
  = DN/10000 in band order **B04, B03, B02, B08**; `sen2sr.predict_large(model, X, overlap=32)`
  tiles 128×128 patches. Output is 4× the input size, same scaling. MIT licence.
  The `SEN2SRLite/main` model also super-resolves the 20 m bands (B05, B06, B07,
  B8A, B11, B12) to 2.5 m; the companion project measured SWIR sharpness gain of
  only 1.08–1.31× versus 1.24× for RGBN, so SWIR stays bilinear by default
  (ablation A1b tries `main`).
- `omnicloudmask`: `predict_from_array(arr)` with `arr` shape `(3, H, W)` in band
  order **Red, Green, NIR**; returns `(1, H, W)` with 0 clear, 1 thick cloud,
  2 thin cloud, 3 shadow. Options: `inference_device="cuda"`,
  `inference_dtype="bf16"`, `batch_size`, `compile_models=True`. Input scaling is
  not documented on the usage page: feed DN (uint16 × 10000 reflectance) exactly
  as the companion project's `ard/core/cloud.py` does, and keep a unit test
  comparing our mask with its mask on one scene.
- `cdsetool`: `query_features("SENTINEL-2", {"contentDateStartGe": ..., "contentDateStartLe": ...,
  "productType": "S2MSI2A", "cloudCover": "[0, 20]", "geometry": wkt})`;
  `download_features(features, path, {"concurrency": 4, "credentials": Credentials(user, pw)})`;
  consume the iterator with `list()`. Credentials from `~/.netrc` when omitted.
  Query cap 10,000 results.
- `timm.create_model("convnext_tiny", pretrained=True, features_only=True, out_indices=(0,1,2,3), in_chans=6)`
  returns four feature maps at strides 4, 8, 16, 32 with 96, 192, 384, 768
  channels. With `in_chans=6`, timm adapts the stem by repeating/averaging RGB
  filters; override to the explicit rule in section 6.1.

### 2.2 Environment variables

```
SCD_DATA=/mnt/nvme/scd_data        # data root (section 3.2)
SCD_RUNS=$SCD_DATA/runs
CDSE_USER / CDSE_PASS              # or ~/.netrc
PC_SDK_SUBSCRIPTION_KEY            # optional, Planetary Computer
HF_TOKEN                           # for gated DINOv3 weights (ablation only)
```

### 2.3 Third-party code

- `third_party/ard/` — git subtree of the companion project's `ard/` package
  and `check_pair.py` (section 17 of the plan doc). Never edit in place; wrap.
- `third_party/v3_roads/` — frozen copy of the previous road project at tag
  `v3.1` (used only by `scd/baseline/pccd.py` and `scripts/12_flicker_baseline.py`).

---

## 3. Repository and data layout

### 3.1 Repository

```
ps10-scd/
├── CLAUDE.md                     # rules for Claude Code sessions (section 15)
├── HANDOFF.md                    # this file
├── README.md                     # 20-line quickstart
├── pyproject.toml
├── environment.yml
├── Makefile                      # setup, test, smoke, lint, notebook
├── labeling_guide.yaml           # class conventions (filled in Phase 0)
├── configs/
│   ├── default.yaml              # everything; other files override sections
│   ├── strata.yaml               # stratum definitions and training shares
│   ├── ablations/A1_bicubic.yaml … A13_dinov3.yaml
│   └── finetune.yaml
├── scd/                          # the package
│   ├── __init__.py  config.py  log.py  constants.py
│   ├── geo/        grid.py  reproject.py  coreg.py  io.py  vector.py
│   ├── imagery/    registry.py  cdse.py  scenes.py  sr.py  cloud.py  preprocess.py
│   ├── labels/     conventions.py  mining.py  rasterize.py  fix_t1.py
│   │               negatives.py  verification.py  synthetic.py
│   │               sources/  gem.py  kruitwagen.py  cbra.py  obt.py  io_lulc.py
│   │                         dynamic_world.py  osm_history.py  spacenet7.py  oscd.py  terracd.py
│   ├── data/       chips.py  manifest.py  dataset.py  sampler.py  transforms.py  splits.py
│   ├── model/      encoder.py  fusion.py  decoder.py  heads.py  net.py  losses.py
│   ├── train/      lit_module.py  metrics.py  robustness.py  calibrate.py  train.py
│   ├── infer/      pipeline.py  tta.py  postprocess.py  export.py  cli.py
│   ├── baseline/   pccd.py
│   └── eval/       flicker.py  report.py  figures.py
├── scripts/                      # thin CLIs, numbered in pipeline order
│   ├── 00_init_registry.py       09_train.py
│   ├── 01_fetch_scenes.py        10_calibrate.py
│   ├── 02_preprocess_pairs.py    11_infer.py
│   ├── 03_mine_candidates.py     12_flicker_baseline.py
│   ├── 04_export_for_review.py   13_evaluate.py
│   ├── 05_import_reviewed.py     14_ablation.py
│   ├── 06_rasterize.py           15_finetune.py
│   ├── 07_build_chips.py         16_report.py
│   └── 08_build_object_library.py
├── notebooks/
│   ├── 00_pipeline.ipynb         # the centralised notebook (section 10)
│   └── qc/                       # small QC notebooks generated by scripts (optional)
├── tests/                        # section 9
├── third_party/  ard/  v3_roads/
└── data -> $SCD_DATA             # symlink, gitignored
```

### 3.2 Data root (`$SCD_DATA`)

```
registry/   aois.parquet  scenes.parquet  pairs.parquet  (+ .csv mirrors for humans)
raw/        s2/<product_id>.zip                    # as downloaded
external/   gem/  kruitwagen/  cbra/  obt/  io_lulc/  dw/  osm/  spacenet7/  oscd/  terracd/  stage2/
pairs/      <aoi_id>/<pair_id>/t1.tif  t2.tif  valid.tif  meta.json  quicklook.png
candidates/ <aoi_id>/<pair_id>.gpkg                # auto-mined polygons, for review
reviewed/   <aoi_id>/<pair_id>.gpkg                # after human correction
labels/     <aoi_id>/<pair_id>/change.tif  sem_t1.tif  sem_t2.tif
chips/      v<N>/images/<chip_id>_t1.tif  _t2.tif  _valid.tif
            v<N>/labels/<chip_id>_change.tif  _sem_t1.tif  _sem_t2.tif
            v<N>/manifest.parquet
objects/    v<N>/<class>/<obj_id>.npz              # synthetic object library
runs/       <run_name>/  checkpoints/  logs/  calibration.json  metrics/
outputs/    <job_id>/  change.tif  change.shp(+shx,dbf,prj)  report.json  quicklook.png
```

All rasters: GeoTIFF, tiled 512, uint16 ×10000 for imagery (nodata 0 is not used;
validity comes from `valid.tif`), uint8 for masks (255 = ignore), DEFLATE for
masks and outputs, uncompressed for chip imagery (the v3 read-speed lesson).

---

## 4. Configuration

One `configs/default.yaml`, loaded by `scd.config.load(path, overrides)` into a
pydantic model (`scd.config.Config`) that validates ranges and band order. Every
script and the notebook take `--config` and `--set key=value` overrides.

```yaml
paths:
  data_root: ${oc.env:SCD_DATA}
  runs: ${oc.env:SCD_RUNS}
grid:
  res_m: 5.0
  snap_m: 5.0
  tile_px: 512
  chip_px: 1024
bands:
  order: [B02, B03, B04, B08, B11, B12]
  scale: 10000.0
  clip_max: 1.2
sr:
  method: sen2sr_lite_rgbn   # | bicubic | sen2sr_lite_main
  model_dir: ${paths.data_root}/models/SEN2SRLite_RGBN_x4
  pool_to_5m: true
cloud:
  method: omnicloudmask       # | scl
  dilate_px: 3
coreg:
  method: phase_corr_half     # | arosics_local_half | none
  max_shift_px: 8
classes:
  ids: {none: 0, buildup: 1, building: 2, road: 3, solar: 4}
  rubric_weights: {buildup: 0.40, building: 0.30, road: 0.20, solar: 0.10}
  priority: [building, solar, road, buildup]     # first wins on overlap
  min_area_m2: {building: 1800, solar: 5000, buildup: 2500}
  min_length_m: {road: 200}
  road_width_m: {highway: 10, access: 6}
strata:
  shares: {A: 0.30, B: 0.30, C: 0.12, D: 0.12, E: 0.08, F: 0.08}
sampler:
  tile_type_shares:
    default: {CHANGE_POS: 0.45, NOCHANGE_HARD: 0.27, NOCHANGE_PLAIN: 0.05,
              IDENTICAL: 0.02, NEAR_IDENTICAL: 0.03, SYNTHETIC: 0.18, STAR: 0.0}
    F:       {CHANGE_POS: 0.10, NOCHANGE_HARD: 0.70, NOCHANGE_PLAIN: 0.05,
              IDENTICAL: 0.02, NEAR_IDENTICAL: 0.03, SYNTHETIC: 0.10, STAR: 0.0}
  class_shares: {buildup: 0.35, building: 0.30, road: 0.20, solar: 0.15}
  density_exponent: 0.3
  change_centred_prob: 0.7
  auto_label_weight: 0.5
model:
  encoder: convnext_tiny
  encoder_init: imagenet       # | v3_roads | dinov3_convnext_tiny
  in_channels: 6
  decoder_channels: [256, 128, 64, 32]
  fusion: absdiff_sum          # | absdiff | absdiff_prod
  full_res_skip: true
  aux_semantic: true
  n_change_classes: 4
  n_sem_classes: 5
train:
  epochs: 100
  batch_size: 16
  crops_per_epoch: 15000
  lr_decoder: 3.0e-4
  lr_encoder: 1.0e-4
  weight_decay: 1.0e-4
  warmup_epochs: 3
  precision: 16-mixed
  ema_decay: 0.999
  loss: {pos_weight: 2.0, lambda_sem: 0.3, class_weight_power: 0.5, cldice_road: false}
  checkpoint: {save_top_k: 5, milestone_every: 10, monitor: val/leaderboard_proxy}
  early_stop_patience: 20
  robustness_subset: 64        # val chips used for the simulated cases each epoch
  seed: 42
augment:
  geometric: {d4: 1.0, rotate_deg: 30, rotate_p: 0.5, scale: [0.9, 1.1], scale_p: 0.3}
  temporal_swap_p: 0.5
  radiometric: {brightness: 0.12, contrast: 0.12, gamma: [0.85, 1.15], band_gain: 0.05,
                haze_p: 0.3, noise_sigma: [0.005, 0.02]}
  shift: {max_px: 2, p: 0.3}
  cloud_paste_p: 0.15
  band_dropout_p: 0.05
infer:
  window_px: 512
  stride_px: 384
  tta_geometric: 8
  tta_temporal: true
  identity_eps: 1.0e-6
  nodata_value: 255
  simplify_m: 2.5
  water_ndwi: 0.3
```

---

## 5. Data layer contracts

### 5.1 AOI registry — `registry/aois.parquet`

| column | type | notes |
|---|---|---|
| aoi_id | str | `A_tibet_shigatse_01` — `<stratum>_<region>_<nn>` |
| stratum | str | A–F (plan doc section 3) |
| target_type | str | airfield, compound, solar, building, road, reclamation, settlement, negative |
| role | str | train, val, targets (the 42), stage3 (the two shapefiles), stage2 |
| centre_lat, centre_lon | float | WGS84 |
| size_km | float | 10 (default) or 5.12 |
| utm_epsg | int | derived |
| geometry | WKB | square polygon in WGS84 |
| lead_source | str | gem, cbra, obt, io_lulc, osm_history, manual, companion |
| lead_ref | str | e.g. GEM location id, OSM way id |
| event_year | int or null | year the lead says the change happened |
| t1_window, t2_window | str | ISO date ranges to search, e.g. `2019-05-01/2019-10-31` |
| notes | str | free text |

`scd.imagery.registry.load()` validates: unique ids, stratum in A–F, no two
train/val AOIs within 10 km, ≤ 8 AOIs per metropolitan cluster (DBSCAN 25 km),
`targets` AOIs never in train/val.

`scenes.parquet`: one row per Sentinel-2 product considered (product id, tile,
relative orbit, date, cloud %, baseline, local path, status). `pairs.parquet`:
one row per (aoi_id, pair_id, t1_product, t2_product, same_orbit, season_match,
valid_fraction, tile_type_hint, status).

### 5.2 Scene selection — `scd.imagery.scenes`

`select_pairs(aoi_row, scenes_df, rules) -> list[PairSpec]` with rules:

1. Same relative orbit preferred; cross-orbit allowed only if none exists.
2. Cloud cover over the AOI (from SCL at query time, not the tile-level %) ≤ 10%.
3. Positives: T1 in `t1_window`, T2 in `t2_window`, both within ±45 days of the
   same day-of-year when possible (season match), else flag `season_match=false`.
4. Negatives (`NOCHANGE_HARD`): the two scenes with maximum seasonal contrast
   (one in each of the stratum's contrast seasons from `strata.yaml`).
5. Stratum E (tropical coast): dry-season months only; accept up to 4 pairs.
6. Never two single-date scenes; composites are never used for training pairs.

### 5.3 Download — `scd.imagery.cdse`

`fetch(product_ids, dest) -> list[Path]` via cdsetool with concurrency 4,
resumable (skip if zip present and size matches), `.SAFE` zips kept under
`raw/s2/`. Read bands directly from the zip with rasterio (`zip+file://`).

Alternative source for dense time series: Planetary Computer STAC
(`sentinel-2-l2a`), used by `scd.labels.sources.io_lulc` and for quick
quicklooks; the training pairs still come from CDSE L2A products so that SCL
and baseline metadata are present.

### 5.4 Preprocessing — `scd.imagery.preprocess`

```
preprocess_pair(t1: ScenePath, t2: ScenePath, aoi: AoiRow, cfg) -> PairProduct
PairProduct: t1.tif, t2.tif (6-band uint16 ×10000, 5 m, canonical grid),
             valid.tif (uint8, 1 valid), meta.json (grid, offsets, cloud %, baselines, band order, hashes)
```

Steps, in this order, each a pure function in its own module:

1. `scd.geo.grid.canonical_grid(footprints: list[Polygon], res_m, snap_m) -> Grid`
   — UTM zone from the centroid of the *union*; origin = floor(min_x / snap) × snap;
   extent = intersection; identical for any ordering of `footprints` (test).
2. Per scene (order-independent): read B02,B03,B04,B08 (10 m), B11,B12 (20 m),
   SCL; apply `BOA_ADD_OFFSET` if baseline ≥ 04.00 (read from MTD; raise if
   missing, never guess); `scd.imagery.sr.super_resolve_rgbn()` → 2.5 m → 2×2 mean
   → 5 m, or bicubic; SWIR bilinear → 5 m; reproject to the canonical grid with
   bilinear; clip to [0, 1.2 × 10000]; write uint16.
3. `scd.imagery.cloud.mask(scene) -> uint8` — OmniCloudMask on (R, G, NIR) at 10 m,
   classes {1,2,3} → invalid, dilate 3 px, resample nearest to 5 m; fall back to
   SCL {0,1,3,8,9,10} when `cloud.method == scl`. Snow stays valid.
4. `scd.geo.coreg.half_shift(a, b) -> (a', b', dx, dy)` — phase correlation on the
   NIR band at 5 m over valid pixels; shift `a` by (−dx/2, −dy/2) and `b` by
   (+dx/2, +dy/2) with bilinear resampling; refuse shifts > `max_shift_px`
   (log and skip correction). `half_shift(b, a)` must return the mirror image of
   `half_shift(a, b)` (test). Optional AROSICS local via `third_party/ard`
   wrapped with the same half-shift contract.
5. `valid = valid_t1 & valid_t2 & inside_both_footprints`.
6. Write `meta.json` with sha256 of the two input products and the config hash
   so the chip builder can detect stale pairs.

Invariant tests (section 9): `preprocess_pair(t1, t2)` and `preprocess_pair(t2, t1)`
produce byte-identical `t1/t2` content (swapped names) and identical `valid.tif`.

### 5.5 Label conventions — `labeling_guide.yaml` and `scd.labels.conventions`

Filled from the Stage-II data in Phase 0 (plan doc Appendix B). Fields the code
reads: class ids, overlap priority, whether removals are labeled, min sizes,
road widths, whether the building rule applies to polygons or change
components, construction-in-progress rule. `conventions.load()` fails if any
field is still `fill`.

### 5.6 Candidate mining — `scd.labels.mining` and `scd.labels.sources.*`

Each source module implements:

```
class Source(Protocol):
    name: str
    def fetch(self, aoi: AoiRow, cache: Path) -> Path          # raw download, cached
    def candidates(self, aoi: AoiRow, pair: PairRow) -> GeoDataFrame
        # columns: geometry (UTM), class_id, source, source_ref, t_event (date or year),
        #          confidence ('auto'), notes
```

| module | what it yields | notes |
|---|---|---|
| `gem.py` | solar: points with capacity, start year → buffered seed polygons (radius from capacity: r = sqrt(cap_MW × 2 ha / π)) | GEM Global Solar Power Tracker xlsx; group rows by location id (phases); start year = latest of a range; status is as-of-research — existence from start/retired year only |
| `kruitwagen.py` | solar polygons as of 2018 with install-date estimates | Zenodo 5005868 |
| `cbra.py` | building: diff of annual 2.5 m rooftop rasters (China 2016–2021) → components ≥ 1,800 m² | Zenodo 7861676; T1/T2 bracket the appearance year with a 1-year buffer |
| `obt.py` | building: Open Buildings 2.5D Temporal annual presence (2016–2023, South/SE Asia) → diff → components ≥ 1,800 m² | via Earth Engine export or the public GCS bucket; the companion project's `fetch_ob_gcs.py` pattern |
| `io_lulc.py` | buildup: Impact Observatory annual 10 m LULC (2017–2023) on Planetary Computer, "Built Area" class diff → clusters | no Earth Engine needed; the default buildup lead source |
| `dynamic_world.py` | buildup: DW built/bare annual medians → Δ clusters | Earth Engine only; optional |
| `osm_history.py` | road: ways created or extended between T1 and T2 (ohsome API), buffered by width class | lead only; the companion project measured 1.3×–29× OSM growth that was mapping, not construction |
| `spacenet7.py`, `oscd.py`, `terracd.py` | external pretraining chips (ablation A12), not candidates | converted to our chip format at 5 m |

`mining.run(aoi, pair, sources) -> candidates/<aoi>/<pair>.gpkg` with one layer
per class plus a `pair` layer (T1/T2 dates, grid). Candidates are never used as
labels without passing through `verification` unless `confidence == 'auto'` is
explicitly allowed by the chip builder (weight 0.5 in the loss).

### 5.7 Verification — `scd.labels.verification`

- `export_for_review(aoi, pair) -> QGIS project (.qgz) + GeoPackage` with T1, T2
  as 8-bit RGB previews, the validity mask, candidate layers styled per class,
  and an empty `added` layer. Swipe via the QGIS "Map Swipe Tool" plugin.
- `import_reviewed(gpkg) -> reviewed/<aoi>/<pair>.gpkg` validating: geometry in
  UTM, class_id in 1–4, confidence ∈ {corrected, double_checked}, annotator,
  timestamp; rejects polygons under the minimum size with a report.
- Companion review app (`third_party/ard` sibling `review_app.py`) is used for
  the accept/reject pass; geometry edits stay in QGIS.
- `agreement(gpkg_a, gpkg_b) -> per-class IoU` for the double-annotated 10%.

### 5.8 Rasterization — `scd.labels.rasterize`

```
rasterize_pair(reviewed_gpkg, pair_product, conventions) -> labels/<aoi>/<pair>/{change.tif, sem_t1.tif, sem_t2.tif}
```

- All geometry ops in the pair's UTM CRS; areas in m².
- Drop building polygons < 1,800 m², solar < 5,000 m², buildup < 2,500 m²,
  roads < 200 m; burn roads as centreline buffers at configured width.
- Burn `change.tif` in priority order (last burned wins → burn lowest priority
  first); `all_touched=False`.
- `sem_t1/sem_t2`: single-date semantic masks when the reviewed file carries
  per-date presence (`present_t1`, `present_t2` booleans per polygon); apply
  `fix_t1`: background at T1 where positive at T2 → 255 at T1.
- Apply `valid.tif`: invalid → 255 in all three label rasters.

### 5.9 Negatives — `scd.labels.negatives`

`stable_sites(stratum, n) -> list[AoiRow]` using CBRA/OBT/IO-LULC stability
(no built change 2017–2024) and no OSM edits in the window; `select_pairs` with
the contrast-season rule; a 5-minute visual check recorded in the registry
(`notes = 'neg_checked:<initials>:<date>'`). Pixels where the image-change gate
fires (mean |x2−x1| over bands > median + 1 std, 3×3 smoothed) are set to 255
in `change.tif` for auto-only negatives (companion project's v3 gate).

### 5.10 Chips and manifest — `scd.data.chips`, `scd.data.manifest`

`build_chips(pairs, labels, cfg) -> chips/v<N>/` cutting 1024 × 1024 windows on a
768 stride where `valid` ≥ 50%, plus `manifest.parquet`:

| column | type |
|---|---|
| chip_id | str `<aoi>_<pair>_r<row>_c<col>` |
| aoi_id, pair_id, stratum, target_type | str |
| t1_date, t2_date, same_orbit, season_match | str/bool |
| tile_type | CHANGE_POS, NOCHANGE_HARD, NOCHANGE_PLAIN, IDENTICAL, NEAR_IDENTICAL |
| classes_present | list[int] |
| change_px_{1..4} | int |
| valid_fraction | float |
| label_confidence | auto, corrected, double_checked |
| split | train, val, stage2, targets |
| row, col, window | ints |

Tile type assignment: CHANGE_POS if any class has ≥ 50 change pixels (≥ 1,250 m²);
else NOCHANGE_HARD if the pair was selected as a contrast pair; else
NOCHANGE_PLAIN. IDENTICAL and NEAR_IDENTICAL are not stored: the dataset
synthesizes them from any chip at draw time.

### 5.11 Synthetic change — `scd.labels.synthetic`

- `build_object_library(chips, manifest) -> objects/v<N>/<class>/*.npz` with keys
  `img_t1`, `img_t2` (6-band uint16), `mask` (uint8), `meta` (class, size_m²,
  stratum, source chip). Buildings ≥ 1,800 m², panel arrays 5,000–50,000 m²,
  road segments 200–1,500 m, cleared pads 2,500–100,000 m². Target ≥ 300
  buildings, ≥ 150 arrays, ≥ 80 roads, ≥ 100 pads, class-balanced across strata.
- `paste(host_t1, host_t2, host_change, objects, rng, cfg) -> (t1, t2, change)`:
  choose target date at random (insertion or removal symmetric); feather 1–2 px;
  per-band gain match within ±10%; optional shadow strip for buildings; never
  over invalid or water (NDWI > 0.3); ≤ 3 objects; class mix per
  `sampler.class_shares`; host labels kept; a 10% share of flat-fill and blurred
  pastes to cover crude synthetic test images.
- `remove(host_t1, host_t2, obj_mask)`: replace the object in one date with
  texture from a neighbouring patch (same chip), label as change of that class.

### 5.12 Splits — `scd.data.splits`

By `aoi_id` within stratum, 80/20 train/val, seeded; `targets` and `stage3`
never in train; `stage2` organiser data as a separate split (`stage2`), used
for validation only until its size is known.

### 5.13 Dataset and sampler — `scd.data.dataset`, `scd.data.sampler`

`PairChipDataset.__getitem__(req: DrawRequest) -> dict(x1, x2, change, sem_t1, sem_t2, valid, weight, meta)`
where `DrawRequest` carries `chip_id`, `tile_type`, `crop_centre`, `synthetic_objs`,
`swap` and the augmentation seed. Tensors are float32 reflectance [0, 1.2],
shape `(6, 512, 512)`; labels int64 with 255 ignore.

`TwoStageSampler(manifest, cfg, epoch_len)` draws `epoch_len` requests:
stratum → tile type → (class for CHANGE_POS) → chip weighted by
`change_px^0.3` → crop centre (within 128 px of a change component with
probability 0.7). IDENTICAL/NEAR_IDENTICAL/SYNTHETIC draws pick a host chip
from the stratum and flag the request. Deterministic given seed and epoch.

`scd.data.transforms`: geometric ops applied identically to x1, x2 and all
labels; asymmetric radiometric per date; relative shift of one date (labels
untouched); temporal swap; cloud paste (marks `valid` 0); band dropout
(identical on both dates). All implemented with torch ops on CPU tensors in
workers; no albumentations dependency required.

---

## 6. Model contracts — `scd.model`

### 6.1 Encoder — `encoder.py`

```
build_encoder(name='convnext_tiny', in_channels=6, init='imagenet') -> nn.Module
# forward(x: [B,6,H,W]) -> list of 4 feature maps at strides 4,8,16,32
```

Stem adaptation rule (explicit, not timm's default): new stem weight `W6` with
`W6[:, 0] = W_rgb[:, 2]` (B02 ← blue), `W6[:, 1] = W_rgb[:, 1]` (B03 ← green),
`W6[:, 2] = W_rgb[:, 0]` (B04 ← red), `W6[:, 3:6] = W_rgb.mean(1, keepdim=True)`.
`init='v3_roads'` loads the frozen v3.1 encoder for the RGBN filters;
`init='dinov3_convnext_tiny'` loads `facebook/dinov3-convnext-tiny-pretrain-lvd1689m`
(gated; ablation A13).

### 6.2 Fusion — `fusion.py`

```
SymmetricFusion(channels_per_level, mode='absdiff_sum')
# forward(f1: list, f2: list) -> list; per level: concat(|f1-f2|, f1+f2) -> 1x1 conv -> LayerNorm(channels-last) -> GELU
```
Invariant: `fusion(f1, f2) == fusion(f2, f1)` exactly (test with
`torch.equal` in fp32, eval mode).

### 6.3 Decoder and heads — `decoder.py`, `heads.py`

`UNetDecoder(in_channels=[96,192,384,768], decoder_channels=[256,128,64,32], full_res_skip=True)`:
top-down with bilinear upsampling and two 3×3 conv-BN-GELU blocks per stage;
the final stage receives, in addition, a 16-channel stride-1 map computed from
`|x1 − x2|` by two 3×3 convs (`full_res_skip`), then upsamples the stride-4
decoder output ×4 and concatenates. Output: `[B, 32, H, W]`.

`ChangeHead`: 3×3 conv → 1 logit. `ClassHead`: 3×3 conv → 4 logits.
`SemanticDecoder`: a second `UNetDecoder(full_res_skip=False)` shared across
dates, applied to `f1` and `f2` separately, → 3×3 conv → 5 logits.

### 6.4 Network — `net.py`

```
class SiameseChangeNet(nn.Module):
    def forward(self, x1, x2) -> dict:
        # 'change': [B,1,H,W], 'cls': [B,4,H,W], 'sem_t1': [B,5,H,W] | None, 'sem_t2': ... | None
```
Invariants (tests): `out(x1,x2)['change']` equals `out(x2,x1)['change']` and
same for `'cls'`; `out(x,x)['change']` is spatially constant (bias-only);
parameters ≈ 45 M.

### 6.5 Losses — `losses.py`

```
ChangeLoss(pos_weight=2.0, dice=True)                         # BCE-with-logits + soft Dice, masked by valid
ClassLoss(class_weights, dice=True)                           # CE + per-class soft Dice on pixels where change GT in {1..4}
SemanticLoss()                                                # CE + Dice, 5-way, masked by valid and 255
TotalLoss = change + cls + lambda_sem * (sem_t1 + sem_t2); per-sample weight from manifest (auto labels 0.5)
```
Class weights = inverse frequency^0.5 over change pixels in the training
manifest, renormalised to mean 1; recomputed when the manifest changes.
Logit clamp ±30 before BCE. Optional `clDice` on the road channel (A10).

---

## 7. Training contracts — `scd.train`

### 7.1 `lit_module.py`

`ChangeLitModule(cfg)` with AdamW param groups (encoder lr 1e-4, rest 3e-4),
cosine with 3-epoch linear warmup, EMA (decay 0.999) evaluated in parallel,
`16-mixed`, gradient clip 1.0, `cudnn.benchmark=True`, TF32 on.

Validation each epoch:

1. Full-chip inference on the `val` split (1024 × 1024, no TTA) accumulating
   `torchmetrics.MulticlassConfusionMatrix(5)` globally and per stratum over
   valid pixels only.
2. Simulated robustness on a fixed `robustness_subset` of 64 val chips
   (`robustness.py`): swapped-order mask agreement, identical-input change
   fraction, pasted-object recovery IoU, near-identical change fraction.
3. Logged: `val/iou_{buildup,building,road,solar}`, `val/weighted_iou`,
   `val/macro_iou`, `val/p_*`, `val/r_*`, `val/stratum_{A..F}_weighted_iou`,
   `val/robust_swap`, `val/robust_identity`, `val/robust_synthetic`,
   `val/robust_near`, `val/leaderboard_proxy = 0.7*weighted_iou + 0.2*mean(robust_*)`.

### 7.2 `metrics.py`

`RubricMetrics` wraps the confusion matrices and exposes `compute() -> dict`;
IoU per class from the accumulated matrix, never a mean of per-chip IoUs.

### 7.3 Checkpointing

`ModelCheckpoint(monitor='val/leaderboard_proxy', mode='max', save_top_k=5)` plus
a milestone callback saving every 10 epochs, plus `last.ckpt`. Early stopping
patience 20 on the proxy.

### 7.4 Calibration — `calibrate.py`

`calibrate(run_dir, val_loader) -> calibration.json {tau, bias:[b1..b4], weighted_iou, per_class}`:
cache `c` and `z` with full TTA on the val split; coordinate ascent — τ in
0.20–0.70 step 0.02, then each `b_k` in −2..2 step 0.1, two passes, maximising
weighted IoU subject to per-class precision ≥ 0.6; then re-run the robustness
simulation and assert it is unchanged.

### 7.5 Ablations — `scripts/14_ablation.py`

Reads `configs/ablations/<id>.yaml`, runs training with the run name
`abl_<id>_<seed>`, and appends a row to `runs/ablations.csv` (proxy, weighted
IoU, per-class IoU, per-stratum IoU, robustness scores, wall-clock). Final
recipe: 3 seeds (42, 43, 44), report mean ± std.

---

## 8. Inference contracts — `scd.infer`

`scd.infer.cli`: `scd-infer --t1 A.tif --t2 B.tif --out outputs/<job> [--config] [--calibration]`

`pipeline.run(t1_path, t2_path, out_dir, cfg, calib) -> Result`:

1. `load_and_identify`: band count, order (from tags or `--bands`), dtype, DN vs
   reflectance (raise on ambiguity unless `--assume-dn`/`--assume-reflectance`),
   SCL presence; both images through the same path as training
   (`preprocess_pair`, which is order-invariant by construction).
2. `identity_guard`: if `max|t1 − t2| < identity_eps` over valid pixels → write
   zeros and the shapefile with no features; return.
3. `tta.predict(net, t1, t2, window=512, stride=384, geometric=8, temporal=True)`:
   average logits over 8 D4 transforms × 2 orders with cosine window blending.
4. `decide`: apply τ and biases; invalid → 0.
5. `postprocess.apply(mask, cfg, t1, t2)`: per-class min area / min skeleton
   length; hole filling; solar closing (3 px disc); water rule (NDWI > 0.3 on
   both dates in a 20 m surround → drop building/solar components; keep buildup
   over former water).
6. `export.write_geotiff(mask, grid, meta)` uint8, nodata 255, DEFLATE, tags:
   model hash, calibration, offsets, band order.
7. `export.write_shapefile(mask, grid)` via `rasterio.features.shapes` per class,
   `simplify(2.5 m)`, fields `class` (int), `area_m2` (float), `conf` (float);
   ESRI Shapefile with `.prj`.
8. `report.json` + `quicklook.png` (T1 | T2 | mask).

Determinism: `torch.use_deterministic_algorithms(True)`, `cudnn.benchmark=False`
at inference; two runs are byte-identical (test).

`scd.baseline.pccd`: wraps the v3.1 road/building/solar checkpoints
(`third_party/v3_roads` and the companion solar v2 when available): run at T1
and T2 on their native inputs, resample masks to the canonical grid, XOR per
class, same post-processing. Used only by `12_flicker_baseline.py` and
`13_evaluate.py --baseline`.

---

## 9. Tests — `tests/`

All tests run on CPU with synthetic data unless marked `@pytest.mark.gpu`.

| test | assertion |
|---|---|
| `test_grid_symmetry` | `canonical_grid([a,b]) == canonical_grid([b,a])` for 50 random footprint pairs across UTM zones; origin on a 5 m multiple |
| `test_coreg_symmetry` | `half_shift(a,b)` offsets are the negation of `half_shift(b,a)`; output arrays swap exactly |
| `test_preprocess_symmetry` (gpu, uses one cached real pair) | `preprocess_pair(t1,t2)` vs `(t2,t1)`: swapped rasters byte-identical, `valid.tif` identical |
| `test_fusion_symmetry` | `torch.equal(fusion(f1,f2), fusion(f2,f1))` |
| `test_net_symmetry` | change and cls logits equal to 1e-5 under swap, eval mode, fp32, batch 2 |
| `test_identity_guard` | `pipeline.run(A, A)` → mask all zero, shapefile has 0 features |
| `test_synthetic_recovery` (gpu, trained ckpt) | 3 pasted objects on a val chip each recovered with IoU ≥ 0.7 and correct class; no other pixels fire |
| `test_near_identical` (gpu, trained ckpt) | JPEG-85 round trip, ±1 px shift, σ=0.01 noise → change fraction < 0.1% |
| `test_determinism` (gpu) | two inference runs byte-identical |
| `test_rasterize` | priority order, min-area filter in m², `fix_t1`, validity → 255 |
| `test_sampler` | over 100k draws, stratum/tile-type/class shares within ±1.5 pts of config; deterministic with seed |
| `test_transforms` | geometric ops identical on x1/x2/labels; shift leaves labels; swap flips tensors; band dropout identical on both dates |
| `test_metrics` | IoU from an accumulated matrix matches a hand computation; per-stratum keys present; proxy formula |
| `test_export` | GeoTIFF values ⊂ {0..4,255}, CRS/transform equal to the canonical grid; shapefile class field ∈ 1..4; polygon count equals component count |
| `test_cloud_parity` (gpu) | our OmniCloudMask wrapper equals the companion `ard` mask on one scene ≥ 99% of pixels |
| `test_config` | invalid band order / shares not summing to 1 / `fill` in labeling guide → validation error |

`make smoke`: builds 4 tiny pairs from fixtures, mines nothing, rasterizes a
fixture GeoPackage, builds chips, trains for 2 steps on CPU, calibrates with a
stub, runs inference, and runs the symmetry/identity/export tests end to end.
Must finish in < 5 minutes.

---

## 10. The centralised notebook — `notebooks/00_pipeline.ipynb`

One notebook runs the whole project; every cell calls a script or package
function, holds no logic of its own, is idempotent (skips work whose outputs
exist and whose config hash matches) and is executable with papermill
(`make notebook STAGE=all`). First cell is the papermill `parameters` cell.

```
# parameters
CONFIG = "configs/default.yaml"
OVERRIDES = []                 # e.g. ["train.epochs=2"]
RUN_NAME = "v1_seed42"
STAGES = "all"                 # or "0,1,2" etc.
CHIPS_VERSION = 1
```

Sections (markdown header → cells):

0. **Setup** — load config, print resolved paths, check GPU, env vars, data root
   free space; `scd.log.banner()`.
1. **Registry** — load `aois.parquet`; show counts per stratum/role/target_type;
   folium map of AOI centres coloured by stratum; registry validation report;
   cell to append AOIs from a CSV.
2. **Scenes** — `01_fetch_scenes.py --query-only` → table of candidate products
   per AOI; then `--download`; progress and failures.
3. **Pairs** — `02_preprocess_pairs.py`; then QC: per-pair valid fraction, coreg
   offsets histogram, a 3-panel quicklook gallery (T1 | T2 | valid) for a random
   sample per stratum; `pytest tests/test_preprocess_symmetry.py` on one pair.
4. **Candidates** — `03_mine_candidates.py --sources gem,cbra,obt,io_lulc,osm_history`;
   per-class candidate counts per stratum; sanity plots.
5. **Review** — `04_export_for_review.py` (writes QGIS projects); a cell listing
   what is pending review; after humans finish, `05_import_reviewed.py` with the
   validation report and the double-annotation agreement table.
6. **Labels** — `06_rasterize.py`; per-class area per stratum; overlay gallery.
7. **Chips** — `07_build_chips.py`; manifest summary: tile types × strata,
   change pixels per class, split sizes; class-weight computation printed.
8. **Object library** — `08_build_object_library.py`; counts per class; a
   gallery of random objects; a cell that pastes 3 objects into a random host
   and shows T1 | T2 | label.
9. **Baseline** — `12_flicker_baseline.py` on NOCHANGE_HARD pairs per stratum
   (PCCD flicker rate) and on val (PCCD rubric IoU); table.
10. **Train** — `09_train.py --run $RUN_NAME` launched as a subprocess with a
    live tail of the metrics CSV; a plot cell for proxy, weighted IoU and the
    four robustness scores by epoch.
11. **Calibrate** — `10_calibrate.py`; prints τ, biases, weighted IoU before/after.
12. **Evaluate** — `13_evaluate.py`; per-class and per-stratum tables; confusion
    matrix; worst-10 val chips gallery (T1 | T2 | GT | pred); robustness table.
13. **Infer one pair** — `11_infer.py` on a chosen pair; show the mask over T2;
    run the swap/identity tests on the same pair and print the result.
14. **Ablations** — build the list from `configs/ablations/`, launch selected
    ones, show `runs/ablations.csv`.
15. **Fine-tune (Stage III)** — `15_finetune.py --config configs/finetune.yaml`;
    V-targets results; forgetting check against V-strata.
16. **Report** — `16_report.py` → `runs/<run>/report.md` (methodology tables,
    figures) for the Approach score.

Every section ends with a `STATUS` cell that writes `runs/pipeline_status.json`
(stage, timestamp, outputs, config hash) so a later session can resume.

---

## 11. Data curation runbook (what exists, what to pull, in what order)

### 11.1 What exists now

| asset | where | use |
|---|---|---|
| 26 Indian AOIs: S2 10 m RGBN + SEN2SR 2.5 m tiles, OSM Tier-A road masks, USEFUL_NEG tiles | previous road project (v3.1) | stratum B (Jaisalmer, Bikaner, Bhuj), C (Karnal, Muzaffarpur, Wardha), D (Shimla, Dehradun), E (Kochi) AOI seeds; v3.1 encoder init (A9); PCCD baseline |
| v3.1 road, building and solar checkpoints | previous project | PCCD baseline; solar pseudo-labels if the transfer test passes |
| Companion project assets (plan doc section 17) | to be received | `ard/` preprocessing, OmniCloudMask integration, solar v2 + `gspt.py`, Open Buildings V3 parquet, OSM extracts 2022/2026, the 42 sample scenes (2022-03 / 2026-06), review app |
| PS-10 Annexure-1 | repo `configs/targets.csv` | the 42 target centres (role `targets`) |
| Stage-II organiser dataset | to be loaded under `external/stage2/` | label conventions; `stage2` validation split |

### 11.2 Order of operations (maps to plan doc phases)

**Phase 0 (4 days)**
1. Load Stage-II data; fill `labeling_guide.yaml`; `conventions.load()` passes.
2. Write the 42 targets into the registry (`role=targets`); pull one cloud-free
   S2 scene per target per year 2019–2026 via Planetary Computer for a visual
   survey; record `target_type` and visible change years in `notes`.
3. Transfer test: run v3.1 road/building/solar and companion solar v2 on 5
   scenes per stratum; keep what reaches ≥ 0.6 precision by eye.
4. Flicker baseline on 10 NOCHANGE_HARD pairs per stratum (needs 3 and 5 below
   for a handful of pairs first; do it on stratum A and B only in Phase 0).

**Phase 1 (1.5 weeks) — dataset v1**
5. Seed the registry to ~250 AOIs: GEM solar sites (start year 2017–2024,
   ≥ 20 MW) in strata A/B/C; airfields from OSM `aeroway=aerodrome` polygons in
   strata A/B/D/E; IO-LULC built-area change clusters (≥ 5 ha) within 30 km of
   the 42 targets and in the analog regions; CBRA diff components in China;
   OBT diff components in Pakistan/Bangladesh/India; stable sites for negatives
   (section 5.9). Deduplicate and cap per section 5.1.
6. Fetch scenes (section 5.2 rules), preprocess pairs, QC quicklooks.
7. Mine candidates; export for review; annotate 150 AOIs (prioritise buildup
   and strata A/B); import; rasterize; build chips v1; build object library v1.
8. Freeze `chips/v1` and the registry snapshot (`registry/aois_v1.parquet`).

**Phase 2 (1.5 weeks) — model v1**: train, calibrate, evaluate, beat PCCD.

**Phase 3 (2 weeks) — dataset v2 + ablations**: registry to ~410 AOIs; second
annotation pass on buildup; A5, A8, A1, A3, A6, A7 (section 12 order).

**Phase 4 (4 days) — inference hardening**: CLI, format detection, edge cases
(all-cloud, partial overlap, 4-band input, DN vs reflectance), runtime < 2 min
per 10 × 10 km chip.

**Phase 5 (1.5 weeks)** — Stage-III fine-tune on the two shapefiles and the 42
targets (plan doc section 12); write-up.

### 11.3 Per-stratum sourcing cheat sheet

| stratum | positives from | negatives from | notes |
|---|---|---|---|
| A plateau | GEM (Qinghai, Tibet), OSM aerodromes, IO-LULC built Δ, OSM highway history (G219/G318/G109), CBRA | snow/no-snow pairs over stable valleys, lake-shore pairs | most annotator time |
| B hot desert | GEM (Xinjiang, Gansu, Ningxia, Rajasthan, Gulf, US SW), CBRA, OBT (India), Microsoft footprints (Gulf, US) | dune fields, oasis crop cycles, salt flats | solar-rich; cap solar-only AOIs so buildup/buildings get draws |
| C plains | OBT (Pakistan, Bangladesh), GEM, OSM history, v3 rural AOIs | flood/dry pairs, kharif/rabi pairs | explicit fallow rule from the guide |
| D mountain | CBRA (Sichuan, Yunnan), OBT (HP, Nepal), OSM aerodromes | cloud/shadow season pairs, landslide areas | cloud-paste augmentation matters here |
| E coast | CBRA (Hainan, Guangdong), OBT (Sri Lanka, Vietnam, Kerala), IO-LULC water→built | aquaculture cycles, anchorage pairs with vessel traffic | dry-season scenes only; water rule |
| F glaciated | few: OSM highway history (new passes) | peak-snow vs bare pairs | almost all negatives |

### 11.4 Licence ledger

Keep `external/LICENSES.md` listing each source and its licence (GEM CC BY 4.0;
Kruitwagen CC BY 4.0; CBRA open via ESSD/Zenodo — confirm; OBT CC BY 4.0;
IO-LULC CC BY 4.0; OSM ODbL; OSCD open; LEVIR-CD/S2Looking research;
TERRA-CD licence not stated — ask or treat as research-only; DINOv3 weights
under the DINOv3 licence, gated). The companion project drops NC/ND sources;
mark which of ours it cannot take back.

---

## 12. Research-informed improvements (adopt / ablate / skip)

Sources are listed at the end of this section.

| # | finding | verdict | where |
|---|---|---|---|
| R1 | A U-Net with a pretrained ImageNet encoder in a Siamese-diff configuration matches or beats BIT, ChangeFormer and TinyCD when all are retrained under one protocol; transformer CD models vary widely across seeds; a common WHU-CD split leaks ~85% of test into train ("A Change Detection Reality Check", Corley et al. 2024) | **adopt**: our architecture stands; report 3 seeds with mean ± std; never copy numbers from papers for the baseline rows | sections 6, 7.5 |
| R2 | SemDINO (June 2026) pairs vision-foundation-model features with CNN detail, uses a *symmetric bi-directional* cross-temporal transformer "to reduce temporal-order bias", and a decoupled head predicting change mask + bi-temporal semantic maps + edges | **adopt the validation**: our symmetric fusion and decoupled heads are the same idea; **ablate** a frozen foundation branch only after the data work (A13) | section 6.2; A13 |
| R3 | DINOv3 ships ConvNeXt-tiny/small/base/large weights distilled from the 7B model (LVD-1689M web images) and a ViT-L/16 trained on 493 M Maxar 0.6 m RGB tiles (SAT-493M); gated licence | **ablate A13a**: `encoder_init=dinov3_convnext_tiny` as a drop-in init (cheapest possible test). **skip** the SAT-493M ViT-L as main encoder: 0.6 m RGB pretraining, 16-px patches = 80 m at 5 m, 300 M params; at most a frozen context branch (A13b) | section 6.1 |
| R4 | ChangeStar2 / STAR trains change detectors from *unpaired* single-date labels and claims generalisation to real pairs | **keep STAR at 0**: the companion project's whole-scene swap pairs mixed with real no-change pairs produced OSCD F1 0.000; if STAR is tried (A7) it must not be mixed with real negatives without the image-change gate | section 5.13, A7 |
| R5 | Changen2 (2024) generates synthetic change pairs with a diffusion model at 0.3–1 m | **skip**: resolution mismatch; our copy-paste library is the 10 m-appropriate version | — |
| R6 | TERRA-CD (May 2026): 5,221 Sentinel-2 10 m pairs (2019 vs 2024) over 232 US/EU cities with a built-area land-cover class and change-type labels; simple LinkNet/U-Net best for multi-class change; HRSCD strategy 4 best for semantic change | **adopt as pretraining source** for the binary change head (A12) alongside OSCD; its built-area transitions also give extra buildup-like positives at 10 m | section 5.6 `terracd.py`; A12 |
| R7 | SEN2SRLite `main` super-resolves the 20 m bands as well; measured SWIR sharpness gain is small (1.08–1.31×) | **ablate A1b** (`sr.method=sen2sr_lite_main`) after A1; default stays bilinear SWIR | section 5.4 |
| R8 | OmniCloudMask works from R/G/NIR only and outputs thick/thin/shadow classes; compiled models cut per-scene time 10–20% | **adopt** as the default cloud mask with `compile_models=True` for batch runs; SCL fallback | section 5.4 |
| R9 | Impact Observatory annual 10 m LULC (2017–2023) is on Planetary Computer without Earth Engine | **adopt** as the default buildup lead source; Dynamic World optional | section 5.6 |
| R10 | TaCo (Nov 2025) adds transition and bi-temporal reconstruction constraints at training time only | **skip for v1**; revisit as a consistency loss between aux semantic maps and the change head if A6 shows the aux head helps | — |

Ablation order (one run each unless noted): A5 (where×what vs 5-way softmax),
A8 (NOCHANGE_HARD 0.15/0.27/0.40), A1 (SR vs bicubic), A3 (5 vs 2.5 m), A6
(aux head on/off), A7 (synthetic 0/0.18/0.30), A2 (normalisation), A11
(full-res skip), A12 (binary-CD pretraining incl. TERRA-CD), A10 (clDice),
A9 (v3 encoder init), A13a (DINOv3 ConvNeXt init), A4 (fusion variants). Final
recipe × 3 seeds.

Sources:
- A Change Detection Reality Check — https://arxiv.org/pdf/2402.06994
- SemDINO: DINOv3-Guided Cross-Temporal Semantic Alignment Network — https://arxiv.org/abs/2606.09772
- DINOv3 ViT-L/16 SAT-493M model card — https://huggingface.co/facebook/dinov3-vitl16-pretrain-sat493m ; DINOv3 ConvNeXt weights — https://huggingface.co/facebook/dinov3-convnext-large-pretrain-lvd1689m (tiny/small/base siblings in the same collection)
- Single-Temporal Supervised Learning for Universal Remote Sensing Change Detection (ChangeStar2) — https://arxiv.org/pdf/2406.15694
- Changen2 — https://arxiv.org/pdf/2406.17998
- TERRA-CD — https://arxiv.org/pdf/2605.14651
- TaCo — https://arxiv.org/abs/2511.20306v1
- sen2sr on PyPI — https://pypi.org/project/sen2sr/
- OmniCloudMask usage — https://omnicloudmask.readthedocs.io/en/latest/usage.html
- cdsetool on PyPI — https://pypi.org/project/cdsetool
- CBRA on Zenodo — https://zenodo.org/records/7861676 ; paper — https://essd.copernicus.org/articles/15/3547/2023/
- GEM Global Solar Power Tracker download — https://globalenergymonitor.org/projects/global-integrated-power-tracker/download-data/

---

## 13. Build order and acceptance criteria

| milestone | build | accept when |
|---|---|---|
| M0 Scaffold | repo layout, `pyproject`, `Makefile`, `CLAUDE.md`, `configs/default.yaml`, `scd.config`, logging, `tests/test_config.py` | `make setup && pytest tests/test_config.py` green; `python -c "import scd"` |
| M1 Geo core | `scd.geo.grid`, `coreg`, `io`, `reproject`, `vector` | `test_grid_symmetry`, `test_coreg_symmetry` green |
| M2 Imagery | `registry`, `cdse`, `scenes`, `sr`, `cloud`, `preprocess`; scripts 00–02; notebook sections 0–3 | one real pair preprocessed end to end; `test_preprocess_symmetry`, `test_cloud_parity` green; quicklook reviewed by the user |
| M3 Labels | `conventions`, `sources/*`, `mining`, `verification`, `rasterize`, `fix_t1`, `negatives`; scripts 03–06; notebook 4–6 | fixture GeoPackage round-trips; `test_rasterize` green; candidates produced for 5 real AOIs per source |
| M4 Chips + synthetic | `chips`, `manifest`, `synthetic`, `dataset`, `sampler`, `transforms`, `splits`; scripts 07–08; notebook 7–8 | `test_sampler`, `test_transforms` green; `make smoke` green; manifest stats cell renders |
| M5 Model | `encoder`, `fusion`, `decoder`, `heads`, `net`, `losses` | `test_fusion_symmetry`, `test_net_symmetry` green; forward/backward on a 2×6×512×512 batch in FP16 under 8 GB |
| M6 Training | `lit_module`, `metrics`, `robustness`, `train.py`; script 09; notebook 10 | 2-epoch run on chips v1 subset logs every metric in 7.1; checkpoints and milestones written; `test_metrics` green |
| M7 Calibration + inference | `calibrate`, `tta`, `postprocess`, `export`, `pipeline`, `cli`; scripts 10–11, 13; notebook 11–13 | `test_identity_guard`, `test_determinism`, `test_export`, `test_synthetic_recovery`, `test_near_identical` green; one 10 × 10 km chip in < 2 min on the target GPU |
| M8 Baseline + report | `baseline.pccd`, `eval.flicker`, `eval.report`, `figures`; scripts 12, 14, 16; notebook 9, 14, 16 | flicker table produced; `ablations.csv` appended by a dry run; `report.md` renders |
| M9 Fine-tune | `configs/finetune.yaml`, script 15, notebook 15 | fine-tune runs from a base checkpoint with the 50/50 sampler; forgetting check implemented |

---

## 14. Known traps (both projects' bug lists, turned into assertions)

- Buffers and areas in a geographic CRS: assert `crs.is_projected` before any
  `buffer`/`area` call (`scd.geo.vector.require_projected`).
- Sentinel-2 baseline ≥ 04.00 offset: read `BOA_ADD_OFFSET` from the L2A MTD
  (`RADIO_ADD_OFFSET` is L1C); raise if absent.
- SEN2SR output has no nodata marker and clips 65535 to bright pixels: derive
  validity from the 10 m source, never from the SR raster.
- GDAL ignores `src_transform` when warping from a dataset band: apply shifts to
  arrays, then write (and assert the shift changed the residual).
- GDAL decimated reads on uncompressed tiled files are single-threaded: use
  strided window reads for statistics.
- Rasterio dataset handles are not thread-safe: one handle per worker.
- A grid chosen from "the first image" is order-dependent: always
  `canonical_grid` over the footprint set.
- Relative normalisation of "subject to reference" is order-dependent: half
  gain/offset each way, or none.
- Shift augmentation must not zero the label on shifted tiles.
- Composites hide single-date artefacts: training pairs are single scenes.
- OSM creation dates are not construction dates.
- Metrics scored on a subset of pixels (strong-vs-background) hide label
  defects: always every valid pixel, every class, accumulated matrix.
- `save_top_k=1` destroys the rare-stratum optimum: top-5 plus milestones.
- A `>` inside text in any generated SVG/JSX QC figure must be escaped (minor,
  from this plan's own drawings).

---

## 15. `CLAUDE.md` (write this file verbatim at M0)

```markdown
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
```

---

## 16. Open items the user must answer before M3

- Stage-II dataset: location, size, format, band order, whether SCL is present,
  label conventions (fills `labeling_guide.yaml`).
- Test chip format from the organisers: bands, dtype, extent, date metadata,
  same-sensor pairs only.
- Companion project handoff (plan doc section 17 checklist): repository access,
  solar v2 checkpoint, OB V3 parquet, OSM extracts, the 42 sample scenes.
- Deadline and whether a third annotator is available.
- CDSE credentials and Planetary Computer key; HF token if A13 is attempted.
