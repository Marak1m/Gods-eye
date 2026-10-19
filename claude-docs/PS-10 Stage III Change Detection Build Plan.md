# PS-10 Stage III: Change Detection Build Plan

Oct 15, 2026 · @YUVRIK

## 1. Executive summary

We are building one Siamese change-detection network for Sentinel-2 image pairs that outputs a 5-valued GeoTIFF (0 no-change, 1 buildup/clearance, 2 building, 3 road, 4 solar) plus a shapefile, trained on a new stratum-balanced bitemporal dataset mined from dated open products over Central and South Asian analog regions, and selected by a proxy of the PS-10 rubric rather than by global pixel IoU.

The previous project subtracted single-date masks from independent specialist models. That design cannot learn temporal consistency, so every pixel where the T1 and T2 predictions disagree becomes false change, and at under 1% true change per chip that collapses precision. The new design looks at both dates at once, decides *where* something changed and *what* it is in two coupled heads, and is order-invariant and identity-consistent by construction because 80% of the robustness score tests exactly those two properties.

| # | Decision | Choice | Why |
| --- | --- | --- | --- |
| 1 | Paradigm | Joint Siamese change detection, not post-classification mask subtraction | Learns which radiometric differences are not change; the old approach has no mechanism for that |
| 2 | Head factorization | Binary change head (where) x 4-way class head on change pixels only (what) | Keeps the 99% background imbalance in one binary head where BCE+Dice is proven; class head trains on a balanced subset; binary head can pretrain on binary CD datasets |
| 3 | Order invariance | Fusion uses only symmetric functions (abs(f1-f2), f1+f2); temporal swap in training and at test time | Test Case-1 (reversed order, 12% of total score) must produce the identical mask |
| 4 | Identity consistency | Change head reads only difference features; hard identity guard zeroes output on pixel-identical inputs | Test Case-3 (identical images, 4% of total score) must produce all zeros |
| 5 | Backbone | ConvNeXt-tiny U-Net, shared weights across dates | Known-good on S2 thin features; transformer decoders regressed on roads at 256 px in v2 |
| 6 | Input | Sentinel-2 L2A, 6 bands (B02, B03, B04, B08, B11, B12), 5 m working grid | SWIR carries the built-vs-bare contrast for the 70% of the score that is buildings and buildup; 5 m makes a 60x30 m building 12x6 px at a quarter of the 2.5 m compute |
| 7 | Data | A few hundred AOIs across six terrain strata, labels mined from dated products (GEM solar tracker, CBRA, Open Buildings Temporal, Dynamic World, OSM history), plus synthetic copy-paste change and mined seasonal no-change pairs | Labels in the new domain are the bottleneck; the 42 eval sites define the distribution, not the dataset |
| 8 | Sampling | Two-stage sampler: stratum shares matched to the eval distribution, then tile-type carve-outs (25-30% hard no-change, 5% identical, 15% synthetic) | Direct extension of the v3 sampler and its calibration lesson |
| 9 | Model selection | Proxy rubric = 0.7 x class-weighted IoU (0.40/0.30/0.20/0.10) + 0.2 x simulated robustness, computed from an accumulated confusion matrix on deployment-style validation pairs | Inversions 1-3 from the road project showed global val IoU picks the wrong checkpoint; here the stakes are higher |
| 10 | LISS-4 | Deferred; a separate model later, warm-started from this one | Different bands and GSD; a unified model would compromise both |

**What changed from the road project**

| Then (roads, v3.1) | Now (PS-10 Stage III) |
| --- | --- |
| One class, one sigmoid, threshold 0.40 | Four change classes, binary change threshold plus four class logit biases |
| 4 bands, 2.5 m via SEN2SR | 6 bands, 5 m (SEN2SR for RGBN then 2x pool; SWIR 20 m to 5 m) |
| Single-date tiles; val on positive tiles | Bitemporal pairs; val on full-AOI pairs including no-change pairs and the three robustness cases |
| 26 Indian AOIs, 7 morphological categories | A few hundred AOIs across 6 terrain strata matched to the Tibet/Xinjiang/Gansu/Pakistan eval geography |
| Minimal colour augmentation | Asymmetric radiometric jitter between dates, relative shift, temporal swap, cloud pasting |
| Checkpoint by global val IoU, save\_top\_k=1 | Checkpoint by proxy rubric, top-5 plus milestones every 10 epochs |

## 2. The problem statement, decoded

PS-10 Stage III scores 70% on per-class change metrics weighted toward buildup and buildings, 20% on three robustness cases that test order invariance and identity consistency, and 10% on the written approach. Every line below is a constraint the design has to satisfy, not background.

**Classes and weights**

| Class ID | Class | Weight | What it looks like at 5 m | Design consequence |
| --- | --- | --- | --- | --- |
| 1 | Clearance area / built-up area | 40% | Leveled or cleared ground, compounds, construction sites, new hardstanding; tens to hundreds of metres across | Change-native: often no single-date class at either date. Needs a direct change-class head, not mask subtraction. Gets the most labeling effort |
| 2 | Stand-alone buildings, 60 m x 30 m or larger | 30% | Hangars, warehouses, barracks blocks, factory halls; 12 x 6 px or more | Minimum-area filter (about 1,800 m2); training labels must exclude small structures or the model learns to fire on villages |
| 3 | Roads | 20% | New highways, access roads, airfield taxiways; 1-3 px wide on bare ground | Thin-feature handling as in v3, but now a fifth of the score, not all of it |
| 4 | Solar farms | 10% | Panel arrays, often square kilometres, strongly regular texture, low NIR and SWIR | Easiest class; abundant dated labels exist; aim for near-perfect precision |

**Robustness cases (20% of total)**

| Case | Share of the 20% | What the organizers do | What the output must be | What that forces |
| --- | --- | --- | --- | --- |
| 1 | 60% | Feed T2 as the first image and T1 as the second | The same mask as the forward order | The raster has no direction field, so change semantics are symmetric (appeared or disappeared). Every pairwise operation in the pipeline must be order-invariant: grid choice, registration reference, normalization, fusion, TTA |
| 2 | 20% | Pair an original image with a synthetic image derived from it | The edits, correctly classed | Everything except the edits is pixel-identical, so a difference-driven model finds them trivially; classification of the pasted object is the real test. Copy-paste training data mirrors this case directly |
| 3 | 20% | Feed the same image twice | All zeros | Zero difference features must yield zero output; a hard identity guard makes this exact rather than probable |

**Metric (70% of total)**

- Mean Jaccard (mIoU), Jaccard per class, precision and recall per class. The class weights most likely apply to a weighted per-class score; whether mIoU is also weighted, and whether it includes no-change, is unknown. We track rubric-weighted IoU, unweighted macro IoU, and per-class P/R, and select on the weighted one.
- Per-class IoU on a chip where a class has no ground truth is zero or undefined the moment we predict a single pixel of it. Rare classes (solar, buildings) need conservative operating points and minimum-area filters.
- Precision and recall are reported separately, so a balanced operating point per class beats a recall-heavy one.

**Output format**

- GeoTIFF mask, pixel values 0-4 (the PS says "05 classes encoded as 1 to 4"; 0 is no change). We write uint8, nodata 255, same CRS as the input, on the canonical grid.
- Shapefile with one feature per polygon and an integer attribute 1-4. Polygonized from the raster per class after post-processing.

**Inputs**

- Sentinel-2 and LISS-4, both stated. LISS-4 (G, R, NIR at 5.8 m; no blue, no SWIR) is deferred to a separate model (section 16). This plan covers the Sentinel-2 model.
- Two images of specified dates over a chip at one of the 42 sample targets (section 3). Date range, chip size, band order and whether a pair can mix sensors are unknown (section 15).

**Stage III specifically**

- "Fine tune the model on Area of Interest" defined by two shapefiles; datasets remain those of Stage II. The base model is trained on the broad dataset in section 7, then fine-tuned per section 12.

## 3. Evaluation geography

Two thirds of the 42 sample targets are arid: 14 on the high plateau (Tibet, Qinghai, Ngari, Aksai Chin) and 14 in hot desert or oasis (Xinjiang, Hexi corridor, Ningxia, the Karachi coast). The rest are temperate mountain (6), irrigated plains (4), tropical coast (2) and glaciated high mountain (2). Settings are read from the coordinates and are unverified; confirm each with imagery during the Phase 0 survey.

| Stratum | Count | Share | Samples | Typical targets |
| --- | --- | --- | --- | --- |
| A. High plateau / cold desert | 14 | 33% | 1, 4, 5, 6, 8, 9, 10, 13, 14, 18, 19, 22, 36, 42 | Airfields and compounds in valley floors, border settlements, plateau solar parks, roads on bare ground |
| B. Hot desert / oasis / loess | 14 | 33% | 2, 3, 11, 12, 23, 24, 25, 30, 31, 32, 33, 34, 35, 38 | Desert airfields, launch and test complexes, utility solar, industrial zones at oasis edges |
| C. Irrigated alluvial plain | 4 | 10% | 26, 39, 40, 41 | Power plants, industrial and airbase construction amid cropland (Pakistani Punjab, Bangladesh) |
| D. Temperate / subtropical mountain | 6 | 14% | 15, 21, 27, 28, 29, 37 | Valley airfields and towns under forested slopes (Sichuan, Yunnan, Himachal, Nyingchi, Chumbi valley) |
| E. Tropical coast | 2 | 5% | 7, 20 | Naval and air facilities, reclamation, aquaculture (Hainan) |
| F. Glaciated high mountain | 2 | 5% | 16, 17 | High-altitude roads and posts; mostly a no-change stratum (Karakoram, Shaksgam) |

**Site list (approximate settings from coordinates)**

| Sample | Lat | Lon | Stratum | Likely setting |
| --- | --- | --- | --- | --- |
| 1 | 29.3517 | 89.3110 | A | Shigatse area airfield and valley, Tibet |
| 2 | 36.6278 | 105.0011 | B | Gansu-Ningxia loess highland |
| 3 | 40.3962 | 99.7771 | B | Dingxin area desert airfield, Gansu |
| 4 | 27.9998 | 89.2250 | A | Khangmar-Gamba valley near the Sikkim-Bhutan border |
| 5 | 29.9097 | 90.3498 | A | Lhasa-Yangbajain corridor |
| 6 | 36.4000 | 94.7900 | A | Golmud, Qaidam basin |
| 7 | 18.2188 | 109.5024 | E | Sanya coast, Hainan |
| 8 | 28.8755 | 89.5437 | A | Gyantse area, Tibet |
| 9 | 28.4232 | 92.3518 | A | Lhuntse county valley airfield, Tibet |
| 10 | 36.3375 | 100.4846 | A | Gonghe / Talatan solar park, Qinghai |
| 11 | 43.9957 | 89.1364 | B | Fukang-Jimsar desert fringe, Xinjiang |
| 12 | 37.8730 | 105.8631 | B | Zhongwei-Zhongning, Ningxia |
| 13 | 29.3241 | 83.9866 | A | Zhongba county near the Nepal border, Tibet |
| 14 | 30.1530 | 81.3261 | A | Purang (Burang) valley, Tibet |
| 15 | 32.2664 | 76.2829 | D | Kangra, Himachal Pradesh |
| 16 | 36.1687 | 76.3939 | F | Shaksgam valley, Trans-Karakoram |
| 17 | 35.5843 | 77.4621 | F | Karakoram Pass / Siachen sector |
| 18 | 30.3959 | 81.1346 | A | Purang-Lipulekh approach, Tibet |
| 19 | 33.6562 | 80.4495 | A | Rutog county, Ngari (east of Pangong) |
| 20 | 18.4967 | 109.9856 | E | Lingshui coast airfield, Hainan |
| 21 | 27.2343 | 89.0173 | D | Chumbi valley / Doklam area |
| 22 | 27.9807 | 90.8986 | A | Lhozhag county near the Bhutan border |
| 23 | 42.1001 | 92.5190 | B | Hami basin south, Xinjiang |
| 24 | 42.2746 | 92.5627 | B | Hami basin south, Xinjiang |
| 25 | 40.2456 | 96.5097 | B | Yumen-Guazhou wind and solar belt, Gansu |
| 26 | 24.0661 | 89.0468 | C | Rooppur nuclear plant site, Bangladesh |
| 27 | 23.4350 | 103.3426 | D | Mengzi basin airfield, Yunnan |
| 28 | 32.1368 | 104.8447 | D | Northern Sichuan mountain valley |
| 29 | 32.4917 | 105.5851 | D | Guangyuan area, Sichuan |
| 30 | 40.2279 | 97.3773 | B | Yumen east, Gansu |
| 31 | 40.3322 | 98.4990 | B | Jiuquan north, Gansu |
| 32 | 40.9136 | 100.2138 | B | Dongfeng / Jiuquan Satellite Launch Center |
| 33 | 41.6390 | 86.3500 | B | Korla east, Xinjiang |
| 34 | 43.8840 | 87.8550 | B | Urumqi east (Midong), Xinjiang |
| 35 | 42.8480 | 93.4860 | B | Hami city, Xinjiang |
| 36 | 36.2070 | 94.7620 | A | Golmud south solar belt, Qinghai |
| 37 | 29.2200 | 94.1850 | D | Nyingchi-Mainling valley, Tibet |
| 38 | 25.0540 | 66.9680 | B | Karachi-Hub arid coast, Pakistan |
| 39 | 29.9900 | 71.0850 | C | Muzaffargarh, Punjab, Pakistan |
| 40 | 32.6290 | 72.7050 | C | Jhelum-Sargodha belt, Punjab, Pakistan |
| 41 | 33.8140 | 72.4130 | C | Attock-Kamra, Punjab, Pakistan |
| 42 | 36.3530 | 78.0260 | A | Shahidulla, northern Aksai Chin |

**Pseudo-change sources per stratum** (what the no-change negatives must cover)

| Stratum | Pseudo-change the model will see | Mitigation in data |
| --- | --- | --- |
| A plateau | Snow cover, lake-level swings (Pangong, Qinghai Lake), braided river shifts, long winter shadows on relief, grassland greening | Snow/no-snow and wet/dry no-change pairs from the same valleys; shadow pairs at low sun angle |
| B hot desert | Dune migration, dust events, salt-flat moisture, oasis crop cycles, ephemeral water, bright construction spoil that is not a target class | Seasonal no-change pairs at oasis margins; dune-field negatives |
| C plains | Kharif/rabi phenology, monsoon flooding, river course change, brick-kiln season, fallow fields that look cleared | Flood-season vs dry-season negatives; explicit rule on fallow vs clearance from the Stage-II labels |
| D mountain | Cloud and haze, slope shadow, landslides, terrace cycles, forest cut | Cloud-pasting augmentation; landslide negatives |
| E tropical coast | Persistent cloud, aquaculture pond cycles, sea state, ships at anchorages the size of target buildings, reclamation that is real buildup | Water mask rule in post-processing; reclamation kept as positive |
| F glaciated | Snow and ice extent, glacier surface change, moraine lakes | Almost entirely negative pairs across extreme seasons |

## 4. Design principles

Six rules decide every later choice; when two options tie on accuracy, the one that satisfies more of these wins.

1. **Symmetric by construction.** No module may see the pair in a fixed order. Fusion uses only symmetric functions of the two feature maps; preprocessing picks grids and registration references without reference to which image came first; training swaps the order at random; inference averages both orders. Reversed order must produce a bit-identical mask.
2. **Identity-consistent.** Identical inputs produce identical features, zero difference features, and an all-zero mask. A hard guard enforces the last step so the property is exact, not learned.
3. **Where before what.** A binary change head answers where the landscape changed; a class head answers what the change is, trained only on changed pixels. The background imbalance lives in one binary problem that we already know how to train.
4. **Change is learned from pairs, not inferred from single-date masks.** Single-date labels are used as auxiliary supervision and as a source for synthetic pairs, never as the primary signal for the change output.
5. **Data over architecture.** The backbone is the known-good one from v3. Effort goes into labels in the right terrain, hard no-change negatives, and synthetic change, in that order.
6. **Validate the way you are scored.** Model selection uses full-AOI pairs that include no-change pairs and the three robustness cases, aggregated into one confusion matrix, weighted as the rubric weights the classes. Tile-level positive-only IoU is a training-time diagnostic only.

## 5. Architecture

The network is a Siamese ConvNeXt-tiny U-Net: both dates pass through one shared encoder, the four feature levels are fused with symmetric functions, a change decoder produces a binary change logit and four class logits at full resolution, and an auxiliary single-date decoder produces semantic maps for each date during training only.

&#91;embedded content: network architecture · shared encoder, symmetric fusion, where and what heads\]

The encoder runs once per date with the same weights; fusion keeps only swap-invariant combinations, so reversed order and identical inputs behave correctly by construction; the class head only ever learns from pixels the change head is trained to flag.

**Encoder.** `convnext_tiny` from timm (28 M parameters), ImageNet-1k weights, stages at strides 4, 8, 16, 32 with 96, 192, 384, 768 channels. The 4x4 stem is widened to 6 input channels: B04, B03, B02 take the pretrained R, G, B filters; B08, B11, B12 take the mean of the three. Weights are shared between dates (true Siamese), so f1 and f2 are comparable by construction. Initializing from the v3 road encoder instead of ImageNet is ablation A9, not the default.

**Fusion, per level l = 1..4.**

```latex
F_l = \mathrm{Conv}_{1\times1}\big(\,[\;|f_l^{1} - f_l^{2}|\;,\; f_l^{1} + f_l^{2}\;]\,\big)
```

Both terms are invariant to swapping the dates, so the change decoder cannot learn an order. Concatenation \[f1, f2\] is forbidden. The difference term carries where things changed; the sum term carries context (what kind of terrain this is) without carrying order. A 1x1 convolution, LayerNorm and GELU bring the channel count back to the encoder's.

**Change decoder.** A U-Net decoder (256, 128, 64, 32 channels) over the fused levels, plus one stride-1 shallow skip computed from |x1 - x2| on the raw 6-band input so that 1-2 px roads are not lost to the stride-4 stem (ablation A11 measures its value). Output: a 32-channel map at the 5 m grid feeding two heads.

- **Change head (where):** 3x3 conv to 1 logit c, sigmoid. Trained with BCE (pos\_weight 2.0, tuned in 1.5-3.0) plus soft Dice, exactly the v3 road recipe applied to "any change".
- **Class head (what):** 3x3 conv to 4 logits z. Trained only on pixels whose ground truth is change, with class-weighted cross-entropy (inverse-frequency^0.5 within change pixels, renormalized) plus per-class soft Dice. It never sees the 99% background, so no class is suppressed by it.

**Auxiliary semantic decoder (training only).** A second U-Net decoder, shared across dates, applied to f1 and to f2 separately, producing 5-way logits (background, buildup, building, road, solar). Supervised where single-date labels exist; ignored elsewhere. Purpose: regularize the encoder toward object semantics on single-date data, which is far more plentiful than paired data. If Stage-II conventions give no usable single-date meaning for buildup, the head is 4-way and buildup is left to the change heads alone. Dropped entirely if it does not help on val (ablation A6).

**Loss.**

```latex
\mathcal{L} = \underbrace{\mathrm{BCE}_{w_+}(c, y_c) + \mathrm{Dice}(\sigma(c), y_c)}_{\mathcal{L}_{\text{chg}}} \;+\; \underbrace{\frac{1}{|\Omega_c|}\sum_{p\in\Omega_c} w_{y_p}\,\mathrm{CE}(z_p, y_p) + \sum_{k=1}^{4}\mathrm{Dice}_k\big|_{\Omega_c}}_{\mathcal{L}_{\text{cls}}} \;+\; \lambda_{\text{sem}}\big(\mathcal{L}_{\text{sem}}^{T_1} + \mathcal{L}_{\text{sem}}^{T_2}\big)
```

where Omega\_c is the set of ground-truth change pixels, lambda\_sem = 0.3, and all terms ignore pixels masked as cloud, shadow or nodata in either date. clDice on the road channel of the class head is ablation A10 (the v3 implementation exists). Lovász is not used from scratch (closed in v2).

**Inference-time output.**

```latex
M(p) = \mathbb{1}\big[\sigma(c_p) > \tau\big]\cdot\Big(1 + \arg\max_{k}\,(z_{p,k} + b_k)\Big)
```

tau and the four biases b\_k are tuned once on the deployment-style validation set by coordinate ascent on the rubric-weighted IoU (section 9). c and z are averaged over 8 geometric transforms and both temporal orders before thresholding.

**Shapes and cost.** Training tiles 512 x 512 px at 5 m (2.56 km), batch 16, FP16 mixed precision; about 45 M parameters in total; roughly 1.3x the per-tile cost of the v3 road model at the same tile size because the encoder runs twice. Inference windows 512 px with 128 px overlap.

**Post-classification baseline.** The v3 specialists (road, building, solar) are kept as a baseline only: run at T1 and T2, XOR the masks, apply the same post-processing. It yields the flicker-rate diagnostic (section 11) and the "before" number for the methodology write-up. It is not part of the submitted model.

**Rejected alternatives**

| Alternative | Why not |
| --- | --- |
| Post-classification mask subtraction (previous approach) | No mechanism to learn temporal consistency; false change from any per-date disagreement; no single-date class for clearance |
| Single 5-way softmax head (no-change plus 4 classes) | Workable, but the class decision competes with the 99% background; kept as ablation A5 |
| Four independent sigmoid class heads | Reintroduces pixel conflicts and needs a priority rule; gains nothing on imbalance that the where/what split does not |
| Concatenation fusion \[f1, f2\] | Learns an order; fails Case-1 unless the swap augmentation happens to cancel it, which is not guaranteed |
| Transformer decoders (ChangeFormer, BIT, SegFormer-style) | SegFormer regressed to 0.26 IoU on thin roads at 256 px in v2; the all-MLP decoder blurs 1-2 px features |
| Mamba backbones (ChangeMamba) | Unproven in our stack; Approach is 10% of the score and does not reward novelty over a well-ablated CNN |
| Sentinel-1 SAR channels | Extra acquisition per test pair, impossible for the synthetic case, adds speckle and geometry pseudo-change |
| One Siamese model per class | 4x training and inference, pixel conflicts at merge, no shared notion of change |
| ConvNeXt-small or base | +0.005 IoU for 2x compute in v2; revisit only after the data work is done |

## 6. Input specification

Every pair enters the network as two 6-band, 5 m, reflectance-scaled rasters on one canonical grid, with a validity mask; the same code path runs in training and at test time so the model never sees a preprocessing it was not trained on.

| Step | Specification | Order-invariant? |
| --- | --- | --- |
| Source product | Sentinel-2 L2A (BOA reflectance), bands B02, B03, B04, B08 at 10 m and B11, B12 at 20 m; SCL layer for cloud, shadow, snow, nodata | n/a |
| Canonical grid | 5 m in the chip's UTM zone; origin snapped to a multiple of 5 m; extent = intersection of the two footprints. Computed from the two footprints as a set, never from "the first image" | Yes |
| Resampling RGBN | SEN2SR-Lite x4 (10 m to 2.5 m, deterministic) then 2x2 mean to 5 m. Ablation A1 compares plain bicubic 10 m to 5 m | Yes (per image) |
| Resampling SWIR | Bilinear 20 m to 5 m | Yes (per image) |
| Radiometric scaling | reflectance / 10000, clipped to \[0, 1.2\], no per-scene stretch. Absolute reflectance is informative for materials; per-scene percentiles shift when a large change (a solar park) enters the chip. Ablation A2 tests per-scene percentile normalization and a symmetric IR-MAD (the companion project's relative normalization with half the fitted gain and offset applied to each date, so the result is order-invariant) | Yes (per image) |
| Co-registration | Global shift estimated by phase correlation on B08 between the two images, using the companion project's ard co-registration step (AROSICS local where the grid allows); half the shift applied to each image (symmetric). Residual local misalignment is handled by the shift augmentation in training. AROSICS local registration is an option if residuals exceed 1 px on the validation pairs | Yes (half-shift) |
| Validity mask | Invalid if either date is thick cloud, thin cloud or shadow under OmniCloudMask (the companion project's choice; it needs only R, G and NIR, so it serves LISS-4 later), or nodata or saturated; SCL classes 8, 9, 10, 3, 0 and 1 are the fallback when the mask cannot run. Snow (11) stays valid: it is a pseudo-change the model must learn, not a hole. Invalid pixels are ignored in the loss and output 0 at test time | Yes (union) |
| Storage | uint16 x10000 uncompressed tiles as in v3 (the 5x read-speed lesson); validity mask uint8; labels uint8 | n/a |
| Band order check | Assert band count, dtype and value range at load; a 4-band or BGRN-ordered test chip must be detected and mapped explicitly, with a manual override flag | n/a |

**Why 5 m and not 2.5 m.** At 5 m a 60 x 30 m building is 12 x 6 px, a solar park is thousands of pixels, and plateau roads (8-12 m) are 2 px. The 2.5 m grid was justified in v3 by 4-6 m Indian lanes; those are not targets here. 5 m is 4x cheaper per tile, which matters because the dataset will be rebuilt several times, and halves super-resolution artifacts that differ between dates. Ablation A3 trains the final recipe at 2.5 m once; if roads or building boundaries gain more than 0.01 rubric-weighted IoU, switch.

**Why SEN2SR at all, if we pool back to 5 m.** Pooling a 2.5 m super-resolved image to 5 m is not the same as bicubic 10 m to 5 m: the SR network has learned S2 point-spread deconvolution. Whether that survives the pooling is exactly what A1 measures; bicubic is the fallback and removes one dependency.

**Test-time variants to detect.** The test chips may arrive as single-date GeoTIFFs without SCL. In that case cloud masking runs OmniCloudMask on the 10 m bands, with SCL as the cheap fallback, and the scaling must detect uint16 DN versus float reflectance (the 8-hour Mumbai bug from v2sr).

## 7. Dataset construction

The dataset is a set of bitemporal chips with a 5-valued change label, built by mining dated open products for candidate change, verifying it by hand, adding synthetic change, and deliberately collecting no-change pairs across seasons; the Stage-II organizer data defines the label conventions and serves as a second validation set.

### 7.1 Units

| Unit | Definition | Stored as |
| --- | --- | --- |
| AOI | A 10 x 10 km box (5.12 km for dense sites) in one stratum, with a known or suspected change event and dates | Row in the AOI registry: id, stratum, target type, centre, T1/T2 candidate dates, source of the lead |
| Pair | Two single-date S2 scenes over an AOI, same relative orbit where possible, preprocessed per section 6 | Two 6-band uint16 rasters, one validity mask |
| Chip | A 1024 x 1024 px (5.12 km) window from a pair with its label raster | Tile files plus a manifest row |
| Manifest row | chip id, AOI id, stratum, T1 date, T2 date, orbits, tile type, classes present, change pixels per class, label source, label confidence (auto / corrected / double-checked), split | Parquet |

Training crops 512 x 512 at random from the 1024 chips (geometric augmentation applied identically to both dates and the label), so chip overlap is not pre-baked.

### 7.2 Label conventions, taken from Stage II first

Nothing is annotated until the labeling guide (Appendix B) is filled in from the Stage-II data. The conventions that change the model's behaviour:

- Is a pixel that was class X at T1 and background at T2 (demolition, decommissioning) labeled as change X? The symmetric design assumes yes. If the GT only marks appearances, reversed order still produces the same mask, so the design holds either way, but the training labels must match.
- Overlap priority: a large building inside a newly built compound is building (2) or buildup (1)? Our default priority is building > solar > road > buildup, pending the GT.
- Buildup/clearance extent: the whole leveled area, the fenced compound, or only the hard-surfaced part? Is a construction site with foundations "clearance" or "building"?
- The 60 x 30 m rule: measured on the polygon or on the change component? Attached rows of smaller buildings that together exceed the size?
- Road width to burn, and whether unpaved tracks count.
- Solar: panel arrays only, or the fenced park including substations?

### 7.3 Tile taxonomy

| Tile type | Definition | Role in training |
| --- | --- | --- |
| CHANGE\_POS(k) | Verified real change of class k, with the stratum recorded | Primary positives; balanced across k by the sampler |
| NOCHANGE\_HARD | Real pair, no man-made change, with strong seasonal, snow, water, shadow or phenology difference | Teaches what is not change; 25-30% of batches. Admitted only after the stability check; auto-mined pairs without a visual pass get loss weight 0.5; pixels where the mean \|x2 - x1\| exceeds one std above the chip median carry no negative target (the companion project's image-change gate) |
| NOCHANGE\_PLAIN | Real pair, no change, similar seasons | Low-cost background; 5% |
| IDENTICAL | The same scene twice | Teaches the exact zero; 2-3% |
| NEAR\_IDENTICAL | The same scene with noise, JPEG-like compression, slight blur or a 1-2 px shift applied to one copy, label all zero | Guards Case-2 and Case-3 against trivial artifacts; 2-3% |
| SYNTHETIC(k) | Copy-paste insertion or removal of a class-k object into a real pair | Mirrors Case-2; expands rare classes; up to 15% |
| STAR | Two unrelated single-date tiles, label = XOR of their class masks (optional, low weight) | Default share 0: the companion project's whole-scene swap pairs taught its head that the same place means no change (OSCD F1 0.000); kept only as ablation A7 |

### 7.4 Label sources and change mining per class

| Class | Primary sources | How candidates are mined | What a human does |
| --- | --- | --- | --- |
| 4 Solar | GEM Global Solar Power Tracker (points, capacity, start year); Kruitwagen et al. 2021 global PV polygons (as of 2018); India AI solar polygons (CDLA-Permissive); the companion project's gspt.py, solar\_dataset.py and solar v2 model for pseudo-labels, with its confuser list (salt works, aquaculture, tank cascades, ash ponds, dark industrial roofs) as hard negatives | Filter tracker to start year 2017-2024 and capacity >= 20 MW in the analog regions; T1 = cloud-free scene 6-12 months before the start year, T2 = 6-12 months after | Digitize or correct the array boundary on T2 (a brightness/NIR rule gives a first polygon); confirm T1 is bare |
| 2 Building | CBRA annual rooftop maps for China 2016-2021 (2.5 m); Google Open Buildings 2.5D Temporal 2016-2023 (Pakistan, Bangladesh, India); SpaceNet 7 monthly footprints (global); Microsoft footprints for T2-era semantics outside China | Difference consecutive annual layers; keep components >= 1,800 m2 in a projected CRS; T1/T2 scenes bracket the appearance year with a one-year buffer on each side | Confirm the building exists at T2 and not at T1; trim the footprint to the roof; reject clusters of small houses that merged |
| 1 Buildup / clearance | Dynamic World per-scene class probabilities (built, bare); GHSL built-up surface; WSF Evolution; OSM aeroway and landuse polygons as site leads | Annual median of built and bare probabilities; flag clusters where built rises > 0.3 or bare rises > 0.3 while vegetation or other falls; at every airfield and military-style compound in the analog regions, compare 2018, 2021 and 2024 scenes directly | Digitize the cleared or built-up extent on T2 per the Appendix B definition; this class takes the most annotator time and gets it |
| 3 Road | OSM full history via ohsome (ways created or extended between T1 and T2); the v3 road model if it passes the transfer test | Buffer new ways by width class (10 m highways, 6 m access roads); keep segments > 200 m | Confirm construction date from imagery: the companion project measured OSM road growth of 1.3x to 29x between 2018 and 2023 that was mapping, not construction, so a creation date is a lead only; delete tracks that are not visible at 5 m |

### 7.5 Synthetic change

An object library is cut from verified positives: buildings >= 1,800 m2 with their 6-band pixels at both dates, panel-array patches, road segments with their burned mask, cleared pads. For a host pair in the same stratum, an object is pasted into one date chosen at random (so insertions and removals occur equally), edges feathered over 1-2 px, per-band gain matched to the host within +-10%, an optional shadow strip offset for buildings; the pasted footprint is labeled with its class and the host's own labels are kept. Removal is the mirror: an object present at both dates is replaced in one date by background texture copied from a neighbouring patch. Rules: at most 3 objects per chip, class mix tracking the rubric weights, never pasted over water or cloud, never more than 15% of training batches.

### 7.6 Negative mining

For each stratum, pick sites that CBRA, Open Buildings or Dynamic World show as stable over 2017-2024 and that have no OSM edits in the window; pair the scenes with the largest seasonal contrast (peak snow vs bare, flood vs dry, green vs fallow, low-sun winter vs summer). A 5-minute visual check per pair confirms no construction. The glaciated stratum is collected this way in full. Target: at least as many NOCHANGE\_HARD chips as CHANGE\_POS chips in strata A, B and F.

### 7.7 Verification workflow

1. Candidate generation writes a GeoPackage per AOI with T1 and T2 rasters, the candidate change polygons, their source and confidence.
2. A QGIS project template loads the pair with a swipe tool and the candidate layer; annotators correct geometry and class, delete false candidates, and add missed change. The companion project's review\_app.py (Flask and SQLite, frozen samples, per-edit log) runs the accept/reject pass across the team once its Verify auto-tick bug is fixed; geometry edits stay in QGIS.
3. 10% of AOIs are annotated twice; agreement below 0.8 IoU on a class triggers a guide revision.
4. Every polygon carries source, annotator, confidence and date; the manifest inherits them so training can weight or exclude auto-only labels (ablation A8).

Budget: correcting a candidate layer takes 10-15 minutes per AOI pair; annotating from scratch takes 45-60. Mining is what makes a few hundred AOIs feasible.

### 7.8 Rasterization rules

- Work in the chip's UTM zone; compute areas and buffers in metres.
- Burn in priority order building > solar > road > buildup (pending Appendix B), all\_touched false, at the 5 m grid.
- Drop building polygons under 1,800 m2 and road segments under 200 m before burning.
- Apply the pair's validity mask to the label (invalid pixels become ignore, value 255). For the auxiliary semantic head, apply the companion project's fix\_t1 rule: a pixel that is background at T1 but positive at T2 becomes ignore at T1, because OSM-derived T1 labels lag reality.

### 7.9 Splits

- Split by AOI within each stratum, 80/20 train/val, so validation measures unseen sites rather than unseen blocks of seen sites.
- The Stage-II organizer data is a second validation set in full until its size is known; if it is large, 70% of its AOIs join training and 30% stay out.
- The 42 sample targets and the two Stage-III shapefiles are excluded from base training and used only in section 12.

### 7.10 Size targets

| Stratum | AOIs | Pairs per AOI | Real chips (approx.) | Of which NOCHANGE\_HARD |
| --- | --- | --- | --- | --- |
| A plateau / cold desert | 120 | 2-3 | 1,200 | 500 |
| B hot desert / oasis | 120 | 2-3 | 1,200 | 450 |
| C alluvial plain | 50 | 2 | 400 | 150 |
| D temperate mountain | 50 | 2 | 400 | 150 |
| E tropical coast | 30 | 2-4 (clouds) | 250 | 100 |
| F glaciated | 40 | 2 | 300 | 280 |
| Total | 410 |  | \~3,750 chips of 1024 px (\~15,000 crops of 512 px per epoch at 4 crops per chip) | \~1,630 |

Synthetic chips are generated on the fly from the object library, not stored, so their count is set by the sampler fraction rather than by disk.

## 8. Additional AOIs needed

We need about 410 training AOIs drawn from regions that share terrain, construction style and pseudo-change with the 42 targets; the targets themselves are reserved for fine-tuning and the final check. Shares follow the eval distribution with a floor per stratum so no stratum is learned from a handful of sites.

| Stratum | AOIs | Analog regions (where to look) | Lead sources for change |
| --- | --- | --- | --- |
| A High plateau / cold desert | 120 | Tibet AR: Lhasa-Shigatse-Nyingchi corridor, Shannan and Ngari border counties, Nagqu; Qinghai: Golmud-Delingha-Dulan (Qaidam), Gonghe-Guinan (Hainan prefecture), Yushu; Ladakh: Leh, Nyoma, Chushul, the DBO road corridor; Mongolia: Ulaanbaatar fringe, Oyu Tolgoi and Tavan Tolgoi mines; Kyrgyz Naryn and Tajik Pamir (Murghab) | GEM solar (Qinghai is dense), OSM aeroway polygons, Dynamic World built/bare, OSM highway history (G219, G318, G109 upgrades) |
| B Hot desert / oasis / loess | 120 | Xinjiang: Kashgar, Hotan, Aksu, Korla, Hami, Turpan, Karamay; Hexi corridor: Dunhuang, Jiuquan, Jiayuguan, Zhangye, Wuwei; Inner Mongolia: Alxa, Bayannur, Ordos; Ningxia: Zhongwei, Wuzhong, Yinchuan; Rajasthan and Gujarat: Bhadla, Jodhpur, Jaisalmer, Bikaner, Khavda, Kutch (existing v3 AOIs reusable); Egypt: Benban, the New Administrative Capital; Saudi Arabia, UAE, Oman: Sakaka, Al Dhafra, Sudair, Duqm; US Southwest: Nevada, Arizona, Mojave solar and logistics parks; Chile: Atacama solar | GEM solar, CBRA (China), Open Buildings Temporal (India), Microsoft footprints (US, Gulf), OSM aeroway, Dynamic World |
| C Irrigated alluvial plain | 50 | Pakistani Punjab: Multan, Bahawalpur, Muzaffargarh, Sargodha, Jhelum, Faisalabad; Sindh: Sukkur, Hyderabad, Karachi fringe; Bangladesh: Pabna, Rajshahi, Khulna, Dhaka fringe; the v3 rural\_agricultural AOIs (Karnal, Muzaffarpur, Wardha); Nile delta | Open Buildings Temporal, GEM solar and thermal plants, OSM history, Dynamic World |
| D Temperate / subtropical mountain | 50 | Sichuan: Guangyuan, Mianyang, Panzhihua, Yibin; Yunnan: Mengzi, Dali, Lijiang; Guizhou: Guiyang fringe; Himachal and Uttarakhand (v3 Shimla and Dehradun reusable, plus Kangra, Mandi); Nepal: Pokhara, Kathmandu fringe; Bhutan: Punatsangchhu | CBRA, Open Buildings Temporal (India, Nepal), OSM aeroway, Dynamic World |
| E Tropical coast | 30 | Hainan: Haikou, Wenchang, Sanya, Lingshui, Danzhou; Guangdong and Guangxi coast: Zhanjiang, Beihai, Yangjiang; Fujian: Xiamen fringe; Vietnam: Da Nang, Cam Ranh; Sri Lanka: Hambantota, Colombo port city; Kerala coast (v3 Kochi reusable) | CBRA, Open Buildings Temporal (Sri Lanka, Vietnam, India), OSM, Dynamic World; cloud-free pairs from dry-season months only |
| F Glaciated high mountain | 40 | Karakoram (Baltoro, Siachen approaches), Shaksgam, western Kunlun, Nyainqentanglha, Pamir glaciers, Zanskar | Almost all negatives; the few positives from OSM highway history (new high passes) and Dynamic World bare-up clusters |

**Target-type coverage** (minimum verified events in training, across strata)

| Target type | Minimum events | Classes exercised | Where they concentrate |
| --- | --- | --- | --- |
| Airfield expansion (aprons, hangars, taxiways, hardstands) | 40 | 1, 2, 3 | OSM aeroway polygons in strata A, B, D, E; the 2017-2024 upgrade wave across western China |
| Compound / base / industrial-park construction | 100 | 1, 2 | Dynamic World built-up clusters near existing facilities in A and B |
| Utility-scale solar park | 150 | 4, 1 | GEM tracker in A, B, C |
| Large stand-alone buildings (warehouses, plants, hangars) | 300 | 2 | CBRA and Open Buildings Temporal in all strata |
| New road segments > 1 km | 80 | 3 | OSM history in A, B, D |
| Land reclamation and port works | 15 | 1, 2 | E, from Dynamic World water-to-built |
| Border and remote settlements | 30 | 1, 2, 3 | A and D, from Dynamic World and manual leads |

**AOI selection procedure**

1. For each stratum, list candidate centres from the lead sources above (tracker rows, aeroway polygons, built-up change clusters, OSM history segments).
2. Deduplicate within 10 km; cap any single metropolitan area at 8 AOIs so one city does not become a stratum.
3. For each candidate, find S2 scenes: same relative orbit, cloud fraction under 10% over the AOI, within the bracketing windows; reject if no valid pair exists.
4. Assign to the target-type table and stop collecting a type once its minimum is met and its stratum quota is full; spend the remaining budget on NOCHANGE\_HARD pairs for A, B and F.
5. Record everything in the AOI registry before any download so the dataset can be rebuilt from the registry alone.

**Training shares.** The sampler draws strata at 30% A, 30% B, 12% C, 12% D, 8% E, 8% F (section 9). This is close to the eval shares, with E and F lifted above their 5% so the tropical cloud regime and the snow-dominated negatives get enough draws to matter.

## 9. Training recipe

Training keeps the v3 optimizer and schedule, replaces the sampler's categories with strata and tile types, inverts the augmentation policy from minimal to deliberately asymmetric, and selects checkpoints by a proxy of the rubric.

### 9.1 Sampler (two-stage, stratum x tile type)

1. **Stage 1, stratum:** draw a stratum with probabilities A 0.30, B 0.30, C 0.12, D 0.12, E 0.08, F 0.08.
2. **Stage 2, tile type within the stratum:** CHANGE\_POS 0.45, NOCHANGE\_HARD 0.27, NOCHANGE\_PLAIN 0.05, IDENTICAL 0.02, NEAR\_IDENTICAL 0.03, SYNTHETIC 0.18, STAR 0 (ablation A7 only). Stratum F uses CHANGE\_POS 0.10 and NOCHANGE\_HARD 0.70.
3. **Within CHANGE\_POS:** draw the class first with probabilities 0.35 buildup, 0.30 building, 0.20 road, 0.15 solar (rubric weights, with solar lifted because its pixels are abundant but its events are not), then a chip containing that class weighted by change\_pixels^0.3 as in v3.
4. **Crop:** a random 512 window; for CHANGE\_POS the window is centred within 128 px of a change component with probability 0.7.

The fractions above are starting points calibrated so that useful-negative time (about 30% of batches) matches what the v3 calibration found necessary for rare terrain; ablation A8 sweeps the NOCHANGE\_HARD share at 0.15, 0.27 and 0.40.

### 9.2 Augmentation

| Family | Applied to | Parameters | Purpose |
| --- | --- | --- | --- |
| Geometric, shared | Both dates and the label identically | D4 (rotations by 90 deg, flips) p=1; rotation +-30 deg p=0.5; scale 0.9-1.1 p=0.3 | Orientation invariance; mild scale robustness |
| Temporal swap | The pair | p=0.5 | Order invariance in the learned weights, on top of the symmetric fusion |
| Radiometric, asymmetric | Each date independently | Brightness +-12%, contrast +-12%, gamma 0.85-1.15, per-band gain +-5%, additive haze (blue-weighted) p=0.3, Gaussian noise sigma 0.005-0.02 | Global radiometric differences between acquisitions are not change |
| Relative shift | One date | Integer shift of +-1 to +-2 px in x and y, p=0.3, label kept as is | Residual misregistration is not change |
| Cloud and shadow paste | One date | Paste a cloud or shadow blob from a library, mark it invalid in the mask, p=0.15 | Model learns to ignore invalid pixels; mask semantics are exercised |
| Band dropout | Both dates identically | Zero B11 and B12 with p=0.05 | Degrades gracefully if a test chip arrives without SWIR |

Not used: mixup, cutmix, channel shuffle, large colour jitter (all hurt thin features in v2); any augmentation that treats the two dates differently in geometry other than the small shift.

### 9.3 Optimizer and schedule

- AdamW, lr 3e-4 for decoders and heads, 1e-4 for the encoder, weight decay 1e-4, cosine to 1e-6 over 100 epochs, 3-epoch linear warmup (the Siamese fusion layers start from scratch; warmup was unnecessary in v3 but costs nothing).
- Batch 16 x 512 px, FP16 mixed, logit clamp +-30 before BCE (v3 bug fix retained), cudnn benchmark and TF32 on.
- EMA of weights (decay 0.999) evaluated alongside the raw weights; the better one on the proxy is kept.
- One epoch = 15,000 crops; about 25 minutes on one H100 at this tile size, so a full run is roughly 40 hours including validation. Early stopping patience 20 on the proxy.

### 9.4 Checkpointing and selection

- `save_top_k=5` on `leaderboard_proxy` plus a milestone every 10 epochs (the v3.1 lesson: save\_top\_k=1 destroyed the epoch-75 rare-terrain optimum).
- `leaderboard_proxy = 0.7 * weighted_iou + 0.2 * robustness`, where `weighted_iou = 0.40 IoU_buildup + 0.30 IoU_building + 0.20 IoU_road + 0.10 IoU_solar` from one confusion matrix accumulated over all deployment-style validation pairs, and `robustness` is the mean over the three simulated cases of section 11. Logged alongside: unweighted macro IoU, per-class P and R, per-stratum weighted IoU.
- Implemented in `on_validation_epoch_end` with torchmetrics confusion matrices accumulated per class and per stratum.

### 9.5 Post-training calibration

1. Run the selected checkpoint with full TTA over the validation pairs and cache c and z.
2. Coordinate ascent: sweep tau in 0.20-0.70 step 0.02, then each bias b\_k in -2 to +2 step 0.1, two passes, maximizing weighted\_iou subject to per-class precision >= 0.6.
3. Freeze tau and b\_k in the inference config; re-run the three robustness simulations to confirm they are unaffected.

### 9.6 Pretraining the change head (optional, ablation A12)

Before the main run, train the encoder, fusion and change head only (binary) for 20 epochs on OSCD at 10 m resampled to 5 m, LEVIR-CD and S2Looking downsampled to 5 m, and DynamicEarthNet impervious-surface change at 5 m. Then start the main run from those weights. Expected effect: faster convergence of the where head; measured by proxy at epoch 20 of the main run.

## 10. Inference pipeline and robustness guarantees

Inference is a fixed, deterministic sequence from two GeoTIFFs to a GeoTIFF mask and a shapefile; every step is either per-image or a symmetric function of the pair, and three unit tests assert the robustness cases before any submission.

1. **Load and identify.** Read both rasters; assert band count and order, dtype and value range; detect uint16 DN vs float reflectance; read SCL if present, else run s2cloudless on the 10 m bands.
2. **Canonical grid.** UTM zone from the footprint centroid; 5 m pixels; origin snapped to a multiple of 5 m; extent = intersection of the two footprints. The same grid results whichever image is listed first.
3. **Resample.** Per image: SEN2SR-Lite on B02/B03/B04/B08 then 2x2 mean, bilinear for B11/B12, scale to reflectance.
4. **Co-register.** Phase correlation on B08; shift each image by half the estimated offset in opposite directions; log the offset.
5. **Validity mask.** Union of cloud, shadow, nodata and saturation over both dates.
6. **Identity guard.** If the two preprocessed arrays are equal everywhere (max absolute difference below 1e-6 in reflectance units over the valid area), write an all-zero mask and skip the network.
7. **Tile and predict.** 512 px windows, stride 384; for each window run 8 geometric transforms x 2 temporal orders = 16 passes; average c and z in logit space; blend windows with a cosine weight.
8. **Decide.** Apply tau and the four biases; set invalid pixels to 0.
9. **Post-process per class.**
   - Building: remove components under 1,800 m2; fill holes under 200 m2.
   - Solar: remove components under 5,000 m2; morphological closing with a 3-px disc to merge panel rows.
   - Buildup: remove components under 2,500 m2; fill holes under 500 m2.
   - Road: remove components whose skeleton is shorter than 150 m; no closing (the v2 lesson: gap-closing after skeletonization destroys recall).
   - Water rule: a building or solar component whose 20 m surround is water (NDWI > 0.3) at both dates is removed (ships, floating structures); buildup over water at T1 but not at T2 is kept (reclamation).
   - Thresholds above are starting values; each is tuned on validation by the same coordinate-ascent routine as tau.
10. **Write raster.** uint8, values 0-4, nodata 255, LZW, input CRS, canonical transform; tags record model hash, tau, biases, offsets.
11. **Write vectors.** `rasterio.features.shapes` per class on the post-processed mask, polygons simplified with a 2.5 m tolerance, fields `class` (int 1-4), `area_m2`, `conf` (mean sigma(c) over the polygon); ESRI Shapefile with .prj.

**Guarantees and the tests that enforce them**

| Property | Why it holds | Unit test (runs on 10 validation pairs before every submission) |
| --- | --- | --- |
| Reversed order gives the same mask | Grid, mask, registration and fusion are symmetric; both orders are averaged in TTA | `assert array_equal(run(A, B), run(B, A))` on raster and on vector geometry |
| Identical inputs give zeros | Difference features are zero; the identity guard makes it exact | `assert run(A, A).max() == 0` |
| Synthetic edits are recovered | Difference-driven change head; copy-paste training | Paste 3 library objects into A, assert each is detected with the right class at IoU >= 0.7 and no other pixels fire |
| Near-identical inputs give zeros | NEAR\_IDENTICAL training tiles | JPEG-quality-85 round trip, +-1 px shift and sigma 0.01 noise on A; assert change fraction < 0.1% |
| Deterministic | No stochastic layers; fixed cudnn algorithms at inference (benchmark off, deterministic on) | Two runs are bit-identical |

Runtime per 10 x 10 km chip: about 2,000 x 2,000 px at 5 m, 36 windows x 16 passes, roughly 40 s on an H100 and 6-8 minutes on a 16 GB consumer GPU, dominated by SEN2SR. Switching A1 to bicubic removes most of that.

## 11. Validation protocol and ablations

Validation reproduces the scoring: whole AOI pairs including no-change pairs, one confusion matrix per class accumulated over everything, the three robustness cases simulated on the same pairs, and results broken out per stratum so a regression on the plateau cannot hide behind a gain on plains.

**Three validation sets**

| Set | Content | Used for |
| --- | --- | --- |
| V-strata | 20% of AOIs per stratum from section 7, whole 10 x 10 km pairs, about 80 AOIs | Checkpoint selection, calibration, ablations |
| V-stage2 | The organizer's Stage-II data, untouched | Sanity check that our label conventions match theirs; reported, never used for selection |
| V-targets | The 42 sample targets with pairs we build (section 12), held-out dates | Final in-domain check after fine-tuning |

**Metrics, all from accumulated confusion matrices (never averaged per tile)**

- Per class: IoU, precision, recall; rubric-weighted IoU; unweighted macro IoU; per-stratum versions of each.
- Robustness: for every V-strata pair, run (A, B), (B, A), (A, A) and (A, A') where A' carries three pasted objects. Score = mean of: 1 - fraction of pixels differing between (A,B) and (B,A); 1 - change fraction on (A,A); mean IoU of the pasted objects on (A,A').
- Flicker rate (diagnostic, section 13 week 1): on 10 NOCHANGE\_HARD pairs per stratum, the fraction of valid pixels flagged as change by the PCCD baseline and by the new model. This is the single number that explains why the architecture changed. The companion project already measured it on Indian metro pairs: about 2% built-up and 0.4% roads false change under post-classification comparison; Phase 0 adds the plateau and desert values.

**Ablation list** (each is one training run on the final recipe unless noted; run in this order, stop when the deadline requires)

| ID | Question | Variants | Decision rule |
| --- | --- | --- | --- |
| A5 | Where x what heads vs single 5-way softmax | Factorized (default) vs 5-way | Keep the higher proxy; report both |
| A8 | NOCHANGE\_HARD share | 0.15 / 0.27 / 0.40 | Pick by proxy; expect the v3 pattern (val IoU falls, deployment precision rises) |
| A1 | SR then pool vs bicubic to 5 m | SEN2SR+pool (default) vs bicubic | Bicubic unless SR gains > 0.01 weighted IoU |
| A3 | 5 m vs 2.5 m working grid | 5 m (default) vs 2.5 m, same recipe | 2.5 m only if gain > 0.01 and inference time stays under 2 min per chip |
| A6 | Auxiliary semantic decoder | On (default, lambda 0.3) vs off | Drop if no gain; it costs 15% training time |
| A7 | Synthetic and STAR shares | SYNTHETIC 0 / 0.15 / 0.30; STAR 0 / 0.03 | Pick by proxy and by the Case-2 simulation specifically |
| A2 | Normalization | Fixed reflectance (default) vs per-scene percentile | Keep fixed unless per-scene wins on strata A and B |
| A11 | Full-resolution difference skip | On (default) vs off | Judge on road IoU |
| A12 | Binary CD pretraining | With vs without | Judge at epoch 20 and at the end |
| A10 | clDice on the road channel | On vs off | Judge on road IoU and on road precision |
| A9 | Encoder init | ImageNet (default) vs v3 road encoder | Judge at epoch 20; unlikely to matter |
| A4 | Fusion variant | abs-diff + sum (default) vs abs-diff only vs abs-diff + product | Keep the default unless a variant wins by > 0.005 |

Each ablation produces a row in the methodology section with proxy, weighted IoU, per-class IoU, per-stratum IoU and the three robustness scores. The methodology write-up (10% of the score) is built from this table and the flicker-rate diagnostic.

## 12. Stage-III fine-tuning

Fine-tuning adapts the base model to the two organizer shapefiles and the 42 targets with a low learning rate, local hard negatives and a forgetting check, so that in-domain precision rises without losing the broad model's recall.

**Data for this phase**

- Inside the two shapefiles: every Stage-II pair, plus pairs we build ourselves from S2 scenes over the shapefile extents, labeled per section 7 with extra annotator time (these are the highest-value labels in the project).
- Over the 42 targets: a 10 x 10 km AOI each, 3-4 S2 dates spanning 2019-2026, pairs chosen to bracket visible change, plus one NOCHANGE\_HARD pair per target. One date pair per target is held out as V-targets and never trained on.
- Local negatives: for each target, the two most seasonally different no-change pairs we can find within 30 km.

**Recipe**

1. Start from the calibrated base checkpoint and its tau and biases.
2. Freeze encoder stages 1-2 for the first 5 epochs, then unfreeze; lr 3e-5 decoders, 1e-5 encoder; 20 epochs cosine; batch 16.
3. Sampler: 50% fine-tune data (local positives and negatives), 50% base data at the section 9 shares, so the broad distribution stays in the gradient.
4. Select by the proxy on V-targets and V-strata jointly (0.6 / 0.4); refuse any checkpoint whose V-strata weighted IoU falls more than 0.02 below the base model.
5. Re-run section 9.5 calibration on V-targets only; expect tau to move and the biases to stay close.
6. Re-run the five robustness unit tests on the fine-tuned model.

**What not to do**

- Do not fine-tune on specific date pairs that could be the test pairs; the test dates are unknown and overfitting to one acquisition's radiometry costs more than it gains.
- Do not drop the base data from the batches; a 20-epoch run on a few dozen AOIs alone will forget the plains and coastal strata.

## 13. Work plan

Six phases over about eight weeks on one H100: specification and the flicker diagnostic first, data before model, model v1 before any ablation, inference hardening before fine-tuning, and the write-up in parallel from week 6. Durations are elapsed time for a two-person team; the deadline (open question, section 15) sets how many ablations survive.

&#91;embedded content: work plan · six phases with their gates\]

Each gate is a yes/no check, and the two tinted phases are where annotation throughput, not compute, sets the pace.

| Phase | Duration | Deliverables | Gate to the next phase |
| --- | --- | --- | --- |
| 0. Specification | 4 days | Labeling guide (Appendix B) filled from Stage-II; AOI registry schema; Stage-II data loaded and inspected; 42-target imagery survey; PCCD flicker-rate measurement; specialist transfer test (one afternoon) | Guide signed off by both team members; flicker number in hand |
| 1. Dataset v1 | 1.5 weeks | AOI registry at \~250 AOIs; mining scripts for all four classes; QGIS correction template; first 150 corrected AOIs; negatives for A, B, F; synthetic object library v1; preprocessing adapted from the companion ard package (section 17), shared with inference | Dataset v1 frozen with manifest; V-strata defined |
| 2. Model v1 | 1.5 weeks | Siamese network, losses, sampler, augmentation, proxy monitor; the five robustness unit tests; first full training run; calibration | Model v1 beats the PCCD baseline on proxy and on flicker rate |
| 3. Dataset v2 and ablations | 2 weeks | Registry to \~410 AOIs; second annotation pass on buildup; A5, A8, A1, A3, A6, A7 run; recipe fixed | Ablations closed; recipe frozen |
| 4. Inference hardening | 4 days | End-to-end CLI from two GeoTIFFs to GeoTIFF plus shapefile, built on check\_pair.py outputs; band and dtype detection; runtime under 2 minutes per chip on the target GPU; tests green on V-strata and on hand-made edge cases (all-cloud, partial overlap, 4-band input) | Robustness tests green; outputs validated in QGIS |
| 5. Fine-tune and write-up | 1.5 weeks | Stage-III fine-tune (section 12); V-targets results; methodology document with the ablation table, flicker diagnostic and failure modes; submission package | Fine-tuned model accepted by the forgetting check; write-up reviewed |

**Compute** (H100 hours): dataset preprocessing \~30 (SEN2SR dominates); model v1 \~45; ablations 6 x 45 = 270 (A3 at 2.5 m is \~150 on its own); fine-tune \~15; validation and calibration \~20. Total \~400 hours, of which ablations are the adjustable part.

**Annotation** (person-hours): \~410 AOIs x 2 pairs x 15 minutes of correction = \~200 hours, plus \~40 hours for the double-annotated 10% and guide revisions. This is the critical path; a third annotator halves phases 1 and 3.

## 14. Risks and failure modes

The three risks that can sink the score are a mismatch between our buildup definition and the organizers', false change on seasonal terrain, and an order- or identity-breaking bug in preprocessing; each has a specific detector and a specific mitigation.

| Risk or failure mode | Where it shows | Detector | Mitigation |
| --- | --- | --- | --- |
| Our buildup/clearance definition differs from the Stage-II GT | 40% of the metric | V-stage2 buildup IoU far below V-strata buildup IoU | Phase 0 guide from Stage-II; re-annotate buildup in phase 3 if V-stage2 disagrees |
| Seasonal false change: snow, lakes, shadow, phenology | Precision on strata A, B, C, F | Flicker rate on NOCHANGE\_HARD pairs; per-stratum precision | NOCHANGE\_HARD share (A8); more negatives in the failing stratum; asymmetric radiometric augmentation |
| Misregistration between dates flags every edge | Thin false change outlining unchanged buildings and roads | Visual on V-strata; change fraction on NEAR\_IDENTICAL shift test | Half-shift registration; shift augmentation; AROSICS local if residual > 1 px |
| Order or identity bug in preprocessing (grid from first image, registration to first image, scene-relative normalization) | Case-1 and Case-3 scores | The five unit tests | Symmetric-by-design rules in section 6; tests run before every submission |
| Ships and floating structures classed as buildings at coastal targets | Building precision at samples 7 and 20 | Water-rule hits logged per chip | Water rule in post-processing; coastal negatives with vessel traffic |
| Small-building false positives in villages and towns | Building precision in strata C and D | Component size histogram of predictions | 1,800 m2 filter; training labels exclude small structures; class bias b\_building tuned with the precision floor |
| Clearance confused with fallow or harvested fields | Buildup precision in stratum C | Per-stratum buildup P/R | Explicit fallow negatives; the Appendix B rule on fallow; SWIR helps (bare soil vs compacted hardstanding) |
| Construction in progress: clearance at T1, building at T2 | Class confusion between 1 and 2 | Confusion matrix off-diagonal 1<->2 | Follow the Stage-II priority rule; if GT marks both, label the building footprint 2 and the rest 1 |
| Cloud-contaminated test pairs on Hainan | Recall at samples 7 and 20 | Validity fraction per chip | Cloud pasting augmentation; invalid pixels output 0; nothing else can be done at inference |
| Synthetic test image has statistics unlike anything in training (GAN texture, flat fills) | Case-2 | Case-2 simulation with flat-fill and texture-swapped objects added to the test suite | Include flat-fill and blurred-paste objects in the synthetic library at low share |
| Test chip arrives in an unexpected format (4 bands, BGRN order, DN scaling, no SCL) | Everything | Load-time assertions | Explicit detection and override flags; s2cloudless fallback; DN detection |
| Annotation throughput slips | Phases 1 and 3 | Registry progress vs plan | Prioritize buildup and the A and B strata; accept auto-only labels for solar; add an annotator |
| Label noise from annual products (CBRA, Open Buildings) bracketing the wrong year | Building recall and precision | Double-annotated 10% agreement | One-year buffer on each side; verification pass; confidence field used to down-weight auto-only labels (A8) |
| Overfitting to the 42 targets in fine-tuning | V-strata regression after fine-tune | Forgetting check in section 12 | 50% base data in fine-tune batches; refuse checkpoints below the threshold |

## 15. Open questions

Seven questions change the plan materially; the first four are answered by reading the Stage-II data, the rest need the organizers or the team.

- [ ] Stage-II label conventions: are removals labeled? What is the overlap priority? What exactly is clearance? Is the 60 x 30 m rule applied to polygons or change components? (Fills Appendix B.)
- [ ] Stage-II dataset size and format: how many pairs, what chip size, which bands, is SCL included, are both sensors present? (Decides V-stage2 vs train use in section 7.9.)
- [ ] Test chip format: band order, dtype, chip extent, whether a pair can mix Sentinel-2 and LISS-4, and whether acquisition dates are in the metadata. (Decides the load-time detection in section 10.)
- [ ] Does the mIoU in the rubric include the no-change class, and do the class weights apply to it? (Decides which number to select on; we log both.)
- [ ] Deadline for Stage III submission. (Decides how many ablations run and whether A3 at 2.5 m is affordable.)
- [ ] Compute available: one H100 continuously, or shared? (Section 13 assumes one H100.)
- [ ] Annotators: two team members, or can a third be added for phases 1 and 3? (Annotation is the critical path.)

## 16. Deferred: LISS-4

LISS-4 gets its own model later, warm-started from the Sentinel-2 model, rather than a unified input; the notes here are what that later work needs to know so nothing in this plan blocks it.

- LISS-4 Mx: three bands (green 0.52-0.59, red 0.62-0.68, NIR 0.77-0.86 um) at 5.8 m, 70 km swath; no blue, no SWIR. Products are typically DN, not reflectance; orthorectification quality varies, so co-registration residuals will be larger than for S2.
- Warm start: copy the full S2 model; rebuild the stem for 3 channels by taking the B03, B04, B08 filters; resample LISS-4 to the same 5 m canonical grid (5.8 to 5 m bilinear) so decoders and heads transfer unchanged.
- Data: the same AOI registry and labels apply; only the imagery changes. Pairs need LISS-4 scenes bracketing the same events, which depends on archive access (Bhoonidhi or the Stage-II data).
- Band dropout in section 9.2 (zeroing SWIR with p=0.05) keeps the S2 model mildly tolerant of missing SWIR, which helps the warm start converge but is not a substitute for LISS-4 training.
- Everything in sections 10 and 11 (pipeline, tests, metrics) is sensor-agnostic once the loader knows which sensor it has.

## 17. Reuse from the companion project

About two weeks of Phase 1 and Phase 4 come ready-made from the companion project (the parallel Sentinel-2 / LISS-4 change-detection effort documented in its project record of 16 Oct 2026); the table lists what to ask for, where it is used, and what must change before use, and the checklist is the handoff. Its change head is not reused: the where x what design and the evaluation protocol in this plan stand.

| Ask for | Files and artefacts | Used in | Must change before use |
| --- | --- | --- | --- |
| Preprocessing package | `ard/` (`sensors/sentinel2.py`, `sensors/liss4.py`, `harmonize.py`, `inputs.py`, `core/*`), `check_pair.py`, its 22 GDAL-free unit checks, the `report.json` schema | Sections 6 and 10, steps 1-5 | Canonical grid from both footprints instead of the reference raster's grid; co-registration as a half-shift; IR-MAD symmetric or off (A2); our 5 m UTM grid instead of the 10 m reference grid; band names written into outputs |
| Cloud masking | OmniCloudMask integration and its GPU path (27 s per date) | Section 6 validity mask; the LISS-4 model later | None; add the SCL fallback |
| Solar pipeline | `gspt.py`, `solar_dataset.py`, `train_solar.py`, the solar v2 checkpoint with its `norm` file and band order, the confuser AOI list, the dust augmentation | Section 7.4 solar mining, pseudo-labels at T1 and T2, hard negatives | Trained on single 2020 scenes; check it on Qinghai and Gansu parks before trusting its pseudo-labels; GSPT traps (phases not plants, start year is the latest of a range, status is as-of-research) are already handled in `gspt.py` |
| Building label tools | The min-rotated-rectangle building metrics in `make_class_vectors.py`, `burn_footprints.py` (footprint cover >= 25% at 4x supersampling), `fetch_ob_gcs.py`, `vector_audit.py` (NDVI audit of large polygons) | Section 7.4 building mining; section 7.8 rasterization | Keep only the strong class (>= 30 x 60 m); our priority order and 5 m grid; Open Buildings V3 has no China coverage, so CBRA fills strata A, B, D, E |
| Review app | `review_app.py`, the `verification.csv` and `.json` schema | Section 7.7 | Fix the Verify auto-tick; add a T1/T2 swipe view for change pairs |
| Data on disk | The 42 sample scenes (2022-03 and 2026-06, full \~100 km scenes), stage-2 metro composites 2018 and 2023 with their SR products, OSM theme extracts for snapshots 180101, 220101, 230101 and 260101, Open Buildings V3 parquet per area, `ob.log` coverage results | Section 12 V-targets and local negatives; section 7.6; stratum C in section 8 | March-vs-June pairs are seasonal pseudo-change by construction: negatives and fine-tune only, with same-season dates pulled for positives; composites are not single-date test chips; EPSG:4326 grids reprojected to UTM 5 m |
| Tilers and inference utilities | Bright-negative mining in `tile_10m.py`, the md5-cell split, Hann blending and D4 TTA in `infer_bb.py`, `vectorize.py` | Sections 7.9, 9.1 and 10 | Add temporal-swap TTA; a `class` integer field and ESRI Shapefile output |
| Change experiments | `train_siamese.py`, `siamese_test.py`, `eval_oscd.py`, the `sbb_v1` and `sbb_v2` results | Section 11 baseline row; negative results in the methodology | Not reused as a model: the gain/loss head is directional and fails Case-1 |
| Measured facts and bug list | SR sharpness per band (B2/B3/B4/B8 1.24x, SWIR 1.08-1.31x), the SR nodata trap, float 2023 exports, `BOA_ADD_OFFSET` vs `RADIO_ADD_OFFSET`, GDAL ignoring `src_transform` on band sources, strided reads, buffering in a geographic CRS | Section 14; load-time assertions in section 10 | None; each becomes a unit test |
| PCC flicker floor | About 2% built-up and 0.4% roads false change on real metro pairs | Section 11 diagnostic | Repeat on strata A and B |

**Handoff checklist**

- [ ] Read access to the repository (`ard/`, label scripts, review app) and the data folders, with a manifest of what exists per sample and per metro
- [ ] The solar v2 checkpoint, its `norm_*.json`, band order and the processing-baseline offset convention
- [ ] The Open Buildings V3 parquet files and `ob.log` for the 42 samples (which samples returned nothing)
- [ ] The OSM theme PBFs for 2022 and 2026 over the sample regions
- [ ] Licence alignment for shared data: the companion project uses permissive licences only and drops ChinaPV; mark in Appendix A which of our sources it cannot take back
- [ ] Run `check_pair.py` on one plateau pair and one Hainan pair; read per-band radiometry, not the overall grade (its grade is band-count-biased)
- [ ] Run the swap and identity unit tests from section 10 against `check_pair.py` outputs before adopting it

**Not inherited**

- The gain/loss change head and post-classification comparison as the primary method.
- The density-derived built-up class: the PS clearance class is change-native and defined by the Stage-II labels.
- Whole-scene swap pairs and OSM label differences as change targets.
- GEE median composites on a lat/lon grid as training imagery (13% pixel anisotropy at Delhi; composites blur single-date artefacts the test will contain).
- Checkpoint selection on strong-vs-background F1, which hid a 67% label-defect rate until a 300-chip manual review.

## Appendix A. External data source catalog

Entries are from memory and marked approximate; each is opened, its licence confirmed and its link added here during Phase 0 before any download.

| Source | Provides | Resolution | Temporal depth | Coverage | Role here | Licence (approx.) |
| --- | --- | --- | --- | --- | --- | --- |
| Sentinel-2 L2A (Copernicus Data Space) | 13-band BOA reflectance, SCL mask | 10 / 20 / 60 m | 2017-present (L2A global from late 2018) | Global | All imagery | Copernicus open |
| SEN2SR-Lite (TACO Foundation) | Super-resolution model for B02/B03/B04/B08 | 10 to 2.5 m | n/a | n/a | RGBN resampling (A1) | Open |
| s2cloudless | Cloud probability from 10 m bands | 10 m | n/a | n/a | Cloud mask fallback when SCL is absent | Open |
| AROSICS | Global and local co-registration | n/a | n/a | n/a | Registration if residual > 1 px | Open |
| Dynamic World | Per-scene LULC class probabilities incl. built and bare | 10 m | 2015-present | Global | Buildup candidate mining; stability check for negatives | CC BY 4.0 |
| GHSL GHS-BUILT-S | Built-up surface fraction | 10 m (2018 epoch), 100 m multi-epoch | 1975-2030 epochs | Global | Coarse buildup confirmation | CC BY 4.0 |
| WSF 2019 and WSF Evolution (DLR) | Settlement footprint; annual settlement growth | 10 m; 30 m | 2019; 1985-2015 | Global | Coarse buildup confirmation, stability | Open |
| CBRA (China Building Rooftop Area) | Annual building rooftop maps | 2.5 m | 2016-2021 | China | Building change mining in strata A, B, D, E | Open (ESSD / Zenodo) |
| Google Open Buildings 2.5D Temporal | Annual building presence, height, counts | 4 m | 2016-2023 | Africa, South and Southeast Asia, Latin America | Building change mining in strata C, D, E, and Indian desert AOIs | CC BY 4.0 |
| Microsoft Global ML Building Footprints | Building polygons, single date | Vector | One epoch per region | Most countries; check China | T2-era single-date semantics; synthetic object library | ODbL |
| SpaceNet 7 | Monthly Planet mosaics with building footprints | 4 m | 24 months, 2018-2020 | About 100 AOIs worldwide | Building change pretraining and library | CC BY-SA 4.0 |
| GEM Global Solar Power Tracker | Plant locations, capacity, status, start year | Point | Updated continually | Global, >= 20 MW (some >= 1 MW) | Solar change mining | CC BY 4.0 |
| Kruitwagen et al. 2021 global PV inventory | PV facility polygons with estimated install dates | Vector | As of 2018 | Global | Solar polygons for pre-2018 parks; library | CC BY 4.0 |
| OpenStreetMap via ohsome (full history) | Roads, aeroway, landuse with edit timestamps | Vector | 2007-present | Global | Road change mining; airfield and compound leads | ODbL |
| OSCD (Onera Satellite Change Detection) | S2 bitemporal pairs, binary urban change | 10 m | 2015-2018 pairs | 24 cities | Binary change-head pretraining (A12) | Open |
| LEVIR-CD, S2Looking, WHU-CD | VHR bitemporal building change | 0.5-0.8 m | Various | Various | Downsampled to 5 m for A12 | Open (research) |
| DynamicEarthNet | Daily Planet imagery with monthly 7-class labels incl. impervious | 3 m | 2018-2019 | 75 AOIs worldwide | Built change pretraining (A12) | CC BY-SA |
| SECOND, HRSCD | Semantic change pairs | 0.5-3 m aerial | Various | China; France | Optional class-head pretraining; low priority | Research |
| Stage-II organizer dataset | Labeled pairs per the PS conventions | Unknown | Unknown | AOIs from Stage II | Label conventions; V-stage2 | Competition terms |

## Appendix B. Labeling guide skeleton

The guide is filled from the Stage-II labels in Phase 0 and signed off before any annotation; every cell marked "fill" is a decision that changes what the model learns.

**General rules**

| Rule | Decision |
| --- | --- |
| Temporal semantics | A pixel is change k if a class-k object is present at exactly one of the two dates (fill: confirm removals are labeled in Stage II) |
| Overlap priority | building > solar > road > buildup (fill: confirm against Stage-II examples where a building sits in a new compound) |
| Validity | Cloud, shadow or nodata at either date = ignore (255); never guess under cloud |
| Under construction | Foundations and framing visible = fill (buildup or building?); roof complete = building |
| Partial change | An object extended by less than the minimum size = fill (labeled or not?) |
| Annotation grid | Digitize at 2.5 m display scale, snap to the 5 m grid on export |

**Per-class definitions**

| Class | Definition (fill from Stage II) | Include | Exclude | Minimum size | Boundary rule | Reference examples (fill with Stage-II chip ids) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 Buildup / clearance | fill | Leveled ground, compounds, hardstanding, construction sites, reclamation, quarry and spoil areas if Stage II includes them | Fallow or harvested fields, dry riverbeds, natural bare ground, dune movement | fill (default 2,500 m2) | Outer edge of the leveled or built-up area | fill |
| 2 Stand-alone building | fill | Single roofs >= 60 x 30 m: hangars, warehouses, plants, barracks blocks | Houses, attached rows of small buildings unless Stage II merges them, tents, vehicles, ships | 1,800 m2 on the roof polygon (fill: polygon or change component?) | Roof outline, no shadow | fill |
| 3 Road | fill | Paved and compacted roads, taxiways and runways if Stage II classes them as roads (fill), tracks visible at 5 m | Field boundaries, canals, pipelines, dry channels | 200 m length; burn width fill (default 10 m highways, 6 m access roads) | Centreline buffered to width | fill |
| 4 Solar farm | fill | Panel arrays; fill: substations and fenced area included or not | Greenhouses, water bodies with glint, reflective roofs | 5,000 m2 | Array outer edge (fill) | fill |

**Annotator workflow checklist**

- [ ] Open the AOI's QGIS project; confirm T1 and T2 dates and that the validity mask is loaded
- [ ] Review each candidate polygon with the swipe tool; keep, reclass, reshape or delete
- [ ] Scan the full AOI once at 1:25,000 for missed change
- [ ] Set confidence (auto, corrected, double-checked) and save
- [ ] Log any case the guide does not cover in the guide's issue list
