"""Inference pipeline (HANDOFF.md 8): two images -> change.tif, valid.tif, change.shp, report.json, quicklook.png.

1. load_and_identify + preprocess_pair (the same order-invariant path as training)
2. identity_guard: max |t1 - t2| < identity_eps over valid pixels -> all zeros, empty shapefile
3. tta.predict (sliding windows, D4 TTA, optional temporal TTA, cosine blending)
4. decide: tau and class biases from calibration; invalid -> 0
5. postprocess.apply
6-8. export GeoTIFF (invalid pixels as infer.invalid_value, DEVIATIONS U3) + valid.tif, shapefile,
     report.json and quicklook.png

Determinism: torch.use_deterministic_algorithms(True) and cudnn.benchmark = False. Two runs are
byte-identical; run(A, B) and run(B, A) produce identical pixel arrays and vectors.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scd import __version__
from scd.config import Config
from scd.constants import CHANGE_CLASSES, CLASS_IDS
from scd.geo.io import read_raster, sha256_file
from scd.imagery.preprocess import PairProduct, preprocess_pair
from scd.infer import export, postprocess, tta
from scd.log import get_logger
from scd.model.net import SiameseChangeNet, build_model
from scd.train.calibrate import Calibration
from scd.train.metrics import decide

log = get_logger(__name__)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


@dataclass
class Result:
    out_dir: Path
    change_tif: Path
    valid_tif: Path
    shapefile: Path
    report: Path
    quicklook: Path
    n_change_px: int
    class_px: dict[str, int] = field(default_factory=dict)
    identity_short_circuit: bool = False
    valid_fraction: float = float("nan")
    seconds: float = float("nan")

    def summary(self) -> str:
        px = ", ".join(f"{k}={v}" for k, v in self.class_px.items())
        guard = " (identity guard: all zeros)" if self.identity_short_circuit else ""
        head = f"{self.out_dir}: {self.n_change_px} change px [{px}]"
        return f"{head} valid {self.valid_fraction:.1%} in {self.seconds:.1f}s{guard}"


def set_deterministic() -> None:
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_model_and_calibration(
    cfg: Config,
    run: str | None = None,
    checkpoint: Path | None = None,
    calibration: Path | None = None,
) -> tuple[SiameseChangeNet, Calibration]:
    """Resolve the network and calibration from a run name or explicit paths; random init if neither."""
    from scd.train.lit_module import net_from_checkpoint
    from scd.train.train import best_checkpoint

    if checkpoint is None and run is not None:
        checkpoint = best_checkpoint(cfg.run_dir(run))
    if calibration is None and run is not None:
        cand = cfg.run_dir(run) / "calibration.json"
        calibration = cand if cand.exists() else None
    if checkpoint is not None:
        calib = Calibration.load(calibration) if calibration else Calibration.default()
        net, _source = net_from_checkpoint(checkpoint, cfg, prefer=calib.weights_source if calibration else None)
    else:
        log.warning("no checkpoint or run given: using a randomly initialised network (tests only)")
        net = build_model(cfg).eval()
        calib = Calibration.load(calibration) if calibration else Calibration.default()
    if calibration is None:
        log.warning("no calibration.json: using tau=0.5 and zero biases")
    return net, calib


def identity_guard(t1: np.ndarray, t2: np.ndarray, valid: np.ndarray, scale: float, eps: float) -> bool:
    v = valid.astype(bool)
    if not v.any():
        return True
    diff = np.abs(t1.astype(np.float32) - t2.astype(np.float32)) / scale
    return float(diff[:, v].max()) < eps


def run(
    t1_path: Path,
    t2_path: Path,
    out_dir: Path,
    cfg: Config,
    calib: Calibration | None = None,
    net: SiameseChangeNet | None = None,
    bands: list[str] | None = None,
    assume: str | None = None,
    device: str | None = None,
) -> Result:
    t0 = time.time()
    set_deterministic()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    calib = calib or Calibration.default()
    if net is None:
        net, _ = load_model_and_calibration(cfg)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    net.to(dev).eval()

    # 1. identify + preprocess (order-invariant)
    scene_kwargs: dict[str, Any] = {}
    if bands:
        scene_kwargs["bands"] = bands
    if assume:
        scene_kwargs["assume"] = assume
    prod: PairProduct = preprocess_pair(
        Path(t1_path), Path(t2_path), None, cfg, out_dir / "pre", scene_kwargs=scene_kwargs
    )
    t1, t2, valid = prod.read()
    grid = prod.grid()
    meta = prod.meta()

    # 2. identity guard
    short = identity_guard(t1, t2, valid, cfg.bands.scale, cfg.infer.identity_eps)
    if short:
        mask = np.zeros(grid.shape, dtype=np.uint8)
        conf = np.zeros(grid.shape, dtype=np.float32)
    else:
        # 3. TTA prediction
        x1 = tta.to_tensor(t1, cfg.bands.scale, cfg.bands.clip_max)
        x2 = tta.to_tensor(t2, cfg.bands.scale, cfg.bands.clip_max)
        c, z = tta.predict(
            net, x1, x2, cfg.infer.window_px, cfg.infer.stride_px, cfg.infer.tta_geometric, cfg.infer.tta_temporal, dev
        )
        # 4. decide
        mask = decide(c[None, None], z[None], calib.tau, calib.bias)[0].numpy().astype(np.uint8)
        mask[valid == 0] = 0
        conf = torch.sigmoid(c).numpy().astype(np.float32)
        # 5. post-process
        mask = postprocess.apply(mask, cfg, t1, t2, valid, grid.res_m)

    # 6-8. export
    tags = {
        "scd_version": __version__,
        "model_hash": _model_hash(net),
        "config_hash": cfg.hash(),
        "tau": f"{calib.tau:.4f}",
        "bias": ",".join(f"{b:.3f}" for b in calib.bias),
        "coreg_dx": f"{meta['coreg']['dx']:.4f}",
        "coreg_dy": f"{meta['coreg']['dy']:.4f}",
        "canonical_first": str(meta["canonical_first"]),
        "identity_guard": str(short),
    }
    change_tif = export.write_geotiff(mask, grid, out_dir / "change.tif", tags, cfg.infer.invalid_value, valid)
    valid_tif = export.write_valid(valid, grid, out_dir / "valid.tif")
    shp = export.write_shapefile(mask, grid, out_dir / "change.shp", conf=conf, simplify_m=cfg.infer.simplify_m)
    quick = export.write_quicklook(t1, t2, mask, out_dir / "quicklook.png")
    class_px = {name: int((mask == CLASS_IDS[name]).sum()) for name in CHANGE_CLASSES}
    seconds = time.time() - t0
    report = {
        "inputs": {
            "t1": str(t1_path),
            "t2": str(t2_path),
            "t1_sha256": sha256_file(Path(t1_path)) if Path(t1_path).is_file() else None,
            "t2_sha256": sha256_file(Path(t2_path)) if Path(t2_path).is_file() else None,
        },
        "grid": grid.to_dict(),
        "valid_fraction": float(valid.mean()),
        "identity_short_circuit": short,
        "coreg": meta["coreg"],
        "calibration": {"tau": calib.tau, "bias": calib.bias, "stub": calib.stub},
        "class_px": class_px,
        "n_components": export.component_count(mask),
        "invalid_value": cfg.infer.invalid_value,
        "tta": {"geometric": cfg.infer.tta_geometric, "temporal": cfg.infer.tta_temporal},
        "seconds": seconds,
        "tags": tags,
    }
    rep = export.write_report(out_dir / "report.json", report)
    return Result(
        out_dir,
        change_tif,
        valid_tif,
        shp,
        rep,
        quick,
        int(sum(class_px.values())),
        class_px,
        short,
        float(valid.mean()),
        seconds,
    )


def _model_hash(net: torch.nn.Module) -> str:
    import hashlib

    h = hashlib.sha256()
    for name, p in sorted(net.state_dict().items()):
        h.update(name.encode())
        h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


# --- notebook helpers (section 13) ---------------------------------------------------------
def pick_pair(cfg: Config, spec: str, chips_version: int = 1) -> Path:
    """'<aoi>/<pair>' or, by default, the first val pair from the manifest (else the first pair)."""
    if spec:
        return cfg.pairs_dir / spec
    from scd.data import manifest as man_mod

    man = cfg.chips_dir(chips_version) / "manifest.parquet"
    if man.exists():
        df = man_mod.load(man)
        sel = df[df["split"] == "val"] if (df["split"] == "val").any() else df
        row = sel.iloc[0]
        return cfg.pairs_dir / str(row["aoi_id"]) / str(row["pair_id"])
    return sorted(cfg.pairs_dir.glob("*/*/meta.json"))[0].parent


def show_result(out_dir: Path) -> None:  # pragma: no cover - notebook display
    from IPython.display import Image, display

    display(Image(filename=str(Path(out_dir) / "quicklook.png"), width=900))
    print(json.dumps(json.loads((Path(out_dir) / "report.json").read_text())["class_px"], indent=2))


def invariance_checks(cfg: Config, run_name: str | None, pair_dir: Path, out_root: Path) -> dict[str, bool]:
    """run(A,B) == run(B,A) on raster and vectors; run(A,A) is all zeros."""
    import geopandas as gpd

    net, calib = load_model_and_calibration(cfg, run=run_name)
    a, b = Path(pair_dir) / "t1.tif", Path(pair_dir) / "t2.tif"
    r_ab = run(a, b, out_root / "ab", cfg, calib, net)
    r_ba = run(b, a, out_root / "ba", cfg, calib, net)
    r_aa = run(a, a, out_root / "aa", cfg, calib, net)
    m_ab, _ = read_raster(r_ab.change_tif)
    m_ba, _ = read_raster(r_ba.change_tif)
    m_aa, _ = read_raster(r_aa.change_tif)
    g_ab = gpd.read_file(r_ab.shapefile)
    g_ba = gpd.read_file(r_ba.shapefile)
    checks = {
        "swap_raster_identical": bool(np.array_equal(m_ab, m_ba)),
        "swap_vectors_identical": bool(
            len(g_ab) == len(g_ba) and all(x.equals(y) for x, y in zip(g_ab.geometry, g_ba.geometry))
        ),
        "identity_all_zero": bool(m_aa.max() == 0),
    }
    log.info("invariance checks: %s", checks)
    return checks
