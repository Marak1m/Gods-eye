"""Post-training calibration of tau and the class biases (HANDOFF.md 7.4; DEVIATIONS C4).

Full-TTA change and class logits are cached for the validation chips; coordinate ascent sweeps tau
in 0.20-0.70 (step 0.02) then each bias in -2..2 (step 0.1), two passes, maximising rubric-weighted
IoU subject to per-class precision >= 0.6 (classes that are never predicted are exempt). The
robustness simulation is run before and after: `robust_swap` must be 1.0 in both (asserted); the
identity, synthetic and near terms depend on tau by definition and are logged before/after. The
inference pipeline's identity guard makes the final identity case exact regardless.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scd.config import Config
from scd.constants import CHANGE_CLASSES, IGNORE, N_SEM_CLASSES
from scd.data import manifest as man_mod
from scd.data.dataset import DrawRequest, PairChipDataset, full_chip_requests
from scd.infer import tta
from scd.labels.synthetic import ObjectLibrary
from scd.log import get_logger
from scd.train import robustness
from scd.train.lit_module import net_from_checkpoint
from scd.train.metrics import decide, metrics_from_cm
from scd.train.train import best_checkpoint

log = get_logger(__name__)

TAU_GRID = [round(x, 2) for x in np.arange(0.20, 0.70 + 1e-9, 0.02)]
BIAS_GRID = [round(x, 1) for x in np.arange(-2.0, 2.0 + 1e-9, 0.1)]
PRECISION_FLOOR = 0.6


@dataclass
class Calibration:
    tau: float = 0.5
    bias: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0])
    weighted_iou: float = float("nan")
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    weights_source: str = "raw"
    checkpoint: str = ""
    robust_before: dict[str, float] = field(default_factory=dict)
    robust_after: dict[str, float] = field(default_factory=dict)
    stub: bool = False
    n_val_chips: int = 0

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str))
        return path

    @classmethod
    def load(cls, path: Path) -> Calibration:
        d = json.loads(Path(path).read_text())
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def default(cls) -> Calibration:
        return cls(stub=True)


def load_calibration(path: Path) -> Calibration:
    return Calibration.load(path)


# --- caching and search ---------------------------------------------------------------------
def cache_logits(
    net: torch.nn.Module,
    dataset: PairChipDataset,
    requests: Sequence[DrawRequest],
    cfg: Config,
    device: torch.device | str = "cpu",
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """[(c (H,W), z (4,H,W), gt (H,W), valid (H,W))] with full TTA for every request."""
    out = []
    for req in requests:
        t1, t2, valid, change, _s1, _s2 = dataset.read_chip(req.chip_id)
        x1 = tta.to_tensor(t1, cfg.bands.scale, cfg.bands.clip_max)
        x2 = tta.to_tensor(t2, cfg.bands.scale, cfg.bands.clip_max)
        c, z = tta.predict(
            net,
            x1,
            x2,
            cfg.infer.window_px,
            cfg.infer.stride_px,
            cfg.infer.tta_geometric,
            cfg.infer.tta_temporal,
            device,
        )
        out.append((c, z, torch.from_numpy(change.astype(np.int64)), torch.from_numpy(valid.astype(bool))))
    return out


def evaluate(
    cache: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]], tau: float, bias: Sequence[float]
) -> dict[str, float]:
    cm = torch.zeros(N_SEM_CLASSES, N_SEM_CLASSES, dtype=torch.float64)
    for c, z, gt, valid in cache:
        pred = decide(c[None, None], z[None], tau, bias)[0]
        m = valid & (gt != IGNORE)
        idx = gt[m] * N_SEM_CLASSES + pred[m]
        cm += (
            torch.bincount(idx, minlength=N_SEM_CLASSES * N_SEM_CLASSES).reshape(N_SEM_CLASSES, N_SEM_CLASSES).double()
        )
    metrics = metrics_from_cm(cm)
    for k, name in enumerate(CHANGE_CLASSES, start=1):
        metrics[f"pred_px_{name}"] = float(cm[:, k].sum())
    return metrics


def _objective(m: dict[str, float], floor: float) -> float:
    feasible = all(m[f"p_{c}"] >= floor or m[f"pred_px_{c}"] == 0 for c in CHANGE_CLASSES)
    return m["weighted_iou"] - (0.0 if feasible else 1.0)


def search(
    cache: Sequence[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
    tau_grid: Sequence[float] = TAU_GRID,
    bias_grid: Sequence[float] = BIAS_GRID,
    precision_floor: float = PRECISION_FLOOR,
    passes: int = 2,
) -> tuple[float, list[float], dict[str, float]]:
    """Coordinate ascent on tau, then each bias, `passes` times."""
    bias = [0.0, 0.0, 0.0, 0.0]
    best_tau = 0.5
    best_obj = -math.inf
    best_m: dict[str, float] = {}
    for tau in tau_grid:
        m = evaluate(cache, tau, bias)
        obj = _objective(m, precision_floor)
        if obj > best_obj:
            best_obj, best_tau, best_m = obj, float(tau), m
    for _ in range(passes):
        for k in range(4):
            for b in bias_grid:
                trial = list(bias)
                trial[k] = float(b)
                m = evaluate(cache, best_tau, trial)
                obj = _objective(m, precision_floor)
                if obj > best_obj + 1e-12:
                    best_obj, bias, best_m = obj, trial, m
    if best_obj < 0:
        log.warning(
            "calibration: no operating point satisfies precision >= %.2f for every predicted class", precision_floor
        )
    return best_tau, bias, best_m


# --- orchestration ----------------------------------------------------------------------------
def calibrate(
    run_dir: Path,
    dataset: PairChipDataset,
    requests: Sequence[DrawRequest],
    cfg: Config,
    net: torch.nn.Module,
    library: ObjectLibrary | None = None,
    device: torch.device | str = "cpu",
) -> Calibration:
    robust_reqs = robustness.pick_subset(requests, cfg.train.robustness_subset, seed=cfg.train.seed)
    before = robustness.simulate(net, dataset, robust_reqs, cfg, tau=0.5, bias=None, library=library, device=device)
    cache = cache_logits(net, dataset, requests, cfg, device)
    tau, bias, m = search(cache)
    after = robustness.simulate(net, dataset, robust_reqs, cfg, tau=tau, bias=bias, library=library, device=device)
    if not (before["robust_swap"] == 1.0 and after["robust_swap"] == 1.0):
        raise AssertionError(
            f"robust_swap must be 1.0 before and after calibration: {before['robust_swap']} / {after['robust_swap']}"
        )
    for key in ("robust_identity", "robust_synthetic", "robust_near"):
        log.info("calibration: %s %.4f -> %.4f", key, before[key], after[key])
    per_class = {c: {"iou": m[f"iou_{c}"], "precision": m[f"p_{c}"], "recall": m[f"r_{c}"]} for c in CHANGE_CLASSES}
    return Calibration(
        tau=tau,
        bias=[float(b) for b in bias],
        weighted_iou=float(m["weighted_iou"]),
        per_class=per_class,
        robust_before=before,
        robust_after=after,
        n_val_chips=len(requests),
    )


def run(cfg: Config, run_name: str, chips_version: int = 1, stub: bool = False) -> Calibration:
    """Calibrate the run's best checkpoint on the val split; write runs/<run>/calibration.json."""
    run_dir = cfg.run_dir(run_name)
    ckpt = best_checkpoint(run_dir)
    net, source = net_from_checkpoint(ckpt, cfg)
    if stub:
        calib = Calibration(weights_source=source, checkpoint=str(ckpt), stub=True)
        calib.save(run_dir / "calibration.json")
        log.info("stub calibration written (tau 0.5, zero biases) -> %s", run_dir / "calibration.json")
        return calib
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net.to(device)
    chips_dir = cfg.chips_dir(chips_version)
    df = man_mod.load(chips_dir / "manifest.parquet")
    val = df[df["split"] == "val"]
    split = "val"
    if len(val) == 0:
        log.warning("no val chips; calibrating on train chips (debug only)")
        val, split = df[df["split"] == "train"], "train"
    dataset = PairChipDataset(chips_dir, val, cfg, tile_px=cfg.grid.chip_px, augment=False)
    requests = full_chip_requests(val, split)
    objects_dir = cfg.objects_dir(chips_version)
    library = ObjectLibrary.load(objects_dir) if objects_dir.exists() else None
    calib = calibrate(run_dir, dataset, requests, cfg, net, library=library, device=device)
    calib.weights_source = source
    calib.checkpoint = str(ckpt)
    calib.save(run_dir / "calibration.json")
    log.info("calibration: tau=%.2f bias=%s weighted_iou=%.4f", calib.tau, calib.bias, calib.weighted_iou)
    return calib


def as_dict(calib: Calibration) -> dict[str, Any]:
    return asdict(calib)
