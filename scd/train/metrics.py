"""RubricMetrics (HANDOFF.md 7.1, 7.2): IoU, precision and recall from accumulated confusion matrices.

Metrics are computed from one 5x5 confusion matrix accumulated over every valid pixel of every
validation chip (globally and per stratum), never from a mean of per-chip IoUs and never over a
subset of classes. `leaderboard_proxy` follows DEVIATIONS U1.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torchmetrics.classification import MulticlassConfusionMatrix

from scd.constants import CHANGE_CLASSES, IGNORE, N_SEM_CLASSES, RUBRIC_WEIGHTS, STRATA


def leaderboard_proxy(
    weighted_iou: float, robust_identity: float, robust_synthetic: float, robust_near: float
) -> float:
    """0.7 * weighted IoU + 0.2 * mean(identity, synthetic, near). `robust_swap` is a guard, not a term."""
    robust = (float(robust_identity) + float(robust_synthetic) + float(robust_near)) / 3.0
    return 0.7 * float(weighted_iou) + 0.2 * robust


def decide(
    change_logit: torch.Tensor, cls_logits: torch.Tensor, tau: float = 0.5, bias: Sequence[float] | None = None
) -> torch.Tensor:
    """M = 1[sigmoid(c) > tau] * (1 + argmax_k(z_k + b_k)) -> (B, H, W) int64 in 0..4."""
    c = change_logit[:, 0] if change_logit.ndim == 4 else change_logit
    z = cls_logits
    if bias is not None:
        z = z + torch.as_tensor(list(bias), dtype=z.dtype, device=z.device).view(1, -1, 1, 1)
    cls = z.argmax(dim=1) + 1
    return torch.where(torch.sigmoid(c) > tau, cls, torch.zeros_like(cls))


def metrics_from_cm(cm: torch.Tensor) -> dict[str, float]:
    """Per-class IoU / precision / recall (classes 1..4), rubric-weighted IoU and macro IoU."""
    cm = cm.double()
    out: dict[str, float] = {}
    ious = []
    for k, name in enumerate(CHANGE_CLASSES, start=1):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        denom = tp + fp + fn
        iou = float(tp / denom) if denom > 0 else 0.0
        out[f"iou_{name}"] = iou
        out[f"p_{name}"] = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        out[f"r_{name}"] = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        out[f"present_{name}"] = float(cm[k, :].sum() > 0)
        ious.append(iou)
    out["weighted_iou"] = float(sum(RUBRIC_WEIGHTS[n] * out[f"iou_{n}"] for n in CHANGE_CLASSES))
    out["macro_iou"] = float(sum(ious) / len(ious))
    total = cm.sum()
    out["pixel_accuracy"] = float(cm.diag().sum() / total) if total > 0 else 0.0
    return out


class RubricMetrics:
    """Accumulates global and per-stratum confusion matrices; `compute()` returns the val/* keys."""

    def __init__(self, strata: Sequence[str] = STRATA, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self.global_cm = MulticlassConfusionMatrix(num_classes=N_SEM_CLASSES, ignore_index=IGNORE).to(self.device)
        self.stratum_cm = {
            s: MulticlassConfusionMatrix(num_classes=N_SEM_CLASSES, ignore_index=IGNORE).to(self.device) for s in strata
        }
        self.n_pixels = 0

    def reset(self) -> None:
        self.global_cm.reset()
        for m in self.stratum_cm.values():
            m.reset()
        self.n_pixels = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, strata: Sequence[str]) -> None:
        """pred/target (B, H, W) int; invalid pixels are ignored."""
        t = target.clone().to(self.device)
        t[~valid.to(self.device)] = IGNORE
        p = pred.to(self.device)
        for i, s in enumerate(strata):
            pi, ti = p[i].flatten(), t[i].flatten()
            self.global_cm.update(pi, ti)
            if s in self.stratum_cm:
                self.stratum_cm[s].update(pi, ti)
            self.n_pixels += int((ti != IGNORE).sum())

    def compute(self, robust: dict[str, float] | None = None, prefix: str = "val/") -> dict[str, float]:
        out = {f"{prefix}{k}": v for k, v in metrics_from_cm(self.global_cm.compute()).items()}
        for s, m in self.stratum_cm.items():
            if getattr(m, "_update_count", 1) == 0:
                continue
            cm = m.compute()
            if cm.sum() > 0:
                out[f"{prefix}stratum_{s}_weighted_iou"] = metrics_from_cm(cm)["weighted_iou"]
        if robust is not None:
            for k, v in robust.items():
                out[f"{prefix}{k}"] = float(v)
            out[f"{prefix}leaderboard_proxy"] = leaderboard_proxy(
                out[f"{prefix}weighted_iou"],
                robust.get("robust_identity", 0.0),
                robust.get("robust_synthetic", 0.0),
                robust.get("robust_near", 0.0),
            )
        out[f"{prefix}n_pixels"] = float(self.n_pixels)
        return out
