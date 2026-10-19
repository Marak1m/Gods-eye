"""Losses (HANDOFF.md 6.5).

ChangeLoss  = BCE-with-logits (pos_weight) + soft Dice on the binary change map, masked by validity.
ClassLoss   = weighted CE + per-class soft Dice on pixels whose change GT is in 1..4.
SemanticLoss= CE + Dice, 5-way, masked by validity and 255.
TotalLoss   = change + cls + lambda_sem * (sem_t1 + sem_t2), weighted per sample (auto labels 0.5).
Logits are clamped to +-30 before BCE. clDice on the road channel (A10) is not built in this pass.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from scd.constants import CHANGE_CLASSES, IGNORE

LOGIT_CLAMP = 30.0


def soft_dice(probs: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    """1 - Dice over the masked pixels of a batch (probs/target/mask broadcastable, same shape)."""
    p = probs * mask
    t = target * mask
    inter = (p * t).sum()
    denom = p.sum() + t.sum()
    return 1.0 - (2.0 * inter + eps) / (denom + eps)


def _sample_weights(weight: torch.Tensor | None, batch: int, device: torch.device) -> torch.Tensor:
    if weight is None:
        return torch.ones(batch, device=device)
    return weight.to(device).float()


class ChangeLoss(nn.Module):
    pos_weight: torch.Tensor

    def __init__(self, pos_weight: float = 2.0, dice: bool = True) -> None:
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        self.dice = dice

    def forward(
        self, logit: torch.Tensor, change_gt: torch.Tensor, valid: torch.Tensor, weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        logit = logit[:, 0].clamp(-LOGIT_CLAMP, LOGIT_CLAMP)
        mask = valid & (change_gt != IGNORE)
        target = ((change_gt > 0) & (change_gt != IGNORE)).float()
        bce = F.binary_cross_entropy_with_logits(logit, target, pos_weight=self.pos_weight, reduction="none")
        w = _sample_weights(weight, logit.shape[0], logit.device)
        per_sample = (bce * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp(min=1)
        loss = (per_sample * w).sum() / w.sum().clamp(min=1e-6)
        if self.dice:
            loss = loss + soft_dice(torch.sigmoid(logit), target, mask.float())
        return loss


class ClassLoss(nn.Module):
    class_weights: torch.Tensor

    def __init__(self, class_weights: torch.Tensor | None = None, dice: bool = True, n_classes: int = 4) -> None:
        super().__init__()
        cw = torch.ones(n_classes) if class_weights is None else class_weights.float()
        self.register_buffer("class_weights", cw)
        self.dice = dice
        self.n_classes = n_classes

    def forward(
        self, cls_logits: torch.Tensor, change_gt: torch.Tensor, valid: torch.Tensor, weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        mask = valid & (change_gt >= 1) & (change_gt <= self.n_classes)
        if not mask.any():
            return cls_logits.sum() * 0.0
        target = (change_gt - 1).clamp(0, self.n_classes - 1)
        ce = F.cross_entropy(cls_logits, target, weight=self.class_weights.to(cls_logits.dtype), reduction="none")
        w = _sample_weights(weight, cls_logits.shape[0], cls_logits.device)
        per_sample = (ce * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp(min=1)
        has = (mask.flatten(1).sum(1) > 0).float()
        loss = (per_sample * w * has).sum() / (w * has).sum().clamp(min=1e-6)
        if self.dice:
            probs = torch.softmax(cls_logits, dim=1)
            m = mask.float()
            d = sum(soft_dice(probs[:, k], (target == k).float(), m) for k in range(self.n_classes))
            loss = loss + d / self.n_classes
        return loss


class SemanticLoss(nn.Module):
    def __init__(self, n_classes: int = 5, dice: bool = True) -> None:
        super().__init__()
        self.n_classes = n_classes
        self.dice = dice

    def forward(
        self, logits: torch.Tensor, sem_gt: torch.Tensor, valid: torch.Tensor, weight: torch.Tensor | None = None
    ) -> torch.Tensor:
        mask = valid & (sem_gt != IGNORE)
        if not mask.any():
            return logits.sum() * 0.0
        target = sem_gt.clamp(0, self.n_classes - 1)
        ce = F.cross_entropy(logits, target, reduction="none")
        w = _sample_weights(weight, logits.shape[0], logits.device)
        per_sample = (ce * mask).flatten(1).sum(1) / mask.flatten(1).sum(1).clamp(min=1)
        has = (mask.flatten(1).sum(1) > 0).float()
        loss = (per_sample * w * has).sum() / (w * has).sum().clamp(min=1e-6)
        if self.dice:
            probs = torch.softmax(logits, dim=1)
            m = mask.float()
            d = sum(soft_dice(probs[:, k], (target == k).float(), m) for k in range(self.n_classes))
            loss = loss + d / self.n_classes
        return loss


class TotalLoss(nn.Module):
    def __init__(
        self,
        pos_weight: float = 2.0,
        lambda_sem: float = 0.3,
        class_weights: torch.Tensor | None = None,
        cldice_road: bool = False,
        dice: bool = True,
    ) -> None:
        super().__init__()
        if cldice_road:
            raise NotImplementedError("clDice on the road channel (ablation A10) is not built in this pass")
        self.change = ChangeLoss(pos_weight, dice)
        self.cls = ClassLoss(class_weights, dice)
        self.sem = SemanticLoss(dice=dice)
        self.lambda_sem = float(lambda_sem)

    def forward(self, out: dict[str, torch.Tensor | None], batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        valid = batch["valid"].bool()
        weight = batch.get("weight")
        change = out["change"]
        cls = out["cls"]
        assert change is not None and cls is not None
        terms: dict[str, torch.Tensor] = {
            "change": self.change(change, batch["change"], valid, weight),
            "cls": self.cls(cls, batch["change"], valid, weight),
        }
        total = terms["change"] + terms["cls"]
        for key in ("sem_t1", "sem_t2"):
            logits = out.get(key)
            if logits is not None and self.lambda_sem > 0:
                terms[key] = self.sem(logits, batch[key], valid, weight)
                total = total + self.lambda_sem * terms[key]
        terms["total"] = total
        return terms


def class_weight_tensor(weights: dict[str, float]) -> torch.Tensor:
    return torch.tensor([float(weights[c]) for c in CHANGE_CLASSES], dtype=torch.float32)
