"""M5: loss masking rules (HANDOFF 6.5)."""

from __future__ import annotations

import pytest
import torch

from scd.model.losses import ChangeLoss, ClassLoss, SemanticLoss, TotalLoss, class_weight_tensor, soft_dice


def _change_gt() -> tuple[torch.Tensor, torch.Tensor]:
    gt = torch.zeros(1, 16, 16, dtype=torch.long)
    gt[0, 2:6, 2:6] = 2
    gt[0, 8:12, 8:12] = 4
    gt[0, 14:, :] = 255
    valid = torch.ones(1, 16, 16, dtype=torch.bool)
    valid[0, :, 14:] = False
    return gt, valid


def test_change_loss_perfect_vs_wrong_and_ignores_masked() -> None:
    gt, valid = _change_gt()
    good = torch.where((gt > 0) & (gt != 255), 20.0, -20.0)[:, None]
    bad = -good
    loss = ChangeLoss(pos_weight=2.0)
    assert loss(good, gt, valid) < 0.05 < loss(bad, gt, valid)
    # changing logits only on ignored/invalid pixels leaves the loss unchanged
    tweaked = good.clone()
    tweaked[:, :, 14:, :] = 7.0
    tweaked[:, :, :, 14:] = -3.0
    assert torch.isclose(loss(good, gt, valid), loss(tweaked, gt, valid))


def test_class_loss_only_sees_change_pixels() -> None:
    gt, valid = _change_gt()
    logits = torch.full((1, 4, 16, 16), -5.0)
    logits[0, 1, 2:6, 2:6] = 5.0  # class 2 -> index 1
    logits[0, 3, 8:12, 8:12] = 5.0  # class 4 -> index 3
    loss = ClassLoss()
    base = loss(logits, gt, valid)
    assert base < 0.1
    noisy = logits.clone()
    noisy[0, :, 0:2, :] = torch.randn(4, 2, 16)  # background pixels: no effect
    assert torch.isclose(loss(noisy, gt, valid), base)
    none = torch.zeros(1, 16, 16, dtype=torch.long)
    z = loss(logits.requires_grad_(), none, valid)
    assert float(z) == 0.0 and z.requires_grad


def test_class_weights_change_the_loss() -> None:
    gt, valid = _change_gt()
    logits = torch.zeros(1, 4, 16, 16)
    w1 = ClassLoss(torch.tensor([1.0, 1.0, 1.0, 1.0]))(logits, gt, valid)
    w2 = ClassLoss(torch.tensor([1.0, 3.0, 1.0, 1.0]))(logits, gt, valid)
    assert not torch.isclose(w1, w2)


def test_semantic_loss_ignores_255_and_invalid() -> None:
    sem = torch.zeros(1, 8, 8, dtype=torch.long)
    sem[0, :4] = 3
    sem[0, 7] = 255
    valid = torch.ones(1, 8, 8, dtype=torch.bool)
    valid[0, :, 7] = False
    logits = torch.full((1, 5, 8, 8), -5.0)
    logits[0, 0] = 5.0
    logits[0, 3, :4] = 10.0
    logits[0, 0, :4] = -5.0
    loss = SemanticLoss()
    base = loss(logits, sem, valid)
    assert base < 0.1
    tweaked = logits.clone()
    tweaked[0, :, 7] = torch.randn(5, 8)
    tweaked[0, :, :, 7] = torch.randn(5, 8)
    assert torch.isclose(loss(tweaked, sem, valid), base)


def test_total_loss_terms_and_lambda() -> None:
    gt, valid = _change_gt()
    out = {
        "change": torch.zeros(1, 1, 16, 16),
        "cls": torch.zeros(1, 4, 16, 16),
        "sem_t1": torch.zeros(1, 5, 16, 16),
        "sem_t2": torch.zeros(1, 5, 16, 16),
    }
    batch = {
        "change": gt,
        "sem_t1": gt.clamp(max=4),
        "sem_t2": gt.clamp(max=4),
        "valid": valid,
        "weight": torch.ones(1),
    }
    t = TotalLoss(lambda_sem=0.3)(out, batch)
    expected = t["change"] + t["cls"] + 0.3 * (t["sem_t1"] + t["sem_t2"])
    assert torch.isclose(t["total"], expected)
    t0 = TotalLoss(lambda_sem=0.0)(out, batch)
    assert torch.isclose(t0["total"], t0["change"] + t0["cls"]) and "sem_t1" not in t0
    out_nosem = dict(out, sem_t1=None, sem_t2=None)
    t2 = TotalLoss()(out_nosem, batch)
    assert "sem_t1" not in t2
    with pytest.raises(NotImplementedError):
        TotalLoss(cldice_road=True)


def test_dice_and_weight_tensor() -> None:
    p = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    assert float(soft_dice(p, p, torch.ones_like(p), eps=0.0)) == 0.0
    assert class_weight_tensor({"buildup": 1.0, "building": 2.0, "road": 3.0, "solar": 4.0}).tolist() == [
        1.0,
        2.0,
        3.0,
        4.0,
    ]
