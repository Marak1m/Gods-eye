"""M6: IoU from an accumulated matrix matches a hand computation; per-stratum keys; proxy formula (HANDOFF 9)."""

from __future__ import annotations

import torch

from scd.constants import IGNORE
from scd.train.metrics import RubricMetrics, decide, leaderboard_proxy, metrics_from_cm


def test_metrics_from_hand_computed_matrix() -> None:
    m = RubricMetrics()
    # chip 1 (stratum A): 4 x 4 pixels, GT building (2) in the top row, prediction partly right
    gt = torch.zeros(1, 4, 4, dtype=torch.long)
    gt[0, 0, :] = 2
    pred = torch.zeros(1, 4, 4, dtype=torch.long)
    pred[0, 0, :3] = 2  # 3 TP, 1 FN
    pred[0, 1, 0] = 2  # 1 FP
    pred[0, 2, 0] = 4  # 1 FP for solar (no GT)
    valid = torch.ones(1, 4, 4, dtype=torch.bool)
    valid[0, 3, 3] = False  # invalid pixel must be ignored even if predicted
    pred[0, 3, 3] = 1
    m.update(pred, gt, valid, ["A"])
    # chip 2 (stratum B): all background, one 255 pixel, perfect prediction
    gt2 = torch.zeros(1, 4, 4, dtype=torch.long)
    gt2[0, 0, 0] = IGNORE
    m.update(torch.zeros(1, 4, 4, dtype=torch.long), gt2, torch.ones(1, 4, 4, dtype=torch.bool), ["B"])
    out = m.compute(robust={"robust_swap": 1.0, "robust_identity": 0.9, "robust_synthetic": 0.6, "robust_near": 0.99})
    assert abs(out["val/iou_building"] - 3 / 5) < 1e-9  # tp 3 / (tp 3 + fp 1 + fn 1)
    assert abs(out["val/p_building"] - 3 / 4) < 1e-9 and abs(out["val/r_building"] - 3 / 4) < 1e-9
    assert out["val/iou_solar"] == 0.0 and out["val/p_solar"] == 0.0 and out["val/iou_road"] == 0.0
    assert abs(out["val/weighted_iou"] - 0.30 * 3 / 5) < 1e-9
    assert abs(out["val/macro_iou"] - (3 / 5) / 4) < 1e-9
    assert "val/stratum_A_weighted_iou" in out and "val/stratum_B_weighted_iou" in out
    assert "val/stratum_C_weighted_iou" not in out  # no data for C
    assert out["val/n_pixels"] == 15 + 15
    expected_proxy = 0.7 * out["val/weighted_iou"] + 0.2 * (0.9 + 0.6 + 0.99) / 3
    assert abs(out["val/leaderboard_proxy"] - expected_proxy) < 1e-9  # DEVIATIONS U1: swap not averaged
    assert out["val/robust_swap"] == 1.0


def test_proxy_formula_and_decide() -> None:
    assert abs(leaderboard_proxy(0.5, 1.0, 1.0, 1.0) - (0.35 + 0.2)) < 1e-12
    assert abs(leaderboard_proxy(0.5, 0.0, 0.0, 0.0) - 0.35) < 1e-12
    c = torch.tensor([[[[-3.0, 3.0], [0.1, -0.1]]]])  # (1,1,2,2)
    z = torch.zeros(1, 4, 2, 2)
    z[0, 2] = 1.0  # road wins everywhere without bias
    m = decide(c, z, tau=0.5)
    assert m.tolist() == [[[0, 3], [3, 0]]]
    m2 = decide(c, z, tau=0.5, bias=[0.0, 2.0, 0.0, 0.0])  # bias lifts building above road
    assert m2.tolist() == [[[0, 2], [2, 0]]]
    assert decide(c, z, tau=0.99).sum() == 0


def test_metrics_from_cm_handles_empty() -> None:
    out = metrics_from_cm(torch.zeros(5, 5))
    assert out["weighted_iou"] == 0.0 and out["macro_iou"] == 0.0 and out["pixel_accuracy"] == 0.0
