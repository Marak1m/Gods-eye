"""OracleNet: a position-independent, exactly symmetric stand-in for the trained network.

It flags change where the mean absolute band difference exceeds a threshold and reads the class
from a code written into B12 (reflectance 0.80 + 0.04 * class on the date that carries the object;
`max` over dates keeps it symmetric). It lets the CPU harness tests exercise TTA inversion,
decision, post-processing, synthetic recovery and near-identical behaviour without a checkpoint
(DEVIATIONS C17). The trained-checkpoint versions of those tests are gpu-marked.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from scd.constants import BAND_INDEX, UINT16_SCALE

CODE_BASE = 0.80
CODE_STEP = 0.04


def class_code(class_id: int) -> float:
    return CODE_BASE + CODE_STEP * class_id


def write_code(img_uint16: np.ndarray, mask: np.ndarray, class_id: int) -> np.ndarray:
    """Write the class code into B12 under `mask` (uint16 DN) and return a copy."""
    out = img_uint16.copy()
    out[BAND_INDEX["B12"]][mask] = int(round(class_code(class_id) * UINT16_SCALE))
    return out


class OracleNet(nn.Module):
    def __init__(self, diff_threshold: float = 0.05, gain: float = 20.0) -> None:
        super().__init__()
        self.diff_threshold = diff_threshold
        self.gain = gain
        self.return_semantic = False
        self.dummy = nn.Parameter(torch.zeros(1))  # so .parameters() is non-empty

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> dict[str, torch.Tensor | None]:
        diff = (x1 - x2).abs().mean(dim=1, keepdim=True)  # symmetric
        change = self.gain * (diff - self.diff_threshold)
        code = torch.maximum(x1[:, BAND_INDEX["B12"]], x2[:, BAND_INDEX["B12"]])  # symmetric
        cls_id = torch.round((code - CODE_BASE) / CODE_STEP).clamp(1, 4)
        cls = torch.full((x1.shape[0], 4, *x1.shape[-2:]), -5.0, device=x1.device)
        for k in range(1, 5):
            cls[:, k - 1] = torch.where(cls_id == k, 5.0, -5.0)
        return {"change": change, "cls": cls, "sem_t1": None, "sem_t2": None}
