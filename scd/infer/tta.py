"""Sliding-window inference with D4 test-time augmentation and cosine blending (HANDOFF.md 8 step 3).

Logits (change c and class z) are averaged over up to 8 dihedral transforms applied identically to
both dates. With `temporal=True` each transform also runs the swapped order and the two results
are summed first (pairwise-then-fixed order, DEVIATIONS U2), so the output stays bit-identical
under swap; because the network is exactly symmetric the second order is redundant and off by
default. Windows are blended with a cosine (Hann) weight.
"""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F

from scd.log import get_logger

log = get_logger(__name__)

D4: tuple[tuple[int, bool], ...] = (
    (0, False),
    (1, False),
    (2, False),
    (3, False),
    (0, True),
    (1, True),
    (2, True),
    (3, True),
)


def d4_forward(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    y = torch.rot90(x, k, dims=(-2, -1))
    return torch.flip(y, dims=(-1,)) if flip else y


def d4_inverse(y: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    x = torch.flip(y, dims=(-1,)) if flip else y
    return torch.rot90(x, -k, dims=(-2, -1))


def cosine_window(h: int, w: int) -> torch.Tensor:
    def hann(n: int) -> torch.Tensor:
        i = torch.arange(n, dtype=torch.float32)
        return 0.5 * (1.0 - torch.cos(2.0 * math.pi * (i + 0.5) / n)) + 1e-3

    return hann(h)[:, None] * hann(w)[None, :]


def window_positions(n: int, win: int, stride: int) -> list[int]:
    if n <= win:
        return [0]
    pos = list(range(0, n - win + 1, stride))
    if pos[-1] != n - win:
        pos.append(n - win)
    return pos


def _pad_to(x: torch.Tensor, h: int, w: int) -> torch.Tensor:
    ph, pw = max(0, h - x.shape[-2]), max(0, w - x.shape[-1])
    if ph == 0 and pw == 0:
        return x
    # reflect padding needs pad < dim; fall back to replicate for tiny inputs
    mode = "reflect" if ph < x.shape[-2] and pw < x.shape[-1] else "replicate"
    return F.pad(x[None], (0, pw, 0, ph), mode=mode)[0]


def predict(
    net: torch.nn.Module,
    t1: torch.Tensor,
    t2: torch.Tensor,
    window: int = 512,
    stride: int = 384,
    geometric: int = 8,
    temporal: bool = False,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """(6,H,W) reflectance tensors -> averaged change logits (H,W) and class logits (4,H,W), on CPU."""
    if geometric not in (1, 4, 8):
        raise ValueError("tta_geometric must be 1, 4 or 8")
    transforms = D4[:geometric]
    net.eval()
    if hasattr(net, "return_semantic"):
        cast(Any, net).return_semantic = False
    dev = torch.device(device)
    h, w = int(t1.shape[-2]), int(t1.shape[-1])
    x1 = _pad_to(t1, window, window).to(dev)
    x2 = _pad_to(t2, window, window).to(dev)
    hp, wp = int(x1.shape[-2]), int(x1.shape[-1])
    acc_c = torch.zeros(hp, wp, dtype=torch.float32)
    acc_z = torch.zeros(4, hp, wp, dtype=torch.float32)
    acc_w = torch.zeros(hp, wp, dtype=torch.float32)
    wgt = cosine_window(window, window)
    with torch.no_grad():
        for r in window_positions(hp, window, stride):
            for c in window_positions(wp, window, stride):
                a = x1[:, r : r + window, c : c + window][None]
                b = x2[:, r : r + window, c : c + window][None]
                sum_c = torch.zeros(1, 1, window, window)
                sum_z = torch.zeros(1, 4, window, window)
                for k, flip in transforms:
                    ta, tb = d4_forward(a, k, flip), d4_forward(b, k, flip)
                    out = net(ta, tb)
                    cc = d4_inverse(out["change"], k, flip).float().cpu()
                    zz = d4_inverse(out["cls"], k, flip).float().cpu()
                    if temporal:
                        out2 = net(tb, ta)
                        cc = (cc + d4_inverse(out2["change"], k, flip).float().cpu()) * 0.5
                        zz = (zz + d4_inverse(out2["cls"], k, flip).float().cpu()) * 0.5
                    sum_c = sum_c + cc
                    sum_z = sum_z + zz
                n = float(len(transforms))
                acc_c[r : r + window, c : c + window] += sum_c[0, 0] / n * wgt
                acc_z[:, r : r + window, c : c + window] += sum_z[0] / n * wgt
                acc_w[r : r + window, c : c + window] += wgt
    c_out = (acc_c / acc_w)[:h, :w]
    z_out = (acc_z / acc_w)[:, :h, :w]
    return c_out, z_out


def to_tensor(arr: np.ndarray, scale: float, clip_max: float) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32) / scale).clamp(0.0, clip_max)
