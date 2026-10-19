"""Simulated robustness cases on a fixed validation subset (HANDOFF.md 7.1, plan doc 11).

For each chip (full, un-augmented): swapped-order agreement, identical-input change fraction,
pasted-object recovery IoU and near-identical change fraction, all on the raw network output
(the identity guard lives in the inference pipeline). Scores are "higher is better":
robust_swap = 1 - differing fraction, robust_identity = 1 - change fraction,
robust_near = 1 - change fraction, robust_synthetic = mean IoU of pasted objects.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np
import torch
from scipy import ndimage

from scd.config import Config
from scd.data.dataset import DrawRequest, PairChipDataset, near_identical
from scd.labels.synthetic import ObjectLibrary, paste
from scd.train.metrics import decide


def to_tensor(arr: np.ndarray, cfg: Config, device: torch.device | str = "cpu") -> torch.Tensor:
    x = torch.from_numpy(arr.astype(np.float32) / cfg.bands.scale).clamp(0.0, cfg.bands.clip_max)
    return x[None].to(device)


def _mask(
    net: torch.nn.Module, x1: torch.Tensor, x2: torch.Tensor, tau: float, bias: Sequence[float] | None
) -> torch.Tensor:
    out = net(x1, x2)
    return decide(out["change"], out["cls"], tau, bias)[0]


def simulate(
    net: torch.nn.Module,
    dataset: PairChipDataset,
    requests: Sequence[DrawRequest],
    cfg: Config,
    tau: float = 0.5,
    bias: Sequence[float] | None = None,
    library: ObjectLibrary | None = None,
    device: torch.device | str = "cpu",
    seed: int = 0,
) -> dict[str, float]:
    was_training = net.training
    net.eval()
    sem_flag = getattr(net, "return_semantic", None)
    if sem_flag is not None:
        cast(Any, net).return_semantic = False
    swap_agree: list[float] = []
    ident_frac: list[float] = []
    near_frac: list[float] = []
    synth_iou: list[float] = []
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for req in requests:
            t1, t2, valid, change, _s1, _s2 = dataset.read_chip(req.chip_id)
            v = torch.from_numpy(valid.astype(bool)).to(device)
            if not bool(v.any()):
                continue
            x1, x2 = to_tensor(t1, cfg, device), to_tensor(t2, cfg, device)
            m_ab = _mask(net, x1, x2, tau, bias)
            m_ba = _mask(net, x2, x1, tau, bias)
            swap_agree.append(float((m_ab == m_ba)[v].float().mean()))
            m_aa = _mask(net, x1, x1.clone(), tau, bias)
            ident_frac.append(float((m_aa > 0)[v].float().mean()))
            x1n = to_tensor(near_identical(t1, rng, cfg, valid), cfg, device)
            m_near = _mask(net, x1, x1n, tau, bias)
            near_frac.append(float((m_near > 0)[v].float().mean()))
            if library is not None and len(library):
                objs = library.sample_mix(rng, cfg.sampler.class_shares, 3)
                p1, p2, _pc, recs = paste(t1, t1.copy(), np.zeros_like(change), valid, objs, rng, cfg)
                if recs:
                    m_syn = _mask(net, to_tensor(p1, cfg, device), to_tensor(p2, cfg, device), tau, bias).cpu().numpy()
                    for r in recs:
                        region = ndimage.binary_dilation(r.mask, iterations=3)
                        pred = (m_syn == r.class_id) & region
                        inter = (pred & r.mask).sum()
                        union = (pred | r.mask).sum()
                        synth_iou.append(float(inter / union) if union > 0 else 0.0)
    if sem_flag is not None:
        cast(Any, net).return_semantic = sem_flag
    if was_training:
        net.train()

    def mean(xs: list[float], default: float) -> float:
        return float(np.mean(xs)) if xs else default

    return {
        "robust_swap": mean(swap_agree, 1.0),
        "robust_identity": 1.0 - mean(ident_frac, 0.0),
        "robust_near": 1.0 - mean(near_frac, 0.0),
        "robust_synthetic": mean(synth_iou, 0.0),
        "robust_n_chips": float(len(swap_agree)),
    }


def pick_subset(requests: Sequence[DrawRequest], n: int, seed: int = 0) -> list[DrawRequest]:
    """A fixed subset of validation chips (deterministic given seed)."""
    if len(requests) <= n:
        return list(requests)
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(requests), size=n, replace=False).tolist())
    return [requests[i] for i in idx]


def summary_row(scores: dict[str, float], **extra: Any) -> dict[str, Any]:
    return {**{k: round(v, 5) for k, v in scores.items()}, **extra}
