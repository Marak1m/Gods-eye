"""Augmentations (HANDOFF.md 5.13, plan doc 9.2) as torch ops on CPU tensors.

Shared geometric ops act identically on x1, x2 and every label; radiometric jitter is asymmetric
per date; the relative shift moves one date only and leaves labels untouched; cloud paste marks
its blob invalid; band dropout zeroes B11/B12 on both dates. Date-asymmetric ops are skipped for
IDENTICAL draws so the exact zero is actually taught (DEVIATIONS C14).

Sample layout: x1, x2 (6, H, W) float32 reflectance; change, sem_t1, sem_t2 (H, W) int64 with 255
ignore; valid (H, W) bool.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from scd.config import Config
from scd.constants import BAND_INDEX, IGNORE, TileType

Sample = dict[str, Any]
LABEL_KEYS = ("change", "sem_t1", "sem_t2")
IMAGE_KEYS = ("x1", "x2")
HAZE_PROFILE = torch.tensor([1.0, 0.8, 0.6, 0.3, 0.1, 0.05]).view(6, 1, 1)


def d4(sample: Sample, k: int, flip: bool) -> Sample:
    """Rotate by k*90 degrees and optionally flip left-right, identically on every tensor."""
    out = dict(sample)
    for key in (*IMAGE_KEYS, *LABEL_KEYS, "valid"):
        t = torch.rot90(sample[key], k, dims=(-2, -1))
        if flip:
            t = torch.flip(t, dims=(-1,))
        out[key] = t.contiguous()
    return out


def affine_shared(sample: Sample, angle_deg: float, scale: float) -> Sample:
    """Rotate/scale about the centre; images bilinear, labels nearest; out-of-bounds -> 255 / invalid."""
    h, w = sample["valid"].shape
    a = math.radians(angle_deg)
    cos, sin = math.cos(a) / scale, math.sin(a) / scale
    theta = torch.tensor([[cos, -sin * h / w, 0.0], [sin * w / h, cos, 0.0]], dtype=torch.float32)[None]
    grid = F.affine_grid(theta, (1, 1, h, w), align_corners=False)

    def warp(t: torch.Tensor, mode: str) -> torch.Tensor:
        return F.grid_sample(t[None].float(), grid, mode=mode, padding_mode="zeros", align_corners=False)[0]

    inb = warp(torch.ones(1, h, w), "nearest")[0] > 0.5
    out = dict(sample)
    for key in IMAGE_KEYS:
        out[key] = warp(sample[key], "bilinear")
    for key in LABEL_KEYS:
        lab = warp(sample[key][None], "nearest")[0].round().long()
        lab[~inb] = IGNORE
        out[key] = lab
    valid = warp(sample["valid"][None].float(), "nearest")[0] > 0.5
    out["valid"] = valid & inb
    return out


def temporal_swap(sample: Sample) -> Sample:
    out = dict(sample)
    out["x1"], out["x2"] = sample["x2"], sample["x1"]
    out["sem_t1"], out["sem_t2"] = sample["sem_t2"], sample["sem_t1"]
    return out


def radiometric(x: torch.Tensor, rng: np.random.Generator, cfg: Config) -> torch.Tensor:
    """Asymmetric per-date jitter: brightness, contrast, gamma, band gain, haze, noise."""
    r = cfg.augment.radiometric
    clip_max = cfg.bands.clip_max
    out = x.clone()
    out = out * float(1.0 + rng.uniform(-r.brightness, r.brightness))
    mean = out.mean(dim=(-2, -1), keepdim=True)
    out = mean + (out - mean) * float(1.0 + rng.uniform(-r.contrast, r.contrast))
    g = float(rng.uniform(*r.gamma))
    out = (out.clamp(0.0, clip_max) / clip_max).pow(g) * clip_max
    gains = torch.tensor(rng.uniform(1.0 - r.band_gain, 1.0 + r.band_gain, size=6), dtype=torch.float32).view(6, 1, 1)
    out = out * gains
    if rng.random() < r.haze_p:
        out = out + float(rng.uniform(0.0, 0.05)) * HAZE_PROFILE
    sigma = float(rng.uniform(*r.noise_sigma))
    out = out + torch.from_numpy(rng.normal(0.0, sigma, size=tuple(out.shape)).astype(np.float32))
    return out.clamp(0.0, clip_max)


def relative_shift(sample: Sample, dx: int, dy: int, which: str = "x2") -> Sample:
    """Integer shift of one date's image; exposed border zero; labels and validity untouched."""
    out = dict(sample)
    x = torch.roll(sample[which], shifts=(dy, dx), dims=(-2, -1)).clone()
    if dy > 0:
        x[..., :dy, :] = 0
    elif dy < 0:
        x[..., dy:, :] = 0
    if dx > 0:
        x[..., :, :dx] = 0
    elif dx < 0:
        x[..., :, dx:] = 0
    out[which] = x
    return out


def cloud_paste(sample: Sample, rng: np.random.Generator, which: str, clip_max: float) -> Sample:
    """Paste a bright elliptical cloud blob on one date; the blob becomes invalid / 255."""
    h, w = sample["valid"].shape
    cy, cx = rng.uniform(0, h), rng.uniform(0, w)
    ry, rx = rng.uniform(h / 10, h / 3), rng.uniform(w / 10, w / 3)
    yy, xx = np.mgrid[0:h, 0:w]
    blob = (((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2) <= 1.0
    blob_t = torch.from_numpy(blob)
    out = dict(sample)
    x = sample[which].clone()
    bright = torch.tensor([0.95, 0.93, 0.92, 0.90, 0.60, 0.50]).view(6, 1, 1).clamp(max=clip_max)
    if blob.any():
        x[:, blob_t] = bright.view(6, 1).expand(6, int(blob.sum()))
    out[which] = x
    out["valid"] = sample["valid"] & ~blob_t
    for key in LABEL_KEYS:
        lab = sample[key].clone()
        lab[blob_t] = IGNORE
        out[key] = lab
    return out


def band_dropout(sample: Sample) -> Sample:
    out = dict(sample)
    for key in IMAGE_KEYS:
        x = sample[key].clone()
        x[[BAND_INDEX["B11"], BAND_INDEX["B12"]]] = 0.0
        out[key] = x
    return out


def apply_all(sample: Sample, req: Any, cfg: Config, rng: np.random.Generator) -> Sample:
    """The full policy for one draw. `req.tile_type` and `req.swap` come from the sampler."""
    a = cfg.augment
    identical = str(req.tile_type) == TileType.IDENTICAL.value
    out = sample
    if rng.random() < a.geometric.d4:
        out = d4(out, int(rng.integers(0, 4)), bool(rng.integers(0, 2)))
    angle = float(rng.uniform(-a.geometric.rotate_deg, a.geometric.rotate_deg)) if rng.random() < a.geometric.rotate_p else 0.0
    scale = float(rng.uniform(*a.geometric.scale)) if rng.random() < a.geometric.scale_p else 1.0
    if angle != 0.0 or scale != 1.0:
        out = affine_shared(out, angle, scale)
    if getattr(req, "swap", False):
        out = temporal_swap(out)
    if not identical:
        out = dict(out)
        out["x1"] = radiometric(out["x1"], rng, cfg)
        out["x2"] = radiometric(out["x2"], rng, cfg)
        if a.shift.max_px > 0 and rng.random() < a.shift.p:
            dx, dy = 0, 0
            while dx == 0 and dy == 0:
                dx, dy = int(rng.integers(-a.shift.max_px, a.shift.max_px + 1)), int(rng.integers(-a.shift.max_px, a.shift.max_px + 1))
            out = relative_shift(out, dx, dy, which="x2" if rng.random() < 0.5 else "x1")
        if rng.random() < a.cloud_paste_p:
            out = cloud_paste(out, rng, "x2" if rng.random() < 0.5 else "x1", cfg.bands.clip_max)
    if rng.random() < a.band_dropout_p:
        out = band_dropout(out)
    return out
