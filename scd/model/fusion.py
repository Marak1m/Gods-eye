"""Symmetric fusion per level (HANDOFF.md 6.2): concat(|f1-f2|, f1+f2) -> 1x1 conv -> LayerNorm -> GELU.

Both terms are exactly invariant to swapping the dates in IEEE arithmetic (|a-b| == |b-a|,
a+b == b+a, a*b == b*a), so `fusion(f1, f2)` equals `fusion(f2, f1)` bit for bit
(`torch.equal`). Concatenation [f1, f2] is forbidden: it would learn an order.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

FUSION_MODES = ("absdiff_sum", "absdiff", "absdiff_prod")


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over channels for (B, C, H, W) tensors (channels-last internally)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class SymmetricFusion(nn.Module):
    def __init__(self, channels_per_level: Sequence[int], mode: str = "absdiff_sum") -> None:
        super().__init__()
        if mode not in FUSION_MODES:
            raise ValueError(f"fusion mode {mode!r} not in {FUSION_MODES}")
        self.mode = mode
        self.channels = tuple(int(c) for c in channels_per_level)
        mult = 1 if mode == "absdiff" else 2
        self.proj = nn.ModuleList([nn.Conv2d(c * mult, c, kernel_size=1) for c in self.channels])
        self.norm = nn.ModuleList([LayerNorm2d(c) for c in self.channels])
        self.act = nn.GELU()

    @staticmethod
    def symmetric_terms(f1: torch.Tensor, f2: torch.Tensor, mode: str) -> torch.Tensor:
        d = (f1 - f2).abs()
        if mode == "absdiff":
            return d
        if mode == "absdiff_sum":
            return torch.cat([d, f1 + f2], dim=1)
        return torch.cat([d, f1 * f2], dim=1)

    def fuse_level(self, i: int, f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        x = self.symmetric_terms(f1, f2, self.mode)
        return self.act(self.norm[i](self.proj[i](x)))

    def forward(self, f1: Sequence[torch.Tensor], f2: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        if len(f1) != len(self.channels) or len(f2) != len(self.channels):
            raise ValueError(f"expected {len(self.channels)} levels, got {len(f1)} and {len(f2)}")
        return [self.fuse_level(i, a, b) for i, (a, b) in enumerate(zip(f1, f2))]

    @staticmethod
    def diff_half(f1: Sequence[torch.Tensor], f2: Sequence[torch.Tensor]) -> list[torch.Tensor]:
        """The |f1-f2| terms alone (zero for identical inputs); used by the identity tests."""
        return [(a - b).abs() for a, b in zip(f1, f2)]
