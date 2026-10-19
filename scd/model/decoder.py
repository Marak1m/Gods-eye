"""U-Net decoder with the full-resolution |x1 - x2| skip (HANDOFF.md 6.3).

Top-down over the four fused levels with bilinear upsampling and two 3x3 conv-BN-GELU blocks per
stage. The final stage upsamples the stride-4 output x4 and, when `full_res_skip` is on,
concatenates a 16-channel stride-1 map computed from |x1 - x2| by two 3x3 convs so that 1-2 px
roads survive the stride-4 stem. Output: (B, decoder_channels[-1], H, W).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class ConvBlock(nn.Sequential):
    """Two 3x3 conv -> BN -> GELU."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )


class UNetDecoder(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int] = (96, 192, 384, 768),
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
        full_res_skip: bool = True,
        skip_in_channels: int = 6,
        skip_channels: int = 16,
    ) -> None:
        super().__init__()
        if len(in_channels) != 4 or len(decoder_channels) != 4:
            raise ValueError("expected four encoder levels and four decoder stages")
        c0, c1, c2, c3 = (int(c) for c in in_channels)
        d0, d1, d2, d3 = (int(c) for c in decoder_channels)
        self.full_res_skip = full_res_skip
        self.stage1 = ConvBlock(c3 + c2, d0)  # stride 16
        self.stage2 = ConvBlock(d0 + c1, d1)  # stride 8
        self.stage3 = ConvBlock(d1 + c0, d2)  # stride 4
        self.skip = (
            nn.Sequential(
                nn.Conv2d(skip_in_channels, skip_channels, 3, padding=1),
                nn.GELU(),
                nn.Conv2d(skip_channels, skip_channels, 3, padding=1),
                nn.GELU(),
            )
            if full_res_skip
            else None
        )
        self.stage4 = ConvBlock(d2 + (skip_channels if full_res_skip else 0), d3)  # stride 1
        self.out_channels = d3

    @staticmethod
    def _up(x: torch.Tensor, ref: torch.Tensor | tuple[int, int]) -> torch.Tensor:
        size = ref.shape[-2:] if isinstance(ref, torch.Tensor) else ref
        return F.interpolate(x, size=tuple(size), mode="bilinear", align_corners=False)

    def forward(self, feats: Sequence[torch.Tensor], x_diff: torch.Tensor | None = None) -> torch.Tensor:
        f0, f1, f2, f3 = feats
        x = self.stage1(torch.cat([self._up(f3, f2), f2], dim=1))
        x = self.stage2(torch.cat([self._up(x, f1), f1], dim=1))
        x = self.stage3(torch.cat([self._up(x, f0), f0], dim=1))
        if self.skip is not None:
            if x_diff is None:
                raise ValueError("full_res_skip is on; pass x_diff = |x1 - x2|")
            full = (int(x_diff.shape[-2]), int(x_diff.shape[-1]))
            x = torch.cat([self._up(x, full), self.skip(x_diff)], dim=1)
        else:
            full = (int(f0.shape[-2]) * 4, int(f0.shape[-1]) * 4)
            x = self._up(x, full)
        return self.stage4(x)
