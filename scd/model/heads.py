"""Heads (HANDOFF.md 6.3): change (1 logit), class (4 logits), auxiliary semantic decoder (5 logits).

The semantic decoder is a second `UNetDecoder` without the full-resolution skip, shared across
dates and applied to f1 and f2 separately; it exists for training-time regularisation only.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from scd.model.decoder import UNetDecoder


class ChangeHead(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ClassHead(nn.Module):
    def __init__(self, in_channels: int, n_classes: int = 4) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, n_classes, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Softmax5Head(nn.Module):
    """Ablation A5: one 5-way head (none + 4 classes) instead of where x what."""

    def __init__(self, in_channels: int, n_classes: int = 5) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, n_classes, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SemanticDecoder(nn.Module):
    def __init__(
        self,
        in_channels: Sequence[int],
        decoder_channels: Sequence[int],
        n_classes: int = 5,
    ) -> None:
        super().__init__()
        self.decoder = UNetDecoder(in_channels, decoder_channels, full_res_skip=False)
        self.head = nn.Conv2d(self.decoder.out_channels, n_classes, 3, padding=1)

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        return self.head(self.decoder(feats))
