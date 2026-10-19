"""ConvNeXt-tiny encoder with the explicit 6-channel stem rule (HANDOFF.md 6.1).

`timm.create_model(..., features_only=True)` gives four feature maps at strides 4, 8, 16, 32 with
96, 192, 384, 768 channels. The stem is widened to 6 input channels by hand, not by timm's
default adaptation: W6[:, 0] = W_rgb[:, 2] (B02 <- blue), W6[:, 1] = W_rgb[:, 1] (B03 <- green),
W6[:, 2] = W_rgb[:, 0] (B04 <- red), W6[:, 3:6] = mean over RGB (B08, B11, B12).
"""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

STRIDES: tuple[int, ...] = (4, 8, 16, 32)
CHANNELS: dict[str, tuple[int, ...]] = {"convnext_tiny": (96, 192, 384, 768)}
DINOV3_REPO = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"


def adapt_stem(w_rgb: torch.Tensor, in_channels: int = 6) -> torch.Tensor:
    """(out, 3, k, k) RGB stem weights -> (out, in_channels, k, k) in our band order."""
    if w_rgb.ndim != 4 or w_rgb.shape[1] != 3:
        raise ValueError(f"expected (out, 3, k, k) weights, got {tuple(w_rgb.shape)}")
    if in_channels < 4:
        raise ValueError("in_channels must be >= 4 (B02, B03, B04, B08)")
    out = torch.zeros(w_rgb.shape[0], in_channels, *w_rgb.shape[2:], dtype=w_rgb.dtype, device=w_rgb.device)
    out[:, 0] = w_rgb[:, 2]  # B02 <- blue
    out[:, 1] = w_rgb[:, 1]  # B03 <- green
    out[:, 2] = w_rgb[:, 0]  # B04 <- red
    out[:, 3:] = w_rgb.mean(dim=1, keepdim=True)  # B08, B11, B12 <- mean of RGB filters
    return out


def _first_conv(module: nn.Module) -> tuple[str, nn.Conv2d]:
    for name, m in module.named_modules():
        if isinstance(m, nn.Conv2d) and m.in_channels == 3:
            return name, m
    raise RuntimeError("no 3-channel stem convolution found")


def _set_module(root: nn.Module, dotted: str, new: nn.Module) -> None:
    parts = dotted.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def build_encoder(name: str = "convnext_tiny", in_channels: int = 6, init: str = "imagenet") -> nn.Module:
    """features_only encoder (strides 4, 8, 16, 32) with the explicit stem adaptation."""
    import timm

    if init == "imagenet":
        enc = timm.create_model(name, pretrained=True, features_only=True, out_indices=(0, 1, 2, 3), in_chans=3)
    elif init == "random":
        enc = timm.create_model(name, pretrained=False, features_only=True, out_indices=(0, 1, 2, 3), in_chans=3)
    elif init == "v3_roads":
        raise NotImplementedError(
            "encoder_init=v3_roads needs the frozen v3.1 checkpoint under third_party/v3_roads (ablation A9); "
            "see third_party/v3_roads/README.md"
        )
    elif init == "dinov3_convnext_tiny":
        raise NotImplementedError(
            f"encoder_init=dinov3_convnext_tiny needs the gated weights {DINOV3_REPO} (HF_TOKEN) and a key "
            "mapping into timm convnext_tiny (ablation A13a); not built in this pass"
        )
    else:
        raise ValueError(f"unknown encoder_init {init!r}")
    stem_name, conv = _first_conv(enc)
    new = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,  # type: ignore[arg-type]
        stride=conv.stride,  # type: ignore[arg-type]
        padding=conv.padding,  # type: ignore[arg-type]
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        new.weight.copy_(adapt_stem(conv.weight.detach(), in_channels))
        if conv.bias is not None and new.bias is not None:
            new.bias.copy_(conv.bias.detach())
    _set_module(enc, stem_name, new)
    info = cast(Any, enc).feature_info
    channels = tuple(int(c) for c in info.channels())
    expected = CHANNELS.get(name)
    if expected is not None and channels != expected:
        raise RuntimeError(f"{name}: feature channels {channels} != expected {expected}")
    cast(Any, enc).feature_channels = channels
    cast(Any, enc).feature_strides = tuple(int(r) for r in info.reduction())
    return enc
