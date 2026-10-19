"""SiameseChangeNet (HANDOFF.md 6.4).

forward(x1, x2) -> {'change': [B,1,H,W], 'cls': [B,4,H,W], 'sem_t1': [B,5,H,W] | None, 'sem_t2': ... | None}

Invariants: change and cls logits are identical under swap (`torch.equal` in eval mode, fp32);
the encoder is called once per date in eval mode so per-date features never depend on batch
position, and stacked into one call in training mode for throughput (DEVIATIONS U4). The
"spatially constant for identical inputs" claim of H6.4 is not an invariant of the default
fusion (DEVIATIONS C1); the identity guard in `scd.infer.pipeline` provides the exact zero.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import torch
from torch import nn

from scd.model.decoder import UNetDecoder
from scd.model.encoder import build_encoder
from scd.model.fusion import SymmetricFusion
from scd.model.heads import ChangeHead, ClassHead, SemanticDecoder, Softmax5Head


class SiameseChangeNet(nn.Module):
    def __init__(
        self,
        encoder: str = "convnext_tiny",
        encoder_init: str = "imagenet",
        in_channels: int = 6,
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
        fusion: str = "absdiff_sum",
        full_res_skip: bool = True,
        aux_semantic: bool = True,
        head_mode: str = "factorized",
        n_change_classes: int = 4,
        n_sem_classes: int = 5,
    ) -> None:
        super().__init__()
        self.encoder = build_encoder(encoder, in_channels, encoder_init)
        channels: tuple[int, ...] = tuple(cast(Any, self.encoder).feature_channels)
        self.fusion = SymmetricFusion(channels, mode=fusion)
        self.decoder = UNetDecoder(
            channels, decoder_channels, full_res_skip=full_res_skip, skip_in_channels=in_channels
        )
        self.full_res_skip = full_res_skip
        self.head_mode = head_mode
        self.n_change_classes = n_change_classes
        d_out = self.decoder.out_channels
        if head_mode == "factorized":
            self.change_head: nn.Module | None = ChangeHead(d_out)
            self.class_head: nn.Module | None = ClassHead(d_out, n_change_classes)
            self.head5: nn.Module | None = None
        elif head_mode == "softmax5":
            self.change_head = None
            self.class_head = None
            self.head5 = Softmax5Head(d_out, n_change_classes + 1)
        else:
            raise ValueError(f"unknown head_mode {head_mode!r}")
        self.semantic = SemanticDecoder(channels, decoder_channels, n_sem_classes) if aux_semantic else None
        self.return_semantic = True  # training and tests; inference turns it off

    # --- encoder calls (DEVIATIONS U4) ----------------------------------------------------
    def encode(self, x1: torch.Tensor, x2: torch.Tensor) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if self.training:
            feats = self.encoder(torch.cat([x1, x2], dim=0))
            n = x1.shape[0]
            return [f[:n] for f in feats], [f[n:] for f in feats]
        return list(self.encoder(x1)), list(self.encoder(x2))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> dict[str, torch.Tensor | None]:
        f1, f2 = self.encode(x1, x2)
        fused = self.fusion(f1, f2)
        x_diff = (x1 - x2).abs() if self.full_res_skip else None
        dec = self.decoder(fused, x_diff)
        out: dict[str, torch.Tensor | None]
        if self.head_mode == "factorized":
            assert self.change_head is not None and self.class_head is not None
            out = {"change": self.change_head(dec), "cls": self.class_head(dec)}
        else:
            assert self.head5 is not None
            logits5 = self.head5(dec)
            # keep the output contract: change = log-odds of "any change", cls = the 4 class logits
            change = torch.logsumexp(logits5[:, 1:], dim=1, keepdim=True) - logits5[:, :1]
            out = {"change": change, "cls": logits5[:, 1:]}
        if self.semantic is not None and self.return_semantic:
            out["sem_t1"] = self.semantic(f1)
            out["sem_t2"] = self.semantic(f2)
        else:
            out["sem_t1"] = None
            out["sem_t2"] = None
        return out

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def param_groups(self, lr_encoder: float, lr_decoder: float, weight_decay: float) -> list[dict[str, Any]]:
        enc = [p for p in self.encoder.parameters() if p.requires_grad]
        enc_ids = {id(p) for p in enc}
        rest = [p for p in self.parameters() if p.requires_grad and id(p) not in enc_ids]
        return [
            {"params": enc, "lr": lr_encoder, "weight_decay": weight_decay, "name": "encoder"},
            {"params": rest, "lr": lr_decoder, "weight_decay": weight_decay, "name": "decoder"},
        ]


def build_model(cfg: Any) -> SiameseChangeNet:
    m = cfg.model
    return SiameseChangeNet(
        encoder=m.encoder,
        encoder_init=m.encoder_init,
        in_channels=m.in_channels,
        decoder_channels=m.decoder_channels,
        fusion=m.fusion,
        full_res_skip=m.full_res_skip,
        aux_semantic=m.aux_semantic,
        head_mode=m.head_mode,
        n_change_classes=m.n_change_classes,
        n_sem_classes=m.n_sem_classes,
    )
