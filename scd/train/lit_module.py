"""ChangeLitModule (HANDOFF.md 7.1, 7.3; DEVIATIONS U1).

AdamW with two parameter groups (encoder 1e-4, rest 3e-4), linear warmup then cosine to lr_min,
EMA of the weights (decay 0.999) evaluated alongside the raw weights, gradient clipping 1.0.
Validation each epoch: full-chip inference on the val split accumulating one confusion matrix
globally and per stratum (`RubricMetrics`), plus the simulated robustness cases on a fixed subset
(`robustness.simulate`). Logged keys: val/iou_*, val/p_*, val/r_*, val/weighted_iou,
val/macro_iou, val/stratum_*_weighted_iou, val/robust_*, val/leaderboard_proxy (the better of raw
and EMA; val/proxy_source 0 = raw, 1 = ema). `val/robust_swap` must be 1.0: a violation is logged
at ERROR level and counted in val/swap_violations rather than aborting a long run.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, cast

import lightning as L
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

from scd.config import Config
from scd.data.dataset import DrawRequest, PairChipDataset
from scd.labels.synthetic import ObjectLibrary
from scd.log import get_logger
from scd.model.losses import TotalLoss, class_weight_tensor
from scd.model.net import SiameseChangeNet, build_model
from scd.train import robustness
from scd.train.metrics import RubricMetrics, decide

log = get_logger(__name__)


class ChangeLitModule(L.LightningModule):
    def __init__(
        self,
        cfg: Config,
        class_weights: dict[str, float],
        val_dataset: PairChipDataset | None = None,
        robust_requests: Sequence[DrawRequest] | None = None,
        library: ObjectLibrary | None = None,
        steps_per_epoch: int = 1,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.class_weights = dict(class_weights)
        self.save_hyperparameters({"config": cfg.to_dict(), "class_weights": self.class_weights})
        self.net: SiameseChangeNet = build_model(cfg)
        self.ema = AveragedModel(self.net, multi_avg_fn=get_ema_multi_avg_fn(cfg.train.ema_decay), use_buffers=True)
        loss_cfg = cfg.train.loss
        self.loss = TotalLoss(
            pos_weight=loss_cfg.pos_weight,
            lambda_sem=loss_cfg.lambda_sem,
            class_weights=class_weight_tensor(self.class_weights),
            cldice_road=loss_cfg.cldice_road,
        )
        self.val_dataset = val_dataset
        self.robust_requests = list(robust_requests or [])
        self.library = library
        self.steps_per_epoch = max(1, int(steps_per_epoch))
        self.metrics_raw = RubricMetrics()
        self.metrics_ema = RubricMetrics()
        self.best_source = "raw"
        self.swap_violations = 0

    # --- training ---------------------------------------------------------------------------
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> dict[str, torch.Tensor | None]:
        return self.net(x1, x2)

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        out = self.net(batch["x1"], batch["x2"])
        terms = self.loss(out, batch)
        bs = batch["x1"].shape[0]
        for k, v in terms.items():
            self.log(f"train/{k}", v.detach(), on_step=True, on_epoch=True, prog_bar=(k == "total"), batch_size=bs)
        return terms["total"]

    def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
        self.ema.update_parameters(self.net)

    def configure_optimizers(self) -> Any:
        t = self.cfg.train
        groups = self.net.param_groups(t.lr_encoder, t.lr_decoder, t.weight_decay)
        opt = torch.optim.AdamW(groups)
        total_steps = max(1, t.epochs * self.steps_per_epoch)
        warmup = int(t.warmup_epochs * self.steps_per_epoch)
        floor = t.lr_min / t.lr_decoder

        def lr_lambda(step: int) -> float:
            if warmup > 0 and step < warmup:
                return (step + 1) / warmup
            progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
            return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    # --- validation -------------------------------------------------------------------------
    def on_validation_epoch_start(self) -> None:
        self.metrics_raw.reset()
        self.metrics_ema.reset()
        self.net.return_semantic = False
        cast(SiameseChangeNet, self.ema.module).return_semantic = False

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> None:
        strata = [m["stratum"] for m in batch["meta"]]
        valid = batch["valid"].bool()
        with torch.no_grad():
            out = self.net(batch["x1"], batch["x2"])
            assert out["change"] is not None and out["cls"] is not None
            self.metrics_raw.update(decide(out["change"], out["cls"]).cpu(), batch["change"].cpu(), valid.cpu(), strata)
            out_e = self.ema.module(batch["x1"], batch["x2"])
            self.metrics_ema.update(
                decide(out_e["change"], out_e["cls"]).cpu(), batch["change"].cpu(), valid.cpu(), strata
            )

    def _robust(self, net: torch.nn.Module) -> dict[str, float]:
        if self.val_dataset is None or not self.robust_requests:
            return {"robust_swap": 1.0, "robust_identity": 1.0, "robust_near": 1.0, "robust_synthetic": 0.0}
        return robustness.simulate(
            net,
            self.val_dataset,
            self.robust_requests,
            self.cfg,
            library=self.library,
            device=self.device,
            seed=self.current_epoch,
        )

    def on_validation_epoch_end(self) -> None:
        raw = self.metrics_raw.compute(self._robust(self.net), prefix="val/")
        ema = self.metrics_ema.compute(self._robust(self.ema.module), prefix="val_ema/")
        proxy_raw = raw["val/leaderboard_proxy"]
        proxy_ema = ema["val_ema/leaderboard_proxy"]
        use_ema = proxy_ema > proxy_raw
        self.best_source = "ema" if use_ema else "raw"
        chosen = {k.replace("val_ema/", "val/"): v for k, v in ema.items()} if use_ema else raw
        for k, v in {**raw, **ema, **chosen}.items():
            self.log(k, float(v), on_epoch=True, prog_bar=False, sync_dist=False)
        self.log("val/proxy_source", 1.0 if use_ema else 0.0, on_epoch=True)
        swap = chosen["val/robust_swap"]
        if swap < 1.0:
            self.swap_violations += 1
            log.error("val/robust_swap = %.6f != 1.0: order invariance broken (epoch %d)", swap, self.current_epoch)
        self.log("val/swap_violations", float(self.swap_violations), on_epoch=True)
        self.net.return_semantic = True
        cast(SiameseChangeNet, self.ema.module).return_semantic = True

    # --- checkpoints ------------------------------------------------------------------------
    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["scd"] = {
            "best_source": self.best_source,
            "config": self.cfg.to_dict(),
            "config_hash": self.cfg.hash(),
            "class_weights": self.class_weights,
            "epoch": self.current_epoch,
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        self.best_source = checkpoint.get("scd", {}).get("best_source", "raw")


def net_from_checkpoint(path: Any, cfg: Config, prefer: str | None = None) -> tuple[SiameseChangeNet, str]:
    """Rebuild the network from a Lightning checkpoint, choosing EMA or raw weights."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    meta = ckpt.get("scd", {})
    source = prefer or meta.get("best_source", "raw")
    state = ckpt["state_dict"]
    prefix = "ema.module." if source == "ema" else "net."
    sub = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    if not sub:
        raise ValueError(f"{path}: no weights with prefix {prefix!r}")
    net = build_model(cfg)
    missing, unexpected = net.load_state_dict(sub, strict=False)
    missing = [m for m in missing if not m.startswith("n_averaged")]
    if missing or unexpected:
        raise ValueError(f"{path}: state mismatch, missing={missing[:5]} unexpected={unexpected[:5]}")
    net.eval()
    return net, source
