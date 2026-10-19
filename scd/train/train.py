"""Training entry points (HANDOFF.md 7, 13 M6/M9).

`run()` builds the data pipeline (two-stage sampler, augmentations, object library), the Lightning
module and trainer (top-5 checkpoints on val/leaderboard_proxy + milestones every N epochs +
last.ckpt, early stopping, CSV metrics) and trains. `finetune()` (Stage III, M9) is a stub.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightning as L
import pandas as pd
import torch
from lightning.pytorch.callbacks import Callback, EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from scd.config import Config
from scd.data import manifest as man_mod
from scd.data.dataset import PairChipDataset, collate, full_chip_requests
from scd.data.sampler import TwoStageSampler
from scd.labels.synthetic import ObjectLibrary
from scd.log import get_logger
from scd.train.lit_module import ChangeLitModule
from scd.train.robustness import pick_subset

log = get_logger(__name__)


class MilestoneCheckpoint(Callback):
    """Save runs/<run>/checkpoints/milestone_epochNNN.ckpt every `every` epochs (HANDOFF 7.3)."""

    def __init__(self, dirpath: Path, every: int) -> None:
        self.dirpath = Path(dirpath)
        self.every = max(1, int(every))

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        epoch = trainer.current_epoch + 1
        if epoch % self.every == 0:
            self.dirpath.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(self.dirpath / f"milestone_epoch{epoch:03d}.ckpt")


class SamplerEpoch(Callback):
    def __init__(self, sampler: TwoStageSampler) -> None:
        self.sampler = sampler

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        self.sampler.set_epoch(trainer.current_epoch)


def _splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    if len(train) == 0:
        raise ValueError("manifest has no train chips")
    if len(val) == 0:
        log.warning("manifest has no val chips; validating on the training chips (smoke/debug only)")
        val = train
    return train, val


def setup_speed(cfg: Config) -> None:
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def build_loaders(cfg: Config, chips_version: int) -> dict[str, Any]:
    chips_dir = cfg.chips_dir(chips_version)
    df = man_mod.load(chips_dir / "manifest.parquet")
    train_df, val_df = _splits(df)
    objects_dir = cfg.objects_dir(chips_version)
    library = ObjectLibrary.load(objects_dir) if objects_dir.exists() else None
    if library is not None and len(library) == 0:
        library = None
    train_ds = PairChipDataset(
        chips_dir, train_df, cfg, tile_px=cfg.grid.tile_px, augment=True, library_dir=objects_dir
    )
    sampler = TwoStageSampler(
        train_df,
        cfg,
        epoch_len=cfg.train.crops_per_epoch,
        seed=cfg.train.seed,
        library=library,
        library_dir=objects_dir,
    )
    nw = cfg.train.num_workers
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        sampler=sampler,
        num_workers=nw,
        collate_fn=collate,
        drop_last=True,
        persistent_workers=nw > 0,
        pin_memory=torch.cuda.is_available(),
    )
    val_ds = PairChipDataset(chips_dir, val_df, cfg, tile_px=cfg.grid.chip_px, augment=False)
    val_requests = full_chip_requests(val_df, str(val_df["split"].iloc[0]))
    val_loader = DataLoader(val_ds, batch_size=1, sampler=val_requests, num_workers=0, collate_fn=collate)
    robust_requests = pick_subset(val_requests, cfg.train.robustness_subset, seed=cfg.train.seed)
    class_weights = man_mod.class_weights(df, cfg.train.loss.class_weight_power)
    return {
        "manifest": df,
        "train_loader": train_loader,
        "val_loader": val_loader,
        "val_dataset": val_ds,
        "robust_requests": robust_requests,
        "library": library,
        "class_weights": class_weights,
        "sampler": sampler,
        "steps_per_epoch": max(1, cfg.train.crops_per_epoch // cfg.train.batch_size),
    }


def run(cfg: Config, run_name: str, chips_version: int = 1, resume: bool = False) -> Path:
    """Train; returns runs/<run_name>/ with checkpoints/, logs/metrics.csv, config.yaml, run.json."""
    L.seed_everything(cfg.train.seed, workers=True)
    setup_speed(cfg)
    run_dir = cfg.run_dir(run_name)
    ckpt_dir = run_dir / "checkpoints"
    run_dir.mkdir(parents=True, exist_ok=True)
    parts = build_loaders(cfg, chips_version)
    module = ChangeLitModule(
        cfg,
        parts["class_weights"],
        val_dataset=parts["val_dataset"],
        robust_requests=parts["robust_requests"],
        library=parts["library"],
        steps_per_epoch=parts["steps_per_epoch"],
    )
    cfg.dump(run_dir / "config.yaml")
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "run_name": run_name,
                "chips_version": chips_version,
                "config_hash": cfg.hash(),
                "class_weights": parts["class_weights"],
                "n_train_chips": int((parts["manifest"]["split"] == "train").sum()),
                "n_val_chips": int((parts["manifest"]["split"] == "val").sum()),
                "params": module.net.param_count(),
            },
            indent=2,
        )
    )
    ck = cfg.train.checkpoint
    callbacks: list[Callback] = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            monitor=ck.monitor,
            mode="max",
            save_top_k=ck.save_top_k,
            save_last=True,
            filename="epoch{epoch:03d}-proxy{val/leaderboard_proxy:.4f}",
            auto_insert_metric_name=False,
        ),
        MilestoneCheckpoint(ckpt_dir, ck.milestone_every),
        EarlyStopping(monitor=ck.monitor, mode="max", patience=cfg.train.early_stop_patience),
        LearningRateMonitor(logging_interval="epoch"),
        SamplerEpoch(parts["sampler"]),
    ]
    trainer = L.Trainer(
        max_epochs=cfg.train.epochs,
        max_steps=cfg.train.max_steps if cfg.train.max_steps else -1,
        precision=cfg.train.precision,  # type: ignore[arg-type]
        accelerator="auto",
        devices=1,
        gradient_clip_val=1.0,
        callbacks=callbacks,
        logger=CSVLogger(save_dir=str(run_dir), name="logs", version=""),
        default_root_dir=str(run_dir),
        log_every_n_steps=1,
        num_sanity_val_steps=0,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    last = ckpt_dir / "last.ckpt"
    trainer.fit(
        module, parts["train_loader"], parts["val_loader"], ckpt_path=str(last) if resume and last.exists() else None
    )
    log.info("training finished: %s (best %s)", run_dir, getattr(trainer.checkpoint_callback, "best_model_path", "-"))
    return run_dir


def best_checkpoint(run_dir: Path) -> Path:
    """The top-k checkpoint with the highest proxy in its name, else last.ckpt."""
    ckpts = sorted((run_dir / "checkpoints").glob("epoch*-proxy*.ckpt"))
    if ckpts:
        return max(ckpts, key=lambda p: float(p.stem.split("proxy")[-1]))
    last = run_dir / "checkpoints" / "last.ckpt"
    if last.exists():
        return last
    raise FileNotFoundError(f"no checkpoints under {run_dir / 'checkpoints'}")


def finetune(cfg: Config, run_name: str) -> Path:
    """Stage-III fine-tune (plan doc 12): 50/50 sampler, frozen early stages, forgetting check."""
    raise NotImplementedError(
        "scd.train.train.finetune: built at M9; see HANDOFF.md section 13 and plan doc section 12"
    )


def forgetting_table(cfg: Config, base_run: str, ft_run: str) -> pd.DataFrame:
    raise NotImplementedError("scd.train.train.forgetting_table: built at M9")
