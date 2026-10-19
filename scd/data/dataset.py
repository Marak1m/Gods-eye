"""PairChipDataset (HANDOFF.md 5.13): DrawRequest -> dict(x1, x2, change, sem_t1, sem_t2, valid, weight, meta).

Tensors are float32 reflectance in [0, clip_max], shape (6, tile, tile); labels int64 with 255
ignore. IDENTICAL, NEAR_IDENTICAL, SYNTHETIC and STAR draws are synthesised here from a host chip.
Everything is deterministic given `req.seed`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from torch.utils.data import Dataset

from scd.config import Config
from scd.constants import IGNORE, TileType
from scd.data import manifest as man_mod
from scd.data import transforms
from scd.geo.io import read_raster
from scd.labels.synthetic import SynthObject, paste


@dataclass(frozen=True)
class DrawRequest:
    """One draw. `centre_class` asks for a crop near a component of that class (CHANGE_POS)."""

    chip_id: str
    tile_type: str
    seed: int
    swap: bool = False
    centre_class: int | None = None
    crop_centre: tuple[int, int] | None = None
    synthetic_objs: tuple[str, ...] = ()
    star_chip_id: str | None = None
    weight: float = 1.0
    augment: bool = True


def near_identical(
    t1: np.ndarray, rng: np.random.Generator, cfg: Config, valid: np.ndarray | None = None
) -> np.ndarray:
    """Noise, optional 1 px shift and optional slight blur on a copy; label stays all-zero.

    With `valid`, pixels whose shifted value would come from invalid (nodata) pixels keep their
    original value, so the perturbation never drags a nodata border into valid data.
    """
    x = t1.astype(np.float32) / cfg.bands.scale
    original = x.copy()
    sigma = float(rng.uniform(*cfg.augment.radiometric.noise_sigma))
    x = x + rng.normal(0.0, sigma, x.shape).astype(np.float32)
    if rng.random() < 0.5:
        dy, dx = int(rng.integers(-1, 2)), int(rng.integers(-1, 2))
        x = ndimage.shift(x, (0, dy, dx), order=0, mode="nearest")  # edge replication, no wrap-around
        if valid is not None:
            src_valid = ndimage.shift(valid.astype(np.uint8), (dy, dx), order=0, mode="nearest").astype(bool)
            x = np.where(src_valid[None], x, original)
    if rng.random() < 0.5:
        x = ndimage.gaussian_filter(x, sigma=(0, 0.6, 0.6))
    x = np.clip(x, 0.0, cfg.bands.clip_max) * cfg.bands.scale
    return np.rint(x).astype(np.uint16)


def star_label(sem_a: np.ndarray, sem_b: np.ndarray) -> np.ndarray:
    """XOR of two single-date semantic masks: class of the date that has it; 255 where either is unknown."""
    out = np.zeros(sem_a.shape, dtype=np.uint8)
    unknown = (sem_a == IGNORE) | (sem_b == IGNORE)
    diff = (sem_a != sem_b) & ~unknown
    out[diff] = np.where(sem_a[diff] > 0, sem_a[diff], sem_b[diff])
    out[unknown] = IGNORE
    return out


class PairChipDataset(Dataset):
    def __init__(
        self,
        chips_dir: Path,
        manifest: pd.DataFrame,
        cfg: Config,
        tile_px: int | None = None,
        augment: bool = True,
        library_dir: Path | None = None,
    ) -> None:
        self.chips_dir = Path(chips_dir)
        self.df = manifest.set_index("chip_id", drop=False)
        self.cfg = cfg
        self.tile = int(tile_px or cfg.grid.tile_px)
        self.augment = augment
        self.library_dir = Path(library_dir) if library_dir else None
        self._objects: dict[str, SynthObject] = {}

    def __len__(self) -> int:
        return len(self.df)

    def read_chip(self, chip_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        p = man_mod.chip_paths(self.chips_dir, chip_id)
        t1, _ = read_raster(p["t1"])
        t2, _ = read_raster(p["t2"])
        valid, _ = read_raster(p["valid"])
        change, _ = read_raster(p["change"])
        s1, _ = read_raster(p["sem_t1"])
        s2, _ = read_raster(p["sem_t2"])
        return t1, t2, valid[0], change[0], s1[0], s2[0]

    def _object(self, rel_path: str) -> SynthObject:
        if rel_path not in self._objects:
            base = self.library_dir if self.library_dir else Path()
            self._objects[rel_path] = SynthObject.load(base / rel_path if not Path(rel_path).is_absolute() else Path(rel_path))
        return self._objects[rel_path]

    def _crop_centre(self, req: DrawRequest, change: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
        h, w = change.shape
        half = self.tile // 2
        lo_r, hi_r = half, max(h - half, half)
        lo_c, hi_c = half, max(w - half, half)
        if req.crop_centre is not None:
            r, c = req.crop_centre
        elif req.centre_class is not None and rng.random() < self.cfg.sampler.change_centred_prob:
            pts = np.argwhere(change == req.centre_class)
            if len(pts) == 0:
                pts = np.argwhere((change > 0) & (change != IGNORE))
            if len(pts):
                pr, pc = pts[int(rng.integers(0, len(pts)))]
                rad = self.cfg.sampler.crop_centre_radius_px
                r, c = int(pr + rng.integers(-rad, rad + 1)), int(pc + rng.integers(-rad, rad + 1))
            else:
                r, c = int(rng.integers(lo_r, hi_r + 1)), int(rng.integers(lo_c, hi_c + 1))
        else:
            r, c = int(rng.integers(lo_r, hi_r + 1)), int(rng.integers(lo_c, hi_c + 1))
        return int(np.clip(r, lo_r, hi_r)), int(np.clip(c, lo_c, hi_c))

    def __getitem__(self, req: DrawRequest) -> dict[str, Any]:  # type: ignore[override]
        cfg = self.cfg
        rng = np.random.default_rng(req.seed)
        t1, t2, valid, change, s1, s2 = self.read_chip(req.chip_id)
        row = self.df.loc[req.chip_id]
        weight = float(req.weight) * (cfg.sampler.auto_label_weight if str(row["label_confidence"]) == "auto" else 1.0)
        tt = str(req.tile_type)
        if tt == TileType.IDENTICAL.value:
            t2 = t1.copy()
            change = np.where(valid == 1, 0, IGNORE).astype(np.uint8)
            s2 = s1.copy()
        elif tt == TileType.NEAR_IDENTICAL.value:
            t2 = near_identical(t1, rng, cfg, valid)
            change = np.where(valid == 1, 0, IGNORE).astype(np.uint8)
            s2 = s1.copy()
        elif tt == TileType.SYNTHETIC.value and req.synthetic_objs:
            objs = [self._object(p) for p in req.synthetic_objs]
            t1, t2, change, _ = paste(t1, t2, change, valid, objs, rng, cfg)
        elif tt == TileType.STAR.value and req.star_chip_id:
            o_t1, _, o_valid, _, o_s1, _ = self.read_chip(req.star_chip_id)
            t2 = o_t1
            valid = (valid.astype(bool) & o_valid.astype(bool)).astype(np.uint8)
            change = star_label(s1, o_s1)
            s2 = o_s1

        r, c = self._crop_centre(req, change, rng)
        half = self.tile // 2
        sl = (slice(r - half, r - half + self.tile), slice(c - half, c - half + self.tile))

        def img(a: np.ndarray) -> torch.Tensor:
            x = torch.from_numpy(a[(slice(None), *sl)].astype(np.float32) / cfg.bands.scale)
            return x.clamp(0.0, cfg.bands.clip_max)

        sample: dict[str, Any] = {
            "x1": img(t1),
            "x2": img(t2),
            "change": torch.from_numpy(change[sl].astype(np.int64)),
            "sem_t1": torch.from_numpy(s1[sl].astype(np.int64)),
            "sem_t2": torch.from_numpy(s2[sl].astype(np.int64)),
            "valid": torch.from_numpy(valid[sl].astype(bool)),
        }
        if self.augment and req.augment:
            sample = transforms.apply_all(sample, req, cfg, rng)
        inv = ~sample["valid"]
        for key in transforms.LABEL_KEYS:
            lab = sample[key].clone()
            lab[inv] = IGNORE
            sample[key] = lab
        sample["weight"] = torch.tensor(weight, dtype=torch.float32)
        sample["meta"] = {"chip_id": req.chip_id, "tile_type": tt, "stratum": str(row["stratum"]), "swap": bool(req.swap), "seed": int(req.seed)}
        return sample


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {k: torch.stack([b[k] for b in batch]) for k in ("x1", "x2", "change", "sem_t1", "sem_t2", "valid", "weight")}
    out["meta"] = [b["meta"] for b in batch]
    return out


def full_chip_requests(manifest: pd.DataFrame, split: str, seed: int = 0) -> list[DrawRequest]:
    """Deterministic, un-augmented full-chip requests for validation (tile_px = chip_px)."""
    rows = manifest[manifest["split"] == split]
    return [
        DrawRequest(chip_id=str(r.chip_id), tile_type=str(r.tile_type), seed=seed + i, swap=False, augment=False)
        for i, r in enumerate(rows.itertuples())
    ]
