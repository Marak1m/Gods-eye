"""Splits (HANDOFF.md 5.12): by aoi_id within stratum, 80/20 train/val, seeded.

`targets` and `stage3` AOIs never train (split "targets"); `stage2` is its own split. AOIs whose
registry role is already `val` stay val; the remaining train-role AOIs are split so each stratum
ends up with about `val_fraction` validation AOIs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from scd.constants import STRATA


def assign_splits(registry: pd.DataFrame, seed: int = 42, val_fraction: float = 0.2) -> dict[str, str]:
    out: dict[str, str] = {}
    for _, r in registry.iterrows():
        role = str(r["role"])
        if role in {"targets", "stage3"}:
            out[str(r["aoi_id"])] = "targets"
        elif role == "stage2":
            out[str(r["aoi_id"])] = "stage2"
        elif role == "val":
            out[str(r["aoi_id"])] = "val"
    rng = np.random.default_rng(seed)
    for stratum in STRATA:
        sub = registry[(registry["stratum"] == stratum) & registry["role"].isin(["train", "val"])]
        ids = sorted(str(a) for a in sub["aoi_id"])
        if not ids:
            continue
        n_val_target = int(round(val_fraction * len(ids)))
        fixed_val = [a for a in ids if out.get(a) == "val"]
        candidates = [a for a in ids if a not in out]
        need = max(n_val_target - len(fixed_val), 0)
        perm = rng.permutation(len(candidates))
        chosen = {candidates[i] for i in perm[:need]}
        for a in candidates:
            out[a] = "val" if a in chosen else "train"
    return out
