"""TwoStageSampler (HANDOFF.md 5.13, plan doc 9.1).

stratum -> tile type -> (class for CHANGE_POS) -> chip weighted by change_px^density_exponent ->
crop request. IDENTICAL / NEAR_IDENTICAL / SYNTHETIC / STAR draws pick a host chip from the
stratum and flag the request. Deterministic given (seed, epoch). When a stratum lacks a requested
type or class the sampler falls back to the next available pool and logs it once (DEVIATIONS C15).
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Sampler

from scd.config import Config
from scd.constants import CHANGE_CLASSES, CLASS_IDS, STRATA, TileType
from scd.data.dataset import DrawRequest
from scd.labels.synthetic import ObjectLibrary
from scd.log import get_logger

log = get_logger(__name__)
ID_TO_NAME = {v: k for k, v in CLASS_IDS.items()}


class TwoStageSampler(Sampler[DrawRequest]):
    def __init__(
        self,
        manifest: pd.DataFrame,
        cfg: Config,
        epoch_len: int,
        seed: int = 42,
        library: ObjectLibrary | None = None,
        library_dir: Path | None = None,
        epoch: int = 0,
    ) -> None:
        self.df = manifest.reset_index(drop=True)
        self.cfg = cfg
        self.epoch_len = int(epoch_len)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.library = library
        self.library_dir = library_dir
        self._logged: set[str] = set()
        self._index()

    # --- indices --------------------------------------------------------------------------
    def _index(self) -> None:
        df = self.df
        self.by_stratum: dict[str, np.ndarray] = {s: df.loc[df["stratum"] == s, "chip_id"].to_numpy() for s in STRATA}
        self.hard = {s: df.loc[(df["stratum"] == s) & (df["tile_type"] == TileType.NOCHANGE_HARD.value), "chip_id"].to_numpy() for s in STRATA}
        self.plain = {s: df.loc[(df["stratum"] == s) & (df["tile_type"] == TileType.NOCHANGE_PLAIN.value), "chip_id"].to_numpy() for s in STRATA}
        self.pos: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}
        exp = self.cfg.sampler.density_exponent
        for s in STRATA:
            for cid in range(1, 5):
                sel = df[(df["stratum"] == s) & (df["tile_type"] == TileType.CHANGE_POS.value) & (df[f"change_px_{cid}"] > 0)]
                if len(sel):
                    w = sel[f"change_px_{cid}"].to_numpy(float) ** exp
                    self.pos[(s, cid)] = (sel["chip_id"].to_numpy(), w / w.sum())
        self.any_pos = df.loc[df["tile_type"] == TileType.CHANGE_POS.value, "chip_id"].to_numpy()
        self.strata_available = [s for s in STRATA if len(self.by_stratum[s])]
        if not self.strata_available:
            raise ValueError("manifest has no chips")
        self._object_paths: dict[int, list[str]] = {}
        if self.library is not None and self.library_dir is not None:
            for cname in CHANGE_CLASSES:
                d = Path(self.library_dir) / cname
                self._object_paths[CLASS_IDS[cname]] = sorted(str(p.relative_to(self.library_dir)) for p in d.glob("*.npz")) if d.exists() else []

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _note(self, key: str, msg: str) -> None:
        if key not in self._logged:
            self._logged.add(key)
            log.info("sampler fallback: %s", msg)

    # --- draws ----------------------------------------------------------------------------
    def _stratum(self, rng: np.random.Generator) -> str:
        shares = self.cfg.strata.shares
        names = list(shares)
        p = np.array([shares[s] for s in names])
        s = names[int(rng.choice(len(names), p=p / p.sum()))]
        if len(self.by_stratum[s]) == 0:
            self._note(f"stratum:{s}", f"stratum {s} has no chips; drawing from {self.strata_available}")
            s = self.strata_available[int(rng.integers(0, len(self.strata_available)))]
        return s

    def _tile_type(self, rng: np.random.Generator, stratum: str) -> str:
        shares = self.cfg.sampler.shares_for(stratum)
        names = list(shares)
        p = np.array([shares[t] for t in names])
        return names[int(rng.choice(len(names), p=p / p.sum()))]

    def _class(self, rng: np.random.Generator) -> int:
        shares = self.cfg.sampler.class_shares
        names = list(shares)
        p = np.array([shares[c] for c in names])
        return CLASS_IDS[names[int(rng.choice(len(names), p=p / p.sum()))]]

    def _pick(self, rng: np.random.Generator, pool: np.ndarray, p: np.ndarray | None = None) -> str:
        return str(pool[int(rng.choice(len(pool), p=p))])

    def _host(self, rng: np.random.Generator, stratum: str) -> str:
        return self._pick(rng, self.by_stratum[stratum])

    def draw(self, rng: np.random.Generator) -> DrawRequest:
        stratum = self._stratum(rng)
        tt = self._tile_type(rng, stratum)
        swap = bool(rng.random() < self.cfg.augment.temporal_swap_p)
        seed = int(rng.integers(0, 2**31 - 1))
        if tt == TileType.CHANGE_POS.value:
            cid = self._class(rng)
            key = (stratum, cid)
            if key in self.pos:
                chips, w = self.pos[key]
                return DrawRequest(self._pick(rng, chips, w), tt, seed, swap, centre_class=cid)
            alt = [k for k in self.pos if k[0] == stratum]
            if alt:
                self._note(f"class:{stratum}:{cid}", f"stratum {stratum} has no {ID_TO_NAME[cid]} chips; using other classes")
                k = alt[int(rng.integers(0, len(alt)))]
                chips, w = self.pos[k]
                return DrawRequest(self._pick(rng, chips, w), tt, seed, swap, centre_class=k[1])
            if len(self.any_pos):
                self._note(f"pos:{stratum}", f"stratum {stratum} has no CHANGE_POS chips; using any stratum")
                return DrawRequest(self._pick(rng, self.any_pos), tt, seed, swap, centre_class=cid)
            self._note("pos:none", "no CHANGE_POS chips at all; serving a host chip as NOCHANGE_PLAIN")
            return DrawRequest(self._host(rng, stratum), TileType.NOCHANGE_PLAIN.value, seed, swap)
        if tt in (TileType.NOCHANGE_HARD.value, TileType.NOCHANGE_PLAIN.value):
            pools = (self.hard, self.plain) if tt == TileType.NOCHANGE_HARD.value else (self.plain, self.hard)
            for pool in pools:
                if len(pool[stratum]):
                    return DrawRequest(self._pick(rng, pool[stratum]), tt, seed, swap)
            self._note(f"neg:{stratum}", f"stratum {stratum} has no no-change chips; using any chip")
            return DrawRequest(self._host(rng, stratum), tt, seed, swap)
        host = self._host(rng, stratum)
        if tt == TileType.SYNTHETIC.value:
            objs: list[str] = []
            if self.library is not None and any(self._object_paths.values()):
                n = int(rng.integers(1, 4))
                for _ in range(n):
                    cid = self._class(rng)
                    paths = self._object_paths.get(cid) or [p for v in self._object_paths.values() for p in v]
                    if paths:
                        objs.append(paths[int(rng.integers(0, len(paths)))])
            else:
                self._note("synthetic:nolib", "no object library; SYNTHETIC draws serve the host unchanged")
            return DrawRequest(host, tt, seed, swap, synthetic_objs=tuple(objs))
        if tt == TileType.STAR.value:
            other = self._host(rng, stratum)
            return DrawRequest(host, tt, seed, swap, star_chip_id=other if other != host else None)
        return DrawRequest(host, tt, seed, swap)  # IDENTICAL / NEAR_IDENTICAL

    def __iter__(self) -> Iterator[DrawRequest]:
        rng = np.random.default_rng([self.seed, self.epoch])
        for _ in range(self.epoch_len):
            yield self.draw(rng)

    def __len__(self) -> int:
        return self.epoch_len


def report(cfg: Config, manifest: pd.DataFrame, n: int = 100_000, seed: int = 0) -> pd.DataFrame:
    """Observed vs configured shares over `n` draws (stratum, tile type, class)."""
    train = manifest[manifest["split"] == "train"]
    sampler = TwoStageSampler(train if len(train) else manifest, cfg, n, seed=seed)
    reqs = list(iter(sampler))
    df = pd.DataFrame({"chip_id": [r.chip_id for r in reqs], "tile_type": [r.tile_type for r in reqs], "centre_class": [r.centre_class for r in reqs]})
    strata = sampler.df.set_index("chip_id")["stratum"]
    df["stratum"] = df["chip_id"].map(strata)
    rows = []
    for s, share in cfg.strata.shares.items():
        rows.append({"kind": "stratum", "key": s, "target": share, "observed": float((df["stratum"] == s).mean())})
    for t, share in cfg.sampler.shares_for("default").items():
        rows.append({"kind": "tile_type", "key": t, "target": share, "observed": float((df["tile_type"] == t).mean())})
    pos = df[df["tile_type"] == TileType.CHANGE_POS.value]
    for cname, share in cfg.sampler.class_shares.items():
        rows.append({"kind": "class|CHANGE_POS", "key": cname, "target": share, "observed": float((pos["centre_class"] == CLASS_IDS[cname]).mean()) if len(pos) else float("nan")})
    out = pd.DataFrame(rows)
    out["diff_pts"] = 100 * (out["observed"] - out["target"])
    return out
