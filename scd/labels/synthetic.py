"""Synthetic change (HANDOFF.md 5.11): object library and copy-paste insertion / removal.

Objects are cut from verified positives with their 6-band pixels at both dates. `paste` inserts
the "present" look of an object into one date chosen at random (so insertions and removals are
equally frequent), feathered over 1-2 px, per-band gain matched within +-10 %, never over invalid
or water pixels, at most 3 objects per chip, with a 10 % share of flat-fill and blurred pastes to
cover crude synthetic test images. Host labels are kept.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage
from skimage import measure

from scd.config import Config
from scd.constants import BAND_INDEX, CHANGE_CLASSES, CLASS_IDS, IGNORE
from scd.log import get_logger

log = get_logger(__name__)

ID_TO_NAME = {v: k for k, v in CLASS_IDS.items()}
SIZE_RULES_M2: dict[int, tuple[float, float | None]] = {
    1: (2500.0, 100_000.0),
    2: (1800.0, None),
    4: (5000.0, 50_000.0),
}
ROAD_LENGTH_M = (200.0, 1500.0)
LIBRARY_TARGETS = {"building": 300, "solar": 150, "road": 80, "buildup": 100}


@dataclass
class SynthObject:
    class_id: int
    img_present: np.ndarray  # (6, h, w) uint16: the date where the object exists
    img_absent: np.ndarray  # (6, h, w) uint16: the other date (context only)
    mask: np.ndarray  # (h, w) bool
    meta: dict[str, Any]

    @property
    def size_px(self) -> int:
        return int(self.mask.sum())

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            img_t1=self.img_absent,
            img_t2=self.img_present,
            mask=self.mask.astype(np.uint8),
            meta=np.array(json.dumps(self.meta)),
        )
        return path

    @classmethod
    def load(cls, path: Path) -> SynthObject:
        with np.load(path) as z:
            meta = json.loads(str(z["meta"]))
            return cls(int(meta["class_id"]), z["img_t2"], z["img_t1"], z["mask"].astype(bool), meta)


class ObjectLibrary:
    def __init__(self, objects: dict[int, list[SynthObject]]) -> None:
        self.objects = {cid: list(objs) for cid, objs in objects.items()}

    @classmethod
    def load(cls, directory: Path) -> ObjectLibrary:
        objs: dict[int, list[SynthObject]] = {cid: [] for cid in range(1, 5)}
        for cname in CHANGE_CLASSES:
            d = Path(directory) / cname
            if d.exists():
                for p in sorted(d.glob("*.npz")):
                    objs[CLASS_IDS[cname]].append(SynthObject.load(p))
        return cls(objs)

    def counts(self) -> dict[str, int]:
        return {ID_TO_NAME[cid]: len(v) for cid, v in sorted(self.objects.items())}

    def __len__(self) -> int:
        return sum(len(v) for v in self.objects.values())

    def sample(self, class_id: int, rng: np.random.Generator) -> SynthObject | None:
        pool = self.objects.get(class_id, [])
        if not pool:
            return None
        return pool[int(rng.integers(0, len(pool)))]

    def sample_mix(self, rng: np.random.Generator, class_shares: dict[str, float], n: int) -> list[SynthObject]:
        """n objects with classes drawn from `class_shares`, falling back to available classes."""
        names = list(class_shares)
        p = np.array([class_shares[c] for c in names], dtype=float)
        p /= p.sum()
        out: list[SynthObject] = []
        avail = [cid for cid, v in self.objects.items() if v]
        if not avail:
            return out
        for _ in range(n):
            cid = CLASS_IDS[names[int(rng.choice(len(names), p=p))]]
            if cid not in avail:
                cid = avail[int(rng.integers(0, len(avail)))]
            obj = self.sample(cid, rng)
            if obj is not None:
                out.append(obj)
        return out


# --- extraction -------------------------------------------------------------------------------
def extract_objects(
    t1: np.ndarray,
    t2: np.ndarray,
    change: np.ndarray,
    sem_t1: np.ndarray | None,
    sem_t2: np.ndarray | None,
    valid: np.ndarray,
    res_m: float,
    chip_id: str,
    stratum: str,
    pad: int = 2,
) -> list[SynthObject]:
    """Connected change components that satisfy the per-class size rules, fully valid."""
    out: list[SynthObject] = []
    h_img, w_img = change.shape
    for cid in range(1, 5):
        lab = measure.label(change == cid, connectivity=1)
        for region in measure.regionprops(lab):
            area_m2 = float(region.area) * res_m * res_m
            if cid == CLASS_IDS["road"]:
                length_m = float(region.axis_major_length) * res_m
                if not (ROAD_LENGTH_M[0] <= length_m <= ROAD_LENGTH_M[1]):
                    continue
            else:
                lo, hi = SIZE_RULES_M2[cid]
                if area_m2 < lo or (hi is not None and area_m2 > hi):
                    continue
            r0, c0, r1, c1 = region.bbox
            r0, c0 = max(r0 - pad, 0), max(c0 - pad, 0)
            r1, c1 = min(r1 + pad, h_img), min(c1 + pad, w_img)
            sl = (slice(r0, r1), slice(c0, c1))
            m = lab[sl] == region.label
            if (valid[sl] == 0).any():
                continue
            present_t2 = True
            if sem_t1 is not None and sem_t2 is not None:
                f2 = float((sem_t2[sl][m] == cid).mean())
                f1 = float((sem_t1[sl][m] == cid).mean())
                present_t2 = f2 >= f1
            img_p = (t2 if present_t2 else t1)[(slice(None), *sl)]
            img_a = (t1 if present_t2 else t2)[(slice(None), *sl)]
            out.append(
                SynthObject(
                    cid,
                    img_p.copy(),
                    img_a.copy(),
                    m.copy(),
                    {
                        "class_id": cid,
                        "class": ID_TO_NAME[cid],
                        "size_m2": area_m2,
                        "stratum": stratum,
                        "chip_id": chip_id,
                        "present_date": "t2" if present_t2 else "t1",
                        "res_m": res_m,
                    },
                )
            )
    return out


def build_object_library(cfg: Config, version: int, include_auto: bool = False) -> dict[str, int]:
    """objects/v<N>/<class>/<chip>_<k>.npz from CHANGE_POS training chips; returns counts per class."""
    from scd.data import manifest as man_mod
    from scd.geo.io import read_raster

    chips_dir = cfg.chips_dir(version)
    df = man_mod.load(chips_dir / "manifest.parquet")
    sel = df[(df["tile_type"] == "CHANGE_POS") & (df["split"] == "train")]
    if not include_auto:
        sel = sel[sel["label_confidence"] != "auto"]
    out_dir = cfg.objects_dir(version)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {c: 0 for c in CHANGE_CLASSES}
    for _, row in sel.iterrows():
        cid = str(row["chip_id"])
        t1, _ = read_raster(chips_dir / "images" / f"{cid}_t1.tif")
        t2, _ = read_raster(chips_dir / "images" / f"{cid}_t2.tif")
        valid, _ = read_raster(chips_dir / "images" / f"{cid}_valid.tif")
        change, _ = read_raster(chips_dir / "labels" / f"{cid}_change.tif")
        s1, _ = read_raster(chips_dir / "labels" / f"{cid}_sem_t1.tif")
        s2, _ = read_raster(chips_dir / "labels" / f"{cid}_sem_t2.tif")
        objs = extract_objects(t1, t2, change[0], s1[0], s2[0], valid[0], cfg.grid.res_m, cid, str(row["stratum"]))
        for k, obj in enumerate(objs):
            obj.save(out_dir / ID_TO_NAME[obj.class_id] / f"{cid}_{k:02d}.npz")
            counts[ID_TO_NAME[obj.class_id]] += 1
    (out_dir / "library.json").write_text(
        json.dumps({"counts": counts, "targets": LIBRARY_TARGETS, "config_hash": cfg.hash()}, indent=2)
    )
    for c, target in LIBRARY_TARGETS.items():
        if counts[c] < target:
            log.warning("object library: %s has %d objects, target >= %d", c, counts[c], target)
    return counts


# --- paste / remove ---------------------------------------------------------------------------
def ndwi(img: np.ndarray) -> np.ndarray:
    g = img[BAND_INDEX["B03"]].astype(np.float32)
    n = img[BAND_INDEX["B08"]].astype(np.float32)
    return (g - n) / np.maximum(g + n, 1.0)


def _feather(mask: np.ndarray, px: int) -> np.ndarray:
    dist = ndimage.distance_transform_edt(mask)
    return np.clip(dist / float(px + 1), 0.0, 1.0).astype(np.float32)


def _gain_match(src: np.ndarray, mask: np.ndarray, host_patch: np.ndarray, max_dev: float = 0.10) -> np.ndarray:
    ring = ndimage.binary_dilation(mask, iterations=3) & ~mask
    if ring.sum() < 4:
        return np.ones(src.shape[0], dtype=np.float32)
    src_mean = src[:, ring].mean(axis=1)
    host_mean = host_patch[:, ring].mean(axis=1)
    gain = np.where(src_mean > 0, host_mean / np.maximum(src_mean, 1.0), 1.0)
    return np.clip(gain, 1.0 - max_dev, 1.0 + max_dev).astype(np.float32)


@dataclass
class PasteRecord:
    class_id: int
    mask: np.ndarray  # full-size bool
    target_date: int  # 1 or 2
    mode: str  # insert | flat | blur | remove


def paste(
    host_t1: np.ndarray,
    host_t2: np.ndarray,
    host_change: np.ndarray,
    host_valid: np.ndarray,
    objects: list[SynthObject],
    rng: np.random.Generator,
    cfg: Config,
    max_objects: int = 3,
    flat_fill_p: float = 0.10,
    shadow_p: float = 0.5,
    max_tries: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[PasteRecord]]:
    """Insert up to 3 objects into one random date each. Returns (t1, t2, change, records)."""
    t1, t2, change = host_t1.copy(), host_t2.copy(), host_change.copy()
    big_h, big_w = change.shape
    water1, water2 = ndwi(t1) > cfg.infer.water_ndwi, ndwi(t2) > cfg.infer.water_ndwi
    records: list[PasteRecord] = []
    for obj in objects[:max_objects]:
        h, w = obj.mask.shape
        if h >= big_h or w >= big_w:
            continue
        m = obj.mask
        placed = None
        for _ in range(max_tries):
            r = int(rng.integers(0, big_h - h + 1))
            c = int(rng.integers(0, big_w - w + 1))
            sl = (slice(r, r + h), slice(c, c + w))
            if (host_valid[sl][m] == 0).any() or (change[sl][m] != 0).any():
                continue
            if water1[sl][m].any() or water2[sl][m].any():
                continue
            placed = sl
            break
        if placed is None:
            continue
        sl = placed
        target = int(rng.integers(1, 3))
        host_img = t1 if target == 1 else t2
        patch = host_img[(slice(None), *sl)].astype(np.float32)
        src = obj.img_present.astype(np.float32)
        u = float(rng.random())
        mode = "insert"
        if u < flat_fill_p / 2:
            src = np.where(m[None], src[:, m].mean(axis=1)[:, None, None], src)
            mode = "flat"
        elif u < flat_fill_p:
            src = ndimage.gaussian_filter(src, sigma=(0, 1.0, 1.0))
            mode = "blur"
        src = src * _gain_match(src, m, patch)[:, None, None]
        alpha = _feather(m, int(rng.integers(1, 3)))[None]
        blended = patch * (1.0 - alpha) + src * alpha
        if obj.class_id == CLASS_IDS["building"] and rng.random() < shadow_p:
            dy, dx = int(rng.integers(1, 3)), int(rng.integers(1, 3))
            shifted = ndimage.shift(m.astype(np.float32), (dy, dx), order=0) > 0.5
            strip = shifted & ~m
            blended[:, strip] *= 0.8
        host_img[(slice(None), *sl)] = np.clip(np.rint(blended), 0, 65535).astype(np.uint16)
        change[sl][m] = obj.class_id
        full = np.zeros((big_h, big_w), dtype=bool)
        full[sl] = m
        records.append(PasteRecord(obj.class_id, full, target, mode))
    return t1, t2, change, records


def remove(
    host_t1: np.ndarray,
    host_t2: np.ndarray,
    obj_mask: np.ndarray,
    rng: np.random.Generator,
    class_id: int = 0,
    host_valid: np.ndarray | None = None,
    host_change: np.ndarray | None = None,
    max_tries: int = 25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replace the object (present at both dates) in one random date with neighbouring texture.

    Returns (t1, t2, change_mask) where change_mask is `obj_mask` labelled `class_id`.
    """
    t1, t2 = host_t1.copy(), host_t2.copy()
    big_h, big_w = obj_mask.shape
    rows, cols = np.nonzero(obj_mask)
    if rows.size == 0:
        return t1, t2, np.zeros_like(obj_mask, dtype=np.uint8)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    h, w = r1 - r0, c1 - c0
    target = int(rng.integers(1, 3))
    img = t1 if target == 1 else t2
    done = False
    for _ in range(max_tries):
        dy = int(rng.integers(-3 * h, 3 * h + 1))
        dx = int(rng.integers(-3 * w, 3 * w + 1))
        if abs(dy) < h and abs(dx) < w:
            continue
        sr0, sc0 = r0 + dy, c0 + dx
        if sr0 < 0 or sc0 < 0 or sr0 + h > big_h or sc0 + w > big_w:
            continue
        src_sl = (slice(sr0, sr0 + h), slice(sc0, sc0 + w))
        m = obj_mask[r0:r1, c0:c1]
        if host_valid is not None and (host_valid[src_sl][m] == 0).any():
            continue
        if host_change is not None and (host_change[src_sl][m] != 0).any():
            continue
        alpha = _feather(m, 1)[None]
        dst = img[:, r0:r1, c0:c1].astype(np.float32)
        src = img[(slice(None), *src_sl)].astype(np.float32)
        img[:, r0:r1, c0:c1] = np.clip(np.rint(dst * (1 - alpha) + src * alpha), 0, 65535).astype(np.uint16)
        done = True
        break
    change = np.zeros_like(obj_mask, dtype=np.uint8)
    if done:
        change[obj_mask] = class_id if class_id else 1
    return t1, t2, change


# --- notebook helpers -------------------------------------------------------------------------
def show_gallery(lib: ObjectLibrary, n: int = 8) -> None:  # pragma: no cover - plotting
    import matplotlib.pyplot as plt

    from scd.imagery.preprocess import _stretch_rgb

    objs = [o for v in lib.objects.values() for o in v][:n]
    if not objs:
        print("empty library")
        return
    fig, axes = plt.subplots(2, len(objs), figsize=(2.2 * len(objs), 4.5))
    for i, o in enumerate(objs):
        axes[0, i].imshow(_stretch_rgb(o.img_present))
        axes[0, i].set_title(f"{o.meta['class']} {o.meta['size_m2']:.0f} m2", fontsize=8)
        axes[1, i].imshow(o.mask, cmap="gray")
        for ax in axes[:, i]:
            ax.axis("off")
    plt.show()


def show_paste_demo(cfg: Config, lib: ObjectLibrary, manifest_path: Path) -> None:  # pragma: no cover - plotting
    import matplotlib.pyplot as plt

    from scd.data import manifest as man_mod
    from scd.geo.io import read_raster
    from scd.imagery.preprocess import _stretch_rgb

    df = man_mod.load(manifest_path)
    row = df.iloc[0]
    chips_dir = manifest_path.parent
    cid = str(row["chip_id"])
    t1, _ = read_raster(chips_dir / "images" / f"{cid}_t1.tif")
    t2, _ = read_raster(chips_dir / "images" / f"{cid}_t2.tif")
    valid, _ = read_raster(chips_dir / "images" / f"{cid}_valid.tif")
    change, _ = read_raster(chips_dir / "labels" / f"{cid}_change.tif")
    rng = np.random.default_rng(0)
    objs = lib.sample_mix(rng, cfg.sampler.class_shares, 3)
    a, b, c, recs = paste(t1, t2, change[0], valid[0], objs, rng, cfg)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(_stretch_rgb(a))
    axes[1].imshow(_stretch_rgb(b))
    axes[2].imshow(np.where(c == IGNORE, 0, c), vmin=0, vmax=4, cmap="tab10")
    for ax, t in zip(axes, ("T1", "T2", f"label ({len(recs)} pasted)")):
        ax.set_title(t)
        ax.axis("off")
    plt.show()
