"""Synthetic Sentinel-2-like scenes and labels for tests and `make smoke` (DEVIATIONS C13).

A scene is a smooth random reflectance field with optional painted objects (class 1-4), a global
gain, a sub-pixel content shift and cloud boxes, written as a `SimpleScene` directory: 10 m
B02/B03/B04/B08, 20 m B11/B12 and SCL, and mtd.json with a BOA offset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from scipy import ndimage

from scd.constants import BAND_ORDER, RGBN_BANDS, SWIR_BANDS, UINT16_SCALE
from scd.geo.grid import Grid

# reflectance deltas painted per class: (B02, B03, B04, B08, B11, B12)
CLASS_DELTA = {
    1: np.array([0.12, 0.12, 0.14, 0.10, 0.12, 0.10], dtype=np.float32),  # buildup: bright bare
    2: np.array([0.25, 0.25, 0.25, 0.20, 0.15, 0.12], dtype=np.float32),  # building: bright roof
    3: np.array([0.10, 0.10, 0.10, 0.08, 0.08, 0.06], dtype=np.float32),  # road: brighter line
    4: np.array([-0.05, -0.06, -0.08, -0.18, -0.20, -0.18], dtype=np.float32),  # solar: dark NIR/SWIR
}


@dataclass
class Obj:
    class_id: int
    r0: int
    c0: int
    r1: int
    c1: int  # 10 m pixel box [r0:r1, c0:c1]


@dataclass
class SceneSpec:
    product_id: str
    date: str
    relative_orbit: int = 5
    tile: str = "T45RVL"
    processing_baseline: float = 5.0
    boa_add_offset: float = -1000.0
    gain: float = 1.0
    shift_px: tuple[float, float] = (0.0, 0.0)  # (dx, dy) at 10 m applied to the base field
    objects: list[Obj] = field(default_factory=list)
    clouds: list[tuple[int, int, int, int]] = field(default_factory=list)  # 10 m boxes
    noise_sigma: float = 0.0


def base_field(rng: np.random.Generator, size10: int, margin: int = 16) -> np.ndarray:
    """(6, size+2m, size+2m) reflectance in roughly [0.05, 0.45] with band correlation."""
    n = size10 + 2 * margin
    shared = ndimage.gaussian_filter(rng.normal(size=(n, n)), 4.0)
    shared = (shared - shared.min()) / (shared.max() - shared.min() + 1e-9)
    bands = []
    for i, _b in enumerate(BAND_ORDER):
        own = ndimage.gaussian_filter(rng.normal(size=(n, n)), 2.0)
        own = (own - own.min()) / (own.max() - own.min() + 1e-9)
        level = [0.08, 0.10, 0.12, 0.28, 0.22, 0.16][i]
        bands.append(level + 0.25 * (0.7 * shared + 0.3 * own))
    return np.stack(bands).astype(np.float32)


def make_simple_scene(out_dir: Path, grid10: Grid, base: np.ndarray, spec: SceneSpec, margin: int = 16) -> Path:
    """Write a SimpleScene from `base` (6, size+2m, size+2m) reflectance and the spec."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    size = grid10.height
    refl = base.copy()
    dx, dy = spec.shift_px
    if dx or dy:
        refl = ndimage.shift(refl, (0.0, dy, dx), order=1, mode="reflect")
    refl = refl[:, margin : margin + size, margin : margin + size] * spec.gain
    for o in spec.objects:
        refl[:, o.r0 : o.r1, o.c0 : o.c1] = np.clip(
            refl[:, o.r0 : o.r1, o.c0 : o.c1] + CLASS_DELTA[o.class_id][:, None, None], 0.01, 1.1
        )
    if spec.noise_sigma > 0:
        rng = np.random.default_rng(abs(hash(spec.product_id)) % (2**32))
        refl = refl + rng.normal(0, spec.noise_sigma, refl.shape).astype(np.float32)
    scl = np.full((size, size), 4, dtype=np.uint8)  # vegetation
    for r0, c0, r1, c1 in spec.clouds:
        refl[:, r0:r1, c0:c1] = 0.9
        scl[r0:r1, c0:c1] = 9  # cloud high probability
    dn_true = np.clip(np.rint(refl * UINT16_SCALE), 1, 65535)
    stored = np.clip(dn_true - spec.boa_add_offset, 1, 65535).astype(np.uint16)  # reader adds the offset back

    def write(path: Path, arr: np.ndarray, transform: Affine, dtype: str) -> None:
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            dtype=dtype,
            count=1,
            width=arr.shape[1],
            height=arr.shape[0],
            crs=grid10.crs.to_wkt(),
            transform=transform,
        ) as ds:
            ds.write(arr.astype(dtype), 1)

    for b in RGBN_BANDS:
        write(out_dir / f"{b}.tif", stored[BAND_ORDER.index(b)], grid10.transform, "uint16")
    t20 = Affine(grid10.transform.a * 2, 0, grid10.transform.c, 0, grid10.transform.e * 2, grid10.transform.f)
    for b in SWIR_BANDS:
        a = stored[BAND_ORDER.index(b)].astype(np.float64)
        a20 = a.reshape(size // 2, 2, size // 2, 2).mean(axis=(1, 3))
        write(out_dir / f"{b}.tif", np.rint(a20), t20, "uint16")
    scl20 = scl.reshape(size // 2, 2, size // 2, 2).max(axis=(1, 3))
    write(out_dir / "SCL.tif", scl20, t20, "uint8")
    (out_dir / "mtd.json").write_text(
        json.dumps(
            {
                "product_id": spec.product_id,
                "date": spec.date,
                "processing_baseline": spec.processing_baseline,
                "boa_add_offset": spec.boa_add_offset,
                "relative_orbit": spec.relative_orbit,
                "tile": spec.tile,
                "cloud_cover": 100.0 * float((scl == 9).mean()),
            }
        )
    )
    return out_dir


def grid10_for(epsg: int = 32645, x0: float = 500000.0, y0: float = 3300000.0, size10: int = 64) -> Grid:
    return Grid(epsg, Affine(10.0, 0.0, x0, 0.0, -10.0, y0), size10, size10)


def default_objects(size10: int = 64) -> list[Obj]:
    """Objects large enough for the 5 m label rules (areas in m2 at 10 m px = 100 m2 each)."""
    q = size10 // 8
    return [
        Obj(2, q, q, q + 4, q + 7),  # building 40 x 70 m = 2,800 m2
        Obj(4, 4 * q, q, 4 * q + 8, q + 10),  # solar 80 x 100 m = 8,000 m2
        Obj(3, 2 * q, 3 * q, 2 * q + 1, 3 * q + 30),  # road 10 m x 300 m
        Obj(1, 5 * q, 4 * q, 5 * q + 7, 4 * q + 8),  # buildup 70 x 80 m = 5,600 m2
    ]


def make_pair(
    root: Path,
    prefix: str,
    seed: int,
    size10: int = 64,
    objects_t2: list[Obj] | None = None,
    objects_t1: list[Obj] | None = None,
    shift: tuple[float, float] = (1.2, -0.6),
    gain_t2: float = 1.04,
    clouds_t2: list[tuple[int, int, int, int]] | None = None,
    identical: bool = False,
    noise_t2: float = 0.0,
    grid10: Grid | None = None,
) -> tuple[Path, Path, Grid]:
    """Two SimpleScene directories sharing a base field; T2 carries the change (and clouds/shift)."""
    rng = np.random.default_rng(seed)
    g10 = grid10 or grid10_for(size10=size10)
    base = base_field(rng, size10)
    s1 = SceneSpec(f"{prefix}_T1", "2019-07-10", objects=objects_t1 or [])
    if identical:
        d1 = make_simple_scene(root / f"{prefix}_T1", g10, base, s1)
        return d1, d1, g10
    s2 = SceneSpec(
        f"{prefix}_T2",
        "2023-07-14",
        gain=gain_t2,
        shift_px=shift,
        objects=(objects_t1 or []) + (objects_t2 if objects_t2 is not None else default_objects(size10)),
        clouds=clouds_t2 or [],
        noise_sigma=noise_t2,
    )
    d1 = make_simple_scene(root / f"{prefix}_T1", g10, base, s1)
    d2 = make_simple_scene(root / f"{prefix}_T2", g10, base, s2)
    return d1, d2, g10


def objects_to_polygons(objects: list[Obj], grid10: Grid):
    """Shapely polygons (UTM) for painted objects, for reviewed-label fixtures."""
    from shapely.geometry import box

    t = grid10.transform
    polys = []
    for o in objects:
        x0, y0 = t @ (o.c0, o.r0)
        x1, y1 = t @ (o.c1, o.r1)
        polys.append((box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)), o.class_id))
    return polys
