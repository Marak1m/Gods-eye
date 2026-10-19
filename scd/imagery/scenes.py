"""Scene readers and pair selection (HANDOFF.md 5.2, 5.4 step 2; DEVIATIONS C5).

Three readers implement the `Scene` protocol so that `preprocess_pair` is identical downstream:

- `SafeZipScene`: a Sentinel-2 L2A `.SAFE` zip read through GDAL's /vsizip/; the BOA offset is
  read from MTD_MSIL2A.xml and the reader raises if a baseline >= 04.00 lacks it (never guess).
- `SimpleScene`: a directory with `B02.tif ... B12.tif`, optional `SCL.tif`, and `mtd.json`
  (fixtures and the smoke test).
- `GeoTiffScene`: a 6- or 4-band GeoTIFF as delivered at inference; DN vs reflectance detected or
  forced; B11/B12 absent => filled with 0 reflectance after scaling (DEVIATIONS U5).

`read_band` always returns DN *after* the BOA offset (reflectance x 10000) as float32.
"""

from __future__ import annotations

import json
import re
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from shapely.geometry.base import BaseGeometry

from scd.constants import BAND_ORDER, RGBN_BANDS, SWIR_BANDS, UINT16_SCALE, TileType
from scd.geo.vector import content_key, raster_footprint_wgs84
from scd.log import get_logger

log = get_logger(__name__)


@dataclass(frozen=True)
class BandData:
    """One band as DN-after-offset float32 with its native georeferencing."""

    array: np.ndarray
    transform: Affine
    epsg: int

    @property
    def res_m(self) -> float:
        return abs(self.transform.a)


@runtime_checkable
class Scene(Protocol):
    product_id: str

    def read_band(self, name: str) -> BandData: ...

    def read_scl(self) -> BandData | None: ...

    def footprint_wgs84(self) -> BaseGeometry: ...

    def available_bands(self) -> tuple[str, ...]: ...

    def content_key(self) -> str: ...

    def meta(self) -> dict[str, Any]: ...


def _read_single(path: str | Path, band: int = 1) -> tuple[np.ndarray, Affine, int]:
    with rasterio.open(path) as ds:
        arr = ds.read(band)
        if ds.crs is None or ds.crs.to_epsg() is None:
            raise ValueError(f"{path}: CRS missing or without EPSG")
        return arr, ds.transform, int(ds.crs.to_epsg())


# --- SimpleScene -----------------------------------------------------------------------------
class SimpleScene:
    """Directory of per-band GeoTIFFs + mtd.json (fixtures)."""

    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)
        mtd_path = self.dir / "mtd.json"
        if not mtd_path.exists():
            raise FileNotFoundError(f"{self.dir} is not a SimpleScene (no mtd.json)")
        self.mtd: dict[str, Any] = json.loads(mtd_path.read_text())
        self.product_id: str = str(self.mtd["product_id"])
        self.offset: float = float(self.mtd.get("boa_add_offset", 0.0))
        self._key: str | None = None

    def available_bands(self) -> tuple[str, ...]:
        return tuple(b for b in BAND_ORDER if (self.dir / f"{b}.tif").exists())

    def read_band(self, name: str) -> BandData:
        arr, t, epsg = _read_single(self.dir / f"{name}.tif")
        dn = arr.astype(np.float32)
        hole = dn == 0
        dn = dn + self.offset
        dn[hole] = 0.0
        return BandData(np.clip(dn, 0, None), t, epsg)

    def read_scl(self) -> BandData | None:
        p = self.dir / "SCL.tif"
        if not p.exists():
            return None
        arr, t, epsg = _read_single(p)
        return BandData(arr.astype(np.float32), t, epsg)

    def footprint_wgs84(self) -> BaseGeometry:
        return raster_footprint_wgs84(self.dir / "B08.tif")

    def content_key(self) -> str:
        if self._key is None:
            files = sorted(p for p in self.dir.iterdir() if p.suffix in {".tif", ".json"})
            self._key = content_key(*files)
        return self._key

    def meta(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "date": self.mtd.get("date"),
            "processing_baseline": self.mtd.get("processing_baseline"),
            "boa_add_offset": self.offset,
            "relative_orbit": self.mtd.get("relative_orbit"),
            "tile": self.mtd.get("tile"),
            "reader": "SimpleScene",
            "path": str(self.dir),
        }


# --- SafeZipScene ----------------------------------------------------------------------------
_BASELINE_RE = re.compile(r"_N(\d{4})_")
_ORBIT_RE = re.compile(r"_R(\d{3})_")
_TILE_RE = re.compile(r"_T(\d{2}[A-Z]{3})_")


class SafeZipScene:
    """Sentinel-2 L2A .SAFE zip (UNVERIFIED here: no product on disk; see DEVIATIONS)."""

    _RES = {b: "R10m" for b in RGBN_BANDS} | {b: "R20m" for b in SWIR_BANDS}

    def __init__(self, zip_path: Path) -> None:
        self.path = Path(zip_path)
        if not zipfile.is_zipfile(self.path):
            raise ValueError(f"{self.path} is not a zip")
        with zipfile.ZipFile(self.path) as z:
            self._names = z.namelist()
            mtd = [n for n in self._names if n.endswith("MTD_MSIL2A.xml")]
            if not mtd:
                raise ValueError(f"{self.path}: MTD_MSIL2A.xml not found")
            self._mtd_xml = z.read(mtd[0]).decode()
        self.product_id = self.path.stem
        self.baseline = self._parse_baseline()
        self.offset = self._parse_offset()
        self._key: str | None = None

    def _parse_baseline(self) -> float:
        root = ElementTree.fromstring(self._mtd_xml)
        el = next(root.iter("PROCESSING_BASELINE"), None)
        if el is not None and el.text:
            return float(el.text)
        m = _BASELINE_RE.search(self.product_id)
        if not m:
            raise ValueError(f"{self.product_id}: processing baseline not found")
        return float(m.group(1)) / 100.0

    def _parse_offset(self) -> float:
        """BOA_ADD_OFFSET from the L2A MTD; RADIO_ADD_OFFSET is L1C and must not be used."""
        root = ElementTree.fromstring(self._mtd_xml)
        offsets = [float(el.text or 0) for el in root.iter("BOA_ADD_OFFSET")]
        if offsets:
            if len(set(offsets)) != 1:
                raise ValueError(f"{self.product_id}: per-band BOA_ADD_OFFSET values differ: {set(offsets)}")
            return offsets[0]
        if self.baseline >= 4.0:
            raise ValueError(
                f"{self.product_id}: baseline {self.baseline:.2f} >= 04.00 but BOA_ADD_OFFSET is missing "
                "from MTD_MSIL2A.xml; refusing to guess (HANDOFF.md section 14)"
            )
        return 0.0

    def _inner(self, band: str) -> str:
        res = self._RES.get(band, "R20m")
        suffix = f"_{band}_{res[1:]}.jp2"
        hits = [n for n in self._names if "IMG_DATA" in n and n.endswith(suffix)]
        if not hits:
            raise FileNotFoundError(f"{self.product_id}: band {band} at {res} not in product")
        return hits[0]

    def _vsi(self, inner: str) -> str:
        return f"/vsizip/{self.path.as_posix()}/{inner}"

    def available_bands(self) -> tuple[str, ...]:
        return tuple(b for b in BAND_ORDER if any(n.endswith(f"_{b}_{self._RES[b][1:]}.jp2") for n in self._names))

    def read_band(self, name: str) -> BandData:
        arr, t, epsg = _read_single(self._vsi(self._inner(name)))
        dn = arr.astype(np.float32)
        hole = dn == 0
        dn = dn + self.offset
        dn[hole] = 0.0
        return BandData(np.clip(dn, 0, None), t, epsg)

    def read_scl(self) -> BandData | None:
        hits = [n for n in self._names if n.endswith("_SCL_20m.jp2")]
        if not hits:
            return None
        arr, t, epsg = _read_single(self._vsi(hits[0]))
        return BandData(arr.astype(np.float32), t, epsg)

    def footprint_wgs84(self) -> BaseGeometry:
        return raster_footprint_wgs84(Path(self._vsi(self._inner("B08"))))

    def content_key(self) -> str:
        if self._key is None:
            self._key = content_key(self.path)
        return self._key

    def meta(self) -> dict[str, Any]:
        root = ElementTree.fromstring(self._mtd_xml)
        start = next(root.iter("PRODUCT_START_TIME"), None)
        orbit = _ORBIT_RE.search(self.product_id)
        tile = _TILE_RE.search(self.product_id)
        return {
            "product_id": self.product_id,
            "date": (start.text or "")[:10] if start is not None else None,
            "processing_baseline": self.baseline,
            "boa_add_offset": self.offset,
            "relative_orbit": int(orbit.group(1)) if orbit else None,
            "tile": tile.group(1) if tile else None,
            "reader": "SafeZipScene",
            "path": str(self.path),
        }


# --- GeoTiffScene ----------------------------------------------------------------------------
class GeoTiffScene:
    """A delivered multi-band GeoTIFF (inference input)."""

    def __init__(
        self,
        path: Path,
        bands: Sequence[str] | None = None,
        assume: str | None = None,
        scl_path: Path | None = None,
    ) -> None:
        self.path = Path(path)
        self.product_id = self.path.stem
        with rasterio.open(self.path) as ds:
            if ds.crs is None or ds.crs.to_epsg() is None:
                raise ValueError(f"{self.path}: CRS missing or without EPSG")
            self.count = ds.count
            self.dtype = np.dtype(ds.dtypes[0])
            tags = ds.tags()
            descs = [d for d in ds.descriptions if d]
            self.transform = ds.transform
            self.epsg = int(ds.crs.to_epsg())
            sample = ds.read(1, out_shape=(min(ds.height, 512), min(ds.width, 512)))
        self.band_names = self._resolve_bands(bands, tags, descs)
        self.scale_to_dn = self._resolve_scaling(assume, sample)
        self.offset_applied = tags.get("boa_offset_applied", "true").lower() != "false"
        self.scl_path = scl_path or self._find_scl()
        self._key: str | None = None

    def _resolve_bands(self, bands: Sequence[str] | None, tags: dict[str, str], descs: list[str]) -> tuple[str, ...]:
        if bands is not None:
            names = tuple(bands)
        elif "band_order" in tags:
            names = tuple(tags["band_order"].split(","))
        elif len(descs) == self.count and all(d in BAND_ORDER for d in descs):
            names = tuple(descs)
        elif self.count == 6:
            log.warning("%s: no band tags; assuming %s", self.path.name, ",".join(BAND_ORDER))
            names = BAND_ORDER
        elif self.count == 4:
            log.warning("%s: 4 bands without tags; assuming B02,B03,B04,B08", self.path.name)
            names = RGBN_BANDS
        else:
            raise ValueError(f"{self.path}: cannot identify {self.count} bands; pass --bands")
        if len(names) != self.count or not set(names) <= set(BAND_ORDER):
            raise ValueError(f"{self.path}: band names {names} do not match {self.count} bands / {BAND_ORDER}")
        for b in RGBN_BANDS:
            if b not in names:
                raise ValueError(f"{self.path}: band {b} is required")
        return names

    def _resolve_scaling(self, assume: str | None, sample: np.ndarray) -> float:
        """Factor to multiply stored values by to get DN (reflectance x 10000)."""
        if assume == "dn":
            return 1.0
        if assume == "reflectance":
            return UINT16_SCALE
        if np.issubdtype(self.dtype, np.integer):
            return 1.0
        finite = sample[np.isfinite(sample)]
        vmax = float(finite.max()) if finite.size else 0.0
        if vmax <= 1.5:
            return UINT16_SCALE
        if vmax > 100.0:
            return 1.0
        raise ValueError(
            f"{self.path}: float values up to {vmax:.2f} are ambiguous between reflectance and DN; "
            "pass --assume-dn or --assume-reflectance"
        )

    def _find_scl(self) -> Path | None:
        for cand in (self.path.with_name(self.path.stem + "_SCL.tif"), self.path.with_name("SCL.tif")):
            if cand.exists():
                return cand
        return None

    def available_bands(self) -> tuple[str, ...]:
        return tuple(b for b in BAND_ORDER if b in self.band_names)

    def read_band(self, name: str) -> BandData:
        if name not in self.band_names:
            raise KeyError(name)
        idx = self.band_names.index(name) + 1
        arr, t, epsg = _read_single(self.path, idx)
        dn = arr.astype(np.float32) * self.scale_to_dn
        dn[~np.isfinite(dn)] = 0.0
        return BandData(np.clip(dn, 0, None), t, epsg)

    def read_scl(self) -> BandData | None:
        if self.scl_path is None:
            return None
        arr, t, epsg = _read_single(self.scl_path)
        return BandData(arr.astype(np.float32), t, epsg)

    def footprint_wgs84(self) -> BaseGeometry:
        return raster_footprint_wgs84(self.path)

    def content_key(self) -> str:
        if self._key is None:
            self._key = content_key(self.path)
        return self._key

    def meta(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "date": None,
            "processing_baseline": None,
            "boa_add_offset": 0.0 if self.offset_applied else None,
            "bands": list(self.band_names),
            "scale_to_dn": self.scale_to_dn,
            "reader": "GeoTiffScene",
            "path": str(self.path),
        }


def open_scene(path: Path | Scene, **kwargs: Any) -> Scene:
    """Pick a reader: a Scene passes through; .zip -> SAFE; directory -> Simple; .tif -> GeoTIFF."""
    if isinstance(path, Scene):
        return path
    p = Path(path)
    if p.is_dir():
        return SimpleScene(p)
    if p.suffix.lower() == ".zip":
        return SafeZipScene(p)
    if p.suffix.lower() in {".tif", ".tiff"}:
        return GeoTiffScene(p, **kwargs)
    raise ValueError(f"cannot open scene {p}")


# --- pair selection (HANDOFF 5.2) ------------------------------------------------------------
SCENES_COLUMNS = (
    "aoi_id",
    "product_id",
    "tile",
    "relative_orbit",
    "date",
    "cloud_cover",
    "baseline",
    "local_path",
    "status",
)
PAIRS_COLUMNS = (
    "aoi_id",
    "pair_id",
    "t1_product",
    "t2_product",
    "t1_date",
    "t2_date",
    "same_orbit",
    "season_match",
    "valid_fraction",
    "tile_type_hint",
    "status",
)


@dataclass(frozen=True)
class PairSpec:
    aoi_id: str
    pair_id: str
    t1_product: str
    t2_product: str
    t1_date: date
    t2_date: date
    same_orbit: bool
    season_match: bool
    tile_type_hint: str = TileType.CHANGE_POS.value
    valid_fraction: float = float("nan")
    status: str = "selected"

    def as_row(self) -> dict[str, Any]:
        return {
            "aoi_id": self.aoi_id,
            "pair_id": self.pair_id,
            "t1_product": self.t1_product,
            "t2_product": self.t2_product,
            "t1_date": self.t1_date.isoformat(),
            "t2_date": self.t2_date.isoformat(),
            "same_orbit": self.same_orbit,
            "season_match": self.season_match,
            "valid_fraction": self.valid_fraction,
            "tile_type_hint": self.tile_type_hint,
            "status": self.status,
        }


@dataclass
class PairRules:
    max_cloud_aoi: float = 10.0
    season_tolerance_days: int = 45
    max_pairs: int = 3
    allowed_months: list[int] | None = None
    contrast_seasons: list[list[int]] = field(default_factory=lambda: [[12, 1, 2], [6, 7, 8]])


def _parse_window(w: str | None) -> tuple[date, date] | None:
    if not w or not isinstance(w, str) or "/" not in w:
        return None
    a, b = w.split("/")
    return date.fromisoformat(a.strip()), date.fromisoformat(b.strip())


def _doy_distance(d1: date, d2: date) -> int:
    a, b = d1.timetuple().tm_yday, d2.timetuple().tm_yday
    d = abs(a - b)
    return min(d, 365 - d)


def _to_date(v: Any) -> date:
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    return pd.Timestamp(v).date()


def select_pairs(
    aoi_row: Any, scenes_df: pd.DataFrame, rules: PairRules | dict[str, Any] | None = None
) -> list[PairSpec]:
    """HANDOFF 5.2 rules 1-6 for one AOI.

    Rule 2 uses `cloud_cover` as stored in `scenes_df` (tile-level at query time; the SCL-based
    fraction over the AOI is only known after download and is re-checked by `preprocess_pair`).
    """
    r = rules if isinstance(rules, PairRules) else PairRules(**(rules or {}))
    aoi_id = str(aoi_row["aoi_id"])
    df = scenes_df[scenes_df["aoi_id"] == aoi_id].copy()
    if df.empty:
        return []
    df["date"] = df["date"].map(_to_date)
    df = df[df["cloud_cover"] <= r.max_cloud_aoi]
    df = df[df.get("status", pd.Series("ok", index=df.index)).ne("composite")]  # rule 6
    if r.allowed_months:
        allowed = set(r.allowed_months)
        df = df[df["date"].map(lambda d: d.month in allowed)]  # rule 5
    if df.empty:
        return []
    negative = str(aoi_row.get("target_type", "")) == "negative"
    specs: list[PairSpec] = []

    def make(t1: pd.Series, t2: pd.Series, hint: str, n: int) -> PairSpec:
        same_orbit = bool(t1["relative_orbit"] == t2["relative_orbit"])
        season = _doy_distance(t1["date"], t2["date"]) <= r.season_tolerance_days
        return PairSpec(
            aoi_id,
            f"p{n:02d}",
            str(t1["product_id"]),
            str(t2["product_id"]),
            t1["date"],
            t2["date"],
            same_orbit,
            season,
            hint,
        )

    if negative:  # rule 4: maximum seasonal contrast
        s_a = df[df["date"].map(lambda d: d.month in set(r.contrast_seasons[0]))]
        s_b = df[df["date"].map(lambda d: d.month in set(r.contrast_seasons[1]))]
        cands: list[tuple[int, int, pd.Series, pd.Series]] = []
        for _, a in s_a.iterrows():
            for _, b in s_b.iterrows():
                if a["product_id"] == b["product_id"]:
                    continue  # rule 6: never the same scene twice
                same = int(a["relative_orbit"] == b["relative_orbit"])
                cands.append((same, _doy_distance(a["date"], b["date"]), a, b))

        def _neg_key(c: tuple[int, int, pd.Series, pd.Series]) -> tuple[int, int, str, str]:
            return (-c[0], -c[1], str(c[2]["product_id"]), str(c[3]["product_id"]))

        cands.sort(key=_neg_key)
        for i, (_, _, a, b) in enumerate(cands[: r.max_pairs], start=1):
            specs.append(make(a, b, TileType.NOCHANGE_HARD.value, i))
        return specs

    w1 = _parse_window(aoi_row.get("t1_window"))
    w2 = _parse_window(aoi_row.get("t2_window"))
    if w1 is None or w2 is None:
        log.warning("%s: no t1/t2 windows in the registry; no positive pairs selected", aoi_id)
        return []
    t1s = df[df["date"].map(lambda d: w1[0] <= d <= w1[1])]
    t2s = df[df["date"].map(lambda d: w2[0] <= d <= w2[1])]
    cands2: list[tuple[int, int, float, pd.Series, pd.Series]] = []
    for _, a in t1s.iterrows():
        for _, b in t2s.iterrows():
            if a["product_id"] == b["product_id"]:
                continue
            same = int(a["relative_orbit"] == b["relative_orbit"])  # rule 1
            season = int(_doy_distance(a["date"], b["date"]) <= r.season_tolerance_days)  # rule 3
            cloud = float(a["cloud_cover"] + b["cloud_cover"])
            cands2.append((same, season, cloud, a, b))
    if not cands2:
        return []
    best_same = max(c[0] for c in cands2)
    cands2 = [c for c in cands2 if c[0] == best_same]  # cross-orbit only if no same-orbit pair

    def _pos_key(c: tuple[int, int, float, pd.Series, pd.Series]) -> tuple[int, float, str, str]:
        return (-c[1], c[2], str(c[3]["product_id"]), str(c[4]["product_id"]))

    cands2.sort(key=_pos_key)
    for i, (_, _, _, a, b) in enumerate(cands2[: r.max_pairs], start=1):
        specs.append(make(a, b, TileType.CHANGE_POS.value, i))
    return specs


def rules_for_stratum(cfg_strata: dict[str, dict[str, Any]], stratum: str) -> PairRules:
    s = cfg_strata[stratum]
    return PairRules(
        allowed_months=s.get("allowed_months"),
        contrast_seasons=[list(x) for x in s["contrast_seasons"]],
        max_pairs=int(s.get("max_pairs", 3)),
    )


def query_and_select(cfg: Any, aoi_ids: Sequence[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Query CDSE per AOI, write scenes.parquet and pairs.parquet (UNVERIFIED path: needs CDSE)."""
    from scd.config import load_strata
    from scd.imagery import cdse, registry

    aois = registry.load(registry.registry_path(cfg))
    if aoi_ids:
        aois = aois[aois["aoi_id"].isin(list(aoi_ids))]
    strata = load_strata()
    scene_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for _, aoi in aois.iterrows():
        maybe = [_parse_window(aoi.get("t1_window")), _parse_window(aoi.get("t2_window"))]
        windows: list[tuple[date, date]] = [w for w in maybe if w is not None] or [(date(2017, 1, 1), date.today())]
        feats: list[dict[str, Any]] = []
        for w in windows:
            feats += cdse.query(aoi["geometry"].wkt, w[0].isoformat(), w[1].isoformat(), cloud_max=20.0)
        for f in feats:
            scene_rows.append({"aoi_id": aoi["aoi_id"], **f, "local_path": "", "status": "queried"})
        sdf = (
            pd.DataFrame(scene_rows, columns=list(SCENES_COLUMNS))
            if scene_rows
            else pd.DataFrame(columns=list(SCENES_COLUMNS))
        )
        for spec in select_pairs(aoi, sdf, rules_for_stratum(strata, str(aoi["stratum"]))):
            pair_rows.append(spec.as_row())
    scenes_df = pd.DataFrame(scene_rows, columns=list(SCENES_COLUMNS)).drop_duplicates(["aoi_id", "product_id"])
    pairs_df = pd.DataFrame(pair_rows, columns=list(PAIRS_COLUMNS))
    cfg.registry_dir.mkdir(parents=True, exist_ok=True)
    scenes_df.to_parquet(cfg.registry_dir / "scenes.parquet", index=False)
    scenes_df.to_csv(cfg.registry_dir / "scenes.csv", index=False)
    pairs_df.to_parquet(cfg.registry_dir / "pairs.parquet", index=False)
    pairs_df.to_csv(cfg.registry_dir / "pairs.csv", index=False)
    return scenes_df, pairs_df
