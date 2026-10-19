"""Human verification (HANDOFF.md 5.7): review export, reviewed import with validation, agreement.

`import_reviewed` validates a corrected GeoPackage (projected CRS, class_id 1-4, confidence in
{corrected, double_checked}, annotator, timestamp, geometry validity, minimum sizes) and writes
the accepted rows to reviewed/<aoi>/<pair>.gpkg with a rejection report. `export_for_review`
(QGIS project + previews) is a stub until the QGIS template from the companion project is in
hand.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

from scd.config import Config
from scd.constants import CHANGE_CLASSES
from scd.geo.vector import GeographicCRSError, require_projected
from scd.labels.conventions import Conventions
from scd.labels.rasterize import REVIEWED_LAYER, filter_by_rules, load_reviewed
from scd.log import get_logger

log = get_logger(__name__)

REQUIRED_COLUMNS = ("class_id", "confidence", "annotator", "timestamp")
ALLOWED_CONFIDENCE = ("corrected", "double_checked")


@dataclass
class ImportReport:
    gpkg: Path
    out_path: Path | None
    accepted: int
    rejected: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        head = f"{self.gpkg.name}: {self.accepted} accepted, {len(self.rejected)} rejected"
        if self.errors:
            head += "; ERRORS: " + "; ".join(self.errors)
        return head


def import_reviewed(
    gpkg: Path, conventions: Conventions, out_path: Path, expect_epsg: int | None = None
) -> ImportReport:
    """Validate a corrected GeoPackage and write the accepted polygons to `out_path`."""
    gpkg = Path(gpkg)
    try:
        gdf = load_reviewed(gpkg)
    except Exception as e:
        return ImportReport(gpkg, None, 0, errors=[f"cannot read: {e}"])
    errors: list[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in gdf]
    if missing:
        errors.append(f"missing columns {missing}")
    if gdf.crs is None:
        errors.append("no CRS")
    else:
        try:
            require_projected(gdf.crs)
        except GeographicCRSError as e:
            errors.append(str(e))
        if expect_epsg is not None and gdf.crs.to_epsg() != expect_epsg:
            errors.append(f"CRS EPSG:{gdf.crs.to_epsg()} differs from the pair grid EPSG:{expect_epsg}")
    if errors:
        return ImportReport(gpkg, None, 0, errors=errors)

    rejected: list[dict[str, Any]] = []
    keep: list[bool] = []
    for i, row in gdf.iterrows():
        why: list[str] = []
        if int(row["class_id"]) not in range(1, 5):
            why.append(f"class_id {row['class_id']} not in 1..4")
        if str(row["confidence"]) not in ALLOWED_CONFIDENCE:
            why.append(f"confidence {row['confidence']!r} not in {ALLOWED_CONFIDENCE}")
        if not str(row["annotator"] or "").strip():
            why.append("annotator empty")
        try:
            pd.Timestamp(row["timestamp"])
        except Exception:
            why.append(f"timestamp {row['timestamp']!r} unparseable")
        if row.geometry is None or row.geometry.is_empty or not row.geometry.is_valid:
            why.append("invalid or empty geometry")
        if why:
            rejected.append({"index": int(i), "reason": "; ".join(why)})
        keep.append(not why)
    valid_rows = gdf[keep].copy() if len(gdf) else gdf.copy()
    if len(valid_rows):
        kept, size_report = filter_by_rules(valid_rows, conventions)
        for d in size_report["dropped"]:
            rejected.append({"index": int(valid_rows.index[d["index"]]), "reason": d["reason"]})
    else:
        kept = valid_rows
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if len(kept):
        kept.to_file(out_path, layer=REVIEWED_LAYER, driver="GPKG")
    else:
        gpd.GeoDataFrame(gdf.iloc[0:0], geometry="geometry", crs=gdf.crs).to_file(
            out_path, layer=REVIEWED_LAYER, driver="GPKG"
        )
    report = ImportReport(gpkg, out_path, int(len(kept)), rejected)
    if rejected:
        pd.DataFrame(rejected).to_csv(out_path.with_suffix(".rejected.csv"), index=False)
    return report


def agreement(gpkg_a: Path, gpkg_b: Path) -> dict[int, float]:
    """Per-class IoU between two annotations of the same pair (projected CRS)."""
    a = load_reviewed(Path(gpkg_a))
    b = load_reviewed(Path(gpkg_b))
    if a.crs != b.crs:
        b = b.to_crs(a.crs)
    require_projected(a.crs)
    out: dict[int, float] = {}
    for cid in range(1, 5):
        sa, sb = (a["class_id"] == cid), (b["class_id"] == cid)
        ga = unary_union(list(a.geometry[sa])) if sa.any() else None
        gb = unary_union(list(b.geometry[sb])) if sb.any() else None
        if ga is None and gb is None:
            continue
        if ga is None or gb is None:
            out[cid] = 0.0
            continue
        union = ga.union(gb).area
        out[cid] = float(ga.intersection(gb).area / union) if union > 0 else 1.0
    return out


# --- orchestration -------------------------------------------------------------------------
def export_for_review(aoi: Any, pair: Any, cfg: Config) -> Path:
    """QGIS project (.qgz) + GeoPackage with T1/T2 previews, validity and candidate layers."""
    raise NotImplementedError(
        "scd.labels.verification.export_for_review: QGIS project export is built at M3 (Phase 1) "
        "once the companion review template is available; see HANDOFF.md section 5.7"
    )


def export_all(cfg: Config, aoi_ids: Sequence[str] | None = None) -> list[Path]:
    from scd.imagery.preprocess import list_products

    out = []
    for prod in list_products(cfg):
        if aoi_ids and prod.aoi_id not in set(aoi_ids):
            continue
        out.append(export_for_review(prod.aoi_id, prod.pair_id, cfg))
    return out


def pending(cfg: Config) -> pd.DataFrame:
    """Pairs with candidates but no reviewed GeoPackage yet."""
    rows = []
    for cand in sorted(cfg.candidates_dir.glob("*/*.gpkg")):
        aoi, pair = cand.parent.name, cand.stem
        reviewed = cfg.reviewed_dir / aoi / f"{pair}.gpkg"
        rows.append({"aoi_id": aoi, "pair_id": pair, "candidates": str(cand), "reviewed": reviewed.exists()})
    df = pd.DataFrame(rows, columns=["aoi_id", "pair_id", "candidates", "reviewed"])
    return df[~df["reviewed"].astype(bool)].reset_index(drop=True) if len(df) else df


def inbox_dir(cfg: Config) -> Path:
    return cfg.data_root / "review_inbox"


def import_all(cfg: Config, conventions: Conventions, gpkgs: Sequence[Path] | None = None) -> list[ImportReport]:
    """Import explicit GeoPackages or everything under <data_root>/review_inbox/<aoi>/<pair>.gpkg."""
    from scd.imagery.preprocess import PairProduct

    files = [Path(p) for p in gpkgs] if gpkgs else sorted(inbox_dir(cfg).glob("*/*.gpkg"))
    reports = []
    for f in files:
        aoi, pair = f.parent.name, f.stem
        prod = PairProduct(cfg.pairs_dir / aoi / pair)
        epsg = prod.grid().epsg if prod.exists() else None
        rep = import_reviewed(f, conventions, cfg.reviewed_dir / aoi / f"{pair}.gpkg", expect_epsg=epsg)
        log.info(rep.summary())
        reports.append(rep)
    return reports


def agreement_table(cfg: Config) -> pd.DataFrame:
    """Double-annotated pairs: <data_root>/review_inbox_double/<aoi>/<pair>.gpkg vs reviewed."""
    rows = []
    for f in sorted((cfg.data_root / "review_inbox_double").glob("*/*.gpkg")):
        other = cfg.reviewed_dir / f.parent.name / f.name
        if other.exists():
            iou = agreement(f, other)
            rows.append(
                {
                    "aoi_id": f.parent.name,
                    "pair_id": f.stem,
                    **{CHANGE_CLASSES[c - 1]: iou.get(c, float("nan")) for c in range(1, 5)},
                }
            )
    return pd.DataFrame(rows, columns=["aoi_id", "pair_id", *CHANGE_CLASSES])
