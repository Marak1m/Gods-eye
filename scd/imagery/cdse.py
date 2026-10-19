"""Copernicus Data Space download via cdsetool (HANDOFF.md 2.1, 5.3).

UNVERIFIED on this machine (no CDSE credentials; DEVIATIONS "Unverified code"). Written against
the API notes in HANDOFF.md 2.1: `query_features("SENTINEL-2", {...})` and
`download_features(features, path, {"concurrency": 4, "credentials": Credentials(user, pw)})`,
consumed with `list()`. Credentials fall back to ~/.netrc when CDSE_USER/CDSE_PASS are unset.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, cast

from scd.log import get_logger

log = get_logger(__name__)

_ORBIT_RE = re.compile(r"_R(\d{3})_")
_TILE_RE = re.compile(r"_T(\d{2}[A-Z]{3})_")
_BASELINE_RE = re.compile(r"_N(\d{4})_")


def _feature_to_row(feat: dict[str, Any]) -> dict[str, Any]:
    props = feat.get("properties", {})
    title = str(props.get("title") or feat.get("id"))
    title = title.removesuffix(".SAFE")
    orbit = _ORBIT_RE.search(title)
    tile = _TILE_RE.search(title)
    base = _BASELINE_RE.search(title)
    return {
        "product_id": title,
        "tile": tile.group(1) if tile else None,
        "relative_orbit": int(orbit.group(1)) if orbit else -1,
        "date": str(props.get("startDate", ""))[:10],
        "cloud_cover": float(props.get("cloudCover", float("nan"))),
        "baseline": float(base.group(1)) / 100.0 if base else None,
    }


def query(
    wkt: str, start: str, end: str, cloud_max: float = 20.0, product_type: str = "S2MSI2A"
) -> list[dict[str, Any]]:
    """S2 L2A products intersecting `wkt` between `start` and `end` (ISO dates). Cap 10,000."""
    from cdsetool.query import query_features

    params: dict[str, Any] = {
        "contentDateStartGe": f"{start}T00:00:00Z",
        "contentDateStartLe": f"{end}T23:59:59Z",
        "productType": product_type,
        "cloudCover": f"[0, {cloud_max:g}]",
        "geometry": wkt,
    }
    feats = list(query_features("SENTINEL-2", params))
    if len(feats) >= 10_000:
        log.warning("CDSE query hit the 10,000 result cap; narrow the window")
    rows = [_feature_to_row(f) | {"_feature": f} for f in feats]
    return rows


def _credentials() -> Any | None:
    user, pw = os.environ.get("CDSE_USER"), os.environ.get("CDSE_PASS")
    if not user or not pw:
        return None  # cdsetool falls back to ~/.netrc
    from cdsetool.credentials import Credentials

    return Credentials(user, pw)


def fetch(product_ids: list[str], dest: Path, concurrency: int = 4) -> list[Path]:
    """Download `.SAFE` zips to `dest`, skipping files already present with a matching size."""
    from cdsetool.download import download_features
    from cdsetool.query import query_features

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    todo: list[Any] = []
    done: list[Path] = []
    for pid in product_ids:
        target = dest / f"{pid}.zip"
        feats = list(query_features("SENTINEL-2", {"productIdentifier": f"%{pid}%"}))
        if not feats:
            log.warning("CDSE: product %s not found", pid)
            continue
        feat = feats[0]
        size = int(feat.get("properties", {}).get("services", {}).get("download", {}).get("size", -1))
        if target.exists() and (size < 0 or target.stat().st_size == size):
            done.append(target)
            continue
        todo.append(feat)
    if todo:
        opts: dict[str, Any] = {"concurrency": concurrency}
        creds = _credentials()
        if creds is not None:
            opts["credentials"] = creds
        list(download_features(cast(Any, todo), str(dest), opts))
        for f in todo:
            title = str(f.get("properties", {}).get("title", "")).removesuffix(".SAFE")
            p = dest / f"{title}.zip"
            if p.exists():
                done.append(p)
            else:
                log.warning("CDSE: download of %s did not produce %s", title, p)
    return done
