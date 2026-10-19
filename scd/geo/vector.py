"""Vector helpers: projected-CRS assertions, footprints, reprojection and canonical content keys.

HANDOFF.md section 14: buffers and areas only in a projected CRS. The content key is the
canonical-ordering primitive used by every pairwise operation (DEVIATIONS C3).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import shapely
from pyproj import CRS, Transformer
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shp_transform


class GeographicCRSError(ValueError):
    """A metric operation was attempted in a geographic CRS."""


def require_projected(crs: Any) -> CRS:
    """Return the CRS if projected, else raise (never buffer or measure area in degrees)."""
    c = CRS.from_user_input(crs)
    if not c.is_projected:
        raise GeographicCRSError(
            f"operation requires a projected CRS, got {c.to_string()!r}; "
            "reproject to the pair's UTM zone first (HANDOFF.md section 14)"
        )
    return c


def reproject_geom(geom: BaseGeometry, src_crs: Any, dst_crs: Any) -> BaseGeometry:
    """Reproject a shapely geometry between CRSs (always x/y order)."""
    src = CRS.from_user_input(src_crs)
    dst = CRS.from_user_input(dst_crs)
    if src == dst:
        return geom
    t = Transformer.from_crs(src, dst, always_xy=True)
    return shp_transform(t.transform, geom)


def raster_footprint_wgs84(path: Path, densify_px: int = 32) -> BaseGeometry:
    """Footprint polygon of a raster in EPSG:4326, edges densified before reprojection."""
    with rasterio.open(path) as ds:
        if ds.crs is None:
            raise ValueError(f"{path} has no CRS")
        poly = box(*ds.bounds)
        seg = max(ds.res) * densify_px
        poly = shapely.segmentize(poly, seg)
        return reproject_geom(poly, ds.crs.to_wkt(), 4326)


def content_key(*parts: Any) -> str:
    """sha256 over the byte content of arrays, geometries, bytes or files.

    Two inputs with identical content get identical keys regardless of the Python object; this is
    what makes `half_shift(a, b)` the exact mirror of `half_shift(b, a)`.
    """
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, np.ndarray):
            h.update(str(p.dtype).encode())
            h.update(str(p.shape).encode())
            h.update(np.ascontiguousarray(p).tobytes())
        elif isinstance(p, BaseGeometry):
            h.update(p.wkb)
        elif isinstance(p, bytes | bytearray):
            h.update(bytes(p))
        elif isinstance(p, Path):
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
        else:
            h.update(repr(p).encode())
    return h.hexdigest()


def canonical_pair(a: Any, b: Any, key_a: str, key_b: str) -> tuple[bool, Any, Any]:
    """Order (a, b) by content key. Returns (swapped, first, second); swapped means b came first."""
    if key_b < key_a:
        return True, b, a
    return False, a, b
