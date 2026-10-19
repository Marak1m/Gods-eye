"""Canonical working grid (HANDOFF.md section 5.4 step 1).

The grid is a pure function of the *set* of footprints: UTM zone from the centroid of their union,
origin snapped to `snap_m`, extent = intersection. Footprints are sorted by WKB before any
overlay so the result is bit-identical for any input order (DEVIATIONS C3).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from affine import Affine
from pyproj import CRS
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from scd.geo.vector import reproject_geom


@dataclass(frozen=True)
class Grid:
    """A north-up raster grid in a projected CRS."""

    epsg: int
    transform: Affine
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"empty grid {self.width}x{self.height}")
        if self.transform.b != 0 or self.transform.d != 0 or self.transform.e >= 0:
            raise ValueError("grid must be north-up with no rotation")

    @property
    def res_m(self) -> float:
        return abs(self.transform.a)

    @property
    def crs(self) -> CRS:
        return CRS.from_epsg(self.epsg)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        left, top = self.transform.c, self.transform.f
        right = left + self.transform.a * self.width
        bottom = top + self.transform.e * self.height
        return (left, bottom, right, top)

    @property
    def polygon(self) -> BaseGeometry:
        return box(*self.bounds)

    @property
    def pixel_area_m2(self) -> float:
        return self.res_m * self.res_m

    def window(self, row_off: int, col_off: int, height: int, width: int) -> Grid:
        """Sub-grid for a chip window."""
        t = self.transform @ Affine.translation(col_off, row_off)
        return Grid(self.epsg, t, width, height)

    def to_dict(self) -> dict[str, Any]:
        return {
            "epsg": self.epsg,
            "transform": list(self.transform)[:6],
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Grid:
        return cls(int(d["epsg"]), Affine(*d["transform"]), int(d["width"]), int(d["height"]))

    def rasterio_profile(self) -> dict[str, Any]:
        return {
            "crs": self.crs.to_wkt(),
            "transform": self.transform,
            "width": self.width,
            "height": self.height,
        }


def utm_epsg(lon: float, lat: float) -> int:
    """EPSG code of the UTM zone containing a WGS84 point (326xx north, 327xx south)."""
    zone = int(math.floor((lon + 180.0) / 6.0)) + 1
    zone = min(max(zone, 1), 60)
    return (32600 if lat >= 0 else 32700) + zone


def canonical_grid(footprints: Sequence[BaseGeometry], res_m: float, snap_m: float) -> Grid:
    """Grid from a set of WGS84 footprints; identical for any ordering of `footprints`."""
    if len(footprints) == 0:
        raise ValueError("canonical_grid needs at least one footprint")
    if res_m <= 0 or snap_m <= 0:
        raise ValueError("res_m and snap_m must be positive")
    fps = sorted(footprints, key=lambda g: g.wkb)  # canonical order
    union = unary_union(fps)
    centre = union.centroid
    epsg = utm_epsg(centre.x, centre.y)
    utm = [reproject_geom(g, 4326, epsg) for g in fps]
    inter = utm[0]
    for g in utm[1:]:
        inter = inter.intersection(g)
    if inter.is_empty or inter.area <= 0:
        raise ValueError("footprints do not overlap")
    minx, miny, maxx, maxy = inter.bounds
    # round the quotients to 1e-6 snap units (5 um at 5 m) so floating-point noise from the
    # WGS84 round trip of footprints cannot add a spurious row or column
    left = math.floor(round(minx / snap_m, 6)) * snap_m
    bottom = math.floor(round(miny / snap_m, 6)) * snap_m
    right = math.ceil(round(maxx / snap_m, 6)) * snap_m
    top = math.ceil(round(maxy / snap_m, 6)) * snap_m
    width = int(round((right - left) / res_m))
    height = int(round((top - bottom) / res_m))
    return Grid(epsg, Affine(res_m, 0.0, left, 0.0, -res_m, top), width, height)
