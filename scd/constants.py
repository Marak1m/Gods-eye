"""Project-wide constants (HANDOFF.md sections 1, 3.2, 5.10).

Everything here is a contract: band order, class ids, the ignore value, tile types. Nothing in
`scd/` may hard-code these values; import them from here.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

# --- bands -----------------------------------------------------------------------------------
BAND_ORDER: Final[tuple[str, ...]] = ("B02", "B03", "B04", "B08", "B11", "B12")
N_BANDS: Final[int] = len(BAND_ORDER)
RGBN_BANDS: Final[tuple[str, ...]] = ("B02", "B03", "B04", "B08")  # native 10 m
SWIR_BANDS: Final[tuple[str, ...]] = ("B11", "B12")  # native 20 m
BAND_INDEX: Final[dict[str, int]] = {b: i for i, b in enumerate(BAND_ORDER)}
# SEN2SR-Lite NonReference_RGBN_x4 input order (HANDOFF 2.1): Red, Green, Blue, NIR
SEN2SR_RGBN_ORDER: Final[tuple[str, ...]] = ("B04", "B03", "B02", "B08")
# OmniCloudMask input order (HANDOFF 2.1): Red, Green, NIR
OMNICLOUD_ORDER: Final[tuple[str, ...]] = ("B04", "B03", "B08")
UINT16_SCALE: Final[float] = 10000.0  # reflectance = DN / 10000
NATIVE_RES_M: Final[dict[str, float]] = {b: 10.0 for b in RGBN_BANDS} | {b: 20.0 for b in SWIR_BANDS}

# --- labels ----------------------------------------------------------------------------------
IGNORE: Final[int] = 255
CLASS_IDS: Final[dict[str, int]] = {"none": 0, "buildup": 1, "building": 2, "road": 3, "solar": 4}
CLASS_NAMES: Final[tuple[str, ...]] = ("none", "buildup", "building", "road", "solar")
CHANGE_CLASSES: Final[tuple[str, ...]] = ("buildup", "building", "road", "solar")
N_CHANGE_CLASSES: Final[int] = 4
N_SEM_CLASSES: Final[int] = 5
RUBRIC_WEIGHTS: Final[dict[str, float]] = {
    "buildup": 0.40,
    "building": 0.30,
    "road": 0.20,
    "solar": 0.10,
}
LABEL_CONFIDENCE: Final[tuple[str, ...]] = ("auto", "corrected", "double_checked")

# --- registry / splits -----------------------------------------------------------------------
STRATA: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E", "F")
ROLES: Final[tuple[str, ...]] = ("train", "val", "targets", "stage3", "stage2")
SPLITS: Final[tuple[str, ...]] = ("train", "val", "stage2", "targets")
TARGET_TYPES: Final[tuple[str, ...]] = (
    "airfield",
    "compound",
    "solar",
    "building",
    "road",
    "reclamation",
    "settlement",
    "negative",
)
LEAD_SOURCES: Final[tuple[str, ...]] = (
    "gem",
    "cbra",
    "obt",
    "io_lulc",
    "osm_history",
    "manual",
    "companion",
    "kruitwagen",
    "dynamic_world",
)


class TileType(StrEnum):
    """Draw-time tile taxonomy (plan doc 7.3). Only the first three are stored in the manifest."""

    CHANGE_POS = "CHANGE_POS"
    NOCHANGE_HARD = "NOCHANGE_HARD"
    NOCHANGE_PLAIN = "NOCHANGE_PLAIN"
    IDENTICAL = "IDENTICAL"
    NEAR_IDENTICAL = "NEAR_IDENTICAL"
    SYNTHETIC = "SYNTHETIC"
    STAR = "STAR"


TILE_TYPES: Final[tuple[str, ...]] = tuple(t.value for t in TileType)
STORED_TILE_TYPES: Final[tuple[str, ...]] = (
    TileType.CHANGE_POS.value,
    TileType.NOCHANGE_HARD.value,
    TileType.NOCHANGE_PLAIN.value,
)

# --- cloud masks -----------------------------------------------------------------------------
# SCL classes treated as invalid when cloud.method == scl (HANDOFF 5.4 step 3); snow (11) stays valid
SCL_INVALID: Final[frozenset[int]] = frozenset({0, 1, 3, 8, 9, 10})
# OmniCloudMask classes: 0 clear, 1 thick cloud, 2 thin cloud, 3 shadow
OMNICLOUD_INVALID: Final[frozenset[int]] = frozenset({1, 2, 3})

# --- misc ------------------------------------------------------------------------------------
DEFAULT_SEED: Final[int] = 42
GEOTIFF_BLOCK: Final[int] = 512
