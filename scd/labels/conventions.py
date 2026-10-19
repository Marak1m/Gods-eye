"""Label conventions read from `labeling_guide.yaml` (HANDOFF.md 5.5, plan doc Appendix B).

`load()` fails while any field is still `fill`: nothing may be rasterised or annotated against an
unfinished guide.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from scd.constants import CHANGE_CLASSES, CLASS_IDS

FILL = "fill"


class ConventionsError(ValueError):
    """The labeling guide is incomplete or inconsistent with the code's class contract."""


@dataclass(frozen=True)
class Conventions:
    class_ids: dict[str, int]
    priority: tuple[str, ...]  # first wins on overlap
    removals_labeled: bool
    min_area_m2: dict[str, float]
    min_length_m: dict[str, float]
    road_width_m: dict[str, float]
    building_rule_applies_to: str  # polygon | component
    construction_in_progress: str  # buildup | building
    partial_change_labeled: bool
    runways_as_roads: bool
    solar_includes_substations: bool
    raw: dict[str, Any]

    def priority_ids(self) -> list[int]:
        """Class ids from highest to lowest priority."""
        return [self.class_ids[c] for c in self.priority]

    def burn_order(self) -> list[int]:
        """Class ids in the order to burn so that the last burned (highest priority) wins."""
        return list(reversed(self.priority_ids()))

    def min_area_for(self, class_id: int) -> float:
        name = next(k for k, v in self.class_ids.items() if v == class_id)
        return float(self.min_area_m2.get(name, 0.0))


def unfilled(doc: Any, path: str = "") -> list[str]:
    """Dotted paths of every value still equal to 'fill'."""
    out: list[str] = []
    if isinstance(doc, dict):
        for k, v in doc.items():
            out += unfilled(v, f"{path}.{k}" if path else str(k))
    elif isinstance(doc, list):
        for i, v in enumerate(doc):
            out += unfilled(v, f"{path}[{i}]")
    elif isinstance(doc, str) and doc.strip().lower() == FILL:
        out.append(path)
    return out


def _bool(doc: dict[str, Any], key: str) -> bool:
    v = doc[key]
    if not isinstance(v, bool):
        raise ConventionsError(f"general.{key} must be true/false, got {v!r}")
    return v


def load(path: str | Path) -> Conventions:
    p = Path(path)
    if not p.exists():
        raise ConventionsError(f"labeling guide not found: {p}")
    doc = yaml.safe_load(p.read_text())
    if not isinstance(doc, dict):
        raise ConventionsError(f"{p} is not a mapping")
    fills = unfilled(doc)
    if fills:
        raise ConventionsError(
            f"labeling guide {p} has {len(fills)} unfilled field(s) still set to 'fill': "
            + ", ".join(fills)
            + ". Fill it from the Stage-II data (plan doc Appendix B) before rasterising."
        )
    general = doc["general"]
    classes = doc["classes"]
    ids = {"none": 0} | {name: int(classes[name]["id"]) for name in CHANGE_CLASSES}
    if ids != CLASS_IDS:
        raise ConventionsError(f"class ids {ids} differ from the code contract {CLASS_IDS}")
    priority = tuple(str(c) for c in general["overlap_priority"])
    if sorted(priority) != sorted(CHANGE_CLASSES):
        raise ConventionsError(f"overlap_priority must permute {CHANGE_CLASSES}, got {priority}")
    applies = str(general["building_rule_applies_to"])
    if applies not in {"polygon", "component"}:
        raise ConventionsError("building_rule_applies_to must be polygon or component")
    cip = str(general["construction_in_progress"])
    if cip not in {"buildup", "building"}:
        raise ConventionsError("construction_in_progress must be buildup or building")
    min_area = {name: float(classes[name]["min_area_m2"]) for name in CHANGE_CLASSES if "min_area_m2" in classes[name]}
    widths = {k: float(v) for k, v in classes["road"]["burn_width_m"].items()}
    if set(widths) != {"highway", "access"}:
        raise ConventionsError("road.burn_width_m must define highway and access")
    return Conventions(
        class_ids=ids,
        priority=priority,
        removals_labeled=_bool(general, "removals_labeled"),
        min_area_m2=min_area,
        min_length_m={"road": float(classes["road"]["min_length_m"])},
        road_width_m=widths,
        building_rule_applies_to=applies,
        construction_in_progress=cip,
        partial_change_labeled=_bool(general, "partial_change_labeled"),
        runways_as_roads=bool(classes["road"]["runways_as_roads"]),
        solar_includes_substations=bool(classes["solar"]["includes_substations"]),
        raw=doc,
    )
