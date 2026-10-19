"""Dynamic World built/bare annual medians -> delta clusters (Earth Engine; optional).

HANDOFF.md section 5.6. Milestone M3. Needs external data / network: stub until Phase 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class DynamicWorldSource:
    """`Source` implementation for dynamic_world (buildup)."""

    name = "dynamic_world"
    kind = "buildup"

    def fetch(self, aoi: Any, cache: Path) -> Path:
        """Raw download, cached under `cache`."""
        raise NotImplementedError(
            "scd.labels.sources.dynamic_world.fetch: built at M3 (Phase 1); see HANDOFF.md section 5.6"
        )

    def candidates(self, aoi: Any, pair: Any) -> Any:
        """GeoDataFrame: geometry (UTM), class_id, source, source_ref, t_event, confidence='auto', notes."""
        raise NotImplementedError(
            "scd.labels.sources.dynamic_world.candidates: built at M3 (Phase 1); see HANDOFF.md section 5.6"
        )
