"""Open Buildings 2.5D Temporal annual presence 2016-2023 -> diff -> components >= 1,800 m2.

HANDOFF.md section 5.6. Milestone M3. Needs external data / network: stub until Phase 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class ObtSource:
    """`Source` implementation for obt (building)."""

    name = "obt"
    kind = "building"

    def fetch(self, aoi: Any, cache: Path) -> Path:
        """Raw download, cached under `cache`."""
        raise NotImplementedError("scd.labels.sources.obt.fetch: built at M3 (Phase 1); see HANDOFF.md section 5.6")

    def candidates(self, aoi: Any, pair: Any) -> Any:
        """GeoDataFrame: geometry (UTM), class_id, source, source_ref, t_event, confidence='auto', notes."""
        raise NotImplementedError(
            "scd.labels.sources.obt.candidates: built at M3 (Phase 1); see HANDOFF.md section 5.6"
        )
