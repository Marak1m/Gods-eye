"""CBRA annual 2.5 m rooftop rasters, China 2016-2021: diff -> components >= 1,800 m2 (Zenodo 7861676).

HANDOFF.md section 5.6. Milestone M3. Needs external data / network: stub until Phase 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class CbraSource:
    """`Source` implementation for cbra (building)."""

    name = "cbra"
    kind = "building"

    def fetch(self, aoi: Any, cache: Path) -> Path:
        """Raw download, cached under `cache`."""
        raise NotImplementedError("scd.labels.sources.cbra.fetch: built at M3 (Phase 1); see HANDOFF.md section 5.6")

    def candidates(self, aoi: Any, pair: Any) -> Any:
        """GeoDataFrame: geometry (UTM), class_id, source, source_ref, t_event, confidence='auto', notes."""
        raise NotImplementedError(
            "scd.labels.sources.cbra.candidates: built at M3 (Phase 1); see HANDOFF.md section 5.6"
        )
