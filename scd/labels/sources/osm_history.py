"""OSM full history via ohsome: ways created or extended between T1 and T2, buffered by width class (lead only).

HANDOFF.md section 5.6. Milestone M3. Needs external data / network: stub until Phase 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class OsmHistorySource:
    """`Source` implementation for osm_history (road)."""

    name = "osm_history"
    kind = "road"

    def fetch(self, aoi: Any, cache: Path) -> Path:
        """Raw download, cached under `cache`."""
        raise NotImplementedError(
            "scd.labels.sources.osm_history.fetch: built at M3 (Phase 1); see HANDOFF.md section 5.6"
        )

    def candidates(self, aoi: Any, pair: Any) -> Any:
        """GeoDataFrame: geometry (UTM), class_id, source, source_ref, t_event, confidence='auto', notes."""
        raise NotImplementedError(
            "scd.labels.sources.osm_history.candidates: built at M3 (Phase 1); see HANDOFF.md section 5.6"
        )
