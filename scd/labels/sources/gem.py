"""GEM Global Solar Power Tracker: points with capacity and start year -> buffered seed polygons.

Seed radius r = sqrt(cap_MW x 2 ha / pi).

HANDOFF.md section 5.6. Milestone M3. Needs external data / network: stub until Phase 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class GemSource:
    """`Source` implementation for gem (solar)."""

    name = "gem"
    kind = "solar"

    def fetch(self, aoi: Any, cache: Path) -> Path:
        """Raw download, cached under `cache`."""
        raise NotImplementedError("scd.labels.sources.gem.fetch: built at M3 (Phase 1); see HANDOFF.md section 5.6")

    def candidates(self, aoi: Any, pair: Any) -> Any:
        """GeoDataFrame: geometry (UTM), class_id, source, source_ref, t_event, confidence='auto', notes."""
        raise NotImplementedError(
            "scd.labels.sources.gem.candidates: built at M3 (Phase 1); see HANDOFF.md section 5.6"
        )
