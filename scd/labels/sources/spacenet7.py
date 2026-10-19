"""SpaceNet 7 monthly footprints converted to our chip format at 5 m (pretraining, A12).

HANDOFF.md section 5.6. Milestone M3. Needs external data / network: stub until Phase 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class Spacenet7Source:
    """`Source` implementation for spacenet7 (external)."""

    name = "spacenet7"
    kind = "external"

    def fetch(self, aoi: Any, cache: Path) -> Path:
        """Raw download, cached under `cache`."""
        raise NotImplementedError(
            "scd.labels.sources.spacenet7.fetch: built at M3 (Phase 1); see HANDOFF.md section 5.6"
        )

    def candidates(self, aoi: Any, pair: Any) -> Any:
        """GeoDataFrame: geometry (UTM), class_id, source, source_ref, t_event, confidence='auto', notes."""
        raise NotImplementedError(
            "scd.labels.sources.spacenet7.candidates: built at M3 (Phase 1); see HANDOFF.md section 5.6"
        )
