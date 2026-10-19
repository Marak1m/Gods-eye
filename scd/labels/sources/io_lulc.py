"""Impact Observatory annual 10 m LULC 2017-2023 on Planetary Computer: Built Area diff -> clusters.

HANDOFF.md section 5.6. Milestone M3. Needs external data / network: stub until Phase 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class IoLulcSource:
    """`Source` implementation for io_lulc (buildup)."""

    name = "io_lulc"
    kind = "buildup"

    def fetch(self, aoi: Any, cache: Path) -> Path:
        """Raw download, cached under `cache`."""
        raise NotImplementedError("scd.labels.sources.io_lulc.fetch: built at M3 (Phase 1); see HANDOFF.md section 5.6")

    def candidates(self, aoi: Any, pair: Any) -> Any:
        """GeoDataFrame: geometry (UTM), class_id, source, source_ref, t_event, confidence='auto', notes."""
        raise NotImplementedError(
            "scd.labels.sources.io_lulc.candidates: built at M3 (Phase 1); see HANDOFF.md section 5.6"
        )
