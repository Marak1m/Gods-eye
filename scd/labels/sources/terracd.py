"""TERRA-CD Sentinel-2 pairs 2019 vs 2024 with built-area transitions, converted to 5 m chips (A12).

HANDOFF.md section 5.6. Milestone M3. Needs external data / network: stub until Phase 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class TerracdSource:
    """`Source` implementation for terracd (external)."""

    name = "terracd"
    kind = "external"

    def fetch(self, aoi: Any, cache: Path) -> Path:
        """Raw download, cached under `cache`."""
        raise NotImplementedError("scd.labels.sources.terracd.fetch: built at M3 (Phase 1); see HANDOFF.md section 5.6")

    def candidates(self, aoi: Any, pair: Any) -> Any:
        """GeoDataFrame: geometry (UTM), class_id, source, source_ref, t_event, confidence='auto', notes."""
        raise NotImplementedError(
            "scd.labels.sources.terracd.candidates: built at M3 (Phase 1); see HANDOFF.md section 5.6"
        )
