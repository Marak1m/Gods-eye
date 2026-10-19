"""Kruitwagen et al. 2021 global PV polygons as of 2018 with install-date estimates (Zenodo 5005868).

HANDOFF.md section 5.6. Milestone M3. Needs external data / network: stub until Phase 1.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class KruitwagenSource:
    """`Source` implementation for kruitwagen (solar)."""

    name = "kruitwagen"
    kind = "solar"

    def fetch(self, aoi: Any, cache: Path) -> Path:
        """Raw download, cached under `cache`."""
        raise NotImplementedError(
            "scd.labels.sources.kruitwagen.fetch: built at M3 (Phase 1); see HANDOFF.md section 5.6"
        )

    def candidates(self, aoi: Any, pair: Any) -> Any:
        """GeoDataFrame: geometry (UTM), class_id, source, source_ref, t_event, confidence='auto', notes."""
        raise NotImplementedError(
            "scd.labels.sources.kruitwagen.candidates: built at M3 (Phase 1); see HANDOFF.md section 5.6"
        )
