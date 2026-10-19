"""Post-classification change baseline from the v3.1 specialists.

HANDOFF.md section 8. Milestone M8. Needs third_party/v3_roads (stub).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scd.config import Config


def run_pccd(t1_path: Path, t2_path: Path, cfg: Config) -> Any:
    """XOR of per-date specialist masks on the canonical grid."""
    raise NotImplementedError("scd.baseline.pccd.run_pccd: built at M8; see HANDOFF.md section 8")
