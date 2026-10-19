"""Per-class morphology, min sizes, water rule.

HANDOFF.md section 8 step 5. Milestone M7.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scd.config import Config


def apply(mask: Any, cfg: Config, t1: Any, t2: Any, valid: Any, res_m: float) -> Any:
    """Post-processed uint8 mask."""
    raise NotImplementedError("scd.infer.postprocess.apply: built at M7; see HANDOFF.md section 8 step 5")
