"""Sliding-window inference with D4 (and optional temporal) TTA and cosine blending.

HANDOFF.md section 8 step 3. Milestone M7.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


def predict(
    net: Any, t1: Any, t2: Any, window: int, stride: int, geometric: int = 8, temporal: bool = False
) -> tuple[Any, Any]:
    """Averaged change and class logits."""
    raise NotImplementedError("scd.infer.tta.predict: built at M7; see HANDOFF.md section 8 step 3")
