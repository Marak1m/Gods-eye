"""Flicker-rate diagnostic on NOCHANGE_HARD pairs.

HANDOFF.md section 11. Milestone M8.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from scd.config import Config


def flicker_rate(mask: Any, valid: Any) -> float:
    """Fraction of valid pixels flagged as change."""
    raise NotImplementedError("scd.eval.flicker.flicker_rate: built at M8; see HANDOFF.md section 11")


def run(cfg: Config, strata: list[str], n_pairs: int, run_name: str | None) -> Any:
    """Table per stratum for PCCD and the model."""
    raise NotImplementedError("scd.eval.flicker.run: built at M8; see HANDOFF.md section 11")
