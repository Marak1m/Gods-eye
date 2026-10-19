"""fix_t1 rule (HANDOFF.md 5.8): background at T1 where positive at T2 becomes ignore at T1.

OSM- and product-derived single-date labels lag reality, so a pixel that is background at T1 but
a positive class at T2 is unreliable at T1 and must not teach "background".
"""

from __future__ import annotations

import numpy as np

from scd.constants import IGNORE


def fix_t1(sem_t1: np.ndarray, sem_t2: np.ndarray, ignore: int = IGNORE) -> np.ndarray:
    """Return a copy of `sem_t1` with (T1 == 0) & (T2 in 1..4) set to `ignore`."""
    if sem_t1.shape != sem_t2.shape:
        raise ValueError(f"shape mismatch {sem_t1.shape} vs {sem_t2.shape}")
    out = sem_t1.copy()
    positive_t2 = (sem_t2 > 0) & (sem_t2 != ignore)
    out[(sem_t1 == 0) & positive_t2] = ignore
    return out
