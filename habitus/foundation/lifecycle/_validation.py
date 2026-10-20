"""本包共用的参数校验。"""

from __future__ import annotations

import math


def require_half_life(half_life: float) -> None:
    """半衰（次数）必须是正的有限数。"""

    if isinstance(half_life, bool) or not isinstance(half_life, int | float) or not math.isfinite(half_life):
        raise ValueError("half_life must be a finite number")
    if half_life <= 0:
        raise ValueError("half_life must be positive")
