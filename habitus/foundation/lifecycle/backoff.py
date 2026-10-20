"""可重试失败的退避：base × 2^(attempt − 1)，封顶 cap。重试账与 memory 的扫描失败集合共用这一份。"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class BackoffPolicy:
    """``max_doublings`` 只防指数溢出：base × 2^max_doublings 不小于 cap 时它不改变任何结果。"""

    base_seconds: float
    cap_seconds: float
    max_doublings: int = 20

    def __post_init__(self) -> None:
        for label, value in (("base_seconds", self.base_seconds), ("cap_seconds", self.cap_seconds)):
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be a positive finite number")
        if self.base_seconds > self.cap_seconds:
            raise ValueError("base_seconds must not exceed cap_seconds")
        if isinstance(self.max_doublings, bool) or not isinstance(self.max_doublings, int) or self.max_doublings < 0:
            raise ValueError("max_doublings must be a non-negative integer")

    def delay(self, attempt: int) -> float:
        """第 ``attempt`` 次失败之后等多少秒再试（``attempt`` 从 1 起）。"""

        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise ValueError("attempt must be a positive integer")
        return min(self.cap_seconds, self.base_seconds * (2 ** min(self.max_doublings, attempt - 1)))


__all__ = ["BackoffPolicy"]
