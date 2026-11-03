"""供应商无关的稠密向量值对象：模型调用层产出它，向量库存取它。"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingVector:
    """经过有限值、非零范数和 L2 归一化的稠密向量。"""

    values: tuple[float, ...]

    def __post_init__(self) -> None:
        if isinstance(self.values, list):
            values = tuple(self.values)
        elif isinstance(self.values, tuple):
            values = self.values
        else:
            raise TypeError("embedding vector values must be a tuple or list")
        if not values:
            raise ValueError("embedding vector must not be empty")
        normalized: list[float] = []
        for value in values:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError("embedding vector values must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("embedding vector values must be finite")
            normalized.append(number)
        norm = math.sqrt(sum(value * value for value in normalized))
        if norm == 0:
            raise ValueError("embedding vector must have a non-zero norm")
        object.__setattr__(self, "values", tuple(value / norm for value in normalized))

    @property
    def dimension(self) -> int:
        return len(self.values)


__all__ = ["EmbeddingVector"]
