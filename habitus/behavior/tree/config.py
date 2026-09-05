"""行为语义树物理枚举的显式容量边界。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BehaviorTreeConfig:
    max_children_per_directory: int = 10_000

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_children_per_directory, bool)
            or not isinstance(self.max_children_per_directory, int)
            or not 2 <= self.max_children_per_directory <= 1_000_000
        ):
            raise ValueError("max_children_per_directory must be between 2 and 1000000")


__all__ = ["BehaviorTreeConfig"]
