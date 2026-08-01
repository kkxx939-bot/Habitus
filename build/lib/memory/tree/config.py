"""长期记忆树物理枚举的运维容量配置。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryTreeConfig:
    """限制单个真实目录允许安全枚举的直接子项数量。"""

    max_children_per_directory: int = 10_000

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_children_per_directory, bool)
            or not isinstance(self.max_children_per_directory, int)
            or not 1 <= self.max_children_per_directory <= 1_000_000
        ):
            raise ValueError("max_children_per_directory must be between 1 and 1000000")


__all__ = ["MemoryTreeConfig"]
