"""情景树物理枚举与留代的显式边界。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SceneTreeConfig:
    max_children_per_directory: int = 10_000
    # 每天保留几代：重建某天时旧代不立即删，供引用它的下游回看（初值，待定档）。
    retained_generations: int = 3

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_children_per_directory, bool)
            or not isinstance(self.max_children_per_directory, int)
            or not 2 <= self.max_children_per_directory <= 1_000_000
        ):
            raise ValueError("max_children_per_directory must be between 2 and 1000000")
        if (
            isinstance(self.retained_generations, bool)
            or not isinstance(self.retained_generations, int)
            or not 2 <= self.retained_generations <= 365
        ):
            # 下界 2：翻指针后立刻删旧代会与不持锁的读侧（预测夜批）撞上读竞争。
            raise ValueError("retained_generations must be between 2 and 365")


__all__ = ["SceneTreeConfig"]
