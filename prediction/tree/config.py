"""预测树物理枚举的显式容量边界。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionTreeConfig:
    max_children_per_directory: int = 10_000
    max_temporary_files_per_directory: int = 64

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_children_per_directory, bool)
            or not isinstance(self.max_children_per_directory, int)
            or not 3 <= self.max_children_per_directory <= 1_000_000
        ):
            raise ValueError("max_children_per_directory must be between 3 and 1000000")
        if (
            isinstance(self.max_temporary_files_per_directory, bool)
            or not isinstance(self.max_temporary_files_per_directory, int)
            or not 2 <= self.max_temporary_files_per_directory <= 10_000
        ):
            raise ValueError("max_temporary_files_per_directory must be between 2 and 10000")


__all__ = ["PredictionTreeConfig"]
