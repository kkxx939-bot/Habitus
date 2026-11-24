"""事件与空白的视图：物理枚举的容量边界与只增写入的锁租约。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BehaviorTreeConfig:
    max_children_per_directory: int = 10_000

    def __post_init__(self) -> None:
        if not 2 <= self.max_children_per_directory <= 1_000_000:
            raise ValueError("max_children_per_directory must be between 2 and 1000000")


@dataclass(frozen=True)
class BehaviorWriteConfig:
    lock_ttl_seconds: int = 30

    def __post_init__(self) -> None:
        if not 1 <= self.lock_ttl_seconds <= 3_600:
            raise ValueError("lock_ttl_seconds must be between 1 and 3600")


__all__ = ["BehaviorTreeConfig", "BehaviorWriteConfig"]
