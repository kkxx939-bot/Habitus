"""记忆目录语义层的显式边界配置。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemorySemanticConfig:
    """限制单个目录概览的条数与字数，以及全树重建一次最多走多少个目录。"""

    max_direct_entries: int = 256
    max_rebuild_directories: int = 10_000
    max_overview_chars: int = 64_000
    max_abstract_chars: int = 800
    lock_ttl_seconds: int = 120

    def __post_init__(self) -> None:
        positive = {
            "max_direct_entries": self.max_direct_entries,
            "max_rebuild_directories": self.max_rebuild_directories,
            "max_overview_chars": self.max_overview_chars,
            "max_abstract_chars": self.max_abstract_chars,
            "lock_ttl_seconds": self.lock_ttl_seconds,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_abstract_chars > self.max_overview_chars:
            raise ValueError("max_abstract_chars cannot exceed max_overview_chars")


__all__ = ["MemorySemanticConfig"]
