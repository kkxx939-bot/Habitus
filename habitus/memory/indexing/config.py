"""远程记忆向量索引的容量、批处理和冲突重试配置。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryVectorIndexConfig:
    """限制全量重建、增量发布和一次远程搜索的资源规模。"""

    max_records: int = 10_000
    max_direct_entries: int = 1_000
    max_directories: int = 2_000
    max_search_hits: int = 10_000
    max_record_chars: int = 16_000
    embedding_batch_size: int = 128
    stale_retries: int = 2

    def __post_init__(self) -> None:
        for name, value, maximum in (
            ("max_records", self.max_records, 1_000_000),
            ("max_direct_entries", self.max_direct_entries, 10_000),
            ("max_directories", self.max_directories, 100_000),
            ("max_search_hits", self.max_search_hits, 100_000),
            ("max_record_chars", self.max_record_chars, 1_000_000),
            ("embedding_batch_size", self.embedding_batch_size, 10_000),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be between one and {maximum}")
        if (
            isinstance(self.stale_retries, bool)
            or not isinstance(self.stale_retries, int)
            or not 0 <= self.stale_retries <= 10
        ):
            raise ValueError("stale_retries must be between zero and ten")


__all__ = ["MemoryVectorIndexConfig"]
