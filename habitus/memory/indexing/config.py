"""远程记忆向量索引的容量、批处理和冲突重试配置。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryVectorIndexConfig:
    """限制全量重建、增量发布和一次远程搜索的资源规模，以及读取记忆树时 memory-root 事务锁的等待参数。"""

    max_records: int = 10_000
    max_direct_entries: int = 1_000
    max_directories: int = 2_000
    max_search_hits: int = 10_000
    max_record_chars: int = 16_000
    embedding_batch_size: int = 128
    stale_retries: int = 2
    lock_ttl_seconds: int = 30
    lock_wait_timeout_seconds: float = 30.0
    lock_retry_delay_seconds: float = 0.02

    def __post_init__(self) -> None:
        for name, value, maximum in (
            ("max_records", self.max_records, 1_000_000),
            ("max_direct_entries", self.max_direct_entries, 10_000),
            ("max_directories", self.max_directories, 100_000),
            ("max_search_hits", self.max_search_hits, 100_000),
            ("max_record_chars", self.max_record_chars, 1_000_000),
            ("embedding_batch_size", self.embedding_batch_size, 10_000),
        ):
            if not 1 <= value <= maximum:
                raise ValueError(f"{name} must be between one and {maximum}")
        if not 0 <= self.stale_retries <= 10:
            raise ValueError("stale_retries must be between zero and ten")
        if not 1 <= self.lock_ttl_seconds <= 3_600:
            raise ValueError("lock_ttl_seconds must be between 1 and 3600")
        for name, float_value, float_maximum in (
            ("lock_wait_timeout_seconds", self.lock_wait_timeout_seconds, 60.0),
            ("lock_retry_delay_seconds", self.lock_retry_delay_seconds, 1.0),
        ):
            if not 0 < float(float_value) <= float_maximum:
                raise ValueError(f"{name} must be greater than zero and at most {float_maximum:g}")


__all__ = ["MemoryVectorIndexConfig"]
