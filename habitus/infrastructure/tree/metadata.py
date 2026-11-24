"""L2 文件的系统字段：版本与建立、更新时间。不由模型或内容 Schema 给。

可改的卡片每次写入推进一个版本（``next_revision``）；只增的事件与空白永远是 ``initial``（版本 1，建立时间由写入方给）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from habitus.foundation.timestamps import require_aware_utc


def _utc(value: datetime, field_name: str) -> datetime:
    return require_aware_utc(value, f"document {field_name} must include a timezone")


@dataclass(frozen=True)
class DocumentMetadata:
    revision: int
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        if self.revision <= 0:
            raise ValueError("document revision must be a positive integer")
        created_at = _utc(self.created_at, "created_at")
        updated_at = _utc(self.updated_at, "updated_at")
        if updated_at < created_at:
            raise ValueError("document updated_at cannot precede created_at")
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)

    @classmethod
    def initial(cls, timestamp: datetime) -> DocumentMetadata:
        normalized = _utc(timestamp, "timestamp")
        return cls(revision=1, created_at=normalized, updated_at=normalized)

    def next_revision(self, timestamp: datetime) -> DocumentMetadata:
        """下一版本；旧版本是否仍是盘上那一份由写入方（记忆的提交写入器）用 CAS 保证。"""

        normalized = _utc(timestamp, "timestamp")
        if normalized < self.updated_at:
            raise ValueError("next document revision timestamp cannot move backwards")
        return DocumentMetadata(revision=self.revision + 1, created_at=self.created_at, updated_at=normalized)


__all__ = ["DocumentMetadata"]
