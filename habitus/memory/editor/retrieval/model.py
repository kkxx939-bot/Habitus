"""提取记忆时查旧卡片用的严格配置和结果模型。"""

from __future__ import annotations

from dataclasses import dataclass

from habitus.memory.snapshot import MemorySnapshotBatch
from habitus.memory.uri import MemoryURI


class MemoryRetrievalError(RuntimeError):
    """旧记忆读取失败，不能安全地继续生成编辑操作。"""


@dataclass(frozen=True)
class MemoryRetrievalConfig:
    """限制按真实工具名直读的工具卡片数，以及名字清单的大小。"""

    max_tool_uris: int = 32
    max_listing_entries: int = 2_000

    def __post_init__(self) -> None:
        for name, value in {
            "max_tool_uris": self.max_tool_uris,
            "max_listing_entries": self.max_listing_entries,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class MemoryNameEntry:
    """名字清单里的一张旧卡片：地址与模板概览里的一小段说明（读不到概览时为空）。"""

    uri: MemoryURI
    description: str = ""

    def line(self) -> str:
        return f"- {self.uri}" + (f" — {self.description}" if self.description else "")


@dataclass(frozen=True)
class MemoryRelatedContext:
    """绑定一个 ConversationSegment 的固定旧卡片快照，以及偏好主题与实体名字清单。"""

    conversation_id: str
    segment_id: str
    source_segment_digest: str
    listing: tuple[MemoryNameEntry, ...]
    snapshots: MemorySnapshotBatch
    omitted_listing_entries: int = 0

    def __post_init__(self) -> None:
        for name, value in {
            "conversation_id": self.conversation_id,
            "segment_id": self.segment_id,
        }.items():
            if not value or value != value.strip():
                raise ValueError(f"{name} must be a non-empty string without surrounding whitespace")
        if not _is_sha256(self.source_segment_digest):
            raise ValueError("source_segment_digest must be a lowercase SHA-256 digest")
        uris = tuple(entry.uri for entry in self.listing)
        if len(set(uris)) != len(uris):
            raise ValueError("memory name listing must be unique")
        if self.omitted_listing_entries < 0:
            raise ValueError("omitted listing entries must be non-negative")


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and value == value.lower() and all(character in "0123456789abcdef" for character in value)


__all__ = [
    "MemoryNameEntry",
    "MemoryRelatedContext",
    "MemoryRetrievalConfig",
    "MemoryRetrievalError",
]
