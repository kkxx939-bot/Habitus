"""MemoryTree 到远程 VectorStore 之间的确定性索引模型。"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from habitus.memory.model import MemoryDirectory, MemoryLevel
from habitus.memory.uri import MemoryURI


@dataclass(frozen=True)
class MemoryIndexSource:
    """一个尚未生成向量的 L0、L1 或 L2 规范快照。"""

    uri: MemoryURI
    level: MemoryLevel
    directory: MemoryDirectory
    content: str
    index_kind: str
    revision: int

    def __post_init__(self) -> None:
        if not isinstance(self.uri, MemoryURI):
            raise TypeError("memory index source uri must be MemoryURI")
        level = MemoryLevel(self.level)
        if not isinstance(self.directory, MemoryDirectory):
            raise TypeError("memory index source directory must be MemoryDirectory")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("memory index source content must be non-empty")
        if (
            not isinstance(self.index_kind, str)
            or not self.index_kind
            or self.index_kind != self.index_kind.strip()
        ):
            raise ValueError("memory index source index_kind must be normalized text")
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision < 0:
            raise ValueError("memory index source revision must be non-negative")
        object.__setattr__(self, "level", level)

    @property
    def identity(self) -> str:
        return str(self.uri)

    @property
    def content_digest(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()

    @property
    def directory_key(self) -> str:
        return str(MemoryURI.from_directory(self.directory))

    @property
    def parent_key(self) -> str:
        if self.level is MemoryLevel.DETAIL:
            return self.directory_key
        parent = self.directory.parent()
        return "" if parent is None else str(MemoryURI.from_directory(parent))

    @property
    def scope_roots(self) -> tuple[str, ...]:
        return tuple(str(MemoryURI.from_directory(directory)) for directory in reversed(self.directory.lineage()))


@dataclass(frozen=True)
class MemoryVectorConsistencyReport:
    """MemoryTree 真相源与远程向量记录的完整差异。"""

    expected_count: int
    indexed_count: int
    missing_identities: tuple[str, ...] = ()
    stale_identities: tuple[str, ...] = ()
    orphan_identities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("expected_count", "indexed_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def ok(self) -> bool:
        return not (self.missing_identities or self.stale_identities or self.orphan_identities)


__all__ = ["MemoryIndexSource", "MemoryVectorConsistencyReport"]
