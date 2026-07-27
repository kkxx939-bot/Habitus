"""记忆领域对远程向量索引的公共搜索契约。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from memory.intention import MemoryIntentionRecallScope
from memory.model import MemoryDirectory, MemoryKind, MemoryLevel
from memory.uri import MemoryURI, MemoryURINodeType
from ModelClient import EmbeddingVector


class MemoryVectorIndexError(RuntimeError):
    """记忆向量索引无法在完整性或资源边界内完成操作。"""


@dataclass(frozen=True)
class MemoryVectorMatch:
    """远程索引返回的一个目录语义节点或 L2 文档候选。"""

    uri: MemoryURI
    level: MemoryLevel
    directory: MemoryDirectory
    content: str
    score: float

    def __post_init__(self) -> None:
        uri = MemoryURI.parse(self.uri)
        level = MemoryLevel(self.level)
        if level is MemoryLevel.DETAIL:
            if uri.node_type is not MemoryURINodeType.DOCUMENT:
                raise ValueError("L2 vector match must identify a memory document")
            if uri.containing_directory != self.directory:
                raise ValueError("L2 vector match directory does not match its URI")
        elif uri.node_type is not MemoryURINodeType.LAYER:
            raise ValueError("L0/L1 vector match must identify a semantic layer")
        else:
            directory, uri_level = uri.to_layer()
            if directory != self.directory or uri_level is not level:
                raise ValueError("semantic vector match does not match its URI")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("memory vector match content must be non-empty")
        if isinstance(self.score, bool) or not isinstance(self.score, int | float):
            raise TypeError("memory vector match score must be numeric")
        score = float(self.score)
        if not math.isfinite(score) or not -1.0 <= score <= 1.0:
            raise ValueError("memory vector match score must be a finite cosine value")
        object.__setattr__(self, "uri", uri)
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "score", score)


class MemoryVectorIndex(Protocol):
    """使用已经生成一次的 query vector 搜索受限记忆节点。"""

    async def search(
        self,
        query_vector: EmbeddingVector,
        *,
        roots: tuple[MemoryURI, ...],
        levels: tuple[MemoryLevel, ...],
        kinds: tuple[MemoryKind, ...],
        intention_scope: MemoryIntentionRecallScope,
        limit: int,
    ) -> Sequence[MemoryVectorMatch]: ...

    async def search_children(
        self,
        query_vector: EmbeddingVector,
        *,
        parent: MemoryURI,
        kinds: tuple[MemoryKind, ...],
        intention_scope: MemoryIntentionRecallScope,
        limit: int,
    ) -> Sequence[MemoryVectorMatch]: ...


__all__ = ["MemoryVectorIndex", "MemoryVectorIndexError", "MemoryVectorMatch"]
