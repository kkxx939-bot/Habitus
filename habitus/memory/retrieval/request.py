"""SearchService 公共入参的规范化与边界校验。"""

from __future__ import annotations

from dataclasses import dataclass

from habitus.memory.model import MemoryKind
from habitus.memory.retrieval.model import MemorySearchServiceConfig


@dataclass(frozen=True)
class MemorySearchRequest:
    """一次 search 调用规范化后的参数：``kinds`` 限定清单给哪几类，``limit`` 限定最终交出的卡片数。"""

    query: str
    limit: int
    kinds: tuple[MemoryKind, ...]


class MemorySearchRequestNormalizer:
    def __init__(self, config: MemorySearchServiceConfig) -> None:
        self.config = config

    def normalize(self, query: str, *, limit: int | None, kinds: tuple[MemoryKind, ...]) -> MemorySearchRequest:
        """依次校验查询、上限和类别，顺序决定多处非法时先报哪一处。"""

        if not query.strip():
            raise ValueError("memory search query must be non-empty text")
        normalized = query.strip()
        if len(normalized) > self.config.max_query_chars:
            raise ValueError("memory search query exceeds its configured character bound")
        maximum = self.config.default_limit if limit is None else limit
        if not 1 <= maximum <= self.config.max_limit:
            raise ValueError("memory search limit is outside its configured bound")
        resolved = tuple(MemoryKind(kind) for kind in kinds)
        if len(resolved) != len(set(resolved)):
            raise ValueError("memory search kinds must be unique")
        return MemorySearchRequest(normalized, maximum, resolved)


__all__ = ["MemorySearchRequest", "MemorySearchRequestNormalizer"]
