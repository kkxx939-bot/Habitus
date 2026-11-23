"""SearchService 公共入参的规范化与边界校验。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from habitus.memory.model import MemoryKind
from habitus.memory.retrieval.model import MemorySearchServiceConfig
from habitus.memory.uri import MemoryURI, MemoryURINodeType


@dataclass(frozen=True, slots=True)
class MemorySearchRequest:
    """一次 find/search 调用规范化后的检索参数。"""

    query: str
    roots: tuple[MemoryURI, ...]
    limit: int
    score_threshold: float | None
    kinds: tuple[MemoryKind, ...]


class MemorySearchRequestNormalizer:
    """按统一检索配置规范查询文本、目标根目录、数量上限、阈值和召回过滤。"""

    def __init__(self, config: MemorySearchServiceConfig) -> None:
        self.config = config

    def normalize(
        self,
        query: str,
        *,
        target_uris: MemoryURI | str | tuple[MemoryURI | str, ...] | None,
        limit: int | None,
        score_threshold: float | None,
        kinds: tuple[MemoryKind, ...],
    ) -> MemorySearchRequest:
        """依次校验查询、根目录、上限、阈值和过滤条件，顺序决定多处非法时先报哪一处。"""

        normalized = self.query(query)
        roots = self.roots(target_uris)
        maximum = self.limit(limit)
        threshold = self.threshold(score_threshold)
        return MemorySearchRequest(normalized, roots, maximum, threshold, self.kinds(kinds))

    def query(self, value: str) -> str:
        if not value.strip():
            raise ValueError("memory search query must be non-empty text")
        normalized = value.strip()
        if len(normalized) > self.config.max_query_chars:
            raise ValueError("memory search query exceeds its configured character bound")
        return normalized

    def roots(
        self,
        value: MemoryURI | str | tuple[MemoryURI | str, ...] | None,
    ) -> tuple[MemoryURI, ...]:
        if value is None:
            raw: tuple[MemoryURI | str, ...] = (MemoryURI.root(),)
        elif isinstance(value, MemoryURI | str):
            raw = (value,)
        else:
            raw = value
        if not raw or len(raw) > self.config.max_target_roots:
            raise ValueError("memory search target root count is outside its configured bound")
        parsed = tuple(
            sorted({MemoryURI.parse(root) for root in raw}, key=lambda root: (len(root.segments), str(root)))
        )
        if any(root.node_type is not MemoryURINodeType.DIRECTORY for root in parsed):
            raise ValueError("memory search target URIs must identify directories")
        roots: list[MemoryURI] = []
        for root in parsed:
            if any(root.matches_prefix(parent) for parent in roots):
                continue
            roots.append(root)
        return tuple(roots)

    def limit(self, value: int | None) -> int:
        limit = self.config.default_limit if value is None else value
        if not 1 <= limit <= self.config.max_limit:
            raise ValueError("memory search limit is outside its configured bound")
        return limit

    @staticmethod
    def threshold(value: float | None) -> float | None:
        if value is None:
            return None
        if not math.isfinite(float(value)):
            raise ValueError("memory search score_threshold must be finite numeric data")
        return float(value)

    @staticmethod
    def kinds(value: tuple[MemoryKind, ...]) -> tuple[MemoryKind, ...]:
        normalized = tuple(MemoryKind(kind) for kind in value)
        if len(normalized) != len(set(normalized)):
            raise ValueError("memory search kinds must be unique")
        return normalized


__all__ = ["MemorySearchRequest", "MemorySearchRequestNormalizer"]
