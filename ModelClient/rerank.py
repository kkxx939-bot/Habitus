"""供应商无关的文本重排调用协议。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class Reranker(Protocol):
    """按照输入原顺序返回相关性分数的异步重排接口。"""

    provider_name: str
    model: str
    is_remote: bool

    async def rerank(self, query: str, documents: Sequence[str]) -> tuple[float, ...]: ...


__all__ = ["Reranker"]
