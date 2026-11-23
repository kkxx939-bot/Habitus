"""最终交出的卡片记一次"被用到"，整个记账限在留出的时间里。

拿不到锁或到点了就跳过这一次记账并写观测，卡片照常交出：插件到点就断，拿到空的比少记一次热度更糟。
失去版本围栏的卡片（记账时发现内容已经变了）不交出。
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Protocol

from habitus.memory.document import MemoryDocument
from habitus.memory.recall import MemoryRecallLifecycle, MemoryRecallTarget
from habitus.memory.retrieval.model import MemoryMatchedMemory
from habitus.memory.uri import MemoryURI


class MemoryContextUseReceipt(Protocol):
    @property
    def documents(self) -> tuple[MemoryDocument, ...]: ...

    @property
    def rejected_uris(self) -> tuple[MemoryURI, ...]: ...

    @property
    def skipped_uris(self) -> tuple[MemoryURI, ...]: ...


class MemoryColdProbeExpander(Protocol):
    """展开压缩过的卡片的活动基线，并在交出前完成 L2 恢复与使用确认（``MemoryLifecycleManager`` 满足它）。"""

    def expand_for_probe(self, document: MemoryDocument) -> MemoryDocument: ...

    async def record_context_use(
        self,
        targets: tuple[MemoryRecallTarget, ...],
        *,
        used_at: datetime | None = None,
        lock_wait_seconds: float = 5.0,
    ) -> MemoryContextUseReceipt: ...


@dataclass(frozen=True)
class UseAccounting:
    """记账之后要交出的卡片，以及这一次没记上账的张数（拿不到锁、到点或出错）。"""

    memories: tuple[MemoryMatchedMemory, ...]
    skipped: int
    error_type: str | None = None


async def account_use(
    memories: tuple[MemoryMatchedMemory, ...],
    *,
    targets: dict[MemoryURI, MemoryRecallTarget],
    used_at: datetime,
    seconds: float,
    recall_lifecycle: MemoryRecallLifecycle,
    expander: MemoryColdProbeExpander | None,
) -> UseAccounting:
    """在 ``seconds`` 之内给交出的卡片记账；超时或出错时原样交出并报告没记上的张数。"""

    owned = tuple(targets[memory.uri] for memory in memories if memory.uri in targets)
    if not owned:
        return UseAccounting(memories, 0)
    try:
        if expander is None:
            await asyncio.wait_for(
                asyncio.to_thread(recall_lifecycle.record_use, owned, used_at=used_at), timeout=seconds
            )
            return UseAccounting(memories, 0)
        receipt = await asyncio.wait_for(
            expander.record_context_use(owned, used_at=used_at, lock_wait_seconds=max(0.0, seconds * 0.8)),
            timeout=seconds,
        )
    except Exception as exc:
        return UseAccounting(memories, len(owned), type(exc).__name__)
    rejected = set(receipt.rejected_uris)
    confirmed = {MemoryURI.from_address(document.address): document for document in receipt.documents}
    kept = tuple(
        replace(memory, document=confirmed.get(memory.uri, memory.document))
        for memory in memories
        if memory.uri not in rejected
    )
    return UseAccounting(kept, len(receipt.skipped_uris), "TimeoutError" if receipt.skipped_uris else None)


__all__ = ["MemoryColdProbeExpander", "MemoryContextUseReceipt", "UseAccounting", "account_use"]
