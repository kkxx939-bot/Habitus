"""卡片写完之后的完成阶段：清冷热状态、刷目录概览、标完成。"""

from __future__ import annotations

import time
from asyncio import to_thread
from typing import Protocol

from habitus.foundation.observability import (
    NullObserver,
    ObservationEvent,
    ObservationStatus,
    Observer,
    observe_operation,
)
from habitus.memory.semantic import MemorySemanticRefresher, MemorySemanticRefreshResult
from habitus.memory.uri import MemoryURI
from habitus.memory.workflow.jobs import (
    MemoryJob,
    MemoryJobError,
    MemoryJobLease,
    MemoryJobStore,
)


class MemoryRecallStateCleaner(Protocol):
    def forget(self, uris: tuple[MemoryURI, ...]) -> int: ...


class MemoryCommittedJobFinalizer:
    """按任务文件里记下的改动地址补齐冷热状态、语义层和 Job 终态；可以幂等重放。"""

    def __init__(
        self,
        store: MemoryJobStore,
        semantic_refresher: MemorySemanticRefresher,
        recall_state_cleaner: MemoryRecallStateCleaner,
        *,
        observer: Observer | None = None,
    ) -> None:
        if semantic_refresher.tree.root != store.memory_root:
            raise ValueError("semantic refresher and MemoryJobStore must use one memory root")
        self.store = store
        self.semantic_refresher = semantic_refresher
        self.recall_state_cleaner = recall_state_cleaner
        self.observer = observer or NullObserver()

    async def finalize(self, lease: MemoryJobLease) -> MemoryJob:
        """卡片已写完的任务：清掉删除卡片的冷热状态、刷改动所在目录的概览，最后标 COMMITTED。"""

        job = self.store.assert_current(lease)
        if not job.cards_written:
            raise MemoryJobError("job finalization requires a job whose cards are written")
        deleted = tuple(MemoryURI.parse(uri) for uri in job.deleted_uris)
        await self._forget_retired_recall_state(deleted)
        with observe_operation(self.observer, "workflow", "semantic_refresh"):
            await to_thread(self._refresh_semantic_layers, job)
        self.store.assert_current(lease)
        with observe_operation(self.observer, "workflow", "job_complete"):
            return self.store.complete(lease)

    async def finalize_skipped(self, lease: MemoryJobLease, reason: str) -> MemoryJob:
        """这一段跳过了（裁定 108）：长期记忆不动；摘要要是已经做出来了，检索直接从摘要文件读到它。任务记成 SKIPPED，
        原文照同一条规则到期释放（裁定 111）。"""

        self.store.assert_current(lease)
        with observe_operation(self.observer, "workflow", "job_skip"):
            return self.store.skip(lease, reason)

    async def _forget_retired_recall_state(
        self,
        deleted_uris: tuple[MemoryURI, ...],
    ) -> None:
        if not deleted_uris:
            return
        started = time.monotonic()
        try:
            deleted = await to_thread(self.recall_state_cleaner.forget, deleted_uris)
        except Exception as exc:
            self.observer.record(
                ObservationEvent(
                    category="retrieval",
                    operation="recall_lifecycle_cleanup",
                    status=ObservationStatus.DEGRADED,
                    duration_seconds=max(0.0, time.monotonic() - started),
                    attributes={
                        "error_type": type(exc).__name__,
                        "requested": len(deleted_uris),
                    },
                )
            )
            return
        self.observer.record(
            ObservationEvent(
                category="retrieval",
                operation="recall_lifecycle_cleanup",
                status=ObservationStatus.SUCCESS,
                duration_seconds=max(0.0, time.monotonic() - started),
                attributes={"requested": len(deleted_uris), "deleted": deleted},
            )
        )

    def _refresh_semantic_layers(self, job: MemoryJob) -> tuple[MemorySemanticRefreshResult, ...]:
        addresses = tuple(MemoryURI.parse(uri).to_address() for uri in (*job.changed_uris, *job.deleted_uris))
        return self.semantic_refresher.refresh_for_many(addresses)


__all__ = ["MemoryCommittedJobFinalizer", "MemoryRecallStateCleaner"]
