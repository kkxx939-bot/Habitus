"""异步 Conversation 写入与长期记忆提交之间的一致性观察语义。"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from enum import Enum

from foundation.observability import NullObserver, ObservationEvent, ObservationStatus, Observer
from memory.workflow import (
    MemoryChangeReceipt,
    MemoryChangeReceiptState,
    MemoryChangeReceiptStore,
    MemoryChangeSource,
    MemoryJob,
    MemoryJobAbandonment,
    MemoryJobStatus,
    MemoryJobStore,
)


class MemoryConsistencyState(str, Enum):
    """调用方能够据此选择继续等待、报错或接受人工处置。"""

    PENDING = "pending"
    COMMITTED = "committed"
    FAILED = "failed"
    ABANDONED = "abandoned"
    UNKNOWN = "unknown"


class MemoryConsistencyTimeoutError(TimeoutError):
    """等待记忆达到终态超过调用方给出的时间预算。"""

    def __init__(self, snapshot: MemoryConsistencySnapshot) -> None:
        self.snapshot = snapshot
        super().__init__(
            f"memory job {snapshot.requested_job.memory_sequence} did not settle before timeout"
        )


@dataclass(frozen=True)
class MemoryConsistencySnapshot:
    """一个写入句柄在队列、回执和人工处置记录中的一致快照。"""

    state: MemoryConsistencyState
    requested_job: MemoryJob
    current_job: MemoryJob | None
    receipt: MemoryChangeReceipt | None
    abandonment: MemoryJobAbandonment | None

    @property
    def terminal(self) -> bool:
        return self.state in {
            MemoryConsistencyState.COMMITTED,
            MemoryConsistencyState.FAILED,
            MemoryConsistencyState.ABANDONED,
            MemoryConsistencyState.UNKNOWN,
        }


class MemoryConsistencyService:
    """只组合耐久事实，不参与 Job 执行或记忆领域写入。"""

    def __init__(
        self,
        jobs: MemoryJobStore,
        receipts: MemoryChangeReceiptStore,
        *,
        observer: Observer | None = None,
    ) -> None:
        if not isinstance(jobs, MemoryJobStore):
            raise TypeError("jobs must be MemoryJobStore")
        if not isinstance(receipts, MemoryChangeReceiptStore):
            raise TypeError("receipts must be MemoryChangeReceiptStore")
        self.jobs = jobs
        self.receipts = receipts
        self.observer = observer or NullObserver()

    def inspect(self, requested_job: MemoryJob) -> MemoryConsistencySnapshot:
        if not isinstance(requested_job, MemoryJob):
            raise TypeError("requested_job must be MemoryJob")
        from memory.conversation import ConversationAddress

        address = ConversationAddress(requested_job.conversation_id, requested_job.started_on)
        current = self.jobs.try_read_source(
            address,
            requested_job.segment_id,
            requested_job.source_segment_digest,
        )
        source = MemoryChangeSource.from_job(requested_job)
        receipt = self.receipts.try_read(source)
        abandonment = self.jobs.try_read_abandonment(requested_job)
        receipt_committed = (
            receipt is not None and receipt.state is MemoryChangeReceiptState.COMMITTED
        )
        if current is not None:
            if current.status is MemoryJobStatus.FAILED:
                state = MemoryConsistencyState.FAILED
            elif current.status is MemoryJobStatus.COMMITTED:
                # Receipt 只证明 L2 事务已提交；Job 终态还要求 Summary、L0/L1 和
                # 远程向量投影全部完成。两者必须同时成立才能对外宣告最终一致。
                state = (
                    MemoryConsistencyState.COMMITTED
                    if receipt_committed
                    else MemoryConsistencyState.UNKNOWN
                )
            else:
                state = MemoryConsistencyState.PENDING
        elif abandonment is not None:
            state = MemoryConsistencyState.ABANDONED
        elif receipt_committed:
            # 生命周期可以先清理已完成 Job，而更长期保留审计 Receipt。
            state = MemoryConsistencyState.COMMITTED
        else:
            state = MemoryConsistencyState.UNKNOWN
        return MemoryConsistencySnapshot(
            state=state,
            requested_job=requested_job,
            current_job=current,
            receipt=receipt,
            abandonment=abandonment,
        )

    async def wait(
        self,
        requested_job: MemoryJob,
        *,
        timeout_seconds: float,
        poll_interval_seconds: float = 0.05,
    ) -> MemoryConsistencySnapshot:
        for name, value in (
            ("timeout_seconds", timeout_seconds),
            ("poll_interval_seconds", poll_interval_seconds),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float) or float(value) <= 0:
                raise ValueError(f"{name} must be a positive number")
        started = time.monotonic()
        deadline = started + float(timeout_seconds)
        while True:
            snapshot = await asyncio.to_thread(self.inspect, requested_job)
            if snapshot.terminal:
                self.observer.record(
                    ObservationEvent(
                        category="consistency",
                        operation="wait",
                        status=(
                            ObservationStatus.SUCCESS
                            if snapshot.state is MemoryConsistencyState.COMMITTED
                            else ObservationStatus.DEGRADED
                        ),
                        duration_seconds=max(0.0, time.monotonic() - started),
                        attributes={
                            "state": snapshot.state.value,
                            "memory_sequence": requested_job.memory_sequence,
                        },
                    )
                )
                return snapshot
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.observer.record(
                    ObservationEvent(
                        category="consistency",
                        operation="wait",
                        status=ObservationStatus.FAILURE,
                        duration_seconds=max(0.0, time.monotonic() - started),
                        attributes={
                            "state": snapshot.state.value,
                            "memory_sequence": requested_job.memory_sequence,
                            "error_type": "MemoryConsistencyTimeoutError",
                        },
                    )
                )
                raise MemoryConsistencyTimeoutError(snapshot)
            await asyncio.sleep(min(float(poll_interval_seconds), remaining))


__all__ = [
    "MemoryConsistencyService",
    "MemoryConsistencySnapshot",
    "MemoryConsistencyState",
    "MemoryConsistencyTimeoutError",
]
