"""LifecycleWorker 的全局 lease、耐久轮转、局部失败与状态机测试。"""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path

import pytest

from Config import ConversationLifecycleConfig
from infrastructure.store.contracts import PathLock
from infrastructure.store.locks import ProcessLocalLockStore
from memory.conversation import (
    ConversationAddress,
    ConversationMessageJournal,
    ConversationRangeSummaryGenerator,
    ConversationRangeSummaryStore,
    ConversationSummaryCompactor,
    ConversationSummaryStore,
    PersistentConversationSummaryVectorIndex,
)
from memory.editor import MemoryTransactionJournal
from memory.workflow import (
    ConversationLifecycleManager,
    MemoryChangeReceiptStore,
    MemoryJobStore,
)
from pre.conversation import ConversationBatch
from Runtime import LifecycleWorker, LifecycleWorkerState, LifecycleWorkerStateError
from tests.helpers import closed_turn, codec


def manager(tmp_path: Path) -> ConversationLifecycleManager:
    path_lock = PathLock(ProcessLocalLockStore())
    journal = ConversationMessageJournal(tmp_path / "conversation", path_lock)
    segment_store = ConversationSummaryStore(journal.layout)
    range_store = ConversationRangeSummaryStore(journal.layout)
    compactor = ConversationSummaryCompactor(
        journal,
        segment_store,
        range_store,
        object.__new__(ConversationRangeSummaryGenerator),
    )
    summary_index = object.__new__(PersistentConversationSummaryVectorIndex)
    summary_index.compactor = compactor

    async def synchronize(_address, *, checkpoint=None):
        return checkpoint

    summary_index.synchronize = synchronize
    document_codec = codec()
    jobs = MemoryJobStore(
        tmp_path / "workflow",
        path_lock,
        memory_root=tmp_path / "memory",
    )
    return ConversationLifecycleManager(
        compactor,
        journal,
        segment_store,
        range_store,
        summary_index,
        jobs,
        MemoryChangeReceiptStore(tmp_path / "workflow", document_codec),
        MemoryTransactionJournal(tmp_path / "workflow" / "transactions", document_codec),
    )


def add_conversation(current: ConversationLifecycleManager, address: ConversationAddress) -> None:
    current.journal.append(address, ConversationBatch(address.conversation_id, closed_turn()))


def config(*, maximum: int = 2) -> ConversationLifecycleConfig:
    return ConversationLifecycleConfig(
        maintenance_interval_seconds=60,
        max_conversations_per_cycle=maximum,
        lease_ttl_seconds=30,
        heartbeat_interval_seconds=10,
        shutdown_timeout_seconds=1,
    )


def test_cycle_rotates_after_durable_cursor_and_one_failure_does_not_skip_other_conversations(
    tmp_path: Path,
) -> None:
    current = manager(tmp_path)
    addresses = (
        ConversationAddress("a", date(2026, 7, 1)),
        ConversationAddress("b", date(2026, 7, 1)),
        ConversationAddress("c", date(2026, 7, 1)),
    )
    for address in addresses:
        add_conversation(current, address)
    calls: list[ConversationAddress] = []

    async def maintain_once(address, *, now=None):
        calls.append(address)
        if address == addresses[0]:
            raise RuntimeError("conversation a failed")
        return object()

    current.maintain_once = maintain_once  # type: ignore[method-assign]
    worker = LifecycleWorker(current, config(maximum=2), worker_id="lifecycle-test")

    first = asyncio.run(worker.run_once())
    second = asyncio.run(worker.run_once())

    assert first.selected_addresses == addresses[:2]
    assert first.maintained_addresses == (addresses[1],)
    assert first.failures[0].address == addresses[0]
    assert first.failures[0].error_type == "RuntimeError"
    assert second.selected_addresses == (addresses[2], addresses[0])
    assert calls == [addresses[0], addresses[1], addresses[2], addresses[0]]
    assert worker._cursor_store.read() == (addresses[0].started_on, addresses[0].conversation_id)


def test_competing_global_lease_skips_cycle_without_moving_cursor_or_running_maintenance(
    tmp_path: Path,
) -> None:
    current = manager(tmp_path)
    address = ConversationAddress("a", date(2026, 7, 1))
    add_conversation(current, address)
    worker = LifecycleWorker(current, config(), worker_id="lifecycle-test")
    token = current.journal.path_lock.lock_store.acquire(worker.lock_key, ttl_seconds=30)

    result = asyncio.run(worker.run_once())

    assert not result.lease_acquired
    assert result.selected_addresses == ()
    assert worker._cursor_store.read() is None
    current.journal.path_lock.lock_store.release(token)


def test_cursor_corruption_is_rejected_instead_of_restarting_rotation_silently(tmp_path: Path) -> None:
    current = manager(tmp_path)
    worker = LifecycleWorker(current, config(), worker_id="lifecycle-test")
    worker._cursor_store.path.parent.mkdir(parents=True, exist_ok=True)
    worker._cursor_store.path.write_text('{"schema":"wrong"}\n', encoding="utf-8")

    with pytest.raises(LifecycleWorkerStateError, match="invalid shape"):
        asyncio.run(worker.run_once())


def test_background_start_is_idempotent_manual_run_is_blocked_and_stop_is_clean(tmp_path: Path) -> None:
    async def scenario() -> None:
        current = manager(tmp_path)
        worker = LifecycleWorker(current, config(), worker_id="lifecycle-test")
        await worker.start()
        await worker.start()
        assert worker.state is LifecycleWorkerState.RUNNING
        with pytest.raises(LifecycleWorkerStateError, match="cannot race"):
            await worker.run_once()
        await worker.stop()
        assert worker.state is LifecycleWorkerState.STOPPED

    asyncio.run(scenario())
