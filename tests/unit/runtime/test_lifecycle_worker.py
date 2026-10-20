"""LifecycleWorker 的全局 lease、参与者失败隔离、会话链耐久轮转与状态机测试。"""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path

import pytest

from habitus.config import ConversationLifecycleConfig, LifecycleWorkerConfig
from habitus.infrastructure.lifecycle import CapacityReading, CapacitySnapshotStore, LifecycleContext, ParticipantReport
from habitus.infrastructure.store.contracts import PathLock
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from habitus.memory.conversation import (
    ConversationAddress,
    ConversationMessageJournal,
    ConversationRangeSummaryGenerator,
    ConversationRangeSummaryStore,
    ConversationSummaryCompactor,
    ConversationSummaryStore,
    PersistentConversationSummaryVectorIndex,
)
from habitus.memory.editor import MemoryTransactionJournal
from habitus.memory.workflow import (
    ConversationLifecycleManager,
    MemoryChangeReceiptStore,
    MemoryJobStore,
)
from habitus.pre.conversation import ConversationBatch
from habitus.runtime import (
    ConversationChainParticipant,
    LifecycleWorker,
    LifecycleWorkerLeaseLostError,
    LifecycleWorkerState,
)
from habitus.runtime.lifecycle_memory import conversation_lease_key
from tests.helpers import closed_turn, codec
from tests.unit.conversation.test_summary_indexing import Embedder, VectorStore
from tests.unit.retrieval.test_search_service import structured


def manager(tmp_path: Path) -> ConversationLifecycleManager:
    path_lock = PathLock(ProcessLocalLockStore())
    journal = ConversationMessageJournal(tmp_path / "conversation", path_lock)
    segment_store = ConversationSummaryStore(journal.layout)
    range_store = ConversationRangeSummaryStore(journal.layout)
    compactor = ConversationSummaryCompactor(
        journal,
        segment_store,
        range_store,
        ConversationRangeSummaryGenerator(structured([])),
    )
    summary_index = PersistentConversationSummaryVectorIndex(
        journal,
        compactor,
        Embedder(),
        VectorStore(),
        dimension=2,
        embedding_fingerprint="lifecycle-worker-test-v1",
    )

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
    return ConversationLifecycleConfig(max_conversations_per_cycle=maximum)


def worker_config() -> LifecycleWorkerConfig:
    return LifecycleWorkerConfig(
        maintenance_interval_seconds=60,
        lease_ttl_seconds=30,
        heartbeat_interval_seconds=10,
        shutdown_timeout_seconds=1,
    )


def chain_worker(
    current: ConversationLifecycleManager,
    *,
    maximum: int = 2,
    extra: tuple[object, ...] = (),
    capacity=None,
    observer=None,
) -> tuple[LifecycleWorker, ConversationChainParticipant]:
    lock_store = current.journal.path_lock.lock_store
    chain = ConversationChainParticipant(
        current, config(maximum=maximum), lock_store=lock_store, lease_ttl_seconds=worker_config().lease_ttl_seconds
    )
    worker = LifecycleWorker(
        (chain, *extra),  # type: ignore[arg-type]
        worker_config(),
        lock_store=lock_store,
        lease_key=conversation_lease_key(current.journal),
        capacity=capacity,
        worker_id="lifecycle-test",
        **({} if observer is None else {"observer": observer}),
    )
    return worker, chain


class _Probe:
    def __init__(
        self,
        name: str,
        *,
        error: Exception | None = None,
        readings: tuple[CapacityReading, ...] = (),
        signals: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.error = error
        self.readings = readings
        self.signals = signals
        self.calls = 0

    async def maintain(self, ctx: LifecycleContext) -> ParticipantReport:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return ParticipantReport(released=1, readings=self.readings, signals=self.signals)


class _StealsLease:
    name = "steals-lease"

    def __init__(self, current: ConversationLifecycleManager) -> None:
        self.current = current

    async def maintain(self, ctx: LifecycleContext) -> ParticipantReport:
        lock_store = self.current.journal.path_lock.lock_store
        lock_store.release(ctx.fence_token())
        lock_store.acquire(ctx.fence_token().lock_key, ttl_seconds=30)
        await asyncio.to_thread(ctx.assert_owned)
        return ParticipantReport()


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
    worker, chain = chain_worker(current, maximum=2)

    asyncio.run(worker.run_once())
    first = chain.last_result
    asyncio.run(worker.run_once())
    second = chain.last_result

    assert first is not None and second is not None
    assert first.selected_addresses == addresses[:2]
    assert first.maintained_addresses == (addresses[1],)
    assert first.failures[0].address == addresses[0]
    assert first.failures[0].error_type == "RuntimeError"
    assert second.selected_addresses == (addresses[2], addresses[0])
    assert calls == [addresses[0], addresses[1], addresses[2], addresses[0]]
    assert chain.cursor_store.read() == (addresses[0].started_on, addresses[0].conversation_id)


def test_competing_global_lease_skips_cycle_without_moving_cursor_or_running_maintenance(
    tmp_path: Path,
) -> None:
    current = manager(tmp_path)
    add_conversation(current, ConversationAddress("a", date(2026, 7, 1)))
    worker, chain = chain_worker(current)
    token = current.journal.path_lock.lock_store.acquire(worker.lock_key, ttl_seconds=30)

    result = asyncio.run(worker.run_once())

    assert not result.lease_acquired
    assert result.outcomes == ()
    assert chain.last_result is None
    assert chain.cursor_store.read() is None
    current.journal.path_lock.lock_store.release(token)


def test_cursor_corruption_fails_the_chain_participant_instead_of_restarting_rotation_silently(
    tmp_path: Path,
) -> None:
    current = manager(tmp_path)
    worker, chain = chain_worker(current)
    chain.cursor_store.path.parent.mkdir(parents=True, exist_ok=True)
    chain.cursor_store.path.write_text('{"schema":"wrong"}\n', encoding="utf-8")

    result = asyncio.run(worker.run_once())

    (failure,) = result.failures
    assert failure.name == "conversation-chain"
    assert failure.error_type == "LifecycleWorkerStateError"
    assert failure.message is not None and "invalid shape" in failure.message


def test_one_failing_participant_does_not_stop_the_others(tmp_path: Path) -> None:
    current = manager(tmp_path)
    broken = _Probe("broken", error=RuntimeError("boom"))
    healthy = _Probe("healthy")
    worker, _chain = chain_worker(current, extra=(broken, healthy))

    result = asyncio.run(worker.run_once())

    assert [item.name for item in result.failures] == ["broken"]
    assert healthy.calls == 1
    outcome = {item.name: item for item in result.outcomes}["healthy"]
    assert outcome.report is not None and outcome.report.released == 1


def test_lost_lease_stops_the_whole_cycle(tmp_path: Path) -> None:
    current = manager(tmp_path)
    after = _Probe("after")
    worker, _chain = chain_worker(current, extra=(_StealsLease(current), after))

    with pytest.raises(LifecycleWorkerLeaseLostError):
        asyncio.run(worker.run_once())
    assert after.calls == 0


def test_capacity_readings_are_published_as_snapshots(tmp_path: Path) -> None:
    current = manager(tmp_path)
    store = CapacitySnapshotStore(tmp_path / "capacity")
    probe = _Probe("sources", readings=(CapacityReading("envelopes", 850, 1000),))
    worker, _chain = chain_worker(current, extra=(probe,), capacity=store)

    asyncio.run(worker.run_once())

    (snapshot,) = store.read_all()
    assert snapshot.participant == "sources"
    assert snapshot.readings[0].share == pytest.approx(0.85)


def test_participant_names_must_be_unique(tmp_path: Path) -> None:
    current = manager(tmp_path)
    with pytest.raises(ValueError, match="unique"):
        chain_worker(current, extra=(_Probe("conversation-chain"),))


def test_background_start_is_idempotent_manual_run_is_blocked_and_stop_is_clean(tmp_path: Path) -> None:
    async def scenario() -> None:
        current = manager(tmp_path)
        worker, _chain = chain_worker(current)
        await worker.start()
        await worker.start()
        assert worker.state is LifecycleWorkerState.RUNNING
        with pytest.raises(Exception, match="cannot race"):
            await worker.run_once()
        await worker.stop()
        assert worker.state is LifecycleWorkerState.STOPPED

    asyncio.run(scenario())


def test_participant_signals_mark_the_cycle_degraded_in_observations(tmp_path: Path) -> None:
    """参与者报的"要人看的情况"原先没人读（L10.10）：这一轮的观测事件按 degraded 记、带上条数。"""

    from habitus.foundation.observability import ObservationStatus

    events = []

    class _Recorder:
        def record(self, event):  # type: ignore[no-untyped-def]
            events.append(event)

    current = manager(tmp_path)
    probe = _Probe("sources", signals=("1 conversation source deliveries are broken and wait for repair",))
    worker, _chain = chain_worker(current, extra=(probe,), observer=_Recorder())

    result = asyncio.run(worker.run_once())

    assert result.signals == ("sources: 1 conversation source deliveries are broken and wait for repair",)
    (cycle,) = [event for event in events if event.operation == "maintenance_cycle"]
    assert cycle.status is ObservationStatus.DEGRADED
    assert cycle.attributes["signals"] == 1
