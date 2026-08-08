from __future__ import annotations

import asyncio

import pytest

from conversation.projection import ConversationBehaviorProjectionConsumer, ConversationBehaviorProjector
from conversation.source import (
    ConversationConsumerDelivery,
    ConversationConsumerExecutionFence,
    ConversationConsumerLeaseLostError,
    ConversationConsumerStateInspector,
    ConversationMemoryPredecessorBrokenError,
    ConversationMemoryPredecessorPendingError,
    ConversationSourceConsumer,
    ConversationSourceCoordinator,
)
from infrastructure.store.contracts import LockToken, PathLock
from infrastructure.store.locks import ProcessLocalLockStore
from tests.unit.conversation.source_v2_helpers import (
    NOW,
    FakeMemoryConsumer,
    WrappedBehaviorConsumer,
    delivery,
    source,
    stores,
)


class CountingLockStore(ProcessLocalLockStore):
    def __init__(self) -> None:
        super().__init__()
        self.renewals = 0

    def renew(self, token: LockToken, ttl_seconds: int = 30) -> None:
        self.renewals += 1
        super().renew(token, ttl_seconds=ttl_seconds)


class FailingRenewLockStore(ProcessLocalLockStore):
    def renew(self, token: LockToken, ttl_seconds: int = 30) -> None:
        raise TimeoutError("lease renewal failed")


def test_same_source_consumer_concurrency_executes_only_once_with_lock_inside_recheck(tmp_path) -> None:
    async def scenario() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()
        service, memory, _behavior = delivery(tmp_path)
        memory.entered = entered
        memory.release = release
        source_value = service.sources.put(source())

        first = asyncio.create_task(
            service.ensure_outcome(source_value, ConversationSourceConsumer.MEMORY)
        )
        await entered.wait()
        second = asyncio.create_task(
            service.ensure_outcome(source_value, ConversationSourceConsumer.MEMORY)
        )
        await asyncio.sleep(0.05)
        assert memory.calls == 1
        release.set()
        first_result, second_result = await asyncio.gather(first, second)
        assert memory.calls == 1
        assert first_result.outcome == second_result.outcome

    asyncio.run(scenario())


def test_different_processor_fingerprints_share_one_execution_lock_and_first_outcome_wins(tmp_path) -> None:
    async def scenario() -> None:
        sources, outcomes, memory_outputs, projection_outputs = stores(tmp_path)
        entered = asyncio.Event()
        release = asyncio.Event()
        first_memory = FakeMemoryConsumer(
            memory_outputs,
            fingerprint_seed="processor-old",
            entered=entered,
            release=release,
        )
        second_memory = FakeMemoryConsumer(memory_outputs, fingerprint_seed="processor-new")
        behavior = WrappedBehaviorConsumer(
            ConversationBehaviorProjectionConsumer(
                ConversationBehaviorProjector(clock=lambda: NOW),
                projection_outputs,
            )
        )
        path_lock = PathLock(ProcessLocalLockStore())
        inspector = ConversationConsumerStateInspector(outcomes)

        def make_service(memory: FakeMemoryConsumer) -> ConversationConsumerDelivery:
            return ConversationConsumerDelivery(
                sources,
                outcomes,
                inspector,
                ConversationConsumerExecutionFence(
                    path_lock,
                    ttl_seconds=3,
                    heartbeat_interval_seconds=0.05,
                    wait_seconds=2.0,
                ),
                memory,
                behavior,
                clock=lambda: NOW,
            )

        first_service = make_service(first_memory)
        second_service = make_service(second_memory)
        source_value = sources.put(source())
        first = asyncio.create_task(
            first_service.ensure_outcome(source_value, ConversationSourceConsumer.MEMORY)
        )
        await entered.wait()
        second = asyncio.create_task(
            second_service.ensure_outcome(source_value, ConversationSourceConsumer.MEMORY)
        )
        await asyncio.sleep(0.05)
        assert second_memory.calls == 0
        release.set()
        left, right = await asyncio.gather(first, second)
        assert left.outcome == right.outcome
        assert left.outcome.processor_fingerprint == first_memory.processor_fingerprint
        assert second_memory.calls == 0

    asyncio.run(scenario())


def test_execution_fence_heartbeats_all_held_memory_leases(tmp_path) -> None:
    async def scenario() -> None:
        lock_store = CountingLockStore()
        entered = asyncio.Event()
        release = asyncio.Event()
        service, memory, _behavior = delivery(
            tmp_path,
            path_lock=PathLock(lock_store),
        )
        memory.entered = entered
        memory.release = release
        source_value = service.sources.put(source())
        running = asyncio.create_task(
            service.ensure_outcome(source_value, ConversationSourceConsumer.MEMORY)
        )
        await entered.wait()
        await asyncio.sleep(0.16)
        assert lock_store.renewals >= 4  # Memory 顺序锁和 Consumer 锁均持续续租。
        release.set()
        await running

    asyncio.run(scenario())


def test_heartbeat_loss_prevents_output_and_outcome_publication(tmp_path) -> None:
    async def scenario() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()
        service, memory, _behavior = delivery(
            tmp_path,
            path_lock=PathLock(FailingRenewLockStore()),
        )
        service.fence.wait_seconds = 0.0
        memory.entered = entered
        memory.release = release
        source_value = service.sources.put(source())
        running = asyncio.create_task(
            service.ensure_outcome(source_value, ConversationSourceConsumer.MEMORY)
        )
        await entered.wait()
        await asyncio.sleep(0.1)
        release.set()
        with pytest.raises(ConversationConsumerLeaseLostError):
            await running
        assert memory.output_store.list(source_value) == ()
        assert service.outcomes.read(source_value.source_id, ConversationSourceConsumer.MEMORY) is None

    asyncio.run(scenario())


def test_cancelled_api_waiter_does_not_cancel_underlying_memory_consumer(tmp_path) -> None:
    async def scenario() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()
        service, memory, _behavior = delivery(tmp_path)
        memory.entered = entered
        memory.release = release
        coordinator = ConversationSourceCoordinator(service.sources, service)
        handle = coordinator.start(source())
        waiter = asyncio.create_task(handle.wait_memory())
        await entered.wait()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert not handle.memory_task.cancelled()
        assert not handle.memory_task.done()
        release.set()
        restored = await handle.wait_memory()
        assert restored.append.next_sequence == 1

    asyncio.run(scenario())


def test_behavior_failure_does_not_delay_or_rollback_memory(tmp_path) -> None:
    async def scenario() -> None:
        service, _memory, behavior = delivery(tmp_path)
        behavior.fail = RuntimeError("projection failed")
        coordinator = ConversationSourceCoordinator(service.sources, service)
        handle = coordinator.start(source())
        memory = await asyncio.wait_for(handle.wait_memory(), timeout=1.0)
        assert memory.append.next_sequence == 1
        with pytest.raises(RuntimeError, match="projection failed"):
            await handle.wait_behavior_projection()
        assert service.restore_terminal(handle.envelope, ConversationSourceConsumer.MEMORY) == memory

    asyncio.run(scenario())


def test_blocked_memory_does_not_prevent_behavior_projection_completion(tmp_path) -> None:
    async def scenario() -> None:
        entered = asyncio.Event()
        release = asyncio.Event()
        service, memory, _behavior = delivery(tmp_path)
        memory.entered = entered
        memory.release = release
        coordinator = ConversationSourceCoordinator(service.sources, service)
        handle = coordinator.start(source())
        await entered.wait()
        projected = await asyncio.wait_for(handle.wait_behavior_projection(), timeout=1.0)
        assert projected is not None
        assert not handle.memory_task.done()
        release.set()
        await handle.wait_memory()

    asyncio.run(scenario())


def test_newer_memory_source_fails_fast_when_predecessor_is_pending_or_broken(tmp_path) -> None:
    async def scenario() -> None:
        service, _memory, _behavior = delivery(tmp_path)
        first = service.sources.put(source(sequence=0, delivery_seed="first"))
        second = service.sources.put(source(sequence=1, delivery_seed="second"))
        with pytest.raises(ConversationMemoryPredecessorPendingError):
            await service.ensure_outcome(second, ConversationSourceConsumer.MEMORY)

        await service.ensure_outcome(first, ConversationSourceConsumer.MEMORY)
        state = service.inspect(first, ConversationSourceConsumer.MEMORY)
        assert state.outcome is not None and state.outcome.output_ref is not None
        output_path = (
            tmp_path
            / "source"
            / "outputs"
            / first.source_id
            / "memory"
            / f"{state.outcome.output_ref.output_id}.json"
        )
        output_path.unlink()
        with pytest.raises(ConversationMemoryPredecessorBrokenError):
            await service.ensure_outcome(second, ConversationSourceConsumer.MEMORY)

    asyncio.run(scenario())
