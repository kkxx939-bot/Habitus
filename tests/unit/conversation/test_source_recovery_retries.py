"""会话源恢复走重试账、周期驱动（26 号清单 L9.2，裁定 60 第 6 条）：

- 更早的坏条目不再占住一批，更新的待恢复项照样被恢复；
- 失败的按退避推迟、次数用完放弃，人工修复后 ``forgive`` 再试；"等前序"不记账；
- 超过宽限期的孤儿标记连同索引标记一起清，宽限期内的不动；
- 周期参与者把一批交给协调器，不等它做完；还在跑的不重复派。
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

from habitus.conversation.source import (
    ConversationConsumerDeliveryState,
    ConversationSourceConsumer,
    ConversationSourceCoordinator,
    ConversationSourceRecoveryDeferredError,
)
from habitus.conversation.source.recovery import recovery_key
from habitus.runtime.lifecycle_sources import ConversationSourceRecoveryParticipant
from tests.unit.conversation.source_v2_helpers import (
    NOW,
    FakeBehaviorConsumer,
    FakeMemoryConsumer,
    delivery,
    recovery,
    source,
    stores,
)

MEMORY = ConversationSourceConsumer.MEMORY
BEHAVIOR = ConversationSourceConsumer.BEHAVIOR_SESSION


class _FailingMemory(FakeMemoryConsumer):
    """指名的那几份源每次都抛可重试的错（像传输失败），其余照常。"""

    def __init__(self, output_store, broken: set[str]) -> None:  # type: ignore[no-untyped-def]
        super().__init__(output_store)
        self.broken = broken
        self.failed = 0

    async def execute(self, envelope, lease):  # type: ignore[no-untyped-def]
        if envelope.source_id in self.broken:
            self.failed += 1
            raise ConnectionError("model transport failed")
        return await super().execute(envelope, lease)


class _FailingBehavior(FakeBehaviorConsumer):
    def __init__(self, output_store, broken: set[str]) -> None:  # type: ignore[no-untyped-def]
        super().__init__(output_store)
        self.broken = broken
        self.failed = 0

    async def execute(self, envelope, lease):  # type: ignore[no-untyped-def]
        if envelope.source_id in self.broken:
            self.failed += 1
            raise ConnectionError("model transport failed")
        return await super().execute(envelope, lease)


def _failing(root: Path, broken: set[str]):  # type: ignore[no-untyped-def]
    _sources, _outcomes, memory_outputs, behavior_outputs = stores(root)
    memory = _FailingMemory(memory_outputs, broken)
    behavior = _FailingBehavior(behavior_outputs, broken)
    service, _memory, _behavior = delivery(root, memory=memory, behavior=behavior)
    return service, memory, behavior


def _state(service, envelope, consumer):  # type: ignore[no-untyped-def]
    return service.inspect(service.sources.read(envelope.source_id), consumer).state


def test_a_hundred_earlier_failing_sources_do_not_starve_a_newer_one(tmp_path: Path) -> None:
    async def scenario() -> None:
        old = [
            source(conversation_id=f"old-{index:03d}", recorded_at=NOW - timedelta(minutes=200 - index))
            for index in range(100)
        ]
        fresh = source(conversation_id="fresh", recorded_at=NOW)
        service, memory, behavior = _failing(tmp_path, {item.source_id for item in old})
        for item in (*old, fresh):
            service.sources.put(item)
        recovering = recovery(service, batch_size=100)

        for _round in range(3):
            await recovering.recover_pending()

        assert _state(service, fresh, MEMORY) is ConversationConsumerDeliveryState.COMMITTED
        assert _state(service, fresh, BEHAVIOR) is ConversationConsumerDeliveryState.COMMITTED
        # 每份坏的只试了一次：退避期内不再占批
        assert (memory.failed, behavior.failed) == (100, 100)

    asyncio.run(scenario())


def test_failures_back_off_then_give_up_and_repair_forgives_them(tmp_path: Path) -> None:
    async def scenario() -> None:
        envelope = source(conversation_id="flaky")
        service, memory, _behavior = _failing(tmp_path, {envelope.source_id})
        service.sources.put(envelope)
        now = [NOW]
        recovering = recovery(service, max_attempts=2, clock=lambda: now[0])
        key = recovery_key(envelope.source_id, MEMORY)

        await recovering.recover_pending()
        assert recovering.retries.entry(key).attempts == 1  # type: ignore[union-attr]
        results = await recovering.recover_pending()  # 还在退避期：不试，但照样出一条"还卡着"
        assert memory.failed == 1
        deferred = [result for result in results if result.consumer is MEMORY]
        assert isinstance(deferred[0].error, ConversationSourceRecoveryDeferredError)

        now[0] = NOW + timedelta(hours=1)
        await recovering.recover_pending()
        assert memory.failed == 2
        assert recovering.retries.entry(key).kind.value == "deterministic"  # type: ignore[union-attr]
        now[0] = NOW + timedelta(days=30)
        await recovering.recover_pending()
        assert memory.failed == 2  # 放弃了：再久也不自己试

        memory.broken.clear()
        assert recovering.forgive(envelope.source_id, MEMORY)
        await recovering.recover_pending()
        assert _state(service, envelope, MEMORY) is ConversationConsumerDeliveryState.COMMITTED
        assert recovering.retries.entry(key) is None

    asyncio.run(scenario())


def test_waiting_for_an_earlier_source_is_not_counted_as_a_failure(tmp_path: Path) -> None:
    async def scenario() -> None:
        first = source(conversation_id="ordered", sequence=0, delivery_seed="a", recorded_at=NOW)
        second = source(conversation_id="ordered", sequence=1, delivery_seed="b", recorded_at=NOW)
        service, memory, _behavior = _failing(tmp_path, {first.source_id})
        service.sources.put(first)
        service.sources.put(second)
        recovering = recovery(service)

        results = await recovering.recover_pending()

        waiting = next(result for result in results if result.envelope == second and result.consumer is MEMORY)
        assert waiting.error is not None
        assert recovering.retries.entry(recovery_key(second.source_id, MEMORY)) is None
        assert recovering.retries.entry(recovery_key(first.source_id, MEMORY)) is not None

    asyncio.run(scenario())


def test_orphan_markers_are_cleared_only_after_the_grace_period(tmp_path: Path) -> None:
    service, _memory, _behavior = delivery(tmp_path)
    orphan = source(conversation_id="orphan")
    service.ledger.admit(orphan)  # put 写到一半崩溃：两样标记都在、信封不在
    now = [datetime.now(UTC)]
    recovering = recovery(service, orphan_grace=timedelta(hours=24), clock=lambda: now[0])

    assert recovering.pending() == ()
    assert [item for item, _address in service.ledger.pending.listing()] == [orphan.source_id]
    assert service.ledger.index.count() == 1

    now[0] += timedelta(hours=25)
    assert recovering.pending() == ()
    assert service.ledger.pending.listing() == ()
    assert service.ledger.index.count() == 0


def test_the_participant_hands_the_batch_to_the_coordinator_without_waiting(tmp_path: Path) -> None:
    async def scenario() -> None:
        _sources, _outcomes, memory_outputs, behavior_outputs = stores(tmp_path)
        release = asyncio.Event()
        memory = FakeMemoryConsumer(memory_outputs)
        behavior = FakeBehaviorConsumer(behavior_outputs, release=release)
        service, _m, _b = delivery(tmp_path, memory=memory, behavior=behavior)
        envelope = source(conversation_id="periodic")
        service.sources.put(envelope)
        coordinator = ConversationSourceCoordinator(service.sources, service)
        woken: list[bool] = []
        participant = ConversationSourceRecoveryParticipant(
            recovery(service), coordinator, wake_memory=lambda: woken.append(True)
        )

        await participant.maintain(_Context())
        await asyncio.sleep(0.05)
        assert {(item.source_id, item.consumer) for item in coordinator.pending_deliveries} == {
            (envelope.source_id, MEMORY),
            (envelope.source_id, BEHAVIOR),
        }
        await participant.maintain(_Context())  # 还在跑的不重复派
        release.set()
        assert await coordinator.wait_for_idle(2.0)

        assert behavior.calls == 1
        assert _state(service, envelope, BEHAVIOR) is ConversationConsumerDeliveryState.COMMITTED
        assert _state(service, envelope, MEMORY) is ConversationConsumerDeliveryState.COMMITTED
        assert woken == []  # 替身的 Memory 不出任务：没有任务就不唤醒

    asyncio.run(scenario())


class _Context:
    def assert_owned(self) -> None:
        return None


def test_followers_of_a_held_memory_source_do_not_crowd_out_other_addresses(tmp_path: Path) -> None:
    """同地址的 Memory 队首在退避：后面的只会报"等前序"，不占这一批；别的地址的新源照样恢复（L10.8）。"""

    async def scenario() -> None:
        head = source(conversation_id="ordered", sequence=0, delivery_seed="h", recorded_at=NOW - timedelta(hours=5))
        followers = [
            source(
                conversation_id="ordered",
                sequence=index,
                delivery_seed=f"f{index}",
                recorded_at=NOW - timedelta(minutes=200 - index),
            )
            for index in range(1, 11)
        ]
        fresh = source(conversation_id="fresh", recorded_at=NOW)
        service, _memory, _behavior = _failing(tmp_path, {head.source_id})
        for item in (head, *followers, fresh):
            service.sources.put(item)
        recovering = recovery(service, batch_size=10)

        for _round in range(3):
            await recovering.recover_pending()

        assert _state(service, fresh, MEMORY) is ConversationConsumerDeliveryState.COMMITTED
        planned = recovering.plan()
        assert not any(entry.envelope in followers and entry.consumer is MEMORY for entry in planned.attempt)

    asyncio.run(scenario())


def test_an_unreadable_envelope_only_holds_itself_back(tmp_path: Path) -> None:
    """一份信封读不出来：恢复把它记成损坏、照常处理别的，不让整轮（以及启动）失败（L10.9）。"""

    async def scenario() -> None:
        service, _memory, _behavior = delivery(tmp_path)
        bad, good = source(conversation_id="bad"), source(conversation_id="good")
        service.sources.put(bad)
        service.sources.put(good)
        path = service.sources._path(bad.source_id)
        path.write_bytes(path.read_bytes()[:-5] + b"xxxxx")
        recovering = recovery(service)

        plan = recovering.plan()
        await recovering.recover_pending()

        assert plan.corrupt == (bad.source_id,)
        assert _state(service, good, MEMORY) is ConversationConsumerDeliveryState.COMMITTED

    asyncio.run(scenario())
