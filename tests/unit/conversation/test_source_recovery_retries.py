"""会话源恢复走重试账、周期驱动（26 号清单 L9.2，裁定 60 第 6 条）：

- 更早的坏条目不再占住一批，更新的待恢复项照样被恢复；
- 失败的按退避推迟、次数用完放弃，人工修复后 ``forgive`` 再试；"等前序"不记账；
- 超过宽限期的孤儿标记连同索引标记一起清，宽限期内的不动；
- 周期参与者把一批交给协调器，不等它做完；还在跑的不重复派。
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from habitus.conversation.source import (
    ConversationConsumerDeliveryState,
    ConversationSourceConsumer,
    ConversationSourceCoordinator,
    ConversationSourceError,
    ConversationSourceRecoveryDeferredError,
    SourceAddress,
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

    assert recovering.plan().attempt == ()
    assert [item for item, _address in service.ledger.pending.listing().sources] == [orphan.source_id]
    assert len(service.ledger.index.address_directories()) == 1

    now[0] += timedelta(hours=25)
    assert recovering.plan().attempt == ()
    assert service.ledger.pending.listing().sources == ()
    assert len(service.ledger.index.address_directories()) == 0


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


def test_an_envelope_over_the_read_bound_only_holds_itself_back(tmp_path: Path) -> None:
    """读上限调低之后，比它大的那份信封是坏的、只挡它自己：进 ``corrupt``，另一份照常恢复，不冲出这一轮。"""

    async def scenario() -> None:
        service, _memory, _behavior = delivery(tmp_path)
        big, small = source(conversation_id="big", content="长" * 3000), source(conversation_id="small")
        service.sources.put(big)
        service.sources.put(small)
        sizes = {item.source_id: service.sources._path(item.source_id).stat().st_size for item in (big, small)}
        service.sources.max_file_bytes = (sizes[big.source_id] + sizes[small.source_id]) // 2
        recovering = recovery(service)

        plan = recovering.plan()
        await recovering.recover_pending()

        assert plan.corrupt == (big.source_id,)
        assert {entry.envelope.source_id for entry in plan.attempt} == {small.source_id}
        assert _state(service, small, MEMORY) is ConversationConsumerDeliveryState.COMMITTED

    asyncio.run(scenario())


def test_an_outcome_over_the_read_bound_is_a_corrupted_delivery_not_a_crash(tmp_path: Path) -> None:
    async def scenario() -> None:
        service, _memory, _behavior = delivery(tmp_path)
        envelope = service.sources.put(source(conversation_id="outcome"))
        await service.ensure_outcome(envelope, MEMORY)
        service.outcomes.max_file_bytes = 16

        assert _state(service, envelope, MEMORY) is ConversationConsumerDeliveryState.CORRUPTED
        plan = recovery(service).plan()
        assert [(entry.envelope.source_id, entry.consumer) for entry in plan.broken] == [(envelope.source_id, MEMORY)]

    asyncio.run(scenario())


def test_an_index_marker_left_without_its_pending_marker_is_cleared_after_the_grace_period(tmp_path: Path) -> None:
    """put 打完索引标记、还没打"还没终态"标记就失败（容量闸、进程被杀），这份源再没重投：同地址后面那份的 Memory
    等的就是它。过了宽限期，恢复删掉这个孤儿索引标记，后面那份照常提交；宽限期内不动（可能是正在进行的 put）。"""

    async def scenario() -> None:
        service, _memory, _behavior = delivery(tmp_path)
        orphan = source(conversation_id="y", sequence=0, delivery_seed="lost")
        service.ledger.index.mark(orphan)  # 只有索引标记：没有"还没终态"标记，也没有信封
        later = service.sources.put(source(conversation_id="y", sequence=1, delivery_seed="later"))
        now = [datetime.now(UTC)]
        recovering = recovery(service, orphan_grace=timedelta(hours=24), clock=lambda: now[0])

        await recovering.recover_pending()
        assert [marker.source_id for marker in service.ledger.earlier(later)] == [orphan.source_id]
        assert _state(service, later, MEMORY) is ConversationConsumerDeliveryState.PENDING

        now[0] += timedelta(hours=25)
        await recovering.recover_pending()
        assert service.ledger.earlier(later) == ()
        assert _state(service, later, MEMORY) is ConversationConsumerDeliveryState.COMMITTED

    asyncio.run(scenario())


# ── 坏条目只挡它自己（裁定 60 第 6 条，U12 / U16） ─────────────────────────────────────────────────────


def _damage_index(service, envelope, how: str) -> None:  # type: ignore[no-untyped-def]
    directory = service.ledger.index.index_root / SourceAddress.of(envelope).digest
    marker = next(path for path in directory.iterdir() if not path.name.startswith("."))
    if how == "truncated":
        marker.write_bytes(marker.read_bytes()[:20])
    elif how == "missing_field":
        payload = json.loads(marker.read_bytes())
        payload.pop("start_sequence")
        marker.write_bytes(json.dumps(payload).encode())
    elif how == "ds_store":
        (directory / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1")
    elif how == "editor_backup":
        (directory / f"{marker.name}~").write_bytes(marker.read_bytes())


@pytest.mark.parametrize("how", ["truncated", "missing_field", "ds_store", "editor_backup"])
def test_a_corrupt_index_marker_only_holds_its_own_address(tmp_path: Path, how: str) -> None:
    """一个地址的索引读不出来（坏标记、目录里的杂文件）：恢复照常处理别的地址，这个地址记进 ``unreadable``，它自己的
    Memory 交付读不了前序、报错记账；不让每一轮恢复都整轮失败。"""

    async def scenario() -> None:
        service, _memory, _behavior = delivery(tmp_path)
        bad = service.sources.put(source(conversation_id="bad", delivery_seed="bad"))
        good = service.sources.put(source(conversation_id="good", delivery_seed="good"))
        _damage_index(service, bad, how)
        recovering = recovery(service)

        plan = recovering.plan()
        results = await recovering.recover_pending()

        assert plan.unreadable == (f"index/{SourceAddress.of(bad).digest}",)
        assert _state(service, good, MEMORY) is ConversationConsumerDeliveryState.COMMITTED
        assert _state(service, bad, MEMORY) is not ConversationConsumerDeliveryState.COMMITTED
        failed = next(result for result in results if result.envelope == bad and result.consumer is MEMORY)
        assert isinstance(failed.error, ConversationSourceError)

    asyncio.run(scenario())


def test_a_corrupt_index_under_a_marker_whose_envelope_is_gone_only_holds_its_address(tmp_path: Path) -> None:
    """"还没终态"标记在、信封不在（put 写到一半），要看索引判它是不是释放过；索引坏了也只挡这个地址。"""

    async def scenario() -> None:
        service, _memory, _behavior = delivery(tmp_path)
        lost = source(conversation_id="bad", delivery_seed="lost")
        service.ledger.admit(lost)
        good = service.sources.put(source(conversation_id="good", delivery_seed="good"))
        _damage_index(service, lost, "ds_store")
        recovering = recovery(service)

        plan = recovering.plan()
        await recovering.recover_pending()

        assert plan.unreadable == (f"index/{SourceAddress.of(lost).digest}",)
        assert _state(service, good, MEMORY) is ConversationConsumerDeliveryState.COMMITTED

    asyncio.run(scenario())


def _damage_pending(service, envelope, how: str) -> None:  # type: ignore[no-untyped-def]
    path = service.ledger.pending.pending_root / f"{envelope.source_id}.json"
    if how == "truncated":
        path.write_bytes(path.read_bytes()[:15])
    elif how == "missing_address":
        payload = json.loads(path.read_bytes())
        payload.pop("address")
        path.write_bytes(json.dumps(payload).encode())
    elif how == "ds_store":
        (service.ledger.pending.pending_root / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1")


@pytest.mark.parametrize("how", ["truncated", "missing_address"])
def test_a_corrupt_pending_marker_only_holds_its_own_source(tmp_path: Path, how: str) -> None:
    async def scenario() -> None:
        service, _memory, _behavior = delivery(tmp_path)
        bad = service.sources.put(source(conversation_id="bad", delivery_seed="bad"))
        good = service.sources.put(source(conversation_id="good", delivery_seed="good"))
        _damage_pending(service, bad, how)
        recovering = recovery(service)

        plan = recovering.plan()
        await recovering.recover_pending()

        assert plan.corrupt == (bad.source_id,)
        assert _state(service, good, MEMORY) is ConversationConsumerDeliveryState.COMMITTED
        marker = service.ledger.pending.pending_root / f"{bad.source_id}.json"
        assert marker.exists()  # 坏的那份留着等人工处置
        with pytest.raises(ConversationSourceError):  # 缺字段也是"坏标记"，不是裸 KeyError
            service.ledger.pending._address(marker)

    asyncio.run(scenario())


def test_a_stray_file_among_the_pending_markers_only_holds_itself(tmp_path: Path) -> None:
    """"还没终态"目录里多一个 ``.DS_Store``：不吞掉（记进 ``unreadable`` 等人处置），也不挡任何一份源。"""

    async def scenario() -> None:
        service, _memory, _behavior = delivery(tmp_path)
        first = service.sources.put(source(conversation_id="first", delivery_seed="first"))
        second = service.sources.put(source(conversation_id="second", delivery_seed="second"))
        _damage_pending(service, first, "ds_store")
        recovering = recovery(service)

        plan = recovering.plan()
        await recovering.recover_pending()

        assert plan.unreadable == ("pending/.DS_Store",)
        assert _state(service, first, MEMORY) is ConversationConsumerDeliveryState.COMMITTED
        assert _state(service, second, MEMORY) is ConversationConsumerDeliveryState.COMMITTED

    asyncio.run(scenario())


def test_a_corrupt_envelope_s_index_marker_is_not_cleared_as_an_orphan(tmp_path: Path) -> None:
    """信封在、只是坏了：它不是孤儿。过了宽限期它的索引标记也留着，同地址后面的 Memory 照旧等它（S2R-13）。"""

    async def scenario() -> None:
        service, _memory, _behavior = delivery(tmp_path)
        broken = service.sources.put(source(conversation_id="y", sequence=0, delivery_seed="broken"))
        later = service.sources.put(source(conversation_id="y", sequence=1, delivery_seed="later"))
        path = service.sources._path(broken.source_id)
        path.write_bytes(path.read_bytes()[:-5] + b"xxxxx")
        service.ledger.pending.clear(broken.source_id)  # 只剩索引标记：正是孤儿清理要看的那种
        now = [datetime.now(UTC) + timedelta(days=30)]
        recovering = recovery(service, orphan_grace=timedelta(hours=24), clock=lambda: now[0])

        results = await recovering.recover_pending()

        assert [marker.source_id for marker in service.ledger.earlier(later)] == [broken.source_id]
        waiting = next(result for result in results if result.envelope == later and result.consumer is MEMORY)
        assert waiting.error is not None
        assert _state(service, later, MEMORY) is ConversationConsumerDeliveryState.PENDING

    asyncio.run(scenario())


# ── 取消立刻生效（U17） ─────────────────────────────────────────────────────────────────────────────


def test_cancelling_a_recovery_batch_stops_it_at_once(tmp_path: Path) -> None:
    """停机时取消一批恢复：取消照常抛出，后面的有序条目不再跑，并发的那几份也一起停。"""

    async def scenario() -> None:
        service, _memory, _behavior = delivery(tmp_path)
        first = service.sources.put(source(conversation_id="ordered", sequence=0, delivery_seed="a"))
        second = service.sources.put(source(conversation_id="ordered", sequence=1, delivery_seed="b"))
        started: list[tuple[str, ConversationSourceConsumer]] = []
        cancelled: list[ConversationSourceConsumer] = []
        entered = asyncio.Event()

        async def ensure_outcome(envelope, consumer):  # type: ignore[no-untyped-def]
            started.append((envelope.source_id, consumer))
            entered.set()
            try:
                await asyncio.sleep(1.0)  # 有界：取消没生效时这一批照跑，测试看到的是正常返回而不是挂住
            except asyncio.CancelledError:
                cancelled.append(consumer)
                raise

        service.ensure_outcome = ensure_outcome  # type: ignore[method-assign]
        recovering = recovery(service)
        entries = recovering.plan().attempt
        task = asyncio.create_task(recovering.recover_all(entries))
        await entered.wait()
        await asyncio.sleep(0)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task
        await asyncio.sleep(0)
        assert (second.source_id, MEMORY) not in started
        assert (first.source_id, MEMORY) in started
        assert cancelled.count(BEHAVIOR) == 2  # 并发的 Behavior 两份没留在后台
        assert recovering.retries.entry(recovery_key(first.source_id, MEMORY)) is None  # 取消不是失败、不记账

    asyncio.run(scenario())


@pytest.mark.parametrize("consumer", [MEMORY, BEHAVIOR])
def test_an_output_over_the_read_bound_is_a_corrupted_delivery_not_a_crash(tmp_path: Path, consumer) -> None:  # type: ignore[no-untyped-def]
    """两个消费者的输出读超上限都转成源错误（U4 的另两处）：交付判为 CORRUPTED，Memory 进 ``broken`` 等人工、Behavior
    当场补救，都不冲出这一轮。"""

    async def scenario() -> None:
        service, _memory, _behavior = delivery(tmp_path)
        envelope = service.sources.put(source(conversation_id="output"))
        await service.ensure_outcome(envelope, consumer)
        service.consumers[consumer].output_store.max_file_bytes = 16

        assert _state(service, envelope, consumer) is ConversationConsumerDeliveryState.CORRUPTED
        plan = recovery(service).plan()
        held = plan.broken if consumer is MEMORY else plan.salvaged
        assert [(entry.envelope.source_id, entry.consumer) for entry in held] == [(envelope.source_id, consumer)]

    asyncio.run(scenario())
