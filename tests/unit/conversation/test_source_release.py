"""会话源的释放：门槛三条、删除顺序崩溃可续、已释放源重投不再执行 Behavior、有序检查读标记、恢复清残留、迁移补标记。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest

from habitus.conversation.source import (
    ConversationConsumerDeliveryState,
    ConversationConsumerOutcomeState,
    ConversationOrderedPredecessorPendingError,
    ConversationSourceConsumer,
    ConversationSourceReleaser,
    SourceAddress,
)
from habitus.conversation.source.delivery import RELEASED_SKIP_REASON
from habitus.conversation.source.quarantine import SourceQuarantine
from habitus.foundation.integrity import canonical_digest
from habitus.infrastructure.lifecycle import Quarantine
from habitus.infrastructure.store.contracts import LockToken
from tests.unit.conversation.source_v2_helpers import behavior_session_output, delivery, source
from tests.unit.conversation.source_v2_helpers import recovery as source_recovery


class _Context:
    """生命周期上下文的替身：持有一把真的全局 lease，fence 才能校验。"""

    def __init__(self, lock_store) -> None:
        self._token = lock_store.acquire("runtime:lifecycle:test", ttl_seconds=60)

    @property
    def now(self) -> datetime:
        return datetime(2026, 8, 9, tzinfo=UTC)

    def assert_owned(self) -> None:
        return None

    def fence_token(self) -> LockToken:
        return self._token


def _setup(root: Path):
    service, memory, behavior = delivery(root)
    lock_store = service.fence.path_lock.lock_store
    releaser = ConversationSourceReleaser(
        service.sources,
        service.outcomes,
        service.ledger,
        {
            ConversationSourceConsumer.MEMORY: memory.output_store,
            ConversationSourceConsumer.BEHAVIOR_SESSION: behavior.output_store,
        },
        lock_store,
        lock_ttl_seconds=30,
        quarantined=SourceQuarantine(
            Quarantine(root, owner="conversation-source-behavior", clock=lambda: datetime(2026, 8, 9, tzinfo=UTC))
        ).contains,
    )
    return service, memory, behavior, releaser, _Context(lock_store)


async def _deliver(service, envelope, *consumers: ConversationSourceConsumer):
    stored = service.sources.put(envelope)
    for consumer in consumers or tuple(ConversationSourceConsumer):
        await service.ensure_outcome(stored, consumer)
    return stored


def _directory(service) -> str:
    (directory,) = service.ledger.index.address_directories()
    return directory


def test_release_keeps_the_last_source_and_removes_everything_of_the_earlier_ones(tmp_path: Path) -> None:
    async def scenario() -> None:
        service, _memory, _behavior, releaser, ctx = _setup(tmp_path)
        envelopes = [await _deliver(service, source(sequence=n, delivery_seed=f"d{n}")) for n in range(3)]
        directory = _directory(service)

        assert releaser.release_address(directory, ctx) == 2

        for gone in envelopes[:2]:
            assert service.sources.read(gone.source_id) is None
            for consumer in ConversationSourceConsumer:
                assert service.outcomes.read(gone.source_id, consumer) is None
            assert not (tmp_path / "source" / "outputs" / gone.source_id).exists()
            assert not (tmp_path / "source" / "outcomes" / gone.source_id).exists()
        assert [marker.source_id for marker in service.ledger.index.markers_in(directory)] == [envelopes[2].source_id]
        address = SourceAddress.of(envelopes[0])
        assert service.ledger.watermarks.released_through(address) == envelopes[1].batch.end_sequence
        assert service.sources.read(envelopes[2].source_id) is not None

    asyncio.run(scenario())


def test_a_later_source_is_not_released_while_an_earlier_one_is_unfinished(tmp_path: Path) -> None:
    async def scenario() -> None:
        service, _memory, _behavior, releaser, ctx = _setup(tmp_path)
        await _deliver(service, source(sequence=0, delivery_seed="d0"), ConversationSourceConsumer.MEMORY)
        for n in (1, 2):
            await _deliver(service, source(sequence=n, delivery_seed=f"d{n}"))

        assert releaser.releasable(_directory(service)) == ()

    asyncio.run(scenario())


def test_a_crash_after_the_watermark_is_written_resumes_on_the_next_round(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def scenario() -> None:
        service, _memory, _behavior, releaser, ctx = _setup(tmp_path)
        first = await _deliver(service, source(sequence=0, delivery_seed="d0"))
        await _deliver(service, source(sequence=1, delivery_seed="d1"))
        directory = _directory(service)
        original = service.outcomes.discard

        def crash(*_args, **_kwargs):
            raise OSError("disk went away")

        monkeypatch.setattr(service.outcomes, "discard", crash)
        with pytest.raises(OSError):
            releaser.release_address(directory, ctx)
        assert service.sources.read(first.source_id) is None  # 信封已删、Outcome 还在

        monkeypatch.setattr(service.outcomes, "discard", original)
        assert releaser.release_address(directory, ctx) == 1
        assert service.outcomes.read(first.source_id, ConversationSourceConsumer.MEMORY) is None
        assert [marker.source_id for marker in service.ledger.index.markers_in(directory)] != [first.source_id]

    asyncio.run(scenario())


def test_a_new_source_under_the_watermark_is_not_released_before_it_is_consumed(tmp_path: Path) -> None:
    """水位盖住的标记不一定是"上一轮删到一半"：插件状态重建后序号从头数，同一地址会来一份内容不同、序号已在水位以下
    的新源。它还没被两个消费者处理，不能删；处理完、到终态之后才按续删放掉（26 号清单 L10.1）。"""

    async def scenario() -> None:
        service, _memory, _behavior, releaser, ctx = _setup(tmp_path)
        await _deliver(service, source(sequence=0, delivery_seed="d0"))
        await _deliver(service, source(sequence=1, delivery_seed="d1"))
        releaser.release_address(_directory(service), ctx)
        fresh = service.sources.put(source(sequence=0, delivery_seed="rewound"))

        assert releaser.release_address(_directory(service), ctx) == 0
        assert service.sources.read(fresh.source_id) is not None

        for consumer in ConversationSourceConsumer:
            await service.ensure_outcome(fresh, consumer)
        assert releaser.release_address(_directory(service), ctx) == 1
        assert service.sources.read(fresh.source_id) is None

    asyncio.run(scenario())


def test_a_broken_behavior_source_is_quarantined_and_stops_holding_up_its_address(tmp_path: Path) -> None:
    """Behavior 坏了不会自愈（两份孤儿输出、没有 Outcome）：不隔离的话它过不了释放门槛第 2 条，同地址后面的源按第 3 条
    也一份都放不了。恢复把信封副本送进隔离区、Behavior 按终态算，地址照常往下放（26 号清单 2.1.7 / L10.2）。"""

    async def scenario() -> None:
        service, _memory, behavior, releaser, ctx = _setup(tmp_path)
        broken = service.sources.put(source(sequence=0, delivery_seed="d0"))
        await service.ensure_outcome(broken, ConversationSourceConsumer.MEMORY)
        for seed in ("processor-a", "processor-b"):
            output = behavior_session_output(broken, canonical_digest(seed))
            assert output is not None
            behavior.output_store.put(broken, output)
        for n in (1, 2):
            await _deliver(service, source(sequence=n, delivery_seed=f"d{n}"))
        state = service.inspect(broken, ConversationSourceConsumer.BEHAVIOR_SESSION).state
        assert state is ConversationConsumerDeliveryState.CORRUPTED
        assert releaser.release_address(_directory(service), ctx) == 0

        plan = source_recovery(service).plan()

        assert [entry.envelope.source_id for entry in plan.isolated] == [broken.source_id]
        assert plan.broken == ()
        assert source_recovery(service).pending() == ()
        assert releaser.release_address(_directory(service), ctx) == 2

    asyncio.run(scenario())


def test_redelivered_released_source_is_skipped_by_behavior_but_memory_runs(tmp_path: Path) -> None:
    async def scenario() -> None:
        service, memory, behavior, releaser, ctx = _setup(tmp_path)
        first = await _deliver(service, source(sequence=0, delivery_seed="d0"))
        await _deliver(service, source(sequence=1, delivery_seed="d1"))
        releaser.release_address(_directory(service), ctx)
        behavior_calls, memory_calls = behavior.calls, memory.calls

        again = await _deliver(service, first)

        outcome = service.outcomes.read(again.source_id, ConversationSourceConsumer.BEHAVIOR_SESSION)
        assert outcome is not None
        assert outcome.state is ConversationConsumerOutcomeState.SKIPPED
        assert outcome.skip_reason == RELEASED_SKIP_REASON
        assert behavior.calls == behavior_calls
        assert memory.calls == memory_calls + 1

    asyncio.run(scenario())


def test_lock_rows_of_released_sources_are_discarded(tmp_path: Path) -> None:
    async def scenario() -> None:
        service, _memory, _behavior, releaser, ctx = _setup(tmp_path)
        first = await _deliver(service, source(sequence=0, delivery_seed="d0"))
        await _deliver(service, source(sequence=1, delivery_seed="d1"))
        releaser.release_address(_directory(service), ctx)
        locks = service.fence.path_lock.lock_store.locks
        assert not any(first.source_id in key for key in locks)

    asyncio.run(scenario())


def test_ordered_check_reads_markers_and_tells_recording_from_released(tmp_path: Path) -> None:
    async def scenario() -> None:
        service, _memory, _behavior, _releaser, _ctx = _setup(tmp_path)
        earlier, later = source(sequence=0, delivery_seed="d0"), source(sequence=1, delivery_seed="d1")
        service.ledger.index.mark(earlier)  # 标记打了、信封还没写：put 进行中
        stored = service.sources.put(later)
        with pytest.raises(ConversationOrderedPredecessorPendingError, match="still being recorded"):
            await service.ensure_outcome(stored, ConversationSourceConsumer.MEMORY)

        service.ledger.watermarks.advance(SourceAddress.of(earlier), earlier.batch.end_sequence)  # 已释放
        result = await service.ensure_outcome(stored, ConversationSourceConsumer.MEMORY)
        assert result.outcome.state is ConversationConsumerOutcomeState.COMMITTED

    asyncio.run(scenario())


def test_recovery_clears_marks_of_terminal_and_released_sources(tmp_path: Path) -> None:
    async def scenario() -> None:
        service, _memory, _behavior, releaser, ctx = _setup(tmp_path)
        first = await _deliver(service, source(sequence=0, delivery_seed="d0"))
        second = await _deliver(service, source(sequence=1, delivery_seed="d1"))
        waiting = service.sources.put(source(sequence=2, delivery_seed="d2"))
        releaser.release_address(_directory(service), ctx)
        service.ledger.pending.mark(first)  # 释放后残留的"还没终态"标记

        recovery = source_recovery(service)
        pending = recovery.pending()

        assert {entry.envelope.source_id for entry in pending} == {waiting.source_id}
        remaining = {source_id for source_id, _address in service.ledger.pending.listing()}
        assert remaining == {waiting.source_id}
        assert second.source_id not in remaining

    asyncio.run(scenario())


def test_the_watermark_is_written_before_the_envelope_is_deleted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2.1.9 的顺序：先写水位、再删信封。反过来时，删了信封、水位没写成就被杀，这份源"信封不在、水位没盖住"，
    被当成 put 写到一半的孤儿，同地址后面的源全被它挡住。这里让写水位失败：信封必须还在（L10.14 补测）。"""

    async def scenario() -> None:
        service, _memory, _behavior, releaser, ctx = _setup(tmp_path)
        first = await _deliver(service, source(sequence=0, delivery_seed="d0"))
        await _deliver(service, source(sequence=1, delivery_seed="d1"))

        def killed(*_args, **_kwargs):
            raise OSError("killed while writing the watermark")

        monkeypatch.setattr(service.ledger.watermarks, "advance", killed)
        with pytest.raises(OSError):
            releaser.release_address(_directory(service), ctx)

        assert service.sources.read(first.source_id) is not None

    asyncio.run(scenario())


def test_behavior_skips_a_redelivered_source_only_because_it_declares_so(tmp_path: Path) -> None:
    """2.1.5：已释放的源重投时，交付层只对声明了 ``skips_released_sources`` 的消费者直接记 SKIPPED（L10.14 补测：
    原先把声明改成 False，测试照样全过）。"""

    async def scenario() -> None:
        service, _memory, behavior, releaser, ctx = _setup(tmp_path)
        assert behavior.skips_released_sources is True
        first = await _deliver(service, source(sequence=0, delivery_seed="d0"))
        await _deliver(service, source(sequence=1, delivery_seed="d1"))
        releaser.release_address(_directory(service), ctx)
        behavior.skips_released_sources = False
        calls = behavior.calls

        again = await _deliver(service, first, ConversationSourceConsumer.BEHAVIOR_SESSION)

        outcome = service.outcomes.read(again.source_id, ConversationSourceConsumer.BEHAVIOR_SESSION)
        assert outcome is not None and outcome.skip_reason != RELEASED_SKIP_REASON
        assert behavior.calls == calls + 1

    asyncio.run(scenario())
