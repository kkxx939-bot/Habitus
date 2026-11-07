from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from habitus.conversation.behavior_session import BehaviorSessionOutput, BehaviorSessionTurnRecord
from habitus.conversation.source import (
    ConsumerOutputRef,
    ConversationConsumerCorruptionError,
    ConversationConsumerDeliveryState,
    ConversationConsumerOutcome,
    ConversationConsumerOutcomeState,
    ConversationSourceConsumer,
    ConversationSourceError,
)
from habitus.foundation.integrity import canonical_digest
from habitus.memory.workflow import MemoryConversationOutput
from habitus.pre.conversation import ConversationMessageRole
from tests.platform_helpers import long_path
from tests.unit.conversation.source_v2_helpers import NOW, FakeMemoryConsumer, delivery, ingest_result, source, stores
from tests.unit.conversation.source_v2_helpers import recovery as source_recovery


async def _write_orphan_memory_output(service, consumer, source_value):
    async with service.fence.acquire(
        source_value,
        ConversationSourceConsumer.MEMORY,
        conversation_ordered=True,
    ) as lease:
        return await consumer.execute(source_value, lease)


def test_recovery_adopts_memory_output_without_rerunning_consumer(tmp_path) -> None:
    async def scenario() -> None:
        service, memory, _behavior = delivery(tmp_path)
        source_value = service.sources.put(source())
        await _write_orphan_memory_output(service, memory, source_value)
        assert memory.calls == 1
        assert (
            service.inspect(
                source_value,
                ConversationSourceConsumer.MEMORY,
            ).state
            is ConversationConsumerDeliveryState.OUTPUT_READY
        )

        recovery = source_recovery(service)
        results = await recovery.recover_pending()
        memory_results = [item for item in results if item.consumer is ConversationSourceConsumer.MEMORY]
        assert len(memory_results) == 1 and memory_results[0].error is None
        assert memory.calls == 1
        assert (
            service.inspect(
                source_value,
                ConversationSourceConsumer.MEMORY,
            ).state
            is ConversationConsumerDeliveryState.COMMITTED
        )

    asyncio.run(scenario())


def test_unique_old_processor_output_is_adopted_after_upgrade(tmp_path) -> None:
    async def scenario() -> None:
        service, old_memory, behavior = delivery(tmp_path)
        source_value = service.sources.put(source())
        await _write_orphan_memory_output(service, old_memory, source_value)

        new_memory = FakeMemoryConsumer(
            old_memory.output_store,
            fingerprint_seed="upgraded-memory-processor",
        )
        upgraded = type(service)(
            service.sources,
            service.outcomes,
            service.inspector,
            service.fence,
            {
                ConversationSourceConsumer.MEMORY: new_memory,
                ConversationSourceConsumer.BEHAVIOR_SESSION: behavior,
            },
            clock=lambda: NOW,
        )
        ensured = await upgraded.ensure_outcome(source_value, ConversationSourceConsumer.MEMORY)
        assert new_memory.calls == 0
        assert ensured.outcome.processor_fingerprint == old_memory.processor_fingerprint

    asyncio.run(scenario())


def test_recovery_adopts_behavior_output_without_running_the_consumer_again(tmp_path) -> None:
    async def scenario() -> None:
        service, _memory, behavior = delivery(tmp_path)
        source_value = service.sources.put(source())
        async with service.fence.acquire(
            source_value,
            ConversationSourceConsumer.BEHAVIOR_SESSION,
        ) as lease:
            await behavior.execute(source_value, lease)
        assert behavior.calls == 1
        assert (
            service.inspect(
                source_value,
                ConversationSourceConsumer.BEHAVIOR_SESSION,
            ).state
            is ConversationConsumerDeliveryState.OUTPUT_READY
        )
        recovery = source_recovery(service)
        await recovery.recover_pending()
        assert behavior.calls == 1
        assert (
            service.inspect(
                source_value,
                ConversationSourceConsumer.BEHAVIOR_SESSION,
            ).state
            is ConversationConsumerDeliveryState.COMMITTED
        )

    asyncio.run(scenario())


def test_multiple_orphan_outputs_are_ambiguous_and_fail_closed(tmp_path) -> None:
    async def scenario() -> None:
        service, memory, _behavior = delivery(tmp_path)
        source_value = service.sources.put(source())
        first = MemoryConversationOutput.create(
            source=source_value,
            processor_fingerprint=memory.processor_fingerprint,
            ingest_result=ingest_result(source_value),
            recorded_at=NOW,
        )
        second_consumer = FakeMemoryConsumer(memory.output_store, fingerprint_seed="second-processor")
        second = MemoryConversationOutput.create(
            source=source_value,
            processor_fingerprint=second_consumer.processor_fingerprint,
            ingest_result=ingest_result(source_value),
            recorded_at=NOW,
        )
        memory.output_store.put(source_value, first)
        memory.output_store.put(source_value, second)
        assert (
            service.inspect(
                source_value,
                ConversationSourceConsumer.MEMORY,
            ).state
            is ConversationConsumerDeliveryState.CORRUPTED
        )
        with pytest.raises(ConversationConsumerCorruptionError, match="multiple orphan"):
            await service.ensure_outcome(source_value, ConversationSourceConsumer.MEMORY)

    asyncio.run(scenario())


def test_committed_outcome_with_missing_output_is_broken_and_not_recreated(tmp_path) -> None:
    async def scenario() -> None:
        service, memory, _behavior = delivery(tmp_path)
        source_value = service.sources.put(source())
        ensured = await service.ensure_outcome(source_value, ConversationSourceConsumer.MEMORY)
        assert ensured.outcome.output_ref is not None
        path = (
            tmp_path
            / "source"
            / "outputs"
            / source_value.source_id
            / "memory"
            / f"{ensured.outcome.output_ref.output_id}.json"
        )
        path.unlink()
        state = service.inspect(source_value, ConversationSourceConsumer.MEMORY)
        assert state.state is ConversationConsumerDeliveryState.BROKEN_OUTCOME
        calls = memory.calls
        recovery = source_recovery(service)
        result = next(
            item for item in await recovery.recover_pending() if item.consumer is ConversationSourceConsumer.MEMORY
        )
        assert result.error is not None
        assert memory.calls == calls

    asyncio.run(scenario())


def test_output_record_digest_mismatch_is_corrupted_and_not_overwritten(tmp_path) -> None:
    async def scenario() -> None:
        service, _memory, _behavior = delivery(tmp_path)
        source_value = service.sources.put(source())
        ensured = await service.ensure_outcome(source_value, ConversationSourceConsumer.MEMORY)
        assert ensured.outcome.output_ref is not None
        path = (
            tmp_path
            / "source"
            / "outputs"
            / source_value.source_id
            / "memory"
            / f"{ensured.outcome.output_ref.output_id}.json"
        )
        value = json.loads(path.read_text(encoding="utf-8"))
        value["output_record_digest"] = "0" * 64
        path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        assert (
            service.inspect(
                source_value,
                ConversationSourceConsumer.MEMORY,
            ).state
            is ConversationConsumerDeliveryState.CORRUPTED
        )
        before = path.read_bytes()
        with pytest.raises(ConversationConsumerCorruptionError):
            await service.ensure_outcome(source_value, ConversationSourceConsumer.MEMORY)
        assert path.read_bytes() == before

    asyncio.run(scenario())


def test_outcome_record_digest_mismatch_is_corrupted(tmp_path) -> None:
    async def scenario() -> None:
        service, _memory, _behavior = delivery(tmp_path)
        source_value = service.sources.put(source())
        await service.ensure_outcome(source_value, ConversationSourceConsumer.MEMORY)
        path = tmp_path / "source" / "outcomes" / source_value.source_id / "memory.json"
        value = json.loads(path.read_text(encoding="utf-8"))
        value["outcome_record_digest"] = "0" * 64
        path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        assert (
            service.inspect(
                source_value,
                ConversationSourceConsumer.MEMORY,
            ).state
            is ConversationConsumerDeliveryState.CORRUPTED
        )

    asyncio.run(scenario())


def test_skipped_outcome_with_behavior_output_is_corrupted(tmp_path) -> None:
    async def scenario() -> None:
        service, _memory, behavior = delivery(tmp_path)
        source_value = service.sources.put(source(role=ConversationMessageRole.COMPLETION))
        skipped = await service.ensure_outcome(
            source_value,
            ConversationSourceConsumer.BEHAVIOR_SESSION,
        )
        assert skipped.outcome.state is ConversationConsumerOutcomeState.SKIPPED
        # 已经落了"跳过"，却又冒出一份回执：两者矛盾，只能判损坏。
        output = BehaviorSessionOutput.create(
            source=source_value,
            processor_fingerprint=behavior.processor_fingerprint,
            turns=(
                BehaviorSessionTurnRecord(
                    start_sequence=0, end_sequence=0, instructed_at=NOW, completed_at=NOW, recorded=True
                ),
            ),
        )
        behavior.output_store.put(source_value, output)
        assert (
            service.inspect(
                source_value,
                ConversationSourceConsumer.BEHAVIOR_SESSION,
            ).state
            is ConversationConsumerDeliveryState.CORRUPTED
        )

    asyncio.run(scenario())


def test_recovery_only_runs_missing_behavior_when_memory_is_committed(tmp_path) -> None:
    async def scenario() -> None:
        service, memory, behavior = delivery(tmp_path)
        source_value = service.sources.put(source())
        await service.ensure_outcome(source_value, ConversationSourceConsumer.MEMORY)
        memory_calls = memory.calls
        recovery = source_recovery(service)
        results = await recovery.recover_pending()
        assert [item.consumer for item in results] == [ConversationSourceConsumer.BEHAVIOR_SESSION]
        assert behavior.calls == 1
        assert memory.calls == memory_calls

    asyncio.run(scenario())


def test_first_memory_output_replay_is_not_changed_by_later_source(tmp_path) -> None:
    async def scenario() -> None:
        service, _memory, _behavior = delivery(tmp_path)
        first = service.sources.put(source(sequence=0, delivery_seed="first"))
        initial = await service.ensure_outcome(first, ConversationSourceConsumer.MEMORY)
        assert initial.outcome.state is ConversationConsumerOutcomeState.COMMITTED
        first_result = service.restore_terminal(first, ConversationSourceConsumer.MEMORY)
        second = service.sources.put(source(sequence=1, delivery_seed="second"))
        await service.ensure_outcome(second, ConversationSourceConsumer.MEMORY)
        assert service.restore_terminal(first, ConversationSourceConsumer.MEMORY) == first_result

    asyncio.run(scenario())


def _rewrite_field(path: Path, field: str, bad: object) -> None:
    """把盘上记录的一个字段（``a.b`` 表示嵌套）换成类型错的值，其余字节照旧规范编码。"""

    value = json.loads(path.read_text(encoding="utf-8"))
    *parents, leaf = field.split(".")
    target = value
    for parent in parents:
        target = target[parent]
    target[leaf] = bad
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("source_id", 1, "source_id must be lowercase SHA-256 text"),
        ("conversation_id", 1, "conversation_id must be non-empty text"),
        ("started_on", 20260808, "started_on must be a calendar date"),
        ("protocol", ["normalized"], "protocol must be non-empty normalized text"),
        ("delivery_id", None, "delivery_id must be lowercase SHA-256 text"),
        ("request_digest", 1, "request_digest must be lowercase SHA-256 text"),
        ("source_payload_digest", 1, "source_payload_digest must be lowercase SHA-256 text"),
        ("recorded_at", 1, "recorded_at must be a datetime"),
        ("source_record_digest", None, "source_record_digest must be lowercase SHA-256 text"),
    ],
)
def test_source_record_with_wrong_field_types_is_corrupt_on_read(
    tmp_path: Path, field: str, bad: object, message: str
) -> None:
    """盘上信封字段类型错：读回时在解码器就被拒，Store 报 corrupt（W16：类型检查在边界）。"""

    sources, *_others = stores(tmp_path)
    stored = sources.put(source())
    _rewrite_field(long_path(sources._path(stored.source_id)), field, bad)  # noqa: SLF001 - 直接改盘上的字节

    with pytest.raises(ConversationSourceError, match="source record is corrupt") as caught:
        sources.read(stored.source_id)
    assert isinstance(caught.value.__cause__, ConversationSourceError)
    assert message in str(caught.value.__cause__)


def test_source_record_with_non_boolean_after_turn_is_rejected_on_read(tmp_path: Path) -> None:
    sources, *_others = stores(tmp_path)
    stored = sources.put(source())
    _rewrite_field(long_path(sources._path(stored.source_id)), "after_turn", "false")  # noqa: SLF001

    with pytest.raises(TypeError, match="after_turn must be boolean"):
        sources.read(stored.source_id)


@pytest.mark.parametrize(
    ("state", "field", "bad", "message"),
    [
        ("skipped", "source_id", 1, "outcome source_id must be lowercase SHA-256 text"),
        ("skipped", "source_payload_digest", None, "outcome source_payload_digest must be lowercase SHA-256 text"),
        ("skipped", "processor_fingerprint", 1, "outcome processor_fingerprint must be lowercase SHA-256 text"),
        ("skipped", "skip_reason", 1, "skip_reason must be non-empty normalized text"),
        ("skipped", "completed_at", 1, "completed_at must be a datetime"),
        ("skipped", "outcome_record_digest", 1, "outcome_record_digest must be lowercase SHA-256 text"),
        ("committed", "output_ref.output_kind", 1, "output_kind must be non-empty normalized text"),
        ("committed", "output_ref.output_id", None, "output_id must be lowercase SHA-256 text"),
        ("committed", "output_ref.output_record_digest", 1, "output_record_digest must be lowercase SHA-256 text"),
        ("committed", "output_ref.processor_fingerprint", 1, "processor_fingerprint must be lowercase SHA-256 text"),
    ],
)
def test_outcome_with_wrong_field_types_is_corrupt_on_read(
    tmp_path: Path, state: str, field: str, bad: object, message: str
) -> None:
    """盘上 Outcome（含其 Output 引用）字段类型错：读回时在解码器就被拒，Store 报 corrupt。"""

    _sources, outcomes, *_others = stores(tmp_path)
    source_value = source()
    fingerprint = canonical_digest("processor")
    committed = state == "committed"
    outcome = ConversationConsumerOutcome.create(
        source_id=source_value.source_id,
        source_payload_digest=source_value.source_payload_digest,
        consumer=ConversationSourceConsumer.MEMORY,
        processor_fingerprint=fingerprint,
        state=ConversationConsumerOutcomeState.COMMITTED if committed else ConversationConsumerOutcomeState.SKIPPED,
        output_ref=(
            ConsumerOutputRef("memory_output", canonical_digest("o"), canonical_digest("r"), fingerprint)
            if committed
            else None
        ),
        skip_reason=None if committed else "nothing to record",
        completed_at=NOW,
    )
    outcomes.create_first(outcome)
    path = long_path(outcomes._path(source_value.source_id, ConversationSourceConsumer.MEMORY))  # noqa: SLF001
    _rewrite_field(path, field, bad)

    with pytest.raises(ConversationSourceError, match="consumer outcome is corrupt") as caught:
        outcomes.read(source_value.source_id, ConversationSourceConsumer.MEMORY)
    assert isinstance(caught.value.__cause__, ConversationSourceError)
    assert message in str(caught.value.__cause__)
