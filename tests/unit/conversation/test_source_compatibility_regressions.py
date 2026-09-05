from __future__ import annotations

import asyncio
import json

import pytest

from habitus.conversation.projection import (
    ConversationBehaviorProjectionBatch,
    ConversationBehaviorProjectionItem,
    ConversationBehaviorProjectionKind,
)
from habitus.conversation.source import (
    ConversationConsumerCorruptionError,
    ConversationConsumerDeliveryState,
    ConversationConsumerOutcomeState,
    ConversationSourceConsumer,
    ConversationSourceRecovery,
)
from habitus.memory.workflow import MemoryConversationOutput
from habitus.pre.conversation import ConversationMessageRole
from tests.unit.conversation.source_v2_helpers import NOW, FakeMemoryConsumer, delivery, ingest_result, source


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
        assert service.inspect(
            source_value,
            ConversationSourceConsumer.MEMORY,
        ).state is ConversationConsumerDeliveryState.OUTPUT_READY

        recovery = ConversationSourceRecovery(service.sources, service, batch_size=10)
        results = await recovery.recover_pending()
        memory_results = [item for item in results if item.consumer is ConversationSourceConsumer.MEMORY]
        assert len(memory_results) == 1 and memory_results[0].error is None
        assert memory.calls == 1
        assert service.inspect(
            source_value,
            ConversationSourceConsumer.MEMORY,
        ).state is ConversationConsumerDeliveryState.COMMITTED

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
                ConversationSourceConsumer.BEHAVIOR_PROJECTION: behavior,
            },
            clock=lambda: NOW,
        )
        ensured = await upgraded.ensure_outcome(source_value, ConversationSourceConsumer.MEMORY)
        assert new_memory.calls == 0
        assert ensured.outcome.processor_fingerprint == old_memory.processor_fingerprint

    asyncio.run(scenario())


def test_recovery_adopts_projection_output_without_duplicate_projection(tmp_path) -> None:
    async def scenario() -> None:
        service, _memory, behavior = delivery(tmp_path)
        source_value = service.sources.put(source())
        async with service.fence.acquire(
            source_value,
            ConversationSourceConsumer.BEHAVIOR_PROJECTION,
        ) as lease:
            await behavior.execute(source_value, lease)
        assert behavior.calls == 1
        assert service.inspect(
            source_value,
            ConversationSourceConsumer.BEHAVIOR_PROJECTION,
        ).state is ConversationConsumerDeliveryState.OUTPUT_READY
        recovery = ConversationSourceRecovery(service.sources, service, batch_size=10)
        await recovery.recover_pending()
        assert behavior.calls == 1
        assert service.inspect(
            source_value,
            ConversationSourceConsumer.BEHAVIOR_PROJECTION,
        ).state is ConversationConsumerDeliveryState.COMMITTED

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
        assert service.inspect(
            source_value,
            ConversationSourceConsumer.MEMORY,
        ).state is ConversationConsumerDeliveryState.CORRUPTED
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
        recovery = ConversationSourceRecovery(service.sources, service, batch_size=10)
        result = next(
            item
            for item in await recovery.recover_pending()
            if item.consumer is ConversationSourceConsumer.MEMORY
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
        assert service.inspect(
            source_value,
            ConversationSourceConsumer.MEMORY,
        ).state is ConversationConsumerDeliveryState.CORRUPTED
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
        assert service.inspect(
            source_value,
            ConversationSourceConsumer.MEMORY,
        ).state is ConversationConsumerDeliveryState.CORRUPTED

    asyncio.run(scenario())


def test_skipped_outcome_with_projection_output_is_corrupted(tmp_path) -> None:
    async def scenario() -> None:
        service, _memory, behavior = delivery(tmp_path)
        source_value = service.sources.put(source(role=ConversationMessageRole.COMPLETION))
        skipped = await service.ensure_outcome(
            source_value,
            ConversationSourceConsumer.BEHAVIOR_PROJECTION,
        )
        assert skipped.outcome.state is ConversationConsumerOutcomeState.SKIPPED
        item = ConversationBehaviorProjectionItem.create(
            source_id=source_value.source_id,
            message=source_value.batch.messages[0],
            projection_kind=ConversationBehaviorProjectionKind.USER_CONVERSATION_INPUT,
            payload={"content": "forced-invalid-output"},
        )
        output = ConversationBehaviorProjectionBatch.create(
            source=source_value,
            processor_fingerprint=behavior.processor_fingerprint,
            projector_version=behavior.inner.projector.projector_version,
            items=(item,),
            recorded_at=NOW,
        )
        behavior.output_store.put(source_value, output)
        assert service.inspect(
            source_value,
            ConversationSourceConsumer.BEHAVIOR_PROJECTION,
        ).state is ConversationConsumerDeliveryState.CORRUPTED

    asyncio.run(scenario())


def test_recovery_only_runs_missing_behavior_when_memory_is_committed(tmp_path) -> None:
    async def scenario() -> None:
        service, memory, behavior = delivery(tmp_path)
        source_value = service.sources.put(source())
        await service.ensure_outcome(source_value, ConversationSourceConsumer.MEMORY)
        memory_calls = memory.calls
        recovery = ConversationSourceRecovery(service.sources, service, batch_size=10)
        results = await recovery.recover_pending()
        assert [item.consumer for item in results] == [ConversationSourceConsumer.BEHAVIOR_PROJECTION]
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
