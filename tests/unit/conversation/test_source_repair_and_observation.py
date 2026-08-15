from __future__ import annotations

import asyncio

import pytest

from conversation.source import (
    ConversationConsumerDeliveryState,
    ConversationOrderedPredecessorPendingError,
    ConversationSourceConsumer,
    ConversationSourceOutputRepair,
    ConversationSourceRepairError,
)
from foundation.observability import ObservationStatus
from tests.unit.conversation.source_v2_helpers import (
    FakeMemoryConsumer,
    RecordingObserver,
    delivery,
    source,
)


async def _write_orphan_output(service, consumer, source_value):
    """绕过交付层直接写一份没有 Outcome 的 Output，模拟被外力放入的产物。"""

    async with service.fence.acquire(
        source_value,
        consumer.consumer,
        conversation_ordered=consumer.ordered_within_conversation,
    ) as lease:
        return await consumer.execute(source_value, lease)


def test_repair_keeps_the_output_matching_the_current_processor(tmp_path) -> None:
    async def scenario() -> None:
        service, current_memory, _behavior = delivery(tmp_path)
        source_value = service.sources.put(source())
        stale_memory = FakeMemoryConsumer(
            current_memory.output_store,
            fingerprint_seed="retired-processor",
        )
        await _write_orphan_output(service, stale_memory, source_value)
        await _write_orphan_output(service, current_memory, source_value)

        outputs = current_memory.output_store
        assert len(outputs.list(source_value)) == 2
        state = service.inspect(source_value, ConversationSourceConsumer.MEMORY)
        assert state.state is ConversationConsumerDeliveryState.CORRUPTED

        result = ConversationSourceOutputRepair(service).repair(
            source_value,
            ConversationSourceConsumer.MEMORY,
        )
        expected = outputs.expected_output_id(source_value, current_memory.processor_fingerprint)
        assert result.retained_output_id == expected
        assert len(result.removed_output_ids) == 1

        remaining = outputs.list(source_value)
        assert len(remaining) == 1
        assert outputs.ref(remaining[0]).output_id == expected
        recovered = service.inspect(source_value, ConversationSourceConsumer.MEMORY)
        assert recovered.state is ConversationConsumerDeliveryState.OUTPUT_READY

        # 修复只保留既有产物，绝不让 Consumer 重新执行。
        calls_before = current_memory.calls
        await service.ensure_outcome(source_value, ConversationSourceConsumer.MEMORY)
        assert current_memory.calls == calls_before

    asyncio.run(scenario())


def test_repair_refuses_when_no_output_matches_the_current_processor(tmp_path) -> None:
    async def scenario() -> None:
        service, current_memory, _behavior = delivery(tmp_path)
        source_value = service.sources.put(source())
        for seed in ("retired-a", "retired-b"):
            await _write_orphan_output(
                service,
                FakeMemoryConsumer(current_memory.output_store, fingerprint_seed=seed),
                source_value,
            )
        with pytest.raises(ConversationSourceRepairError):
            ConversationSourceOutputRepair(service).repair(
                source_value,
                ConversationSourceConsumer.MEMORY,
            )
        # 拒绝修复时一个文件都不能被删除。
        assert len(current_memory.output_store.list(source_value)) == 2

    asyncio.run(scenario())


def test_repair_refuses_a_single_output_and_a_committed_outcome(tmp_path) -> None:
    async def scenario() -> None:
        service, memory, _behavior = delivery(tmp_path)
        source_value = service.sources.put(source())
        repair = ConversationSourceOutputRepair(service)
        await _write_orphan_output(service, memory, source_value)
        with pytest.raises(ConversationSourceRepairError):
            repair.repair(source_value, ConversationSourceConsumer.MEMORY)
        await service.ensure_outcome(source_value, ConversationSourceConsumer.MEMORY)
        with pytest.raises(ConversationSourceRepairError):
            repair.repair(source_value, ConversationSourceConsumer.MEMORY)

    asyncio.run(scenario())


def test_delivery_observes_every_consumer_terminal(tmp_path) -> None:
    async def scenario() -> None:
        observer = RecordingObserver()
        service, _memory, _behavior = delivery(tmp_path, observer=observer)
        source_value = service.sources.put(source())
        for consumer in ConversationSourceConsumer:
            await service.ensure_outcome(source_value, consumer)
        observed = {
            event.attributes["consumer"]: event
            for event in observer.events
            if event.operation == "consumer_delivery"
        }
        assert set(observed) == {consumer.value for consumer in ConversationSourceConsumer}
        for event in observed.values():
            assert event.category == "conversation.source"
            assert event.status is ObservationStatus.SUCCESS
            assert event.attributes["source_id"] == source_value.source_id
            assert event.attributes["outcome_state"] == "committed"

    asyncio.run(scenario())


def test_delivery_observes_behavior_failure_that_nobody_awaits(tmp_path) -> None:
    async def scenario() -> None:
        observer = RecordingObserver()
        service, _memory, behavior = delivery(tmp_path, observer=observer)
        behavior.fail = RuntimeError("projection backend is down")
        source_value = service.sources.put(source())
        with pytest.raises(RuntimeError):
            await service.ensure_outcome(
                source_value,
                ConversationSourceConsumer.BEHAVIOR_PROJECTION,
            )
        failures = [
            event
            for event in observer.events
            if event.status is ObservationStatus.FAILURE
            and event.attributes["consumer"] == ConversationSourceConsumer.BEHAVIOR_PROJECTION.value
        ]
        assert len(failures) == 1
        assert failures[0].attributes["error_type"] == "RuntimeError"
        assert failures[0].attributes["source_id"] == source_value.source_id

    asyncio.run(scenario())


def test_ordered_predecessor_wait_is_degraded_not_failure(tmp_path) -> None:
    async def scenario() -> None:
        observer = RecordingObserver()
        service, memory, _behavior = delivery(tmp_path, observer=observer)
        earlier = service.sources.put(source(sequence=0, delivery_seed="delivery-earlier"))
        later = service.sources.put(source(sequence=1, delivery_seed="delivery-later"))
        memory.fail = RuntimeError("earlier source cannot finish")
        with pytest.raises(RuntimeError):
            await service.ensure_outcome(earlier, ConversationSourceConsumer.MEMORY)
        with pytest.raises(ConversationOrderedPredecessorPendingError):
            await service.ensure_outcome(later, ConversationSourceConsumer.MEMORY)
        statuses = [
            event.status
            for event in observer.events
            if event.attributes["consumer"] == ConversationSourceConsumer.MEMORY.value
        ]
        # 前序未完成是排队，不是损坏：两者不能塌缩成同一个状态。
        assert ObservationStatus.DEGRADED in statuses
        assert ObservationStatus.FAILURE in statuses

    asyncio.run(scenario())
