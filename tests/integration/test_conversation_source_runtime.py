"""Runtime 对 Conversation Source 恢复与双 Consumer 装配的集成验证。"""

import asyncio
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from types import MethodType

import pytest

from habitus.conversation import (
    ConversationConsumerDeliveryState,
    ConversationSourceConsumer,
    ConversationSourceEnvelope,
    conversation_source_request_digest,
)
from habitus.infrastructure.store.contracts import PathLock
from habitus.infrastructure.store.locks import ProcessLocalLockStore
from habitus.memory.conversation import ConversationAddress
from habitus.pre.conversation import ConversationBatch
from habitus.runtime import RuntimeShutdownTimeoutError, RuntimeState, build_runtime
from tests.helpers import closed_turn
from tests.integration.test_runtime_assembly import runtime_config, runtime_dependencies


def test_runtime_start_recovers_durable_source_with_only_missing_outcomes(tmp_path: Path) -> None:
    async def scenario() -> None:
        providers, vectors = runtime_dependencies()
        runtime = build_runtime(
            runtime_config(tmp_path),
            providers=providers,
            vector_stores=vectors,
            path_lock=PathLock(ProcessLocalLockStore()),
        )
        runtime.initialize()
        source_config = runtime.config.conversation.source
        projection_config = runtime.config.conversation.behavior_projection
        assert runtime.components.conversation.sources.max_file_bytes == source_config.max_envelope_bytes
        assert runtime.components.conversation.source_outcomes.max_file_bytes == source_config.max_outcome_bytes
        assert runtime.components.conversation.memory_outputs.max_file_bytes == source_config.max_memory_output_bytes
        assert (
            runtime.components.conversation.behavior_projections.max_file_bytes
            == projection_config.max_projection_output_bytes
        )
        started_on = date(2026, 8, 7)
        batch = ConversationBatch("source-recovery", closed_turn())
        request_digest = conversation_source_request_digest(
            conversation_id=batch.conversation_id,
            started_on=started_on,
            protocol="normalized",
            batch=batch,
            after_turn=False,
            omit_tool_call_ids=frozenset(),
        )
        envelope = ConversationSourceEnvelope.create(
            conversation_id=batch.conversation_id,
            started_on=started_on,
            protocol="normalized",
            batch=batch,
            after_turn=False,
            omit_tool_call_ids=frozenset(),
            delivery_id=request_digest,
            request_digest=request_digest,
            recorded_at=datetime(2026, 8, 7, 1, 0, tzinfo=UTC),
        )
        runtime.components.conversation.sources.put(envelope)
        assert tuple(entry.consumer for entry in runtime.components.conversation.source_recovery.pending()) == (
            ConversationSourceConsumer.MEMORY,
            ConversationSourceConsumer.BEHAVIOR_PROJECTION,
        )

        await runtime.start()

        assert runtime.components.conversation.source_recovery.pending() == ()
        assert await runtime.read_live_conversation(ConversationAddress(batch.conversation_id, started_on)) == batch
        memory_outcome = runtime.components.conversation.source_outcomes.read(
            envelope.source_id,
            ConversationSourceConsumer.MEMORY,
        )
        projection_outcome = runtime.components.conversation.source_outcomes.read(
            envelope.source_id,
            ConversationSourceConsumer.BEHAVIOR_PROJECTION,
        )
        assert memory_outcome is not None
        assert projection_outcome is not None
        assert projection_outcome.output_ref is not None
        projection = runtime.components.conversation.behavior_projections.read(
            envelope,
            projection_outcome.output_ref.output_id,
        )
        assert projection is not None and projection.source_id == envelope.source_id
        await runtime.close()

    asyncio.run(scenario())


def test_runtime_append_returns_after_memory_without_waiting_for_blocked_behavior(tmp_path: Path) -> None:
    async def scenario() -> None:
        providers, vectors = runtime_dependencies()
        runtime = build_runtime(
            runtime_config(tmp_path),
            providers=providers,
            vector_stores=vectors,
            path_lock=PathLock(ProcessLocalLockStore()),
        )
        runtime.initialize()
        consumer = runtime.components.conversation.behavior_projection_consumer
        original = consumer.execute
        entered = asyncio.Event()
        release = asyncio.Event()

        async def blocked(_self, envelope, lease):
            entered.set()
            await release.wait()
            return await original(envelope, lease)

        consumer.execute = MethodType(blocked, consumer)  # type: ignore[method-assign]
        address = ConversationAddress("runtime-behavior-blocked", date(2026, 8, 8))
        batch = ConversationBatch(address.conversation_id, closed_turn())
        append = asyncio.create_task(runtime.append_conversation(address, batch))
        await entered.wait()
        memory = await asyncio.wait_for(append, timeout=1.0)
        assert memory.append.next_sequence == batch.end_sequence + 1
        envelope = runtime.components.conversation.sources.list()[0]
        assert runtime.components.conversation.source_delivery.inspect(
            envelope,
            ConversationSourceConsumer.BEHAVIOR_PROJECTION,
        ).state is ConversationConsumerDeliveryState.PENDING
        release.set()
        assert await runtime.components.conversation.source_coordinator.wait_for_idle(1.0)
        await runtime.close()

    asyncio.run(scenario())


def test_runtime_memory_wait_can_remain_blocked_while_behavior_finishes(tmp_path: Path) -> None:
    async def scenario() -> None:
        providers, vectors = runtime_dependencies()
        runtime = build_runtime(
            runtime_config(tmp_path),
            providers=providers,
            vector_stores=vectors,
            path_lock=PathLock(ProcessLocalLockStore()),
        )
        runtime.initialize()
        consumer = runtime.components.workflow.conversation_consumer
        original = consumer.execute
        entered = asyncio.Event()
        release = asyncio.Event()

        async def blocked(_self, envelope, lease):
            entered.set()
            await release.wait()
            return await original(envelope, lease)

        consumer.execute = MethodType(blocked, consumer)  # type: ignore[method-assign]
        address = ConversationAddress("runtime-memory-blocked", date(2026, 8, 8))
        batch = ConversationBatch(address.conversation_id, closed_turn())
        append = asyncio.create_task(runtime.append_conversation(address, batch))
        await entered.wait()
        for _attempt in range(100):
            envelope = runtime.components.conversation.sources.list()[0]
            state = runtime.components.conversation.source_delivery.inspect(
                envelope,
                ConversationSourceConsumer.BEHAVIOR_PROJECTION,
            ).state
            if state is ConversationConsumerDeliveryState.COMMITTED:
                break
            await asyncio.sleep(0.01)
        assert state is ConversationConsumerDeliveryState.COMMITTED
        assert not append.done()
        release.set()
        await append
        await runtime.close()

    asyncio.run(scenario())


def test_runtime_close_timeout_is_explicit_and_does_not_cancel_behavior_consumer(tmp_path: Path) -> None:
    async def scenario() -> None:
        providers, vectors = runtime_dependencies()
        config = runtime_config(tmp_path)
        config = replace(
            config,
            conversation=replace(
                config.conversation,
                source=replace(config.conversation.source, shutdown_timeout_seconds=0.05),
            ),
        )
        runtime = build_runtime(
            config,
            providers=providers,
            vector_stores=vectors,
            path_lock=PathLock(ProcessLocalLockStore()),
        )
        runtime.initialize()
        consumer = runtime.components.conversation.behavior_projection_consumer
        original = consumer.execute
        entered = asyncio.Event()
        release = asyncio.Event()

        async def blocked(_self, envelope, lease):
            entered.set()
            await release.wait()
            return await original(envelope, lease)

        consumer.execute = MethodType(blocked, consumer)  # type: ignore[method-assign]
        address = ConversationAddress("runtime-close-blocked", date(2026, 8, 8))
        batch = ConversationBatch(address.conversation_id, closed_turn())
        await runtime.append_conversation(address, batch)
        await entered.wait()
        envelope = runtime.components.conversation.sources.list()[0]
        with pytest.raises(RuntimeShutdownTimeoutError) as captured:
            await asyncio.wait_for(runtime.close(), timeout=1.0)
        assert runtime.state is RuntimeState.CLOSING
        assert tuple(
            (item.source_id, item.consumer) for item in captured.value.pending_deliveries
        ) == ((envelope.source_id, ConversationSourceConsumer.BEHAVIOR_PROJECTION),)
        assert runtime.components.conversation.source_coordinator.running_task_count == 1
        release.set()
        assert await runtime.components.conversation.source_coordinator.wait_for_idle(1.0)
        await runtime.close()
        assert runtime.state is RuntimeState.CLOSED

    asyncio.run(scenario())
