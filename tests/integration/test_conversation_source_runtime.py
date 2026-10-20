"""Runtime 对 Conversation Source 恢复与双 Consumer 装配的集成验证。"""

import asyncio
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from types import MethodType

import pytest

from habitus.conversation import (
    ConversationConsumerDeliveryState,
    ConversationConsumerOutcomeState,
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


def only_envelope(runtime):
    """这些用例只写过一份源：经生命周期账的标记找到它（信封存储不再提供全量列举）。"""

    conversation = runtime.components.conversation
    (directory,) = conversation.source_ledger.index.address_directories()
    (marker,) = conversation.source_ledger.index.markers_in(directory)
    envelope = conversation.sources.read(marker.source_id)
    assert envelope is not None
    return envelope


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
        assert runtime.components.conversation.sources.max_file_bytes == source_config.max_envelope_bytes
        assert runtime.components.conversation.source_outcomes.max_file_bytes == source_config.max_outcome_bytes
        assert runtime.components.conversation.memory_outputs.max_file_bytes == source_config.max_memory_output_bytes
        assert (
            runtime.components.conversation.behavior_session_outputs.max_file_bytes
            == source_config.max_behavior_session_output_bytes
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
            ConversationSourceConsumer.BEHAVIOR_SESSION,
        )

        await runtime.start()

        assert runtime.components.conversation.source_recovery.pending() == ()
        assert await runtime.read_live_conversation(ConversationAddress(batch.conversation_id, started_on)) == batch
        memory_outcome = runtime.components.conversation.source_outcomes.read(
            envelope.source_id,
            ConversationSourceConsumer.MEMORY,
        )
        behavior_outcome = runtime.components.conversation.source_outcomes.read(
            envelope.source_id,
            ConversationSourceConsumer.BEHAVIOR_SESSION,
        )
        assert memory_outcome is not None
        # 这份配置没有启用行为侧：会话 lane 的消费者照样注册，把这份源落成"跳过"，不留回执。
        assert behavior_outcome is not None
        assert behavior_outcome.state is ConversationConsumerOutcomeState.SKIPPED
        assert behavior_outcome.skip_reason == "BEHAVIOR_DISABLED"
        assert runtime.components.conversation.behavior_session_outputs.list(envelope) == ()
        await runtime.close()

    asyncio.run(scenario())


def _durable_source(conversation_id: str) -> ConversationSourceEnvelope:
    started_on = date(2026, 8, 7)
    batch = ConversationBatch(conversation_id, closed_turn())
    request_digest = conversation_source_request_digest(
        conversation_id=conversation_id,
        started_on=started_on,
        protocol="normalized",
        batch=batch,
        after_turn=False,
        omit_tool_call_ids=frozenset(),
    )
    return ConversationSourceEnvelope.create(
        conversation_id=conversation_id,
        started_on=started_on,
        protocol="normalized",
        batch=batch,
        after_turn=False,
        omit_tool_call_ids=frozenset(),
        delivery_id=request_digest,
        request_digest=request_digest,
        recorded_at=datetime(2026, 8, 7, 1, 0, tzinfo=UTC),
    )


def test_a_broken_memory_source_degrades_only_its_address_at_startup(tmp_path: Path) -> None:
    """裁定 60 第 6 条：一份 Memory 源恢复不成不再整体阻断启动；别的地址照常恢复，出事的那一份留给周期恢复按退避再试。"""

    async def scenario() -> None:
        providers, vectors = runtime_dependencies()
        runtime = build_runtime(
            runtime_config(tmp_path),
            providers=providers,
            vector_stores=vectors,
            path_lock=PathLock(ProcessLocalLockStore()),
        )
        runtime.initialize()
        conversation = runtime.components.conversation
        broken, healthy = _durable_source("startup-broken"), _durable_source("startup-healthy")
        consumer = conversation.source_delivery.consumers[ConversationSourceConsumer.MEMORY]
        original = consumer.execute

        async def failing(_self, envelope, lease):
            if envelope.source_id == broken.source_id:
                raise ConnectionError("vector store unreachable")
            return await original(envelope, lease)

        consumer.execute = MethodType(failing, consumer)  # type: ignore[method-assign]
        conversation.sources.put(broken)
        conversation.sources.put(healthy)

        # 释放与恢复认同一个隔离区（L10.2：组合根原先没接，坏掉的 Behavior 永远挡住地址）
        assert conversation.source_releaser.quarantined == conversation.source_recovery.quarantine.contains

        await runtime.start()

        assert runtime.state is RuntimeState.RUNNING
        delivery = conversation.source_delivery
        assert (
            delivery.inspect(healthy, ConversationSourceConsumer.MEMORY).state
            is ConversationConsumerDeliveryState.COMMITTED
        )
        assert (
            delivery.inspect(broken, ConversationSourceConsumer.MEMORY).state
            is ConversationConsumerDeliveryState.PENDING
        )
        key = f"{broken.source_id}/{ConversationSourceConsumer.MEMORY.value}"
        assert conversation.source_recovery.retries.entry(key) is not None
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
        consumer = runtime.components.conversation.behavior_session_consumer
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
        envelope = only_envelope(runtime)
        assert (
            runtime.components.conversation.source_delivery.inspect(
                envelope,
                ConversationSourceConsumer.BEHAVIOR_SESSION,
            ).state
            is ConversationConsumerDeliveryState.PENDING
        )
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
            envelope = only_envelope(runtime)
            state = runtime.components.conversation.source_delivery.inspect(
                envelope,
                ConversationSourceConsumer.BEHAVIOR_SESSION,
            ).state
            # 这份配置没有启用行为侧，会话 lane 的终态是"跳过"；要验的是它不必等记忆那边。
            if state is ConversationConsumerDeliveryState.SKIPPED:
                break
            await asyncio.sleep(0.01)
        assert state is ConversationConsumerDeliveryState.SKIPPED
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
        consumer = runtime.components.conversation.behavior_session_consumer
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
        envelope = only_envelope(runtime)
        with pytest.raises(RuntimeShutdownTimeoutError) as captured:
            await asyncio.wait_for(runtime.close(), timeout=1.0)
        assert runtime.state is RuntimeState.CLOSING
        assert tuple((item.source_id, item.consumer) for item in captured.value.pending_deliveries) == (
            (envelope.source_id, ConversationSourceConsumer.BEHAVIOR_SESSION),
        )
        assert runtime.components.conversation.source_coordinator.running_task_count == 1
        release.set()
        assert await runtime.components.conversation.source_coordinator.wait_for_idle(1.0)
        await runtime.close()
        assert runtime.state is RuntimeState.CLOSED

    asyncio.run(scenario())


def test_a_corrupt_envelope_does_not_stop_the_runtime_from_starting(tmp_path: Path) -> None:
    """一份信封读不出来：启动照常，别的地址照常恢复（L10.9；原先 ``Runtime.start`` 整体失败）。"""

    async def scenario() -> None:
        providers, vectors = runtime_dependencies()
        runtime = build_runtime(
            runtime_config(tmp_path),
            providers=providers,
            vector_stores=vectors,
            path_lock=PathLock(ProcessLocalLockStore()),
        )
        runtime.initialize()
        conversation = runtime.components.conversation
        bad, good = _durable_source("startup-corrupt"), _durable_source("startup-fine")
        conversation.sources.put(bad)
        conversation.sources.put(good)
        path = conversation.sources._path(bad.source_id)
        path.write_bytes(path.read_bytes()[:-5] + b"xxxxx")

        await runtime.start()

        assert runtime.state is RuntimeState.RUNNING
        memory = ConversationSourceConsumer.MEMORY
        assert conversation.source_delivery.inspect(good, memory).state is ConversationConsumerDeliveryState.COMMITTED
        await runtime.close()

    asyncio.run(scenario())
