from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, date, datetime
from pathlib import Path

from conversation.projection import (
    ConversationBehaviorProjectionConsumer,
    ConversationBehaviorProjectionStore,
    ConversationBehaviorProjector,
)
from conversation.source import (
    ConversationConsumerDelivery,
    ConversationConsumerExecutionFence,
    ConversationConsumerOutcomeStore,
    ConversationConsumerRunDisposition,
    ConversationConsumerRunResult,
    ConversationConsumerStateInspector,
    ConversationSourceConsumer,
    ConversationSourceEnvelope,
    ConversationSourceStore,
    conversation_source_request_digest,
)
from conversation.source.fence import ConversationConsumerExecutionLease
from foundation.integrity import canonical_digest
from foundation.observability import ObservationEvent, Observer
from infrastructure.store.contracts import PathLock
from infrastructure.store.locks import ProcessLocalLockStore
from memory.conversation import ConversationAppendResult, ConversationAppendStatus, ConversationRetentionPlan
from memory.workflow import ConversationMemoryIngestResult, MemoryConversationOutput, MemoryConversationOutputStore
from pre.conversation import ConversationBatch, ConversationMessage, ConversationMessageRole

NOW = datetime(2026, 8, 8, 1, 0, tzinfo=UTC)
STARTED_ON = date(2026, 8, 8)


def message(
    sequence: int = 0,
    *,
    role: ConversationMessageRole = ConversationMessageRole.PROMPT,
    content: str = "hello",
    conversation_id: str = "conversation-source-v2",
) -> ConversationMessage:
    return ConversationMessage(
        message_id=canonical_digest({"conversation_id": conversation_id, "sequence": sequence, "role": role.value}),
        sequence=sequence,
        role=role,
        content=content,
        occurred_at=NOW,
    )


def source(
    *,
    conversation_id: str = "conversation-source-v2",
    sequence: int = 0,
    delivery_seed: str = "delivery-a",
    role: ConversationMessageRole = ConversationMessageRole.PROMPT,
    content: str = "hello",
    recorded_at: datetime = NOW,
) -> ConversationSourceEnvelope:
    batch = ConversationBatch(
        conversation_id,
        (message(sequence, role=role, content=content, conversation_id=conversation_id),),
    )
    request_digest = conversation_source_request_digest(
        conversation_id=conversation_id,
        started_on=STARTED_ON,
        protocol="normalized",
        batch=batch,
        after_turn=False,
        omit_tool_call_ids=frozenset(),
    )
    return ConversationSourceEnvelope.create(
        conversation_id=conversation_id,
        started_on=STARTED_ON,
        protocol="normalized",
        batch=batch,
        after_turn=False,
        omit_tool_call_ids=frozenset(),
        delivery_id=canonical_digest(delivery_seed),
        request_digest=request_digest,
        recorded_at=recorded_at,
    )


def ingest_result(source_value: ConversationSourceEnvelope) -> ConversationMemoryIngestResult:
    live = source_value.batch
    return ConversationMemoryIngestResult(
        append=ConversationAppendResult(
            status=ConversationAppendStatus.CREATED,
            appended_count=len(live.messages),
            live=live,
            next_sequence=live.end_sequence + 1,
        ),
        jobs=(),
        retention=ConversationRetentionPlan(
            through_sequence=None,
            archive_messages=(),
            retained_messages=live.messages,
            triggered=False,
            flush=False,
            pending_tokens=1,
            budget_exceeded=False,
            reason="below threshold",
        ),
    )


class FakeMemoryConsumer:
    consumer = ConversationSourceConsumer.MEMORY
    ordered_within_conversation = True

    def __init__(
        self,
        output_store: MemoryConversationOutputStore,
        *,
        fingerprint_seed: str = "memory-processor-a",
        entered: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
        fail: BaseException | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.output_store = output_store
        self.processor_fingerprint = canonical_digest(fingerprint_seed)
        self.entered = entered
        self.release = release
        self.fail = fail
        self.clock = clock or (lambda: NOW)
        self.calls = 0

    async def execute(
        self,
        envelope: ConversationSourceEnvelope,
        lease: ConversationConsumerExecutionLease,
    ) -> ConversationConsumerRunResult:
        self.calls += 1
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            await self.release.wait()
        if self.fail is not None:
            raise self.fail
        result = ingest_result(envelope)
        output = MemoryConversationOutput.create(
            source=envelope,
            processor_fingerprint=self.processor_fingerprint,
            ingest_result=result,
            recorded_at=self.clock(),
        )
        stored = await lease.run_fenced(lambda: self.output_store.put(envelope, output))
        return ConversationConsumerRunResult(
            disposition=ConversationConsumerRunDisposition.OUTPUT_WRITTEN,
            output_ref=self.output_store.ref(stored),
            skip_reason=None,
            runtime_result=result,
        )


class WrappedBehaviorConsumer:
    consumer = ConversationSourceConsumer.BEHAVIOR_PROJECTION
    ordered_within_conversation = False

    def __init__(
        self,
        inner: ConversationBehaviorProjectionConsumer,
        *,
        entered: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
        fail: BaseException | None = None,
    ) -> None:
        self.inner = inner
        self.output_store = inner.output_store
        self.processor_fingerprint = inner.processor_fingerprint
        self.entered = entered
        self.release = release
        self.fail = fail
        self.calls = 0

    async def execute(
        self,
        envelope: ConversationSourceEnvelope,
        lease: ConversationConsumerExecutionLease,
    ) -> ConversationConsumerRunResult:
        self.calls += 1
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            await self.release.wait()
        if self.fail is not None:
            raise self.fail
        return await self.inner.execute(envelope, lease)


def stores(root: Path):
    sources = ConversationSourceStore(root, max_files=100, max_file_bytes=2_000_000)
    outcomes = ConversationConsumerOutcomeStore(root, max_file_bytes=100_000)
    memory_outputs = MemoryConversationOutputStore(
        root,
        max_files_per_source=4,
        max_file_bytes=2_000_000,
    )
    projection_outputs = ConversationBehaviorProjectionStore(
        root,
        max_files_per_source=4,
        max_file_bytes=2_000_000,
        max_items=1_000,
    )
    return sources, outcomes, memory_outputs, projection_outputs


class RecordingObserver:
    """收集交付层观察事件，供断言使用。"""

    def __init__(self) -> None:
        self.events: list[ObservationEvent] = []

    def record(self, event: ObservationEvent) -> None:
        self.events.append(event)


def delivery(
    root: Path,
    *,
    memory: FakeMemoryConsumer | None = None,
    behavior: WrappedBehaviorConsumer | None = None,
    path_lock: PathLock | None = None,
    observer: Observer | None = None,
):
    sources, outcomes, memory_outputs, projection_outputs = stores(root)
    memory_consumer = memory or FakeMemoryConsumer(memory_outputs)
    behavior_consumer = behavior or WrappedBehaviorConsumer(
        ConversationBehaviorProjectionConsumer(
            ConversationBehaviorProjector(),
            projection_outputs,
        )
    )
    inspector = ConversationConsumerStateInspector(outcomes)
    fence = ConversationConsumerExecutionFence(
        path_lock or PathLock(ProcessLocalLockStore()),
        ttl_seconds=3,
        heartbeat_interval_seconds=0.05,
        wait_seconds=2.0,
    )
    service = ConversationConsumerDelivery(
        sources,
        outcomes,
        inspector,
        fence,
        {
            ConversationSourceConsumer.MEMORY: memory_consumer,
            ConversationSourceConsumer.BEHAVIOR_PROJECTION: behavior_consumer,
        },
        clock=lambda: NOW,
        observer=observer,
    )
    return service, memory_consumer, behavior_consumer
