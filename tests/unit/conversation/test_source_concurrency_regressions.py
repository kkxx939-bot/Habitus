"""Conversation Source 并发重放与部分恢复的回归测试。"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from conversation import (
    ConversationBehaviorProjectionConsumer,
    ConversationBehaviorProjectionStore,
    ConversationBehaviorProjector,
    ConversationConsumerExecution,
    ConversationConsumerReceiptState,
    ConversationSourceConsumer,
    ConversationSourceCoordinator,
    ConversationSourceEnvelope,
    ConversationSourceReceiptStore,
    ConversationSourceRecovery,
    ConversationSourceStore,
    conversation_source_request_digest,
)
from foundation.integrity import canonical_digest
from infrastructure.store.contracts import PathLock
from infrastructure.store.locks import ProcessLocalLockStore
from memory.conversation import (
    ConversationAddress,
    ConversationMessageJournal,
    ConversationRetentionPlanner,
    ConversationSegmentationConfig,
    ConversationSemanticBoundaryScorer,
)
from memory.workflow import ConversationMemoryEnqueuer, MemoryConversationConsumer, MemoryJobStore
from ModelClient import EmbeddingVector
from pre.conversation import ConversationBatch
from tests.helpers import closed_turn

STARTED_ON = date(2026, 8, 8)


def _envelope(*, after_turn: bool = False) -> ConversationSourceEnvelope:
    batch = ConversationBatch(
        "source-concurrency",
        (
            *closed_turn(prompt="first request"),
            *closed_turn(start_sequence=2, prompt="second request"),
        ),
    )
    request_digest = conversation_source_request_digest(
        conversation_id=batch.conversation_id,
        started_on=STARTED_ON,
        protocol="normalized",
        batch=batch,
        after_turn=after_turn,
        omit_tool_call_ids=frozenset(),
    )
    return ConversationSourceEnvelope.create(
        conversation_id=batch.conversation_id,
        started_on=STARTED_ON,
        protocol="normalized",
        batch=batch,
        after_turn=after_turn,
        omit_tool_call_ids=frozenset(),
        delivery_id=request_digest,
        request_digest=request_digest,
        created_at=datetime(2026, 8, 8, 1, 0, tzinfo=timezone.utc),
    )


class _DeterministicEmbedder:
    async def embed_documents(self, texts: tuple[str, ...]) -> tuple[EmbeddingVector, ...]:
        return tuple(EmbeddingVector((1.0, 0.0)) for _ in texts)


class _BarrierMemoryConsumer:
    """让两个 Coordinator 都在 Memory Receipt 缺失时进入真实 Consumer。"""

    consumer = ConversationSourceConsumer.MEMORY

    def __init__(self, inner: MemoryConversationConsumer) -> None:
        self.inner = inner
        self.arrivals = 0
        self.both_arrived = asyncio.Event()

    async def consume(self, envelope: ConversationSourceEnvelope) -> ConversationConsumerExecution:
        self.arrivals += 1
        if self.arrivals == 2:
            self.both_arrived.set()
        await self.both_arrived.wait()
        return await self.inner.consume(envelope)

    async def completed(self, envelope: ConversationSourceEnvelope, receipt: object) -> object:
        return await self.inner.completed(envelope, receipt)  # type: ignore[arg-type]


class _SuccessfulConsumer:
    def __init__(self, consumer: ConversationSourceConsumer) -> None:
        self.consumer = consumer
        self.consume_count = 0

    async def consume(self, envelope: ConversationSourceEnvelope) -> ConversationConsumerExecution:
        self.consume_count += 1
        result = {"consumer": self.consumer.value, "source_id": envelope.source_id}
        return ConversationConsumerExecution(
            ConversationConsumerReceiptState.SUCCEEDED,
            result,
            canonical_digest({"result": result}),
            canonical_digest(result),
        )

    async def completed(self, envelope: ConversationSourceEnvelope, _receipt: object) -> object:
        return {"consumer": self.consumer.value, "source_id": envelope.source_id, "restored": True}


class _FailingConsumer(_SuccessfulConsumer):
    async def consume(self, envelope: ConversationSourceEnvelope) -> ConversationConsumerExecution:
        self.consume_count += 1
        raise RuntimeError(f"{self.consumer.value} failed for {envelope.source_id}")


class _BlockingCompletedConsumer(_SuccessfulConsumer):
    def __init__(self, consumer: ConversationSourceConsumer, release: asyncio.Event) -> None:
        super().__init__(consumer)
        self.release = release
        self.completed_started = asyncio.Event()

    async def consume(self, _envelope: ConversationSourceEnvelope) -> ConversationConsumerExecution:
        raise AssertionError("terminal consumer must not be consumed again")

    async def completed(self, envelope: ConversationSourceEnvelope, _receipt: object) -> object:
        self.completed_started.set()
        await self.release.wait()
        return {"consumer": self.consumer.value, "source_id": envelope.source_id, "restored": True}


class _SignalingConsumer(_SuccessfulConsumer):
    def __init__(self, consumer: ConversationSourceConsumer, finished: asyncio.Event) -> None:
        super().__init__(consumer)
        self.finished = finished

    async def consume(self, envelope: ConversationSourceEnvelope) -> ConversationConsumerExecution:
        result = await super().consume(envelope)
        self.finished.set()
        return result


def _stores(
    root: Path,
) -> tuple[ConversationSourceStore, ConversationSourceReceiptStore]:
    return (
        ConversationSourceStore(root, max_entries=100, max_file_bytes=2_000_000),
        ConversationSourceReceiptStore(root, max_file_bytes=100_000),
    )


def test_same_source_concurrent_dispatch_is_idempotent_across_real_memory_consumer(tmp_path: Path) -> None:
    async def scenario() -> None:
        conversation_root = tmp_path / "conversation"
        path_lock = PathLock(ProcessLocalLockStore())
        journal = ConversationMessageJournal(conversation_root, path_lock)
        jobs = MemoryJobStore(tmp_path / "workflow", path_lock, memory_root=tmp_path / "memory")
        planner = ConversationRetentionPlanner(
            ConversationSegmentationConfig(
                commit_token_threshold=1,
                keep_recent_turn_count=1,
                retained_message_token_budget=100,
                max_segment_tokens=100,
                max_live_messages=1_000,
                max_live_bytes=1_000_000,
                max_segment_messages=1_000,
                max_segment_bytes=1_000_000,
            ),
            token_estimator=lambda item: len(item.content) if isinstance(item.content, str) else 1,
        )
        real_memory = MemoryConversationConsumer(
            ConversationMemoryEnqueuer(journal, jobs, planner),
            journal,
            ConversationSemanticBoundaryScorer(
                _DeterministicEmbedder(),
                embedding_fingerprint="deterministic-v1",
                max_unit_chars=256,
            ),
        )
        memory = _BarrierMemoryConsumer(real_memory)
        sources, receipts = _stores(conversation_root)
        projection_store = ConversationBehaviorProjectionStore(
            conversation_root,
            max_file_bytes=2_000_000,
        )
        behavior = ConversationBehaviorProjectionConsumer(
            ConversationBehaviorProjector(),
            projection_store,
        )
        first = ConversationSourceCoordinator(sources, receipts, memory, behavior)
        second = ConversationSourceCoordinator(sources, receipts, memory, behavior)
        envelope = _envelope(after_turn=True)

        first_result, second_result = await asyncio.gather(
            first.dispatch(envelope),
            second.dispatch(envelope),
        )

        assert memory.arrivals == 2
        assert first_result.memory_error is None
        assert second_result.memory_error is None
        address = ConversationAddress(envelope.conversation_id, envelope.started_on)
        history = journal.list_history(address)
        live = journal.read_live(address)
        persisted_messages = tuple(message for segment in history for message in segment.messages) + (
            () if live is None else live.messages
        )
        assert len(persisted_messages) == len(envelope.batch.messages)
        persisted_jobs = jobs.list_for_conversation(address)
        assert persisted_jobs
        assert len({job.source_identity for job in persisted_jobs}) == len(persisted_jobs)
        memory_receipt = receipts.read(envelope.source_id, ConversationSourceConsumer.MEMORY)
        projection_receipt = receipts.read(
            envelope.source_id,
            ConversationSourceConsumer.BEHAVIOR_PROJECTION,
        )
        assert memory_receipt is not None
        assert projection_receipt is not None
        assert projection_receipt.result_id is not None
        assert projection_store.read(projection_receipt.result_id) is not None
        assert len(tuple(projection_store.projection_root.glob("*.json"))) == 1

    asyncio.run(scenario())


@pytest.mark.parametrize(
    "terminal_consumer",
    [
        ConversationSourceConsumer.MEMORY,
        ConversationSourceConsumer.BEHAVIOR_PROJECTION,
    ],
)
def test_partial_recovery_does_not_wait_for_terminal_consumer_restoration(
    tmp_path: Path,
    terminal_consumer: ConversationSourceConsumer,
) -> None:
    async def scenario() -> None:
        sources, receipts = _stores(tmp_path)
        envelope = _envelope()
        missing_consumer = (
            ConversationSourceConsumer.BEHAVIOR_PROJECTION
            if terminal_consumer is ConversationSourceConsumer.MEMORY
            else ConversationSourceConsumer.MEMORY
        )
        initial = {
            terminal_consumer: _SuccessfulConsumer(terminal_consumer),
            missing_consumer: _FailingConsumer(missing_consumer),
        }
        first = ConversationSourceCoordinator(
            sources,
            receipts,
            initial[ConversationSourceConsumer.MEMORY],
            initial[ConversationSourceConsumer.BEHAVIOR_PROJECTION],
        )
        await first.dispatch(envelope)
        assert receipts.read(envelope.source_id, terminal_consumer) is not None
        assert receipts.read(envelope.source_id, missing_consumer) is None

        release_terminal = asyncio.Event()
        missing_finished = asyncio.Event()
        recovered = {
            terminal_consumer: _BlockingCompletedConsumer(terminal_consumer, release_terminal),
            missing_consumer: _SignalingConsumer(missing_consumer, missing_finished),
        }
        coordinator = ConversationSourceCoordinator(
            sources,
            receipts,
            recovered[ConversationSourceConsumer.MEMORY],
            recovered[ConversationSourceConsumer.BEHAVIOR_PROJECTION],
        )
        recovery = ConversationSourceRecovery(sources, receipts, coordinator)
        running = asyncio.create_task(recovery.recover_pending())
        blocking = recovered[terminal_consumer]
        assert isinstance(blocking, _BlockingCompletedConsumer)
        await asyncio.wait_for(blocking.completed_started.wait(), timeout=1.0)
        try:
            await asyncio.wait_for(missing_finished.wait(), timeout=0.2)
            missing_completed_before_release = True
        except TimeoutError:
            missing_completed_before_release = False
        finally:
            release_terminal.set()
        results = await running

        assert missing_completed_before_release
        assert len(results) == 1
        assert receipts.read(envelope.source_id, missing_consumer) is not None
        assert results[0].memory_error is None
        assert results[0].behavior_projection_error is None

    asyncio.run(scenario())


def test_blocked_behavior_consumer_does_not_delay_memory_receipt(tmp_path: Path) -> None:
    async def scenario() -> None:
        sources, receipts = _stores(tmp_path)
        envelope = _envelope()
        memory_finished = asyncio.Event()
        behavior_started = asyncio.Event()
        release_behavior = asyncio.Event()

        class BlockingBehavior(_SuccessfulConsumer):
            async def consume(self, value: ConversationSourceEnvelope) -> ConversationConsumerExecution:
                behavior_started.set()
                await release_behavior.wait()
                return await super().consume(value)

        memory = _SignalingConsumer(ConversationSourceConsumer.MEMORY, memory_finished)
        behavior = BlockingBehavior(ConversationSourceConsumer.BEHAVIOR_PROJECTION)
        coordinator = ConversationSourceCoordinator(sources, receipts, memory, behavior)
        running = asyncio.create_task(coordinator.dispatch(envelope))

        await asyncio.wait_for(behavior_started.wait(), timeout=1.0)
        await asyncio.wait_for(memory_finished.wait(), timeout=1.0)
        assert not running.done()
        assert receipts.read(envelope.source_id, ConversationSourceConsumer.MEMORY) is not None
        assert receipts.read(envelope.source_id, ConversationSourceConsumer.BEHAVIOR_PROJECTION) is None
        release_behavior.set()
        result = await running

        assert result.memory_error is None
        assert result.behavior_projection_error is None
        assert receipts.read(
            envelope.source_id,
            ConversationSourceConsumer.BEHAVIOR_PROJECTION,
        ) is not None

    asyncio.run(scenario())
