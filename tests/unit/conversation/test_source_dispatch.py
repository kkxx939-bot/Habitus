"""Conversation Source 双 Consumer 分发、投影、隔离与恢复。"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from conversation import (
    ConversationBehaviorProjectionBatch,
    ConversationBehaviorProjectionConsumer,
    ConversationBehaviorProjectionKind,
    ConversationBehaviorProjectionStore,
    ConversationBehaviorProjector,
    ConversationConsumerExecution,
    ConversationConsumerReceiptState,
    ConversationSourceConsumer,
    ConversationSourceCoordinator,
    ConversationSourceEnvelope,
    ConversationSourceError,
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
    ConversationIngressRequest,
    ConversationMessageJournal,
    ConversationRetentionPlanner,
    ConversationSegmentationConfig,
    ConversationSemanticBoundaryScorer,
)
from memory.workflow import ConversationMemoryEnqueuer, MemoryConversationConsumer, MemoryJobStore
from ModelClient import EmbeddingVector
from pre.conversation import ConversationBatch, ConversationMessageRole
from tests.helpers import BASE_TIME, closed_turn, message, tool_turn

STARTED_ON = date(2026, 8, 7)


def source_envelope(
    batch: ConversationBatch | None = None,
    *,
    delivery_id: str = "a" * 64,
    request_digest: str | None = None,
    after_turn: bool = False,
) -> ConversationSourceEnvelope:
    resolved_batch = batch or ConversationBatch("conversation-1", closed_turn())
    digest = request_digest or conversation_source_request_digest(
        conversation_id=resolved_batch.conversation_id,
        started_on=STARTED_ON,
        protocol="normalized",
        batch=resolved_batch,
        after_turn=after_turn,
        omit_tool_call_ids=frozenset(),
    )
    return ConversationSourceEnvelope.create(
        conversation_id=resolved_batch.conversation_id,
        started_on=STARTED_ON,
        protocol="normalized",
        batch=resolved_batch,
        after_turn=after_turn,
        omit_tool_call_ids=frozenset(),
        delivery_id=delivery_id,
        request_digest=digest,
        created_at=datetime(2026, 8, 7, 1, 0, tzinfo=timezone.utc),
    )


def stores(tmp_path: Path) -> tuple[ConversationSourceStore, ConversationSourceReceiptStore]:
    return (
        ConversationSourceStore(tmp_path, max_entries=100, max_file_bytes=2_000_000),
        ConversationSourceReceiptStore(tmp_path, max_file_bytes=100_000),
    )


class SuccessfulConsumer:
    def __init__(self, consumer: ConversationSourceConsumer) -> None:
        self.consumer = consumer
        self.consume_count = 0
        self.completed_count = 0
        self.envelopes: list[ConversationSourceEnvelope] = []

    async def consume(self, envelope: ConversationSourceEnvelope) -> ConversationConsumerExecution:
        self.consume_count += 1
        self.envelopes.append(envelope)
        result_id = canonical_digest({"consumer": self.consumer.value, "source_id": envelope.source_id})
        result = {"consumer": self.consumer.value, "source_id": envelope.source_id}
        return ConversationConsumerExecution(
            ConversationConsumerReceiptState.SUCCEEDED,
            result,
            result_id,
            canonical_digest(result),
        )

    async def completed(self, envelope: ConversationSourceEnvelope, _receipt: object) -> object:
        self.completed_count += 1
        return {"consumer": self.consumer.value, "source_id": envelope.source_id, "restored": True}


class FailingConsumer(SuccessfulConsumer):
    async def consume(self, envelope: ConversationSourceEnvelope) -> ConversationConsumerExecution:
        self.consume_count += 1
        self.envelopes.append(envelope)
        raise RuntimeError(f"{self.consumer.value} failed")


def test_source_store_reuses_same_content_and_rejects_same_identity_with_different_content(
    tmp_path: Path,
) -> None:
    source_store, _ = stores(tmp_path)
    original = source_envelope()
    replay = ConversationSourceEnvelope.create(
        conversation_id=original.conversation_id,
        started_on=original.started_on,
        protocol=original.protocol,
        batch=original.batch,
        after_turn=original.after_turn,
        omit_tool_call_ids=original.omit_tool_call_ids,
        delivery_id=original.delivery_id,
        request_digest=original.request_digest,
        created_at=datetime(2026, 8, 7, 2, 0, tzinfo=timezone.utc),
    )
    conflicting_batch = ConversationBatch("conversation-1", closed_turn(prompt="different"))
    conflicting_request = conversation_source_request_digest(
        conversation_id="conversation-1",
        started_on=STARTED_ON,
        protocol="normalized",
        batch=conflicting_batch,
        after_turn=False,
        omit_tool_call_ids=frozenset(),
    )
    conflicting = source_envelope(
        conflicting_batch,
        delivery_id=original.delivery_id,
        request_digest=conflicting_request,
    )

    stored = source_store.put(original)
    assert source_store.put(replay) == stored
    assert conflicting.source_id == original.source_id
    assert source_store.list() == (stored,)
    with pytest.raises(ConversationSourceError, match="different source content"):
        source_store.put(conflicting)


def test_behavior_projection_reads_raw_batch_bounds_tool_payloads_and_skips_completion(
    tmp_path: Path,
) -> None:
    envelope = source_envelope(ConversationBatch("conversation-1", tool_turn()))
    projected = ConversationBehaviorProjector(clock=lambda: BASE_TIME).project(envelope)

    assert isinstance(projected, ConversationBehaviorProjectionBatch)
    assert tuple(item.projection_kind for item in projected.items) == (
        ConversationBehaviorProjectionKind.USER_CONVERSATION_INPUT,
        ConversationBehaviorProjectionKind.AGENT_TOOL_CALL,
        ConversationBehaviorProjectionKind.TOOL_EXECUTION_RESULT,
    )
    tool_call = projected.items[1].payload
    assert set(tool_call) == {"tool_call_id", "tool_name", "arguments_digest"}
    assert "path" not in tool_call
    tool_result = projected.items[2].payload
    assert set(tool_result) == {
        "tool_call_id",
        "tool_name",
        "tool_status",
        "content_mode",
        "result_digest",
        "source_ref",
        "original_size_bytes",
        "original_sha256",
    }
    assert "工作区正常" not in str(tool_result)
    store = ConversationBehaviorProjectionStore(tmp_path, max_file_bytes=2_000_000)
    assert store.put(projected) == store.read(projected.projection_id)


def test_completion_only_source_writes_skipped_receipt_without_empty_projection(tmp_path: Path) -> None:
    batch = ConversationBatch(
        "conversation-1",
        (message(0, ConversationMessageRole.COMPLETION, "internal agent activity"),),
    )
    envelope = source_envelope(batch)
    source_store, receipts = stores(tmp_path)
    memory = SuccessfulConsumer(ConversationSourceConsumer.MEMORY)
    projection_store = ConversationBehaviorProjectionStore(tmp_path, max_file_bytes=2_000_000)
    behavior = ConversationBehaviorProjectionConsumer(ConversationBehaviorProjector(), projection_store)
    coordinator = ConversationSourceCoordinator(source_store, receipts, memory, behavior)

    result = asyncio.run(coordinator.dispatch(envelope))

    assert result.behavior_projection_result is None
    assert result.behavior_projection_receipt is not None
    assert result.behavior_projection_receipt.state is ConversationConsumerReceiptState.SKIPPED
    assert not projection_store.projection_root.exists()


def test_coordinator_starts_both_from_same_durable_source_and_does_not_wait_for_memory(
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        source_store, receipts = stores(tmp_path)
        memory_started = asyncio.Event()
        release_memory = asyncio.Event()
        behavior_completed = asyncio.Event()
        seen: list[ConversationSourceEnvelope] = []

        class BlockingMemory(SuccessfulConsumer):
            async def consume(self, envelope: ConversationSourceEnvelope) -> ConversationConsumerExecution:
                seen.append(envelope)
                memory_started.set()
                await release_memory.wait()
                return await super().consume(envelope)

        class ImmediateBehavior(SuccessfulConsumer):
            async def consume(self, envelope: ConversationSourceEnvelope) -> ConversationConsumerExecution:
                await memory_started.wait()
                seen.append(envelope)
                result = await super().consume(envelope)
                behavior_completed.set()
                return result

        memory = BlockingMemory(ConversationSourceConsumer.MEMORY)
        behavior = ImmediateBehavior(ConversationSourceConsumer.BEHAVIOR_PROJECTION)
        coordinator = ConversationSourceCoordinator(source_store, receipts, memory, behavior)
        envelope = source_envelope()
        running = asyncio.create_task(coordinator.dispatch(envelope))

        await asyncio.wait_for(behavior_completed.wait(), timeout=1.0)
        assert not running.done()
        assert receipts.read(envelope.source_id, ConversationSourceConsumer.BEHAVIOR_PROJECTION) is not None
        assert source_store.read(envelope.source_id) is not None
        assert len(seen) == 2 and seen[0] is seen[1]
        release_memory.set()
        result = await running
        assert result.memory_error is None and result.behavior_projection_error is None

    asyncio.run(scenario())


@pytest.mark.parametrize(
    ("failing", "successful"),
    [
        (ConversationSourceConsumer.MEMORY, ConversationSourceConsumer.BEHAVIOR_PROJECTION),
        (ConversationSourceConsumer.BEHAVIOR_PROJECTION, ConversationSourceConsumer.MEMORY),
    ],
)
def test_consumer_failure_does_not_rollback_other_output(
    tmp_path: Path,
    failing: ConversationSourceConsumer,
    successful: ConversationSourceConsumer,
) -> None:
    source_store, receipts = stores(tmp_path)
    consumers = {
        name: (FailingConsumer(name) if name is failing else SuccessfulConsumer(name))
        for name in ConversationSourceConsumer
    }
    coordinator = ConversationSourceCoordinator(
        source_store,
        receipts,
        consumers[ConversationSourceConsumer.MEMORY],
        consumers[ConversationSourceConsumer.BEHAVIOR_PROJECTION],
    )
    envelope = source_envelope()

    result = asyncio.run(coordinator.dispatch(envelope))

    assert getattr(result, f"{failing.value}_error") is not None
    assert receipts.read(envelope.source_id, failing) is None
    assert receipts.read(envelope.source_id, successful) is not None


def test_behavior_failure_keeps_real_memory_journal_and_memory_receipt(tmp_path: Path) -> None:
    path_lock = PathLock(ProcessLocalLockStore())
    conversation_root = tmp_path / "conversation"
    journal = ConversationMessageJournal(conversation_root, path_lock)
    jobs = MemoryJobStore(tmp_path / "workflow", path_lock, memory_root=tmp_path / "memory")
    enqueuer = ConversationMemoryEnqueuer(journal, jobs)
    memory = MemoryConversationConsumer(
        enqueuer,
        journal,
        ConversationSemanticBoundaryScorer(
            FakeEmbedder(),
            embedding_fingerprint="fake-v1",
            max_unit_chars=256,
        ),
    )
    source_store = ConversationSourceStore(conversation_root, max_entries=100, max_file_bytes=2_000_000)
    receipts = ConversationSourceReceiptStore(conversation_root, max_file_bytes=100_000)
    behavior = FailingConsumer(ConversationSourceConsumer.BEHAVIOR_PROJECTION)
    coordinator = ConversationSourceCoordinator(source_store, receipts, memory, behavior)
    envelope = source_envelope()

    result = asyncio.run(coordinator.dispatch(envelope))

    assert result.memory_error is None
    assert result.behavior_projection_error is not None
    assert journal.read_live(ConversationAddress("conversation-1", STARTED_ON)) is not None
    assert receipts.read(envelope.source_id, ConversationSourceConsumer.MEMORY) is not None
    assert receipts.read(envelope.source_id, ConversationSourceConsumer.BEHAVIOR_PROJECTION) is None


def test_memory_failure_keeps_real_projection_outbox_and_projection_receipt(tmp_path: Path) -> None:
    source_store, receipts = stores(tmp_path)
    memory = FailingConsumer(ConversationSourceConsumer.MEMORY)
    projection_store = ConversationBehaviorProjectionStore(tmp_path, max_file_bytes=2_000_000)
    behavior = ConversationBehaviorProjectionConsumer(ConversationBehaviorProjector(), projection_store)
    coordinator = ConversationSourceCoordinator(source_store, receipts, memory, behavior)
    envelope = source_envelope(ConversationBatch("conversation-1", tool_turn()))

    result = asyncio.run(coordinator.dispatch(envelope))

    assert result.memory_error is not None
    assert result.behavior_projection_error is None
    projected = result.behavior_projection_result
    assert isinstance(projected, ConversationBehaviorProjectionBatch)
    assert projection_store.read(projected.projection_id) == projected
    assert receipts.read(envelope.source_id, ConversationSourceConsumer.MEMORY) is None
    assert receipts.read(envelope.source_id, ConversationSourceConsumer.BEHAVIOR_PROJECTION) is not None


def test_recovery_consumes_only_missing_terminal_receipt(tmp_path: Path) -> None:
    source_store, receipts = stores(tmp_path)
    memory = SuccessfulConsumer(ConversationSourceConsumer.MEMORY)
    failing_behavior = FailingConsumer(ConversationSourceConsumer.BEHAVIOR_PROJECTION)
    envelope = source_envelope()
    first = ConversationSourceCoordinator(source_store, receipts, memory, failing_behavior)
    asyncio.run(first.dispatch(envelope))
    assert memory.consume_count == 1

    recovered_behavior = SuccessfulConsumer(ConversationSourceConsumer.BEHAVIOR_PROJECTION)
    second = ConversationSourceCoordinator(source_store, receipts, memory, recovered_behavior)
    recovery = ConversationSourceRecovery(source_store, receipts, second)
    pending = recovery.pending()
    assert len(pending) == 1
    assert pending[0].missing_consumers == (ConversationSourceConsumer.BEHAVIOR_PROJECTION,)

    asyncio.run(recovery.recover_pending())

    assert memory.consume_count == 1
    assert memory.completed_count == 1
    assert recovered_behavior.consume_count == 1
    assert recovery.pending() == ()


def test_terminal_source_replay_does_not_reapply_old_after_turn_to_newer_live_state(
    tmp_path: Path,
) -> None:
    path_lock = PathLock(ProcessLocalLockStore())
    conversation_root = tmp_path / "conversation"
    journal = ConversationMessageJournal(conversation_root, path_lock)
    jobs = MemoryJobStore(tmp_path / "workflow", path_lock, memory_root=tmp_path / "memory")
    enqueuer = ConversationMemoryEnqueuer(journal, jobs)
    memory = MemoryConversationConsumer(
        enqueuer,
        journal,
        ConversationSemanticBoundaryScorer(
            FakeEmbedder(),
            embedding_fingerprint="fake-v1",
            max_unit_chars=256,
        ),
    )
    source_store = ConversationSourceStore(conversation_root, max_entries=100, max_file_bytes=2_000_000)
    receipts = ConversationSourceReceiptStore(conversation_root, max_file_bytes=100_000)
    behavior = ConversationBehaviorProjectionConsumer(
        ConversationBehaviorProjector(),
        ConversationBehaviorProjectionStore(conversation_root, max_file_bytes=2_000_000),
    )
    coordinator = ConversationSourceCoordinator(source_store, receipts, memory, behavior)
    first = source_envelope(after_turn=True)
    second = source_envelope(
        ConversationBatch(
            "conversation-1",
            (message(2, ConversationMessageRole.PROMPT, "new incomplete turn"),),
        ),
        delivery_id="b" * 64,
        after_turn=False,
    )

    asyncio.run(coordinator.dispatch(first))
    asyncio.run(coordinator.dispatch(second))
    replay = asyncio.run(coordinator.dispatch(first))

    assert replay.memory_error is None
    assert replay.memory_result is not None
    assert replay.memory_result.append.next_sequence == 2
    assert replay.memory_result.jobs == ()
    assert journal.next_sequence(ConversationAddress("conversation-1", STARTED_ON)) == 3


class FakeEmbedder:
    async def embed_documents(self, texts: tuple[str, ...]) -> tuple[object, ...]:
        raise AssertionError(f"boundary scoring was not expected for {len(texts)} texts")


class DeterministicEmbedder:
    async def embed_documents(self, texts: tuple[str, ...]) -> tuple[EmbeddingVector, ...]:
        return tuple(EmbeddingVector((1.0, 0.0)) for _ in texts)


def test_memory_consumer_is_artifact_equivalent_to_previous_runtime_orchestration(tmp_path: Path) -> None:
    async def scenario() -> None:
        config = ConversationSegmentationConfig(
            commit_token_threshold=1,
            keep_recent_turn_count=1,
            retained_message_token_budget=40,
            max_segment_tokens=40,
            max_live_messages=1_000,
            max_live_bytes=1_000_000,
            max_segment_messages=1_000,
            max_segment_bytes=1_000_000,
            max_inline_tool_result_bytes=8,
            max_tool_result_summary_chars=128,
        )

        def chain(root: Path) -> tuple[
            ConversationMessageJournal,
            ConversationMemoryEnqueuer,
            ConversationSemanticBoundaryScorer,
        ]:
            path_lock = PathLock(ProcessLocalLockStore())
            journal = ConversationMessageJournal(root / "conversation", path_lock)
            jobs = MemoryJobStore(root / "workflow", path_lock, memory_root=root / "memory")
            planner = ConversationRetentionPlanner(
                config,
                token_estimator=lambda item: len(item.content) if isinstance(item.content, str) else 1,
            )
            return (
                journal,
                ConversationMemoryEnqueuer(journal, jobs, planner),
                ConversationSemanticBoundaryScorer(
                    DeterministicEmbedder(),
                    embedding_fingerprint="deterministic-v1",
                    max_unit_chars=256,
                ),
            )

        messages = (*closed_turn(prompt="A" * 100), *tool_turn(start_sequence=2))
        batch = ConversationBatch("conversation-1", messages)
        envelope = source_envelope(batch, after_turn=True)
        address = ConversationAddress("conversation-1", STARTED_ON)
        legacy_journal, legacy_enqueuer, legacy_scorer = chain(tmp_path / "legacy")
        current_journal, current_enqueuer, current_scorer = chain(tmp_path / "current")

        legacy_append = await asyncio.to_thread(
            legacy_enqueuer.append,
            address,
            batch,
            omit_tool_call_ids=envelope.omit_tool_call_ids,
            ingress=ConversationIngressRequest(envelope.delivery_id, envelope.request_digest),
        )
        legacy_preview = await asyncio.to_thread(
            legacy_enqueuer.preview_retention,
            address,
            after_turn=True,
        )
        legacy_hints = None
        if legacy_preview.should_seal:
            legacy_hints = await legacy_scorer.score(legacy_journal.read_live(address))
        legacy_jobs, legacy_retention = await asyncio.to_thread(
            legacy_enqueuer.enqueue_ready_segments,
            address,
            after_turn=True,
            flush=False,
            boundary_hints=legacy_hints,
        )

        consumer = MemoryConversationConsumer(current_enqueuer, current_journal, current_scorer)
        execution = await consumer.consume(envelope)
        current_result = execution.result
        assert current_result is not None
        assert current_result.append == legacy_append
        assert current_result.retention == legacy_retention
        assert tuple(job.source_identity for job in current_result.jobs) == tuple(
            job.source_identity for job in legacy_jobs
        )
        assert tuple(job.memory_sequence for job in current_result.jobs) == tuple(
            job.memory_sequence for job in legacy_jobs
        )
        assert current_journal.read_live(address) == legacy_journal.read_live(address)
        assert current_journal.list_history(address) == legacy_journal.list_history(address)
        legacy_jsonl = tuple(
            path.read_bytes()
            for path in sorted((tmp_path / "legacy" / "conversation").rglob("*.jsonl"))
        )
        current_jsonl = tuple(
            path.read_bytes()
            for path in sorted((tmp_path / "current" / "conversation").rglob("*.jsonl"))
        )
        assert current_jsonl == legacy_jsonl

    asyncio.run(scenario())


def test_long_prompt_is_chunked_only_in_memory_and_stays_one_raw_projection_item(tmp_path: Path) -> None:
    path_lock = PathLock(ProcessLocalLockStore())
    journal = ConversationMessageJournal(tmp_path / "conversation", path_lock)
    jobs = MemoryJobStore(tmp_path / "workflow", path_lock, memory_root=tmp_path / "memory")
    planner = ConversationRetentionPlanner(
        ConversationSegmentationConfig(
            commit_token_threshold=10_000,
            keep_recent_turn_count=1,
            retained_message_token_budget=10_000,
            max_segment_tokens=24,
            max_live_messages=1_000,
            max_live_bytes=1_000_000,
            max_segment_messages=1_000,
            max_segment_bytes=1_000_000,
        ),
        token_estimator=lambda item: len(item.content) if isinstance(item.content, str) else 1,
    )
    enqueuer = ConversationMemoryEnqueuer(journal, jobs, planner)
    memory = MemoryConversationConsumer(
        enqueuer,
        journal,
        ConversationSemanticBoundaryScorer(
            FakeEmbedder(),
            embedding_fingerprint="fake-v1",
            max_unit_chars=256,
        ),
    )
    conversation_root = tmp_path / "conversation"
    source_store = ConversationSourceStore(conversation_root, max_entries=100, max_file_bytes=2_000_000)
    receipts = ConversationSourceReceiptStore(conversation_root, max_file_bytes=100_000)
    projection_store = ConversationBehaviorProjectionStore(conversation_root, max_file_bytes=2_000_000)
    behavior = ConversationBehaviorProjectionConsumer(ConversationBehaviorProjector(), projection_store)
    coordinator = ConversationSourceCoordinator(source_store, receipts, memory, behavior)
    original_prompt = "A" * 100
    envelope = source_envelope(
        ConversationBatch("conversation-1", closed_turn(prompt=original_prompt)),
    )

    result = asyncio.run(coordinator.dispatch(envelope))

    live = journal.read_live(ConversationAddress("conversation-1", STARTED_ON))
    assert live is not None
    memory_prompt_parts = tuple(item for item in live.messages if item.role is ConversationMessageRole.PROMPT)
    assert len(memory_prompt_parts) > 1
    assert "".join(str(item.content) for item in memory_prompt_parts) == original_prompt
    projected = result.behavior_projection_result
    assert isinstance(projected, ConversationBehaviorProjectionBatch)
    projected_prompts = tuple(
        item
        for item in projected.items
        if item.projection_kind is ConversationBehaviorProjectionKind.USER_CONVERSATION_INPUT
    )
    assert len(projected_prompts) == 1
    assert projected_prompts[0].source_message_id == envelope.batch.messages[0].message_id
    assert projected_prompts[0].payload == {"content": original_prompt}
