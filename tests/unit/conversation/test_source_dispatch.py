from __future__ import annotations

import asyncio
from dataclasses import FrozenInstanceError
from datetime import timedelta

import pytest

from habitus.conversation.projection import (
    ConversationBehaviorProjectionKind,
    ConversationBehaviorProjector,
)
from habitus.conversation.source import (
    ConversationConsumerDeliveryState,
    ConversationConsumerOutcomeState,
    ConversationSourceConsumer,
    ConversationSourceCoordinator,
    ConversationSourceEnvelope,
    ConversationSourceError,
)
from habitus.foundation.integrity import canonical_digest
from habitus.memory.workflow import (
    ConversationMemoryIngestResult,
    MemoryConversationOutput,
    MemoryJob,
    MemoryJobStatus,
)
from habitus.pre.conversation import ConversationBatch, ConversationMessageRole
from tests.unit.conversation.source_v2_helpers import NOW, delivery, ingest_result, message, source, stores


def test_source_identity_and_record_digests_have_separate_duties(tmp_path) -> None:
    first = source(recorded_at=NOW)
    replay = source(recorded_at=NOW + timedelta(hours=1))

    assert first.delivery_id == replay.delivery_id
    assert first.request_digest == replay.request_digest
    assert first.source_id == replay.source_id
    assert first.source_payload_digest == replay.source_payload_digest
    assert first.source_record_digest != replay.source_record_digest

    source_store, _outcomes, _memory_outputs, _projections = stores(tmp_path)
    stored = source_store.put(first)
    assert source_store.put(replay) == stored
    assert source_store.read(first.source_id) == first
    assert (tmp_path / "source" / "envelopes" / f"{first.source_id}.json").is_file()

    with pytest.raises(FrozenInstanceError):
        first.protocol = "changed"  # type: ignore[misc]


def test_same_source_identity_with_different_payload_fails_explicitly(tmp_path) -> None:
    original = source()
    changed_batch = ConversationBatch(
        original.conversation_id,
        (message(content="different"),),
    )
    changed = ConversationSourceEnvelope.create(
        conversation_id=original.conversation_id,
        started_on=original.started_on,
        protocol=original.protocol,
        batch=changed_batch,
        after_turn=original.after_turn,
        omit_tool_call_ids=original.omit_tool_call_ids,
        delivery_id=original.delivery_id,
        request_digest=canonical_digest(
            {
                "conversation_id": original.conversation_id,
                "started_on": original.started_on.isoformat(),
                "protocol": original.protocol,
                "batch": changed_batch.to_dict(),
                "after_turn": original.after_turn,
                "omit_tool_call_ids": [],
            }
        ),
        recorded_at=NOW,
    )
    source_store, _outcomes, _memory_outputs, _projections = stores(tmp_path)
    source_store.put(original)
    with pytest.raises(ConversationSourceError, match="different source payload"):
        source_store.put(changed)


def test_source_rejects_request_digest_that_is_not_the_canonical_request() -> None:
    original = source()
    with pytest.raises(ConversationSourceError, match="canonical source request"):
        ConversationSourceEnvelope.create(
            conversation_id=original.conversation_id,
            started_on=original.started_on,
            protocol=original.protocol,
            batch=original.batch,
            after_turn=original.after_turn,
            omit_tool_call_ids=original.omit_tool_call_ids,
            delivery_id=original.delivery_id,
            request_digest="0" * 64,
            recorded_at=NOW,
        )


def test_source_read_verifies_full_record_digest(tmp_path) -> None:
    value = source()
    source_store, _outcomes, _memory_outputs, _projections = stores(tmp_path)
    source_store.put(value)
    path = tmp_path / "source" / "envelopes" / f"{value.source_id}.json"
    encoded = path.read_text(encoding="utf-8")
    path.write_text(encoded.replace(value.source_record_digest, "0" * 64), encoding="utf-8")
    with pytest.raises(ConversationSourceError, match="corrupt"):
        source_store.read(value.source_id)


def test_behavior_projector_reads_original_source_batch_and_keeps_mapping_rules() -> None:
    conversation_id = "behavior-source"
    messages = (
        message(0, role=ConversationMessageRole.PROMPT, content="p" * 20_000, conversation_id=conversation_id),
        message(1, role=ConversationMessageRole.COMPLETION, content="internal", conversation_id=conversation_id),
    )
    batch = ConversationBatch(conversation_id, messages)
    request = canonical_digest(
        {
            "conversation_id": conversation_id,
            "started_on": "2026-08-08",
            "protocol": "normalized",
            "batch": batch.to_dict(),
            "after_turn": False,
            "omit_tool_call_ids": [],
        }
    )
    envelope = ConversationSourceEnvelope.create(
        conversation_id=conversation_id,
        started_on=messages[0].occurred_at.date(),
        protocol="normalized",
        batch=batch,
        after_turn=False,
        omit_tool_call_ids=frozenset(),
        delivery_id=canonical_digest("behavior-delivery"),
        request_digest=request,
        recorded_at=NOW,
    )
    projected = ConversationBehaviorProjector().project(envelope)
    assert projected is not None
    assert len(projected.items) == 1
    assert projected.items[0].source_message_id == messages[0].message_id
    assert projected.items[0].projection_kind is ConversationBehaviorProjectionKind.USER_CONVERSATION_INPUT
    assert projected.items[0].payload == {"content": "p" * 20_000}


def test_completion_only_source_creates_skipped_outcome_without_output(tmp_path) -> None:
    async def scenario() -> None:
        source_value = source(role=ConversationMessageRole.COMPLETION)
        service, _memory, behavior = delivery(tmp_path)
        service.sources.put(source_value)
        ensured = await service.ensure_outcome(
            source_value,
            ConversationSourceConsumer.BEHAVIOR_PROJECTION,
        )
        assert ensured.outcome.state is ConversationConsumerOutcomeState.SKIPPED
        assert ensured.outcome.skip_reason == "NO_ELIGIBLE_MESSAGES"
        assert behavior.output_store.list(source_value) == ()
        assert service.inspect(
            source_value,
            ConversationSourceConsumer.BEHAVIOR_PROJECTION,
        ).state is ConversationConsumerDeliveryState.SKIPPED

    asyncio.run(scenario())


def test_coordinator_start_returns_before_blocked_behavior_and_memory_wait_is_independent(tmp_path) -> None:
    async def scenario() -> None:
        release = asyncio.Event()
        entered = asyncio.Event()
        service, _memory, behavior = delivery(tmp_path)
        behavior.entered = entered
        behavior.release = release
        coordinator = ConversationSourceCoordinator(service.sources, service)
        source_value = source()

        handle = coordinator.start(source_value)
        memory = await asyncio.wait_for(handle.wait_memory(), timeout=1.0)
        assert memory.append.next_sequence == 1
        await asyncio.wait_for(entered.wait(), timeout=1.0)
        assert not handle.behavior_projection_task.done()
        assert handle.inspect_memory().state is ConversationConsumerDeliveryState.COMMITTED
        release.set()
        projected = await handle.wait_behavior_projection()
        assert projected is not None

    asyncio.run(scenario())


def test_memory_output_codec_restores_full_ingest_result_including_jobs(tmp_path) -> None:
    source_value = source()
    base = ingest_result(source_value)
    job = MemoryJob(
        memory_sequence=1,
        conversation_id=source_value.conversation_id,
        started_on=source_value.started_on,
        segment_id="0-0",
        source_segment_digest=canonical_digest("segment"),
        transaction_id="a" * 32,
        status=MemoryJobStatus.QUEUED,
        attempts=0,
        claim_id=None,
        claim_generation=0,
        worker_id=None,
        lease_expires_at=None,
        next_attempt_at=None,
        last_error=None,
        created_at=NOW,
        updated_at=NOW,
    )
    expected = ConversationMemoryIngestResult(base.append, (job,), base.retention)
    _sources, _outcomes, output_store, _projections = stores(tmp_path)
    fingerprint = canonical_digest("memory-output-codec")
    output = MemoryConversationOutput.create(
        source=source_value,
        processor_fingerprint=fingerprint,
        ingest_result=expected,
        recorded_at=NOW,
    )
    stored = output_store.put(source_value, output)
    assert output_store.restore(stored) == expected
    assert output_store.read(source_value, output.output_id) == output
