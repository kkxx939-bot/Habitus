"""Conversation Source 重构不得改变既有 Ingress 幂等和返回值契约。"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from conversation import (
    ConversationBehaviorProjectionConsumer,
    ConversationBehaviorProjectionStore,
    ConversationBehaviorProjector,
    ConversationSourceCoordinator,
    ConversationSourceEnvelope,
    ConversationSourceReceiptStore,
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
    ConversationWriteConflictError,
)
from memory.workflow import (
    ConversationMemoryEnqueuer,
    ConversationMemoryIngestResult,
    MemoryConversationConsumer,
    MemoryJobStore,
)
from ModelClient import EmbeddingVector
from pre.conversation import ConversationBatch
from tests.helpers import closed_turn

STARTED_ON = date(2026, 8, 8)
CREATED_AT = datetime(2026, 8, 8, 1, 0, tzinfo=timezone.utc)


def _legacy_runtime_request_digest(
    *,
    address: ConversationAddress,
    protocol: str,
    batch: ConversationBatch,
    after_turn: bool,
    omit_tool_call_ids: frozenset[str],
) -> str:
    """逐字段保留 Source 重构前 Runtime 使用的公开投递摘要公式。"""

    return canonical_digest(
        {
            "conversation_id": address.conversation_id,
            "started_on": address.started_on,
            "protocol": protocol,
            "batch": batch.to_dict(),
            "after_turn": after_turn,
            "omit_tool_call_ids": omit_tool_call_ids,
        }
    )


def _envelope(
    batch: ConversationBatch,
    *,
    delivery_id: str,
    after_turn: bool,
) -> ConversationSourceEnvelope:
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
        delivery_id=delivery_id,
        request_digest=request_digest,
        created_at=CREATED_AT,
    )


class _DeterministicEmbedder:
    async def embed_documents(self, texts: tuple[str, ...]) -> tuple[EmbeddingVector, ...]:
        return tuple(EmbeddingVector((1.0, 0.0)) for _ in texts)


def test_source_request_digest_preserves_existing_runtime_formula() -> None:
    address = ConversationAddress("source-digest-compatibility", STARTED_ON)
    batch = ConversationBatch(address.conversation_id, closed_turn())
    omitted = frozenset({"call-2", "call-1"})
    legacy = _legacy_runtime_request_digest(
        address=address,
        protocol="openai_chat_completions",
        batch=batch,
        after_turn=True,
        omit_tool_call_ids=omitted,
    )

    current = conversation_source_request_digest(
        conversation_id=address.conversation_id,
        started_on=address.started_on,
        protocol="openai_chat_completions",
        batch=batch,
        after_turn=True,
        omit_tool_call_ids=omitted,
    )

    assert current == legacy, (
        "显式 delivery_id 的 request_digest 是既有耐久 Ingress 身份，"
        "Source 重构不得通过新增摘要字段改变它"
    )


def test_existing_durable_ingress_receipt_replays_after_source_upgrade(tmp_path: Path) -> None:
    path_lock = PathLock(ProcessLocalLockStore())
    journal = ConversationMessageJournal(tmp_path / "conversation", path_lock)
    enqueuer = ConversationMemoryEnqueuer(
        journal,
        MemoryJobStore(tmp_path / "workflow", path_lock, memory_root=tmp_path / "memory"),
    )
    address = ConversationAddress("durable-ingress-upgrade", STARTED_ON)
    batch = ConversationBatch(address.conversation_id, closed_turn())
    delivery_id = "a" * 64
    legacy_digest = _legacy_runtime_request_digest(
        address=address,
        protocol="openai_chat_completions",
        batch=batch,
        after_turn=False,
        omit_tool_call_ids=frozenset(),
    )
    first = enqueuer.append(
        address,
        batch,
        ingress=ConversationIngressRequest(delivery_id, legacy_digest),
    )
    upgraded_digest = conversation_source_request_digest(
        conversation_id=address.conversation_id,
        started_on=address.started_on,
        protocol="openai_chat_completions",
        batch=batch,
        after_turn=False,
        omit_tool_call_ids=frozenset(),
    )

    try:
        replayed = enqueuer.append(
            address,
            batch,
            ingress=ConversationIngressRequest(delivery_id, upgraded_digest),
        )
    except ConversationWriteConflictError:
        pytest.fail(
            "同一显式 delivery_id 和同一 Conversation 请求在 Source 升级后必须继续命中旧耐久 Receipt，"
            "不能变成永久写入冲突"
        )

    assert replayed.status.value == "unchanged"
    assert replayed.next_sequence == first.next_sequence == 2


def test_terminal_source_replay_keeps_ingress_next_sequence_without_reapplying_after_turn(
    tmp_path: Path,
) -> None:
    path_lock = PathLock(ProcessLocalLockStore())
    conversation_root = tmp_path / "conversation"
    journal = ConversationMessageJournal(conversation_root, path_lock)
    jobs = MemoryJobStore(tmp_path / "workflow", path_lock, memory_root=tmp_path / "memory")
    planner = ConversationRetentionPlanner(
        ConversationSegmentationConfig(
            commit_token_threshold=1,
            keep_recent_turn_count=1,
            retained_message_token_budget=1_000,
            max_live_messages=100,
            max_live_bytes=1_000_000,
            max_segment_messages=100,
            max_segment_bytes=1_000_000,
            max_segment_tokens=1_000,
        )
    )
    enqueuer = ConversationMemoryEnqueuer(journal, jobs, planner)
    memory = MemoryConversationConsumer(
        enqueuer,
        journal,
        ConversationSemanticBoundaryScorer(
            _DeterministicEmbedder(),
            embedding_fingerprint="deterministic-v1",
            max_unit_chars=256,
        ),
    )
    source_store = ConversationSourceStore(
        conversation_root,
        max_entries=100,
        max_file_bytes=2_000_000,
    )
    receipts = ConversationSourceReceiptStore(conversation_root, max_file_bytes=100_000)
    projection_store = ConversationBehaviorProjectionStore(
        conversation_root,
        max_file_bytes=2_000_000,
    )
    coordinator = ConversationSourceCoordinator(
        source_store,
        receipts,
        memory,
        ConversationBehaviorProjectionConsumer(
            ConversationBehaviorProjector(),
            projection_store,
        ),
    )
    address = ConversationAddress("source-replay-compatibility", STARTED_ON)
    source_a = _envelope(
        ConversationBatch(address.conversation_id, closed_turn()),
        delivery_id="a" * 64,
        after_turn=True,
    )
    source_b = _envelope(
        ConversationBatch(address.conversation_id, closed_turn(start_sequence=2, prompt="第二轮")),
        delivery_id="b" * 64,
        after_turn=False,
    )

    async def scenario() -> None:
        first = await coordinator.dispatch(source_a)
        assert isinstance(first.memory_result, ConversationMemoryIngestResult)
        assert first.memory_result.append.next_sequence == 2
        assert first.memory_result.jobs == ()
        second = await coordinator.dispatch(source_b)
        assert isinstance(second.memory_result, ConversationMemoryIngestResult)
        assert second.memory_result.append.next_sequence == 4
        live_before = journal.read_live(address)
        history_before = journal.list_history(address)

        replayed = await coordinator.dispatch(source_a)

        assert isinstance(replayed.memory_result, ConversationMemoryIngestResult)
        assert journal.read_live(address) == live_before
        assert journal.list_history(address) == history_before == ()
        assert replayed.memory_result.jobs == ()
        assert replayed.memory_result.append.next_sequence == 2, (
            "终态 Source A 的重放必须返回 A 的耐久 Ingress Receipt 所绑定的 next_sequence；"
            "不能用追加 Source B 后的当前 cursor 伪造 A 的原始 ConversationAppendResult"
        )
        assert journal.next_sequence(address) == 4

    asyncio.run(scenario())
