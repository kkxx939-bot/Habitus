"""W1 第 4 项 A1：会话号凡参与算键、摘要、路径、锁、去重、回执号，一律用归一身份（NFC + casefold）。

原始 ``conversation_id`` 照原文保存，只是不再参与算身份。全小写的会话号（例如 UUID）改动前后算出的键完全一样。
"""

from __future__ import annotations

import asyncio
import hashlib
import unicodedata
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from habitus.conversation import (
    ConversationConsumerDeliveryState,
    ConversationSourceConsumer,
    ConversationSourceEnvelope,
    conversation_source_request_digest,
)
from habitus.conversation.source.address import SourceAddress
from habitus.conversation.source.fence import ConversationConsumerExecutionFence
from habitus.conversation.source.recovery import ConversationSourceRecoveryEntry, _in_recovery_order
from habitus.conversation.source.watermark import AddressWatermarks
from habitus.foundation.integrity import canonical_digest
from habitus.infrastructure.store.contracts import PathLock
from habitus.integrations.http_api import create_http_app
from habitus.memory.conversation import ConversationAddress, ConversationWriteConflictError
from habitus.memory.conversation.indexing.model import ConversationSummaryReference, ConversationSummaryStage
from habitus.memory.workflow.jobs.store import MemoryJobStore
from habitus.memory.workflow.receipt.model import MemoryChangeSource
from habitus.pre.conversation import ConversationBatch
from habitus.runtime import build_runtime
from habitus.runtime.session_lane import session_turns
from tests.helpers import closed_turn
from tests.integration.test_runtime_assembly import runtime_config, runtime_dependencies
from tests.lock_store import ProcessLocalLockStore

DAY = date(2026, 8, 7)
UUID = "0b68e0c8-46f9-424b-a99e-9244159db256"
SEGMENT = "000000000000-000000000001"
DIGEST = "a" * 64


def _envelope(conversation_id: str, *, start_sequence: int = 0, prompt: str = "请记住我喜欢简洁回答"):
    batch = ConversationBatch(conversation_id, closed_turn(start_sequence=start_sequence, prompt=prompt))
    digest = conversation_source_request_digest(
        conversation_id=conversation_id,
        started_on=DAY,
        protocol="normalized",
        batch=batch,
        after_turn=False,
        omit_tool_call_ids=frozenset(),
    )
    return ConversationSourceEnvelope.create(
        conversation_id=conversation_id,
        started_on=DAY,
        protocol="normalized",
        batch=batch,
        after_turn=False,
        omit_tool_call_ids=frozenset(),
        delivery_id=digest,
        request_digest=digest,
        recorded_at=datetime(2026, 8, 7, 1, 0, tzinfo=UTC),
    )


def _change_source(conversation_id: str) -> MemoryChangeSource:
    return MemoryChangeSource(
        memory_sequence=1,
        transaction_id="b" * 32,
        conversation_id=conversation_id,
        started_on=DAY,
        segment_id=SEGMENT,
        source_segment_digest=DIGEST,
    )


def _keys(conversation_id: str) -> dict[str, object]:
    """一个会话号在各处算出来的身份键。"""

    address = ConversationAddress(conversation_id, DAY)
    envelope = _envelope(conversation_id)
    return {
        "source_digest": SourceAddress(conversation_id, DAY).digest,
        "source_address": SourceAddress(conversation_id, DAY),
        "ordered_lock": ConversationConsumerExecutionFence._keys(
            envelope, ConversationSourceConsumer.MEMORY, conversation_ordered=True
        )[0],
        "source_id": ConversationSourceEnvelope.source_identity(conversation_id, DAY, "c" * 64),
        "job_key": MemoryJobStore._source_key_values(address, SEGMENT, DIGEST),
        "receipt_id": _change_source(conversation_id).receipt_id,
        "summary_identity": ConversationSummaryReference(address, ConversationSummaryStage.SEGMENT, SEGMENT).identity,
        "summary_scope": ConversationSummaryReference(
            address, ConversationSummaryStage.SEGMENT, SEGMENT
        ).conversation_scope,
        "address": address,
    }


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("Conv-ABC", "conv-abc"),
        ("Conv-ABC", "CONV-abc"),
        # 同一个中文会话号的 NFC / NFD 两种写法（带一个可分解的字母）
        (unicodedata.normalize("NFC", "会话-café"), unicodedata.normalize("NFD", "会话-café")),
        # 韩文：NFD 会拆成字母
        (unicodedata.normalize("NFC", "대화-1"), unicodedata.normalize("NFD", "대화-1")),
    ],
)
def test_case_and_normalization_variants_compute_the_same_identity_keys(left: str, right: str) -> None:
    assert left != right
    assert _keys(left) == _keys(right)


def test_lowercase_uuid_keys_are_byte_for_byte_what_they_were_before() -> None:
    """全小写 UUID 的归一身份就是原文：这些公式的结果与改动前（拿原文算）完全一样。"""

    keys = _keys(UUID)
    assert keys["source_digest"] == canonical_digest(
        {"schema": "conversation_source_address_v1", "conversation_id": UUID, "started_on": DAY.isoformat()}
    )[:32]
    assert keys["job_key"] == hashlib.sha256(f"{UUID}\0{DAY.isoformat()}\0{SEGMENT}\0{DIGEST}".encode()).hexdigest()
    assert keys["receipt_id"] == hashlib.sha256(f"{UUID}\0{DAY.isoformat()}\0{SEGMENT}\0{DIGEST}".encode()).hexdigest()
    assert keys["source_id"] == canonical_digest(
        {
            "schema_version": "conversation_source_identity_v1",
            "conversation_id": UUID,
            "started_on": DAY.isoformat(),
            "delivery_id": "c" * 64,
        }
    )
    lock_digest = canonical_digest(
        {
            "schema_version": "conversation_order_lock_v2",
            "consumer": ConversationSourceConsumer.MEMORY.value,
            "conversation_id": UUID,
            "started_on": DAY.isoformat(),
        }
    )
    assert keys["ordered_lock"] == f"conversation-ordered:{lock_digest}"
    assert keys["summary_identity"] == f"conversation-summary:{DAY.isoformat()}:{UUID}:segment:{SEGMENT}"


def test_the_original_conversation_id_is_kept_verbatim_in_records() -> None:
    address = SourceAddress("Conv-ABC", DAY)
    assert address.conversation_id == "Conv-ABC"
    assert address.to_dict() == {"conversation_id": "Conv-ABC", "started_on": DAY.isoformat()}
    assert SourceAddress.from_dict(address.to_dict()).conversation_id == "Conv-ABC"
    assert _envelope("Conv-ABC").to_dict()["conversation_id"] == "Conv-ABC"
    assert _change_source("Conv-ABC").to_dict()["conversation_id"] == "Conv-ABC"
    (turn,) = session_turns(_envelope("Conv-ABC"))  # 行为侧的轮记录也保留原文，只有轮的身份按归一身份
    assert (turn.conversation_id, turn.key) == ("Conv-ABC", "conv-abc#0-1")


def test_one_watermark_serves_both_spellings(tmp_path: Path) -> None:
    """水位文件按归一身份命名、按归一身份认：大写写进去的，小写读得出来，不会被判成"别的地址"。"""

    watermarks = AddressWatermarks(tmp_path)
    watermarks.advance(SourceAddress("Conv-ABC", DAY), 5)
    assert watermarks.released_through(SourceAddress("conv-abc", DAY)) == 5
    assert watermarks.covers(SourceAddress("conv-abc", DAY), 5)


def _pending(envelope: ConversationSourceEnvelope) -> ConversationSourceRecoveryEntry:
    return ConversationSourceRecoveryEntry(
        envelope, ConversationSourceConsumer.MEMORY, ConversationConsumerDeliveryState.PENDING
    )


def test_recovery_orders_both_spellings_as_one_conversation() -> None:
    later_upper = _pending(_envelope("Conv-ABC", start_sequence=2))
    # 中间夹一个按原文排会落在两者之间的会话号（"Conv-ABC" < "Conv-ABD" < "conv-abc"）
    other = _pending(_envelope("Conv-ABD", start_sequence=0))
    earlier_lower = _pending(_envelope("conv-abc", start_sequence=0))
    ordered = _in_recovery_order((later_upper, other, earlier_lower), lambda consumer: True)
    assert ordered.index(earlier_lower) < ordered.index(later_upper)


def _runtime(tmp_path: Path):
    providers, vectors = runtime_dependencies()
    runtime = build_runtime(
        runtime_config(tmp_path),
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )
    runtime.initialize()
    return runtime


def test_one_conversation_sent_under_two_spellings_is_one_source_stream_and_one_job(tmp_path: Path) -> None:
    async def scenario() -> None:
        runtime = _runtime(tmp_path)
        upper = ConversationAddress("Conv-ABC", DAY)
        lower = ConversationAddress("conv-abc", DAY)
        await runtime.append_conversation(upper, ConversationBatch("Conv-ABC", closed_turn()))
        await runtime.append_conversation(lower, ConversationBatch("conv-abc", closed_turn(start_sequence=2)))

        conversation = runtime.components.conversation
        # 一条来源流：一个地址目录、两份标记，按序号排；信封里各自保留原文
        (directory,) = conversation.source_ledger.index.address_directories()
        markers = conversation.source_ledger.index.markers_in(directory)
        assert [marker.start_sequence for marker in markers] == [0, 2]
        envelopes = [conversation.sources.read(marker.source_id) for marker in markers]
        assert [envelope.conversation_id for envelope in envelopes if envelope is not None] == ["Conv-ABC", "conv-abc"]
        # 一个会话目录、一份 live
        day_directory = conversation.journal.layout.messages_root() / "2026" / "08" / "07"
        assert [path.name for path in day_directory.iterdir()] == ["conv-abc"]
        live = await runtime.read_live_conversation(upper)
        assert live is not None and [message.sequence for message in live.messages] == [0, 1, 2, 3]
        assert conversation.journal.layout.lock_key(upper) == conversation.journal.layout.lock_key(lower)

        # 同一轮以另一种写法重投：按重放处理，不加消息、不加 Job
        replay = await runtime.append_conversation(upper, ConversationBatch("Conv-ABC", closed_turn(start_sequence=2)))
        assert replay.append.appended_count == 0

        first = await runtime.flush_conversation(lower)
        second = await runtime.flush_conversation(upper)
        assert len(first.jobs) == 1 and second.jobs == ()
        jobs = runtime.components.workflow.jobs
        assert jobs.list_for_conversation(upper) == jobs.list_for_conversation(lower) == first.jobs
        await runtime.close()

    asyncio.run(scenario())


def test_two_different_conversations_differing_only_in_case_reject_the_second(tmp_path: Path) -> None:
    """不另造冲突检测：第二个按同一会话的后续算，被序号连续性拦下；报错写明会话号与原因。"""

    async def scenario() -> None:
        runtime = _runtime(tmp_path)
        await runtime.append_conversation(
            ConversationAddress("Conv-ABC", DAY), ConversationBatch("Conv-ABC", closed_turn(prompt="第一个会话"))
        )
        with pytest.raises(ConversationWriteConflictError) as rejected:
            await runtime.append_conversation(
                ConversationAddress("conv-abc", DAY), ConversationBatch("conv-abc", closed_turn(prompt="另一个会话"))
            )
        text = str(rejected.value)
        assert "'conv-abc'" in text
        assert "sequence 0" in text
        await runtime.close()

    asyncio.run(scenario())


def test_http_rejects_the_second_case_variant_as_a_non_retryable_conflict(tmp_path: Path) -> None:
    runtime = _runtime(tmp_path)
    runtime.start = AsyncMock()  # type: ignore[method-assign]
    close = runtime.close
    runtime.close = AsyncMock()  # type: ignore[method-assign]

    def body(conversation_id: str, content: str) -> dict[str, object]:
        return {
            "conversation_id": conversation_id,
            "started_on": DAY.isoformat(),
            "protocol": "openai_chat_completions",
            "payload": {"messages": [{"role": "user", "content": content}, {"role": "assistant", "content": "好的"}]},
            "start_sequence": 0,
            "occurred_at": "2026-08-07T01:00:00+00:00",
        }

    try:
        with TestClient(create_http_app(runtime), base_url="http://127.0.0.1:8787", raise_server_exceptions=False) as client:
            accepted = client.post("/api/v1/memory/remember", json=body("Conv-ABC", "第一个会话"))
            assert accepted.status_code == 200, accepted.text
            rejected = client.post("/api/v1/memory/remember", json=body("conv-abc", "另一个会话"))
    finally:
        asyncio.run(close())
    assert rejected.status_code == 409, rejected.text
    error = rejected.json()["error"]
    assert error["retryable"] is False
    assert "'conv-abc'" in error["message"] and "sequence 0" in error["message"]
