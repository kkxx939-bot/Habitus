"""记忆桥接(检索、上下文供给与审计绑定)的确定性合同测试。"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from infrastructure.editor.snapshot import SnapshotState, VersionedSnapshot
from memory.intention.review import MemoryIntentionReviewer
from memory.model import MemoryKind
from memory.retrieval import MemoryMatchedMemory, MemorySearchHit
from memory.uri import MemoryURI
from prediction import PredictionRunSourceBinding
from Runtime.prediction_bridge import (
    MemoryBridgeConfig,
    MemoryBridgeError,
    derive_memory_context,
    memory_provenance,
    memory_query_for_context,
    memory_source_bindings,
    persona_context,
)
from tests.helpers import BASE_TIME, document, memory_snapshot


def _match(doc, *, score: float = 0.9, now: datetime = BASE_TIME) -> MemoryMatchedMemory:
    review = (
        MemoryIntentionReviewer().review(doc, now=now)
        if doc.kind is MemoryKind.INTENTION
        else None
    )
    return MemoryMatchedMemory(
        hit=MemorySearchHit(MemoryURI.from_address(doc.address), score),
        document=doc,
        matched_queries=("下一步",),
        intention_review=review,
    )


def test_memory_context_renders_entries_without_semantic_judgement() -> None:
    intention = document(MemoryKind.INTENTION)
    stale = document(
        MemoryKind.INTENTION,
        fields={"intent_name": "整理相册", "status": "waiting"},
    )
    preference = document(MemoryKind.PREFERENCE)
    entity = document(MemoryKind.ENTITY)
    now = BASE_TIME + timedelta(days=45)

    entries = derive_memory_context(
        (_match(intention, now=now), _match(stale, now=now), _match(preference), _match(entity)),
        now=now,
    )

    assert "未完成事项「完成记忆系统重构」(状态 open,45 天未确认);下一步:补齐测试体系" in entries
    assert "未完成事项「整理相册」(状态 waiting,45 天未确认)" in entries
    assert "偏好简洁直接的回答" in entries
    assert all("Habitus" not in entry for entry in entries)


def test_memory_context_drops_low_relevance_and_completed_items() -> None:
    weak = _match(document(MemoryKind.INTENTION), score=0.2)
    completed = _match(
        document(MemoryKind.INTENTION, fields={"intent_name": "旧事项", "status": "completed"})
    )
    strong = _match(document(MemoryKind.PREFERENCE))

    entries = derive_memory_context((weak, completed, strong), now=BASE_TIME)

    assert entries == ("偏好简洁直接的回答",)


def test_prose_documents_split_into_sentence_entries() -> None:
    prose = document(
        MemoryKind.PREFERENCE,
        fields={
            "topic": "作息",
            "content": "晚上到家一般先打开空调乘凉。为了保证睡眠他会避免深夜喝咖啡。睡前会在床头倒一杯水。",
        },
    )

    entries = derive_memory_context((_match(prose),), now=BASE_TIME)

    assert entries == (
        "晚上到家一般先打开空调乘凉",
        "为了保证睡眠他会避免深夜喝咖啡",
        "睡前会在床头倒一杯水",
    )


def test_persona_context_reads_profile_and_preferences_only() -> None:
    profile = document(
        MemoryKind.PROFILE,
        fields={"content": "- 重视测试的软件开发者\n- 独居"},
    )
    preference = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "开发流程", "content": "- 代码执行后先运行对应测试用例"},
    )

    entries = persona_context((profile, preference))

    assert entries == (
        "重视测试的软件开发者",
        "独居",
        "代码执行后先运行对应测试用例",
    )

    with pytest.raises(MemoryBridgeError, match="only profile and preference"):
        persona_context((document(MemoryKind.INTENTION),))


def test_memory_query_is_built_from_the_behavior_context() -> None:
    from prediction import PredictionContext, PredictionKind
    from tests.unit.prediction.test_pattern_learning import _AC_STEP, _transition

    context = PredictionContext.from_document(
        _transition("query-ctx", prefix=1, history=(_AC_STEP,))
    )

    query = memory_query_for_context(context)

    assert "打开空调" in query
    assert "空调" in query
    assert context.kind is PredictionKind.TRANSITION


def test_memory_provenance_covers_config_and_entries() -> None:
    entries = ("未完成事项「完成记忆系统重构」(状态 open)", "偏好简洁直接的回答")

    baseline = memory_provenance(entries, now=BASE_TIME)

    assert baseline == memory_provenance(entries, now=BASE_TIME)
    assert baseline != memory_provenance(entries[:1], now=BASE_TIME)
    assert baseline != memory_provenance(
        entries, now=BASE_TIME, config=MemoryBridgeConfig(relevance_threshold=0.5)
    )
    assert baseline != memory_provenance(entries, now=BASE_TIME + timedelta(minutes=1))


def test_source_bindings_use_snapshot_identity_revision_digest() -> None:
    snapshot = memory_snapshot(document(MemoryKind.INTENTION))

    bindings = memory_source_bindings((snapshot, snapshot))

    assert bindings == (
        PredictionRunSourceBinding(
            "memory://intentions/完成记忆系统重构.md",
            1,
            "0" * 64,
        ),
    )


def test_bridge_rejects_invalid_inputs() -> None:
    with pytest.raises(MemoryBridgeError, match="sequence"):
        derive_memory_context("not-a-sequence", now=BASE_TIME)  # type: ignore[arg-type]

    with pytest.raises(MemoryBridgeError, match="timezone-aware"):
        derive_memory_context((), now=datetime(2026, 7, 1, 8, 0))

    with pytest.raises(MemoryBridgeError, match="between zero and one"):
        MemoryBridgeConfig(relevance_threshold=1.5)

    missing = VersionedSnapshot(
        identity="memory://intentions/缺失.md",
        state=SnapshotState.MISSING,
        value=None,
        revision=None,
        source_digest=None,
        size_bytes=0,
    )
    with pytest.raises(MemoryBridgeError, match="missing memory snapshot"):
        memory_source_bindings((missing,))
