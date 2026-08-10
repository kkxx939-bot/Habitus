"""记忆检索命中到预测信号桥接的确定性折算测试。"""

from __future__ import annotations

import math
from datetime import datetime, timedelta

import pytest

from infrastructure.editor.snapshot import SnapshotState, VersionedSnapshot
from memory.intention.review import MemoryIntentionReviewer
from memory.model import MemoryKind
from memory.retrieval import MemoryMatchedMemory, MemorySearchHit
from memory.uri import MemoryURI
from prediction import PredictionRunSourceBinding
from Runtime.prediction_bridge import (
    MemorySignalBridgeConfig,
    MemorySignalBridgeError,
    PredictionStanceTable,
    derive_memory_signals,
    derive_persona_signals,
    memory_query_for_context,
    memory_source_bindings,
    signal_provenance,
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


def test_signals_rank_by_kind_prior_status_and_relevance() -> None:
    fresh = document(MemoryKind.INTENTION)
    preference = document(MemoryKind.PREFERENCE)
    blocked = document(
        MemoryKind.INTENTION,
        fields={"intent_name": "等待快递", "status": "blocked", "next_step": "去驿站取快递"},
    )
    entity = document(MemoryKind.ENTITY)

    signals = derive_memory_signals(
        (_match(fresh), _match(preference), _match(blocked), _match(entity)),
        now=BASE_TIME,
    )

    by_semantics = {item.semantics: item for item in signals}
    assert set(by_semantics) == {
        "补齐测试体系",
        "完成记忆系统重构",
        "偏好简洁直接的回答",
        "去驿站取快递",
        "等待快递",
    }
    assert by_semantics["补齐测试体系"].kind == "intention"
    assert by_semantics["补齐测试体系"].weight == pytest.approx(0.9)
    assert by_semantics["偏好简洁直接的回答"].weight == pytest.approx(0.7 * 0.9)
    assert by_semantics["去驿站取快递"].weight == pytest.approx(0.5 * 0.9)
    assert signals[0].weight >= signals[-1].weight
    assert by_semantics["补齐测试体系"].source_uri == "memory://intentions/完成记忆系统重构.md"


def test_unconfirmed_intentions_decay_exponentially() -> None:
    stale = document(MemoryKind.INTENTION)
    now = BASE_TIME + timedelta(days=120)

    signals = derive_memory_signals((_match(stale, now=now),), now=now)

    assert signals[0].weight == pytest.approx(0.9 * math.exp(-2), rel=1e-6)


def test_completed_intentions_bullet_caps_and_dedup() -> None:
    completed = document(
        MemoryKind.INTENTION,
        fields={"intent_name": "旧事项", "status": "completed"},
    )
    chores = document(
        MemoryKind.PREFERENCE,
        fields={
            "topic": "家务",
            "content": "- 回家先开空调\n- 晚上九点后不用洗衣机\n- 周末大扫除\n- 第四条应被截断",
        },
    )
    duplicate = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "环境", "content": "- 回家先开空调"},
    )

    signals = derive_memory_signals(
        (_match(completed), _match(chores), _match(duplicate, score=0.5)),
        now=BASE_TIME,
    )

    semantics = [item.semantics for item in signals]
    assert "旧事项" not in semantics
    assert "第四条应被截断" not in semantics
    assert semantics.count("回家先开空调") == 1
    assert next(item for item in signals if item.semantics == "回家先开空调").weight == pytest.approx(0.7 * 0.9)

    capped = derive_memory_signals(
        (_match(chores),),
        now=BASE_TIME,
        config=MemorySignalBridgeConfig(max_signals=2),
    )
    assert len(capped) == 2


def test_low_relevance_hits_are_dropped_not_floored() -> None:
    weak = _match(document(MemoryKind.INTENTION), score=0.2)
    strong = _match(document(MemoryKind.PREFERENCE), score=0.9)

    signals = derive_memory_signals((weak, strong), now=BASE_TIME)

    assert {item.kind for item in signals} == {"preference"}


def test_negative_preferences_become_oppose_signals() -> None:
    negative = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "洗衣", "content": "- 晚上九点后不用洗衣机\n- 回家先开空调"},
    )

    signals = derive_memory_signals((_match(negative),), now=BASE_TIME)

    by_semantics = {item.semantics: item for item in signals}
    assert by_semantics["晚上九点后不用洗衣机"].stance == "oppose"
    assert by_semantics["回家先开空调"].stance == "support"


def test_stance_markers_do_not_flip_positive_compound_words() -> None:
    tricky = document(
        MemoryKind.PREFERENCE,
        fields={
            "topic": "生活",
            "content": (
                "- 特别喜欢睡前喝一杯牛奶\n"
                "- 按类别整理相册\n"
                "- 别忘了倒垃圾"
            ),
        },
    )
    ambiguous = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "早饭", "content": "- 不吃早饭会低血糖，要记得吃早饭"},
    )

    signals = derive_memory_signals((_match(tricky), _match(ambiguous)), now=BASE_TIME)

    by_semantics = {item.semantics: item.stance for item in signals}
    assert by_semantics["特别喜欢睡前喝一杯牛奶"] == "support"
    assert by_semantics["按类别整理相册"] == "support"
    assert by_semantics["别忘了倒垃圾"] == "support"
    assert by_semantics["不吃早饭会低血糖，要记得吃早饭"] == "support"


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


def test_persona_signals_come_from_profile_and_preferences_with_standing_weights() -> None:
    profile = document(
        MemoryKind.PROFILE,
        fields={"content": "- 重视测试的软件开发者\n- 独居"},
    )
    preference = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "开发流程", "content": "- 代码执行后先运行对应测试用例\n- 不要直接推送到主分支"},
    )

    signals = derive_persona_signals((profile, preference))

    by_semantics = {item.semantics: item for item in signals}
    assert by_semantics["重视测试的软件开发者"].kind == "profile"
    assert by_semantics["重视测试的软件开发者"].weight == pytest.approx(0.35)
    assert by_semantics["代码执行后先运行对应测试用例"].kind == "preference"
    assert by_semantics["代码执行后先运行对应测试用例"].weight == pytest.approx(0.5)
    assert by_semantics["代码执行后先运行对应测试用例"].stance == "support"
    assert by_semantics["不要直接推送到主分支"].stance == "oppose"

    with pytest.raises(MemorySignalBridgeError, match="only profile and preference"):
        derive_persona_signals((document(MemoryKind.INTENTION),))


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


def test_signal_provenance_covers_config_table_and_signals() -> None:
    signals = derive_memory_signals((_match(document(MemoryKind.INTENTION)),), now=BASE_TIME)

    baseline = signal_provenance(signals, now=BASE_TIME)
    assert baseline == signal_provenance(signals, now=BASE_TIME)
    assert baseline != signal_provenance((), now=BASE_TIME)
    assert baseline != signal_provenance(
        signals, now=BASE_TIME, stance_table=PredictionStanceTable(stances={"某条": "oppose"})
    )
    assert baseline != signal_provenance(
        signals, now=BASE_TIME, config=MemorySignalBridgeConfig(relevance_threshold=0.5)
    )


def test_bridge_rejects_invalid_inputs() -> None:
    with pytest.raises(MemorySignalBridgeError, match="sequence"):
        derive_memory_signals("not-a-sequence", now=BASE_TIME)  # type: ignore[arg-type]

    with pytest.raises(MemorySignalBridgeError, match="timezone-aware"):
        derive_memory_signals((), now=datetime(2026, 7, 1, 8, 0))

    missing = VersionedSnapshot(
        identity="memory://intentions/缺失.md",
        state=SnapshotState.MISSING,
        value=None,
        revision=None,
        source_digest=None,
        size_bytes=0,
    )
    with pytest.raises(MemorySignalBridgeError, match="missing memory snapshot"):
        memory_source_bindings((missing,))
