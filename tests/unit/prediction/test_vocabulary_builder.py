"""行为词清单提取与合并守门的合同测试。"""

from __future__ import annotations

import pytest
from test_pattern_learning import _AC_STEP, _WATER, LEARNED_AT, _transition

from prediction.learning import (
    PredictionLearningError,
    PredictionPatternLearner,
    PredictionVocabularyMergeConfig,
    backtest,
    behavior_token_inventory,
    validate_merge_proposals,
)

_AC_ALIAS_STEP = ("开空调", "device_control", ("空调",))
_LIGHT_STEP = ("关灯", "device_control", ("灯",))
_SLEEP = ("上床睡觉", "daily_activity", ())


def _corpus() -> tuple:
    return (
        *(
            _transition(f"inv-ac-{index}", prefix=1, history=(_AC_STEP,), label=_WATER)
            for index in range(3)
        ),
        *(
            _transition(f"inv-alias-{index}", prefix=1, history=(_AC_ALIAS_STEP,), label=_WATER)
            for index in range(3)
        ),
        *(
            _transition(f"inv-light-{index}", prefix=1, history=(_LIGHT_STEP,), label=_SLEEP)
            for index in range(3)
        ),
    )


def test_inventory_reports_usage_and_successor_distributions() -> None:
    documents = (
        _transition("inv-a"),
        _transition("inv-b"),
        _transition("inv-c", prefix=1, history=(_AC_STEP,), label=_WATER),
    )

    usages = {usage.token: usage for usage in behavior_token_inventory(documents)}

    assert usages["打开空调"].label_count == 2
    assert usages["打开空调"].context_count == 1
    assert len(usages["打开空调"].next_distribution) == 1
    assert usages["打开空调"].next_distribution[0][1] == 1
    assert usages["倒一杯水"].label_count == 1


def test_merge_gate_accepts_same_distribution_and_rejects_divergent_pairs() -> None:
    report = validate_merge_proposals(
        _corpus(),
        {"开空调": "打开空调", "关灯": "打开空调", "冲咖啡": "打开空调"},
    )

    assert ("开空调", "打开空调") in report.accepted
    assert ("冲咖啡", "打开空调") in report.accepted
    assert len(report.rejected) == 1
    source, target, reason = report.rejected[0]
    assert (source, target) == ("关灯", "打开空调")
    assert "diverge" in reason
    assert report.vocabulary.canonical_token("开空调", "token") == "打开空调"
    assert report.vocabulary.canonical_token("关灯", "token") == "关灯"


def test_gate_evaluates_on_the_merged_successor_key_space() -> None:
    documents = (
        *(
            _transition(
                f"pair-a-{index}",
                prefix=1,
                history=(("动作甲", "daily_activity", ()),),
                label=("喝水", "daily_activity", ("水杯",)),
            )
            for index in range(3)
        ),
        *(
            _transition(
                f"pair-b-{index}",
                prefix=1,
                history=(("动作乙", "daily_activity", ()),),
                label=("饮水", "daily_activity", ("水杯",)),
            )
            for index in range(3)
        ),
    )

    report = validate_merge_proposals(
        documents,
        {"喝水": "饮水", "动作甲": "动作乙"},
    )

    assert ("喝水", "饮水") in report.accepted
    assert ("动作甲", "动作乙") in report.accepted
    assert report.rejected == ()


def test_merged_vocabulary_closes_the_escape_gap_in_backtest() -> None:
    documents = (
        *(
            _transition(f"merge-{index}", cutoff_offset_minutes=index)
            for index in range(4)
        ),
        _transition(
            "merge-eval",
            label=("开空调", "device_control", ("空调",)),
            cutoff_offset_minutes=30,
        ),
    )

    plain = backtest(documents, learned_at=LEARNED_AT)
    assert plain.escape_rate == 1.0
    assert dict(plain.top_k_hit_rates)[1] == 0.0

    report = validate_merge_proposals(documents, {"开空调": "打开空调"})
    merged = backtest(
        documents,
        learned_at=LEARNED_AT,
        learner=PredictionPatternLearner(vocabulary=report.vocabulary),
    )
    assert merged.escape_rate == 0.0
    assert dict(merged.top_k_hit_rates)[1] == 1.0
    assert merged.log_loss is not None and plain.log_loss is not None
    assert merged.log_loss < plain.log_loss


def test_merge_validation_rejects_invalid_inputs() -> None:
    with pytest.raises(PredictionLearningError, match="map source tokens"):
        validate_merge_proposals((_transition("bad-a"), _transition("bad-b")), {"a": 1})  # type: ignore[dict-item]

    with pytest.raises(PredictionLearningError, match="positive integer"):
        PredictionVocabularyMergeConfig(min_context_support=0)

    with pytest.raises(PredictionLearningError, match="chains"):
        validate_merge_proposals(
            (_transition("chain-a"), _transition("chain-b")),
            {"甲": "乙", "乙": "丙"},
        )
