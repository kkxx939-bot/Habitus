"""回测评估框架的切分、指标与熵上限测试。"""

from __future__ import annotations

import math

import pytest
from test_pattern_learning import _AC_STEP, _WATER, LEARNED_AT, _transition

from prediction.learning import (
    PredictionEvaluationConfig,
    PredictionLearningError,
    backtest,
    entropy_report,
    temporal_split,
)


def test_temporal_split_orders_by_cutoff_and_never_randomizes() -> None:
    documents = (
        _transition("split-a", cutoff_offset_minutes=5),
        _transition("split-b", cutoff_offset_minutes=1),
        _transition("split-c", cutoff_offset_minutes=9),
        _transition("split-d", cutoff_offset_minutes=3),
    )

    train, holdout = temporal_split(documents, train_fraction=0.75)

    assert len(train) == 3 and len(holdout) == 1
    train_cutoffs = [str(item.fields["anchor"]["cutoff_at"]) for item in train]
    holdout_cutoffs = [str(item.fields["anchor"]["cutoff_at"]) for item in holdout]
    assert max(train_cutoffs) < min(holdout_cutoffs)
    assert temporal_split(documents, train_fraction=0.75) == (train, holdout)


def test_backtest_reports_perfect_metrics_for_a_stable_routine() -> None:
    documents = (
        *(
            _transition(f"routine-{index}", cutoff_offset_minutes=index)
            for index in range(4)
        ),
        _transition("routine-eval", cutoff_offset_minutes=30),
    )

    report = backtest(documents, learned_at=LEARNED_AT)

    assert report.train_count == 4 and report.holdout_count == 1
    assert report.labeled_count == 1
    assert report.escape_rate == 0.0
    assert dict(report.top_k_hit_rates)[1] == 1.0
    assert report.log_loss is not None and report.log_loss > 0
    curve = {threshold: (coverage, precision) for threshold, coverage, precision in report.coverage_precision}
    assert curve[0.05] == (1.0, 1.0)
    assert curve[0.95] == (0.0, None)


def test_backtest_counts_escape_unmatched_and_censored_separately() -> None:
    documents = (
        *(
            _transition(f"base-{index}", cutoff_offset_minutes=index)
            for index in range(4)
        ),
        _transition(
            "novel-label",
            label=("关灯", "device_control", ("灯",)),
            cutoff_offset_minutes=31,
        ),
        _transition(
            "other-domain",
            label=("推送代码", "coding_action", ("git",)),
            domain="software_delivery",
            cutoff_offset_minutes=32,
        ),
        _transition("late-censored", censored=True, cutoff_offset_minutes=33),
    )

    report = backtest(
        documents,
        learned_at=LEARNED_AT,
        config=PredictionEvaluationConfig(train_fraction=0.6),
    )

    assert report.train_count == 4
    assert report.censored_count == 1
    assert report.unmatched_count == 1
    assert report.labeled_count == 1
    assert report.escape_rate == 1.0
    assert dict(report.top_k_hit_rates)[1] == 0.0
    assert report.log_loss == pytest.approx(-math.log(1e-6))


def test_empty_distributions_count_as_unmatched_not_perfect() -> None:
    documents = (
        *(
            _transition(f"cens-train-{index}", censored=True, cutoff_offset_minutes=index)
            for index in range(4)
        ),
        _transition("cens-hold", cutoff_offset_minutes=30),
    )

    report = backtest(documents, learned_at=LEARNED_AT)

    assert report.unmatched_count == 1
    assert report.labeled_count == 0
    assert report.log_loss is None


def test_backtest_matches_time_of_day_states_before_the_root() -> None:
    documents = (
        *(
            _transition(
                f"tmp-eval-morning-{index}",
                label=("出门晨跑", "daily_activity", ("公园",)),
                cutoff_offset_minutes=-210 + index,
            )
            for index in range(2)
        ),
        *(
            _transition(f"tmp-eval-mid-{index}", cutoff_offset_minutes=index)
            for index in range(3)
        ),
        _transition(
            "tmp-eval-next-morning",
            label=("出门晨跑", "daily_activity", ("公园",)),
            cutoff_offset_minutes=1230,
        ),
    )

    report = backtest(
        documents,
        learned_at=LEARNED_AT,
        config=PredictionEvaluationConfig(train_fraction=0.84),
    )

    assert report.train_count == 5 and report.holdout_count == 1
    assert dict(report.top_k_hit_rates)[1] == 1.0


def test_entropy_report_separates_root_and_context_ceilings() -> None:
    documents = (
        _transition("ent-a"),
        _transition("ent-b"),
        _transition("ent-c", prefix=1, history=(_AC_STEP,), label=_WATER),
        _transition("ent-d", prefix=1, history=(_AC_STEP,), label=_WATER),
    )

    (ceiling,) = entropy_report(documents)

    assert ceiling.state_level == "action"
    assert ceiling.target_domain == "device_control"
    assert ceiling.labeled_count == 4
    assert ceiling.root_entropy_bits == pytest.approx(1.0)
    assert ceiling.bayes_top1_root == pytest.approx(0.5)
    assert ceiling.context_entropy_bits == pytest.approx(0.0)
    assert ceiling.bayes_top1_context == pytest.approx(1.0)


def test_evaluation_rejects_invalid_inputs() -> None:
    with pytest.raises(PredictionLearningError, match="at least two"):
        temporal_split((_transition("solo"),))

    with pytest.raises(PredictionLearningError, match="strictly between"):
        PredictionEvaluationConfig(train_fraction=1.0)

    with pytest.raises(PredictionLearningError, match="strictly increasing"):
        PredictionEvaluationConfig(top_k=(2, 1))
