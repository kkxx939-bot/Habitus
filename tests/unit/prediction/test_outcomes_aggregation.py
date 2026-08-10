"""Consequence → outcomes 聚合与风险门激活的合同测试。"""

from __future__ import annotations

import pytest
from test_pattern_learning import LEARNED_AT, _consequence, _transition

from prediction import (
    PredictionContext,
    PredictionPatternDocument,
    PredictionPatternGenerationPublisher,
    PredictionPatternGraph,
    PredictionPatternKind,
    PredictionPublisher,
    PredictionRunSourceBinding,
    PredictionTree,
)
from prediction.learning import PredictionLearningError, PredictionPatternLearner
from prediction.predictor import PatternGraphPredictor, PredictionDecisionConfig

_BINDINGS = (
    PredictionRunSourceBinding("behavior://behaviors/events/2026/08/08/runtime.md", 1, "b" * 64),
)


def _ac_branch(patterns) -> PredictionPatternDocument:
    return next(
        item
        for item in patterns
        if item.kind is PredictionPatternKind.BRANCH
        and item.fields["behavior"]["semantics"] == "打开空调"
        and item.fields["outcomes"]
    )


def test_consequences_aggregate_into_branch_outcome_distributions() -> None:
    transitions = tuple(_transition(f"oc-{index}") for index in range(4))
    consequences = (
        _consequence("oc-0", delay_seconds=300.0),
        _consequence("oc-1", delay_seconds=600.0),
        _consequence(
            "oc-2",
            outcome_type="correction",
            valence="negative",
            outcome_semantics="用户随即关闭了空调",
            delay_seconds=60.0,
        ),
    )

    patterns = PredictionPatternLearner().learn(
        transitions, learned_at=LEARNED_AT, consequences=consequences
    )

    branch = _ac_branch(patterns)
    outcomes = branch.fields["outcomes"]
    assert len(outcomes) == 2
    assert sum(item["probability"] for item in outcomes) < 1
    positive = next(item for item in outcomes if item["valence"] == "positive")
    negative = next(item for item in outcomes if item["valence"] == "negative")
    assert positive["probability"] > negative["probability"]
    assert positive["semantics"] == "室温下降"
    assert positive["average_delay_seconds"] == pytest.approx(450.0)
    assert negative["average_delay_seconds"] == pytest.approx(60.0)

    plain = PredictionPatternLearner().learn(transitions, learned_at=LEARNED_AT)
    assert (
        plain[0].fields["pattern_generation"] != patterns[0].fields["pattern_generation"]
    )


def test_outcome_revisions_converge_to_the_latest_materialization() -> None:
    transitions = tuple(_transition(f"rev-{index}") for index in range(4))
    stale = _consequence(
        "rev-0",
        valence="negative",
        outcome_type="correction",
        outcome_semantics="旧修订：被用户纠正",
        revision=1,
    )
    fresh = _consequence(
        "rev-0",
        valence="positive",
        outcome_semantics="新修订：状态正常变化",
        revision=2,
    )

    patterns = PredictionPatternLearner().learn(
        transitions, learned_at=LEARNED_AT, consequences=(stale, fresh)
    )

    outcomes = _ac_branch(patterns).fields["outcomes"]
    assert [item["valence"] for item in outcomes] == ["positive"]
    assert outcomes[0]["semantics"] == "新修订：状态正常变化"


def test_negative_outcome_history_blocks_execution_through_the_risk_gate(tmp_path) -> None:
    transitions = tuple(_transition(f"risk-{index}") for index in range(4))
    consequences = tuple(
        _consequence(
            f"risk-{index}",
            outcome_type="correction",
            valence="negative",
            outcome_semantics="用户随即关闭了空调",
            delay_seconds=30.0,
        )
        for index in range(3)
    )
    tree = PredictionTree(tmp_path / "prediction")
    PredictionPublisher(tree).publish(transitions)
    patterns = PredictionPatternLearner().learn(
        transitions, learned_at=LEARNED_AT, consequences=consequences
    )
    PredictionPatternGenerationPublisher(tree).publish(patterns)
    graph = PredictionPatternGraph(tree)
    context = PredictionContext.from_document(_transition("risk-query"))
    predictor = PatternGraphPredictor(
        graph,
        config=PredictionDecisionConfig(
            execute_probability_threshold=0.4,
            execute_margin=0.1,
            min_execute_support=2,
        ),
    )

    decision = predictor.predict(
        context,
        predicted_at=LEARNED_AT,
        horizon_seconds=3600,
        source_bindings=_BINDINGS,
    )

    assert decision.run.candidates[0].semantics == "打开空调"
    assert not decision.execute
    assert decision.blocked_gates == ("negative_outcome_risk",)


def test_learner_rejects_non_consequence_documents_in_consequences() -> None:
    with pytest.raises(PredictionLearningError, match="ConsequenceSamples"):
        PredictionPatternLearner().learn(
            (_transition("bad-mix"),),
            learned_at=LEARNED_AT,
            consequences=(_transition("bad-mix-2"),),
        )
