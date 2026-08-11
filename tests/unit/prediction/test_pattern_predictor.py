"""Pattern 图预测判决链的匹配、记忆调整与执行门槛测试。"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from test_pattern_learning import _AC_STEP, _WATER, LEARNED_AT, _transition

from prediction import (
    PredictionAbstentionReason,
    PredictionContext,
    PredictionPatternGenerationPublisher,
    PredictionPatternGraph,
    PredictionPublisher,
    PredictionRunSourceBinding,
    PredictionTree,
)
from prediction.learning import PredictionPatternLearner
from prediction.learning.keys import (
    logical_state_key,
    sequence_state_identity,
    temporal_state_identity,
)
from prediction.predictor import (
    PatternGraphPredictor,
    PredictionDecisionConfig,
    PredictionPredictorError,
)

_BINDINGS = (
    PredictionRunSourceBinding("behavior://behaviors/events/2026/08/08/runtime.md", 1, "b" * 64),
)
_RELAXED = PredictionDecisionConfig(
    execute_probability_threshold=0.4,
    execute_margin=0.1,
    min_execute_support=2,
)


def _published_graph(tmp_path) -> PredictionPatternGraph:
    documents = (
        _transition("ev-a"),
        _transition("ev-b"),
        _transition("ev-c"),
        _transition("ev-d", label=_WATER),
        _transition("ev-a", prefix=1, history=(_AC_STEP,), label=_WATER),
        _transition("ev-b", prefix=1, history=(_AC_STEP,), label=_WATER),
        _transition("ev-c", prefix=1, history=(_AC_STEP,), terminal=True),
        _transition("ev-d", prefix=1, history=(_WATER,), label=("关灯", "device_control", ("灯",))),
    )
    tree = PredictionTree(tmp_path / "prediction")
    PredictionPublisher(tree).publish(documents)
    patterns = PredictionPatternLearner().learn(documents, learned_at=LEARNED_AT)
    PredictionPatternGenerationPublisher(tree).publish(patterns)
    return PredictionPatternGraph(tree)


def _context(token: str, **kwargs) -> PredictionContext:
    return PredictionContext.from_document(_transition(token, **kwargs))


def test_predictor_matches_context_state_and_gates_execution(tmp_path) -> None:
    graph = _published_graph(tmp_path)
    context = _context("rt-child", prefix=1, history=(_AC_STEP,), label=_WATER)

    relaxed = PatternGraphPredictor(graph, config=_RELAXED)
    decision = relaxed.predict(
        context,
        predicted_at=LEARNED_AT,
        horizon_seconds=3600,
        source_bindings=_BINDINGS,
    )

    run = decision.run
    assert run.abstention_reason is None
    assert run.candidates[0].semantics == "倒一杯水"
    assert run.candidates[0].evidence_refs[0].startswith("prediction://patterns/branches/")
    assert sum(item.probability for item in run.candidates) < 1
    child_key = logical_state_key(
        sequence_state_identity(
            "action",
            "device_control",
            {"behavior_type": "device_control", "semantics": "打开空调", "target_refs": ["空调"]},
        )
    )
    assert child_key in {binding.logical_state_key for binding in run.pattern_bindings}
    assert decision.execute and decision.blocked_gates == ()

    strict = PatternGraphPredictor(
        graph,
        config=PredictionDecisionConfig(execute_probability_threshold=0.9, execute_margin=0.5),
    )
    blocked = strict.predict(
        context,
        predicted_at=LEARNED_AT,
        horizon_seconds=3600,
        source_bindings=_BINDINGS,
    )
    assert not blocked.execute
    assert "probability_below_threshold" in blocked.blocked_gates
    assert blocked.run.candidates == run.candidates


def test_no_context_queries_match_the_time_of_day_state(tmp_path) -> None:
    documents = (
        *(
            _transition(
                f"tmp-morning-{index}",
                label=("出门晨跑", "daily_activity", ("公园",)),
                cutoff_offset_minutes=-210 + index,
            )
            for index in range(3)
        ),
        _transition("tmp-mid-a"),
        _transition("tmp-mid-b"),
    )
    tree = PredictionTree(tmp_path / "prediction")
    PredictionPublisher(tree).publish(documents)
    patterns = PredictionPatternLearner().learn(documents, learned_at=LEARNED_AT)
    PredictionPatternGenerationPublisher(tree).publish(patterns)
    graph = PredictionPatternGraph(tree)

    context = _context("tmp-query", cutoff_offset_minutes=-210)
    morning_at = datetime(2026, 8, 8, 7, 30, tzinfo=timezone.utc)
    decision = PatternGraphPredictor(graph, config=_RELAXED).predict(
        context,
        predicted_at=morning_at,
        horizon_seconds=3600,
        source_bindings=_BINDINGS,
    )

    assert decision.run.candidates[0].semantics == "出门晨跑"
    temporal_key = logical_state_key(
        temporal_state_identity("action", "device_control", "06-09")
    )
    assert temporal_key in {
        binding.logical_state_key for binding in decision.run.pattern_bindings
    }


def test_unseen_context_falls_back_to_the_root_state(tmp_path) -> None:
    graph = _published_graph(tmp_path)
    context = _context(
        "rt-unseen",
        prefix=1,
        history=(("看电视", "daily_activity", ("电视",)),),
        label=_WATER,
    )

    decision = PatternGraphPredictor(graph, config=_RELAXED).predict(
        context,
        predicted_at=LEARNED_AT,
        horizon_seconds=3600,
        source_bindings=_BINDINGS,
    )

    root_key = logical_state_key(sequence_state_identity("action", "device_control", None))
    assert decision.run.pattern_bindings[0].logical_state_key == root_key
    assert len(decision.run.pattern_bindings) == 1
    assert {item.semantics for item in decision.run.candidates} == {"打开空调", "倒一杯水", "关灯"}


def test_sparse_states_inherit_parent_branches_into_the_candidate_set(tmp_path) -> None:
    documents = (
        *(_transition(f"inh-{index}") for index in range(8)),
        *(
            _transition(
                f"inh-tv-{index}",
                prefix=1,
                history=(("看电视", "daily_activity", ("电视",)),),
                label=_WATER,
            )
            for index in range(2)
        ),
    )
    tree = PredictionTree(tmp_path / "prediction")
    PredictionPublisher(tree).publish(documents)
    patterns = PredictionPatternLearner().learn(documents, learned_at=LEARNED_AT)
    PredictionPatternGenerationPublisher(tree).publish(patterns)
    graph = PredictionPatternGraph(tree)
    context = _context(
        "inh-query",
        prefix=1,
        history=(("看电视", "daily_activity", ("电视",)),),
        label=_WATER,
    )

    run = PatternGraphPredictor(graph, config=_RELAXED).predict(
        context,
        predicted_at=LEARNED_AT,
        horizon_seconds=3600,
        source_bindings=_BINDINGS,
    ).run

    semantics = [item.semantics for item in run.candidates]
    assert semantics[0] == "倒一杯水"
    assert "打开空调" in semantics
    inherited = next(item for item in run.candidates if item.semantics == "打开空调")
    assert inherited.probability > 0.2
    assert sum(item.probability for item in run.candidates) < 1
    assert len(run.pattern_bindings) == 2


def test_excluded_branch_keys_remove_candidates_and_regate(tmp_path) -> None:
    graph = _published_graph(tmp_path)
    context = _context("rt-exclude")
    predictor = PatternGraphPredictor(graph, config=_RELAXED)

    baseline = predictor.predict(
        context,
        predicted_at=LEARNED_AT,
        horizon_seconds=3600,
        source_bindings=_BINDINGS,
    ).run
    water_key = next(
        str(item.payload["branch_key"])
        for item in baseline.candidates
        if item.semantics == "倒一杯水"
    )

    filtered = predictor.predict(
        context,
        predicted_at=LEARNED_AT,
        horizon_seconds=3600,
        source_bindings=_BINDINGS,
        excluded_branch_keys=(water_key,),
    ).run

    assert all(item.semantics != "倒一杯水" for item in filtered.candidates)
    assert filtered.candidates
    assert filtered.run_id != baseline.run_id
    assert filtered.excluded_branch_keys == (water_key,)
    assert baseline.excluded_branch_keys == ()


def test_advisor_adjustment_is_frozen_into_the_run_record(tmp_path) -> None:
    graph = _published_graph(tmp_path)
    context = _context("rt-adjrec")
    predictor = PatternGraphPredictor(graph, config=_RELAXED, advisor_digest="a" * 64)

    plain = predictor.predict(
        context,
        predicted_at=LEARNED_AT,
        horizon_seconds=3600,
        source_bindings=_BINDINGS,
    ).run
    water_key = next(
        str(item.payload["branch_key"])
        for item in plain.candidates
        if item.semantics == "倒一杯水"
    )
    weighted = predictor.predict(
        context,
        predicted_at=LEARNED_AT,
        horizon_seconds=3600,
        source_bindings=_BINDINGS,
        advisor_weights={water_key: 1.0},
    ).run

    assert plain.advisor_adjustment is None
    assert weighted.advisor_adjustment is not None
    assert dict(weighted.advisor_adjustment) == {water_key: 1.0}
    assert weighted.run_id != plain.run_id
    from prediction import PredictionRun

    assert PredictionRun.from_dict(weighted.to_dict()) == weighted


def test_terminal_scope_queries_reach_termination_branches(tmp_path) -> None:
    graph = _published_graph(tmp_path)
    context = _context("rt-terminal", prefix=1, history=(_AC_STEP,), terminal=True)

    decision = PatternGraphPredictor(
        graph,
        config=PredictionDecisionConfig(
            execute_probability_threshold=0.1,
            execute_margin=0.01,
            min_execute_support=1,
        ),
    ).predict(
        context,
        predicted_at=LEARNED_AT,
        horizon_seconds=3600,
        source_bindings=_BINDINGS,
    )

    assert decision.run.abstention_reason is None
    assert decision.run.candidates
    assert all(
        item.payload["target_kind"] == "termination" for item in decision.run.candidates
    )


def test_predictor_handshakes_the_learning_identity(tmp_path) -> None:
    documents = tuple(_transition(f"hs-{index}") for index in range(4))
    tree = PredictionTree(tmp_path / "prediction")
    PredictionPublisher(tree).publish(documents)
    learner = PredictionPatternLearner()
    patterns = learner.learn(documents, learned_at=LEARNED_AT)
    PredictionPatternGenerationPublisher(tree).publish(
        patterns, learning_identity=learner.match_identity()
    )
    graph = PredictionPatternGraph(tree)
    context = _context("hs-query")

    aligned = PatternGraphPredictor(graph, config=_RELAXED).predict(
        context,
        predicted_at=LEARNED_AT,
        horizon_seconds=3600,
        source_bindings=_BINDINGS,
    )
    assert aligned.run.candidates

    drifted = PatternGraphPredictor(
        graph,
        config=PredictionDecisionConfig(temporal_utc_offset_minutes=480),
    )
    with pytest.raises(PredictionPredictorError, match="does not match the learned generation"):
        drifted.predict(
            context,
            predicted_at=LEARNED_AT,
            horizon_seconds=3600,
            source_bindings=_BINDINGS,
        )


def test_memory_provenance_is_recorded_and_changes_the_run_identity(tmp_path) -> None:
    graph = _published_graph(tmp_path)
    context = _context("prov-query")
    predictor = PatternGraphPredictor(graph, config=_RELAXED)

    plain = predictor.predict(
        context,
        predicted_at=LEARNED_AT,
        horizon_seconds=3600,
        source_bindings=_BINDINGS,
    ).run
    stamped = predictor.predict(
        context,
        predicted_at=LEARNED_AT,
        horizon_seconds=3600,
        source_bindings=_BINDINGS,
        memory_provenance="a" * 64,
    ).run

    assert plain.memory_provenance is None
    assert stamped.memory_provenance == "a" * 64
    assert plain.run_id != stamped.run_id


def test_store_integrity_failures_are_not_swallowed(tmp_path) -> None:
    from prediction import PredictionPatternGenerationStoreError

    graph = _published_graph(tmp_path)
    context = _context("rt-corrupt")
    predictor = PatternGraphPredictor(graph, config=_RELAXED)

    def _corrupt(_key: str):
        raise PredictionPatternGenerationStoreError(
            "active pattern shard is not canonically serialized"
        )

    graph.generation_store.read_active_state = _corrupt  # type: ignore[method-assign]
    with pytest.raises(PredictionPredictorError, match="integrity"):
        predictor.predict(
            context,
            predicted_at=LEARNED_AT,
            horizon_seconds=3600,
            source_bindings=_BINDINGS,
        )


def test_advisor_weights_reorder_candidates_within_bounds(tmp_path) -> None:
    graph = _published_graph(tmp_path)
    context = _context("rt-advisor")
    plain = PatternGraphPredictor(graph, config=_RELAXED)
    advised = PatternGraphPredictor(graph, config=_RELAXED, advisor_digest="a" * 64)
    assert plain.predictor_digest != advised.predictor_digest

    baseline = advised.predict(
        context,
        predicted_at=LEARNED_AT,
        horizon_seconds=3600,
        source_bindings=_BINDINGS,
    ).run
    water_key = next(
        str(item.payload["branch_key"])
        for item in baseline.candidates
        if item.semantics == "倒一杯水"
    )
    boosted = advised.predict(
        context,
        predicted_at=LEARNED_AT,
        horizon_seconds=3600,
        source_bindings=_BINDINGS,
        advisor_weights={water_key: 1.0},
    ).run

    assert boosted.candidates[0].semantics == "倒一杯水"
    assert sum(item.probability for item in boosted.candidates) == pytest.approx(
        sum(item.probability for item in baseline.candidates)
    )

    with pytest.raises(PredictionPredictorError, match="advisor_digest"):
        plain.predict(
            context,
            predicted_at=LEARNED_AT,
            horizon_seconds=3600,
            source_bindings=_BINDINGS,
            advisor_weights={water_key: 1.0},
        )


def test_predictor_abstains_with_controlled_reasons(tmp_path) -> None:
    empty_graph = PredictionPatternGraph(PredictionTree(tmp_path / "empty"))
    context = _context("rt-empty")

    missing = PatternGraphPredictor(empty_graph).predict(
        context,
        predicted_at=LEARNED_AT,
        horizon_seconds=3600,
        source_bindings=_BINDINGS,
    )
    assert missing.run.abstention_reason is PredictionAbstentionReason.NO_MATCHING_STATE
    assert not missing.execute
    assert missing.blocked_gates == ("abstained:no_matching_state",)

    guarded = PatternGraphPredictor(
        _published_graph(tmp_path),
        config=PredictionDecisionConfig(min_coverage_score=0.95),
    ).predict(
        context,
        predicted_at=LEARNED_AT,
        horizon_seconds=3600,
        source_bindings=_BINDINGS,
    )
    assert guarded.run.abstention_reason is PredictionAbstentionReason.INSUFFICIENT_CONTEXT

    with pytest.raises(PredictionPredictorError, match="PredictionContext"):
        PatternGraphPredictor(empty_graph).predict(
            object(),  # type: ignore[arg-type]
            predicted_at=LEARNED_AT,
            horizon_seconds=3600,
            source_bindings=_BINDINGS,
        )
