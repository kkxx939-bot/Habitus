"""等渗概率校准的拟合、应用与判决链注入测试。"""

from __future__ import annotations

import pytest
from test_pattern_learning import _WATER, LEARNED_AT, _transition

from prediction import (
    PredictionContext,
    PredictionPatternGenerationPublisher,
    PredictionPatternGraph,
    PredictionPublisher,
    PredictionRunSourceBinding,
    PredictionTree,
)
from prediction.learning import (
    PredictionLearningError,
    PredictionPatternLearner,
    PredictionProbabilityCalibration,
    backtest,
    fit_probability_calibration,
)
from prediction.predictor import PatternGraphPredictor, PredictionDecisionConfig

_BINDINGS = (
    PredictionRunSourceBinding("behavior://behaviors/events/2026/08/08/runtime.md", 1, "b" * 64),
)


def test_pav_fits_a_monotone_map_from_outcomes() -> None:
    calibration = PredictionProbabilityCalibration.from_outcomes(
        (
            (0.9, False),
            (0.9, False),
            (0.9, True),
            (0.2, False),
        )
    )

    assert calibration.sample_count == 4
    assert calibration.points == ((0.2, 0.0), (0.9, pytest.approx(1 / 3)))
    assert calibration.apply(0.9) == pytest.approx(1 / 3)
    assert calibration.apply(0.99) == pytest.approx(1 / 3)
    assert calibration.apply(0.05) == pytest.approx(0.0)
    values = [calibration.apply(x) for x in (0.05, 0.2, 0.55, 0.9)]
    assert values == sorted(values)


def test_apply_interpolates_and_identity_when_empty() -> None:
    calibration = PredictionProbabilityCalibration(points=((0.2, 0.1), (0.8, 0.5)))

    assert calibration.apply(0.1) == pytest.approx(0.1)
    assert calibration.apply(0.5) == pytest.approx(0.3)
    assert calibration.apply(0.9) == pytest.approx(0.5)
    assert PredictionProbabilityCalibration().apply(0.42) == pytest.approx(0.42)
    assert calibration.version != PredictionProbabilityCalibration().version

    with pytest.raises(PredictionLearningError, match="strictly increasing"):
        PredictionProbabilityCalibration(points=((0.5, 0.1), (0.5, 0.2)))
    with pytest.raises(PredictionLearningError, match="non-decreasing"):
        PredictionProbabilityCalibration(points=((0.2, 0.5), (0.8, 0.1)))


def test_pav_survives_massive_duplicate_probabilities() -> None:
    outcomes = []
    for value, hits, total in ((1 / 3, 9, 28), (2 / 3, 20, 28), (0.9999991267577347, 27, 28)):
        outcomes.extend((value, index < hits) for index in range(total))

    calibration = PredictionProbabilityCalibration.from_outcomes(outcomes)

    assert calibration.sample_count == 84
    values = [y for _x, y in calibration.points]
    assert values == sorted(values)


def test_calibration_runs_before_reallocation_and_ties_break_by_raw_probability(tmp_path) -> None:
    documents = (
        *(_transition(f"order-{index}") for index in range(3)),
        *(_transition(f"order-w{index}", label=_WATER) for index in range(2)),
    )
    tree = PredictionTree(tmp_path / "prediction")
    PredictionPublisher(tree).publish(documents)
    patterns = PredictionPatternLearner().learn(documents, learned_at=LEARNED_AT)
    PredictionPatternGenerationPublisher(tree).publish(patterns)
    graph = PredictionPatternGraph(tree)
    context = PredictionContext.from_document(_transition("order-query"))
    config = PredictionDecisionConfig(
        execute_probability_threshold=0.3,
        execute_margin=0.05,
        min_execute_support=2,
    )
    shrink = PredictionProbabilityCalibration(points=((0.2, 0.2), (0.8, 0.5)))
    flat = PredictionProbabilityCalibration(points=((0.05, 0.45), (0.95, 0.45)))

    def _run(calibration, weights=None):
        predictor = PatternGraphPredictor(
            graph,
            config=config,
            calibration=calibration,
            advisor_digest="a" * 64 if weights is not None else None,
        )
        return predictor.predict(
            context,
            predicted_at=LEARNED_AT,
            horizon_seconds=3600,
            source_bindings=_BINDINGS,
            advisor_weights=weights,
        ).run

    plain = _run(shrink)
    water_key = next(
        str(item.payload["branch_key"]) for item in plain.candidates if item.semantics == "倒一杯水"
    )
    advised = _run(shrink, weights={water_key: 1.0})
    plain_total = sum(item.probability for item in plain.candidates)
    advised_total = sum(item.probability for item in advised.candidates)
    assert advised_total == pytest.approx(plain_total)

    flat_run = _run(flat)
    raw_order = [item.semantics for item in _run(None).candidates]
    assert [item.semantics for item in flat_run.candidates] == raw_order


def test_calibration_is_fitted_without_seeing_the_final_holdout() -> None:
    shared = (
        *(
            _transition(f"cal-{index}", cutoff_offset_minutes=index)
            for index in range(6)
        ),
    )
    documents = (
        *shared,
        _transition("cal-hold-a", cutoff_offset_minutes=30),
        _transition("cal-hold-b", label=_WATER, cutoff_offset_minutes=31),
    )
    flipped_holdout = (
        *shared,
        _transition("cal-hold-c", label=_WATER, cutoff_offset_minutes=30),
        _transition("cal-hold-d", label=_WATER, cutoff_offset_minutes=31),
    )

    calibration = fit_probability_calibration(documents, learned_at=LEARNED_AT)
    assert calibration.sample_count > 0
    assert calibration.version == fit_probability_calibration(
        flipped_holdout, learned_at=LEARNED_AT
    ).version

    calibrated = backtest(documents, learned_at=LEARNED_AT, calibration=calibration)
    assert calibrated.expected_calibration_error is not None
    assert calibrated.coverage_precision


def test_predictor_applies_calibration_to_candidates_and_identity(tmp_path) -> None:
    documents = tuple(_transition(f"pred-cal-{index}") for index in range(4))
    tree = PredictionTree(tmp_path / "prediction")
    PredictionPublisher(tree).publish(documents)
    patterns = PredictionPatternLearner().learn(documents, learned_at=LEARNED_AT)
    PredictionPatternGenerationPublisher(tree).publish(patterns)
    graph = PredictionPatternGraph(tree)
    context = PredictionContext.from_document(_transition("pred-cal-query"))
    flat = PredictionProbabilityCalibration(points=((0.0, 0.2), (1.0, 0.2)))
    config = PredictionDecisionConfig(
        execute_probability_threshold=0.5,
        execute_margin=0.1,
        min_execute_support=2,
    )

    plain = PatternGraphPredictor(graph, config=config)
    calibrated = PatternGraphPredictor(graph, config=config, calibration=flat)
    assert plain.predictor_digest != calibrated.predictor_digest

    plain_run = plain.predict(
        context, predicted_at=LEARNED_AT, horizon_seconds=3600, source_bindings=_BINDINGS
    ).run
    calibrated_decision = calibrated.predict(
        context, predicted_at=LEARNED_AT, horizon_seconds=3600, source_bindings=_BINDINGS
    )

    assert plain_run.candidates[0].probability > 0.5
    assert calibrated_decision.run.candidates[0].probability == pytest.approx(0.2)
    assert not calibrated_decision.execute
    assert "probability_below_threshold" in calibrated_decision.blocked_gates
