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


def test_fitted_calibration_shrinks_overconfident_probabilities() -> None:
    documents = (
        *(
            _transition(f"cal-{index}", cutoff_offset_minutes=index)
            for index in range(4)
        ),
        _transition("cal-hold-a", cutoff_offset_minutes=30),
        _transition("cal-hold-b", label=_WATER, cutoff_offset_minutes=31),
    )

    calibration = fit_probability_calibration(documents, learned_at=LEARNED_AT)
    assert calibration.sample_count > 0

    raw = backtest(documents, learned_at=LEARNED_AT)
    calibrated = backtest(documents, learned_at=LEARNED_AT, calibration=calibration)
    assert raw.expected_calibration_error is not None
    assert calibrated.expected_calibration_error is not None
    assert calibrated.expected_calibration_error <= raw.expected_calibration_error


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
