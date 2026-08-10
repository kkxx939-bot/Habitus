"""不可变 PredictionRun 的身份、上下文和持久化测试。"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pytest
from prediction_test_payloads import transition_payload

from prediction import (
    PredictionAbstentionReason,
    PredictionCandidate,
    PredictionContext,
    PredictionDocumentCodec,
    PredictionKind,
    PredictionRun,
    PredictionRunPatternBinding,
    PredictionRunSourceBinding,
    PredictionRunStore,
    PredictionRunStoreError,
    PredictionSchemaRegistry,
    PredictionTargetLevel,
)


def _context() -> PredictionContext:
    document = PredictionDocumentCodec(PredictionSchemaRegistry.load_default()).build(
        PredictionKind.TRANSITION,
        transition_payload("prediction-run"),
    )
    return PredictionContext.from_document(document)


def _binding() -> PredictionRunSourceBinding:
    return PredictionRunSourceBinding("sensor://living-room/temperature", 3, "a" * 64)


def _candidate() -> PredictionCandidate:
    return PredictionCandidate(
        rank=1,
        target_level=PredictionTargetLevel.ACTION,
        semantics="打开空调",
        probability=0.72,
        expected_delay_seconds=120.0,
        payload={"behavior_type": "device_control", "target_ref": "device:air-conditioner"},
        evidence_refs=("prediction://patterns/branches/example",),
    )


def _consequence_context() -> PredictionContext:
    context = _context()
    cutoff = context.anchor["cutoff_at"]
    return PredictionContext(
        kind=PredictionKind.CONSEQUENCE,
        prediction_scope={
            **context.prediction_scope,
            "target_level": "outcome",
            "target_domain": None,
            "prediction_mode": "consequence",
        },
        anchor={
            **context.anchor,
            "decision_basis": "treatment_observed",
        },
        input=context.input,
        treatment={
            "step_kind": "action",
            "step_ref": "action:act_0001",
            "semantics": "打开空调",
            "actor": "主人",
            "behavior_type": "device_control",
            "target_refs": ("空调",),
            "status": "active",
            "started_at": cutoff,
            "ended_at": None,
            "available_at": cutoff,
        },
    )


def test_prediction_run_freezes_context_candidates_and_round_trips(tmp_path) -> None:
    run = PredictionRun.create(
        predicted_at=datetime(2026, 8, 8, 10, 30, tzinfo=timezone.utc),
        horizon_seconds=300,
        context=_context(),
        predictor_id="personal-pattern-graph",
        predictor_version="v1",
        predictor_digest="d" * 64,
        pattern_bindings=(PredictionRunPatternBinding("a" * 64, "1" * 64, "e" * 64),),
        source_bindings=(_binding(),),
        candidates=(_candidate(),),
    )
    store = PredictionRunStore(tmp_path / "prediction-runs")

    assert store.create(run) == run
    assert store.create(run) == run
    assert store.read(run.predicted_at.date(), run.run_id) == run
    assert run.kind is PredictionKind.TRANSITION
    assert run.context["input"]["observation_frame"]["facts"][0]["value"] == 29


def test_consequence_prediction_run_round_trips_treatment_observed_context(tmp_path) -> None:
    run = PredictionRun.create(
        predicted_at=datetime(2026, 8, 8, 10, 30, tzinfo=timezone.utc),
        horizon_seconds=300,
        context=_consequence_context(),
        predictor_id="personal-consequence-model",
        predictor_version="v1",
        predictor_digest="d" * 64,
        source_bindings=(_binding(),),
        candidates=(
            PredictionCandidate(
                rank=1,
                target_level=PredictionTargetLevel.OUTCOME,
                semantics="室内开始降温",
                probability=0.65,
                expected_delay_seconds=60,
                payload={"outcome_type": "state_change"},
            ),
        ),
    )
    store = PredictionRunStore(tmp_path / "consequence-runs")

    assert store.read(run.predicted_at.date(), store.create(run).run_id) == run
    assert run.context["treatment"]["step_ref"] == "action:act_0001"


def test_prediction_run_requires_candidates_or_one_abstention_reason() -> None:
    abstained = PredictionRun.create(
        predicted_at=datetime(2026, 8, 8, 10, 30, tzinfo=timezone.utc),
        horizon_seconds=300,
        context=_context(),
        predictor_id="personal-pattern-graph",
        predictor_version="v1",
        predictor_digest="d" * 64,
        source_bindings=(_binding(),),
        abstention_reason=PredictionAbstentionReason.NO_MATCHING_STATE,
    )

    assert abstained.candidates == ()
    assert abstained.abstention_reason is PredictionAbstentionReason.NO_MATCHING_STATE
    with pytest.raises(ValueError, match="candidates or one abstention"):
        PredictionRun.create(
            predicted_at=datetime(2026, 8, 8, 10, 30, tzinfo=timezone.utc),
            horizon_seconds=300,
            context=_context(),
            predictor_id="personal-pattern-graph",
            predictor_version="v1",
            predictor_digest="d" * 64,
            source_bindings=(_binding(),),
        )
    with pytest.raises(ValueError, match="candidates or one abstention"):
        PredictionRun.create(
            predicted_at=datetime(2026, 8, 8, 10, 30, tzinfo=timezone.utc),
            horizon_seconds=300,
            context=_context(),
            predictor_id="personal-pattern-graph",
            predictor_version="v1",
            predictor_digest="d" * 64,
            source_bindings=(_binding(),),
            candidates=(_candidate(),),
            abstention_reason=PredictionAbstentionReason.LOW_CONFIDENCE,
        )


def test_prediction_run_rejects_identity_tampering_and_out_of_horizon_candidate(tmp_path) -> None:
    run = PredictionRun.create(
        predicted_at=datetime(2026, 8, 8, 10, 30, tzinfo=timezone.utc),
        horizon_seconds=300,
        context=_context(),
        predictor_id="personal-pattern-graph",
        predictor_version="v1",
        predictor_digest="d" * 64,
        source_bindings=(_binding(),),
        candidates=(_candidate(),),
    )
    with pytest.raises(ValueError, match="run ID"):
        replace(run, run_id="f" * 64)
    with pytest.raises(ValueError, match="run horizon"):
        PredictionRun.create(
            predicted_at=datetime(2026, 8, 8, 10, 30, tzinfo=timezone.utc),
            horizon_seconds=60,
            context=_context(),
            predictor_id="personal-pattern-graph",
            predictor_version="v1",
            predictor_digest="d" * 64,
            source_bindings=(_binding(),),
            candidates=(_candidate(),),
        )

    store = PredictionRunStore(tmp_path / "prediction-runs")
    path = store.path_for(run.predicted_at.date(), run.run_id)
    store.create(run)
    path.write_text(path.read_text(encoding="utf-8").replace("0.72", "0.71"), encoding="utf-8")
    with pytest.raises(PredictionRunStoreError, match="read safely"):
        store.read(run.predicted_at.date(), run.run_id)


def test_prediction_context_requires_treatment_exactly_for_consequence() -> None:
    context = _context()
    treatment = {
        "step_kind": "action",
        "step_ref": "action:act_0001",
        "semantics": "打开空调",
        "actor": "主人",
        "behavior_type": "device_control",
        "target_refs": ("空调",),
        "status": "active",
        "started_at": context.anchor["cutoff_at"],
        "ended_at": None,
        "available_at": context.anchor["cutoff_at"],
    }

    with pytest.raises(ValueError, match="exactly for Consequence"):
        replace(context, treatment=treatment)
    with pytest.raises(ValueError, match="exactly for Consequence"):
        replace(context, kind=PredictionKind.CONSEQUENCE)
    future_treatment = {
        **treatment,
        "available_at": datetime(2026, 8, 8, 10, 30, 1, tzinfo=timezone.utc),
    }
    with pytest.raises(ValueError, match="cutoff must equal treatment available_at"):
        replace(
            context,
            kind=PredictionKind.CONSEQUENCE,
            prediction_scope={
                **context.prediction_scope,
                "target_level": "outcome",
                "target_domain": None,
                "prediction_mode": "consequence",
            },
            anchor={
                **context.anchor,
                "decision_basis": "treatment_observed",
            },
            treatment=future_treatment,
        )
    with pytest.raises(ValueError, match="Trajectory context requires a Phase anchor"):
        replace(context, kind=PredictionKind.TRAJECTORY)
    with pytest.raises(ValueError, match="incompatible target level"):
        replace(
            context,
            prediction_scope={
                **context.prediction_scope,
                "target_level": "outcome",
            },
        )


def test_prediction_context_rejects_malformed_scope_anchor_and_decision_basis() -> None:
    context = _context()

    with pytest.raises(ValueError, match="precision is not an allowed value"):
        replace(context, anchor={**context.anchor, "precision": "bogus"})
    with pytest.raises(ValueError, match="scope is missing keys"):
        replace(
            context,
            prediction_scope={
                "target_level": "action",
                "prediction_mode": "next_step",
            },
        )
    with pytest.raises(ValueError, match="anchor contains unsupported keys"):
        replace(context, anchor={**context.anchor, "future_label": "打开空调"})
    with pytest.raises(ValueError, match="exact prediction anchor requires only cutoff_at"):
        replace(context, anchor={**context.anchor, "cutoff_at": None})
    with pytest.raises(ValueError, match="previous-step anchor requires a non-empty prefix"):
        replace(
            context,
            anchor={
                **context.anchor,
                "decision_basis": "previous_step_observed",
            },
        )
    with pytest.raises(ValueError, match="prefix_length must equal"):
        replace(
            context,
            anchor={
                **context.anchor,
                "prefix_length": 1,
                "previous_step_ref": "action:act_0000",
                "decision_basis": "previous_step_observed",
            },
        )


def test_prediction_context_closes_container_and_previous_step_time_identity() -> None:
    context = _context()
    with pytest.raises(ValueError, match="container anchor requires exact precision"):
        replace(
            context,
            anchor={
                **context.anchor,
                "precision": "order_only",
                "cutoff_at": None,
            },
        )

    earlier = datetime(2026, 8, 8, 10, 29, 59, tzinfo=timezone.utc)
    with pytest.raises(ValueError, match="cutoff must equal frame observed_at"):
        replace(
            context,
            input={
                **context.input,
                "observation_frame": {
                    **context.input["observation_frame"],
                    "observed_at": earlier,
                },
            },
        )

    real_step = {
        "step_kind": "action",
        "step_ref": "action:real-step",
        "source_uri": None,
        "local_id": "real-step",
        "sequence": 1,
        "semantics": "已经可见的真实动作",
        "actor": "主人",
        "behavior_type": "daily_activity",
        "target_refs": (),
        "status": "active",
        "started_at": context.anchor["cutoff_at"],
        "ended_at": None,
        "available_at": context.anchor["cutoff_at"],
    }
    prediction_input = {
        **context.input,
        "behavior_history": {
            **context.input["behavior_history"],
            "active_behaviors": (real_step,),
        },
    }
    with pytest.raises(ValueError, match="must identify one visible history step"):
        replace(
            context,
            input=prediction_input,
            anchor={
                **context.anchor,
                "prefix_length": 1,
                "previous_step_ref": "action:forged-step",
                "decision_basis": "previous_step_observed",
            },
        )
    with pytest.raises(ValueError, match="prefix_length must equal"):
        replace(
            context,
            input=prediction_input,
            anchor={
                **context.anchor,
                "prefix_length": 7,
                "previous_step_ref": "action:real-step",
                "decision_basis": "previous_step_observed",
            },
        )
    with pytest.raises(ValueError, match="referenced step at cutoff"):
        replace(
            context,
            input={
                **prediction_input,
                "behavior_history": {
                    **prediction_input["behavior_history"],
                    "active_behaviors": (
                        {
                            **real_step,
                            "available_at": earlier,
                        },
                    ),
                },
            },
            anchor={
                **context.anchor,
                "prefix_length": 1,
                "previous_step_ref": "action:real-step",
                "decision_basis": "previous_step_observed",
            },
        )


def test_consequence_context_rejects_forged_treatment_prefix() -> None:
    context = _context()
    treatment = {
        "step_kind": "action",
        "step_ref": "action:act_0001",
        "semantics": "打开空调",
        "actor": "主人",
        "behavior_type": "device_control",
        "target_refs": ("空调",),
        "status": "active",
        "started_at": context.anchor["cutoff_at"],
        "ended_at": None,
        "available_at": context.anchor["cutoff_at"],
    }

    with pytest.raises(ValueError, match="prefix_length must equal"):
        replace(
            context,
            kind=PredictionKind.CONSEQUENCE,
            prediction_scope={
                **context.prediction_scope,
                "target_level": "outcome",
                "target_domain": None,
                "prediction_mode": "consequence",
            },
            anchor={
                **context.anchor,
                "prefix_length": 999,
                "previous_step_ref": "action:does-not-exist",
                "decision_basis": "treatment_observed",
            },
            treatment=treatment,
        )


def test_prediction_run_closes_candidate_source_and_pattern_bindings() -> None:
    def create(
        *,
        source_bindings: tuple[PredictionRunSourceBinding, ...],
        candidates: tuple[PredictionCandidate, ...],
        pattern_bindings: tuple[PredictionRunPatternBinding, ...] = (),
    ) -> PredictionRun:
        return PredictionRun.create(
            predicted_at=datetime(2026, 8, 8, 10, 30, tzinfo=timezone.utc),
            horizon_seconds=300,
            context=_context(),
            predictor_id="personal-pattern-graph",
            predictor_version="v1",
            predictor_digest="d" * 64,
            pattern_bindings=pattern_bindings,
            source_bindings=source_bindings,
            candidates=candidates,
        )

    with pytest.raises(ValueError, match="target level"):
        create(
            source_bindings=(_binding(),),
            candidates=(
                replace(_candidate(), target_level=PredictionTargetLevel.OUTCOME),
            ),
        )
    with pytest.raises(ValueError, match="source revision"):
        create(
            source_bindings=(
                _binding(),
                PredictionRunSourceBinding(
                    "sensor://living-room/temperature",
                    3,
                    "b" * 64,
                ),
            ),
            candidates=(_candidate(),),
        )
    with pytest.raises(ValueError, match="exactly one Pattern generation"):
        create(
            pattern_bindings=(
                PredictionRunPatternBinding("a" * 64, "1" * 64, "e" * 64),
                PredictionRunPatternBinding("a" * 64, "2" * 64, "f" * 64),
            ),
            source_bindings=(_binding(),),
            candidates=(_candidate(),),
        )
