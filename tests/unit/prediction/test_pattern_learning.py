"""Sample → Pattern 学习聚合器的估计与确定性合同测试。"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
from typing import Any

import pytest

from foundation.integrity import canonical_digest
from prediction import (
    PredictionDocument,
    PredictionDocumentCodec,
    PredictionKind,
    PredictionPatternDocument,
    PredictionPatternGenerationPublisher,
    PredictionPatternGraph,
    PredictionPatternKind,
    PredictionPublisher,
    PredictionSchemaRegistry,
    PredictionTree,
    derive_sample_identity,
)
from prediction.learning import (
    PredictionLearningConfig,
    PredictionLearningError,
    PredictionPatternLearner,
)

UTC = timezone.utc
CUTOFF = datetime(2026, 8, 8, 10, 30, tzinfo=UTC)
LEARNED_AT = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)
_CODEC = PredictionDocumentCodec(PredictionSchemaRegistry.load_default())


def _event_uri(token: str) -> str:
    suffix = sha256(token.encode("utf-8")).hexdigest()[:16]
    return (
        f"behavior://behaviors/events/2026/08/08/主人回家-{suffix}"
        "--20260808T103000000000%2B0000.md"
    )


def _history_step(
    index: int,
    semantics: str,
    behavior_type: str,
    refs: tuple[str, ...],
    *,
    cutoff: datetime,
    is_last: bool,
) -> dict[str, Any]:
    started = datetime(2026, 8, 8, 10, index * 2, tzinfo=UTC)
    ended = started + timedelta(minutes=1)
    return {
        "step_kind": "action",
        "step_ref": f"action:act_{index:04d}",
        "source_uri": None,
        "local_id": f"act_{index:04d}",
        "sequence": index + 1,
        "semantics": semantics,
        "actor": "主人",
        "behavior_type": behavior_type,
        "target_refs": list(refs),
        "status": "completed",
        "started_at": started,
        "ended_at": ended,
        "available_at": cutoff if is_last else ended,
    }


def _transition(
    token: str,
    *,
    prefix: int = 0,
    history: tuple[tuple[str, str, tuple[str, ...]], ...] = (),
    label: tuple[str, str, tuple[str, ...]] = ("打开空调", "device_control", ("空调",)),
    domain: str = "device_control",
    terminal: bool = False,
    censored: bool = False,
    cutoff_offset_minutes: int = 0,
    source_confidence: float = 0.96,
    projection_version: str = "behavior-prediction-v2",
) -> PredictionDocument:
    if len(history) != prefix:
        raise AssertionError("test history must match the requested prefix length")
    cutoff = CUTOFF + timedelta(minutes=cutoff_offset_minutes)
    event_uri = _event_uri(token)
    target_id = f"act_{prefix:04d}"
    target_ref = "terminal" if terminal else f"{event_uri}#action:{target_id}"
    identity_material = {
        "container_uri": event_uri,
        "anchor_type": "action",
        "prefix_length": prefix,
        "target_ref": target_ref,
    }
    identity = derive_sample_identity(
        PredictionKind.TRANSITION, identity_material, projection_version, {}
    )
    semantics, behavior_type, target_refs = label
    if terminal:
        label_fields: dict[str, Any] = {
            "target_kind": "terminal",
            "source_ref": None,
            "actor": None,
            "behavior_type": None,
            "semantics": "completed terminal state",
            "target_refs": [],
            "parameters": {},
            "started_at": None,
            "delay_seconds": None,
            "relations": [],
            "terminal": {"status": "completed", "reason": None},
        }
    else:
        label_fields = {
            "target_kind": "action",
            "source_ref": target_ref,
            "actor": "主人",
            "behavior_type": behavior_type,
            "semantics": semantics,
            "target_refs": list(target_refs),
            "parameters": {},
            "started_at": None,
            "delay_seconds": None,
            "relations": [],
            "terminal": None,
        }
    payload = {
        "sample_date": date(2026, 8, 8),
        "logical_sample_id": identity.logical_sample_id,
        "materialization_id": identity.materialization_id,
        "prediction_scope": {
            "participants": ["主人"],
            "target_level": "terminal" if terminal else "action",
            "target_domain": domain,
            "prediction_mode": "termination" if terminal else "next_step",
        },
        "anchor": {
            "anchor_type": "action",
            "container_ref": "behavior-container:" + canonical_digest({"uri": event_uri}),
            "prefix_length": prefix,
            "previous_step_ref": f"action:act_{prefix - 1:04d}" if prefix else None,
            "decision_basis": "previous_step_observed" if prefix else "container_started",
            "cutoff_at": cutoff,
            "precision": "exact",
            "lower_bound_at": None,
            "upper_bound_at": None,
        },
        "input": {
            "observation_frame": {
                "observed_at": cutoff,
                "available_at": cutoff,
                "observer": "camera:living-room",
                "subjects": ["主人"],
                "facts": [],
                "active_goals": [],
                "constraints": [],
                "coverage": {
                    "available_modalities": ["vision"],
                    "missing_modalities": [],
                    "blind_intervals": [],
                    "coverage_score": 0.9,
                },
            },
            "behavior_history": {
                "completed_events": [],
                "completed_actions": [
                    _history_step(index, *step, cutoff=cutoff, is_last=index == prefix - 1)
                    for index, step in enumerate(history)
                ],
                "completed_phases": [],
                "active_behaviors": [],
                "parallel_behaviors": [],
                "interruptions": [],
                "resumptions": [],
            },
            "decision_space": {
                "known_available": [],
                "known_unavailable": [],
                "prohibited": [],
                "unknown": ["counterfactual_action_space"],
            },
        },
        "label": label_fields,
        "supervision": {
            "label_status": "terminal" if terminal else "observed",
            "window_started_at": cutoff,
            "window_closed_at": cutoff + timedelta(minutes=2),
            "censored": censored,
            "censoring_reason": "observation window closed" if censored else None,
        },
        "lineage": {
            "behavior_root_uri": event_uri,
            "event_uri": event_uri,
            "episode_uri": None,
            "outcome_uri": None,
            "occurrence_group_id": event_uri,
            "consequence_group_id": None,
        },
        "identity_material": identity_material,
        "materialization_context": {},
        "provenance": {
            "source_bindings": [
                {
                    "uri": event_uri,
                    "revision": 1,
                    "digest": "b" * 64,
                    "member_type": "action",
                    "member_id": target_id,
                }
            ],
            "projection_version": projection_version,
            "projector_digest": "c" * 64,
        },
        "quality": {
            "source_confidence": source_confidence,
            "evidence_coverage": 0.9,
            "context_completeness": 0.8,
            "conflict_count": 0,
            "inferred_fact_ratio": 0.0,
        },
    }
    return _CODEC.build(PredictionKind.TRANSITION, payload)


_AC_STEP = ("打开空调", "device_control", ("空调",))
_WATER = ("倒一杯水", "daily_activity", ("水杯",))


def _consequence(
    token: str,
    *,
    treatment: tuple[str, str, tuple[str, ...]] = _AC_STEP,
    outcome_type: str = "state_change",
    valence: str = "positive",
    outcome_semantics: str = "室温下降",
    delay_seconds: float = 300.0,
    outcome_id: str = "out_0001",
    revision: int = 1,
    domain: str = "device_control",
    projection_version: str = "behavior-prediction-v2",
) -> PredictionDocument:
    event_uri = _event_uri(token)
    outcome_uri = event_uri.replace("behaviors/events", "behaviors/outcomes")
    identity_material = {"outcome_uri": outcome_uri, "outcome_id": outcome_id}
    identity = derive_sample_identity(
        PredictionKind.CONSEQUENCE,
        identity_material,
        projection_version,
        {"outcome_revision": revision},
    )
    semantics, behavior_type, target_refs = treatment
    payload = {
        "sample_date": date(2026, 8, 8),
        "logical_sample_id": identity.logical_sample_id,
        "materialization_id": identity.materialization_id,
        "prediction_scope": {
            "participants": ["主人"],
            "target_level": "outcome",
            "target_domain": domain,
            "prediction_mode": "consequence",
        },
        "anchor": {
            "anchor_type": "action",
            "container_ref": "behavior-container:" + canonical_digest({"uri": event_uri}),
            "prefix_length": 0,
            "previous_step_ref": None,
            "decision_basis": "treatment_observed",
            "cutoff_at": CUTOFF,
            "precision": "exact",
            "lower_bound_at": None,
            "upper_bound_at": None,
        },
        "input": {
            "observation_frame": {
                "observed_at": CUTOFF,
                "available_at": CUTOFF,
                "observer": "camera:living-room",
                "subjects": ["主人"],
                "facts": [],
                "active_goals": [],
                "constraints": [],
                "coverage": {
                    "available_modalities": ["vision"],
                    "missing_modalities": [],
                    "blind_intervals": [],
                    "coverage_score": 0.9,
                },
            },
            "behavior_history": {
                "completed_events": [],
                "completed_actions": [],
                "completed_phases": [],
                "active_behaviors": [],
                "parallel_behaviors": [],
                "interruptions": [],
                "resumptions": [],
            },
            "decision_space": {
                "known_available": [],
                "known_unavailable": [],
                "prohibited": [],
                "unknown": ["counterfactual_action_space"],
            },
        },
        "treatment": {
            "step_kind": "action",
            "step_ref": "action:act_0001",
            "source_uri": event_uri,
            "local_id": "act_0001",
            "sequence": 1,
            "semantics": semantics,
            "actor": "主人",
            "behavior_type": behavior_type,
            "target_refs": list(target_refs),
            "status": "active",
            "started_at": CUTOFF,
            "ended_at": None,
            "available_at": CUTOFF,
        },
        "label": {
            "outcome": {
                "outcome_id": outcome_id,
                "occurred_at": CUTOFF + timedelta(seconds=delay_seconds),
                "outcome_type": outcome_type,
                "semantics": outcome_semantics,
                "valence": valence,
                "knowledge_state": "observed",
                "confidence": 0.9,
                "delay_seconds": delay_seconds,
            },
            "attribution": "temporal_only",
        },
        "supervision": {
            "label_status": "observed",
            "window_started_at": CUTOFF,
            "window_closed_at": CUTOFF + timedelta(minutes=30),
            "censored": False,
            "censoring_reason": None,
        },
        "lineage": {
            "behavior_root_uri": event_uri,
            "event_uri": event_uri,
            "episode_uri": None,
            "outcome_uri": outcome_uri,
            "occurrence_group_id": event_uri,
            "consequence_group_id": f"{event_uri}#treatment:act_0001",
        },
        "identity_material": identity_material,
        "materialization_context": {"outcome_revision": revision},
        "provenance": {
            "source_bindings": [
                {
                    "uri": event_uri,
                    "revision": 1,
                    "digest": "b" * 64,
                    "member_type": "action",
                    "member_id": "act_0001",
                },
                {
                    "uri": outcome_uri,
                    "revision": revision,
                    "digest": "d" * 64,
                    "member_type": "outcome",
                    "member_id": outcome_id,
                },
            ],
            "projection_version": projection_version,
            "projector_digest": "c" * 64,
        },
        "quality": {
            "source_confidence": 0.96,
            "evidence_coverage": 0.9,
            "context_completeness": 0.8,
            "conflict_count": 0,
            "inferred_fact_ratio": 0.0,
        },
    }
    return _CODEC.build(PredictionKind.CONSEQUENCE, payload)


def _states(patterns: tuple[PredictionPatternDocument, ...]) -> list[PredictionPatternDocument]:
    return [item for item in patterns if item.kind is PredictionPatternKind.STATE]


def _sequence_states(
    patterns: tuple[PredictionPatternDocument, ...],
) -> list[PredictionPatternDocument]:
    return [
        item
        for item in _states(patterns)
        if item.fields["identity_material"]["family"] == "sequence-context"
    ]


def _root_state(patterns: tuple[PredictionPatternDocument, ...]) -> PredictionPatternDocument:
    return next(
        item
        for item in _sequence_states(patterns)
        if item.fields["identity_material"]["context"] is None
    )


def _branch_probability(
    graph: PredictionPatternGraph,
    logical_state_key: str,
    semantics: str,
) -> float:
    expansion = graph.expand(logical_state_key)
    branch = next(
        item for item in expansion.branches if item.fields["behavior"]["semantics"] == semantics
    )
    return branch.fields["conditional_probability"]


def test_learner_aggregates_transitions_into_publishable_pattern_generations(tmp_path) -> None:
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

    states = _states(patterns)
    assert len(states) == 3
    sequence_states = _sequence_states(patterns)
    assert len(sequence_states) == 2
    root = _root_state(patterns)
    child = next(item for item in sequence_states if item is not root)
    temporal = next(item for item in states if item not in sequence_states)
    assert temporal.fields["identity_material"]["family"] == "temporal"
    assert temporal.fields["identity_material"]["hour_bucket"] == "09-12"
    assert temporal.fields["support_count"] == 8
    assert root.fields["support_count"] == 8
    assert root.fields["statistics"]["source_count"] == 4
    assert child.fields["identity_material"]["context"]["semantics"] == "打开空调"
    assert child.fields["support_count"] == 3

    publisher = PredictionPatternGenerationPublisher(tree)
    publisher.publish(patterns)
    graph = PredictionPatternGraph(tree)

    root_expansion = graph.expand(root.fields["logical_state_key"])
    assert sum(item.fields["conditional_probability"] for item in root_expansion.branches) < 1
    root_semantics = {item.fields["behavior"]["semantics"] for item in root_expansion.branches}
    assert {"打开空调", "倒一杯水", "关灯"} <= root_semantics
    assert {item.fields["target_kind"] for item in root_expansion.branches} == {"action", "termination"}
    ac_branch = next(
        item for item in root_expansion.branches if item.fields["behavior"]["semantics"] == "打开空调"
    )
    assert ac_branch.fields["support_count"] == 3

    child_expansion = graph.expand(child.fields["logical_state_key"])
    assert len(child_expansion.branches) == 2
    assert sum(item.fields["conditional_probability"] for item in child_expansion.branches) < 1
    child_water = _branch_probability(graph, child.fields["logical_state_key"], "倒一杯水")
    root_water = _branch_probability(graph, root.fields["logical_state_key"], "倒一杯水")
    assert child_water > root_water

    assert len(publisher.store.active_logical_state_keys()) == 3


def test_domains_form_separate_states_with_independent_baselines() -> None:
    documents = (
        _transition("dom-a"),
        _transition("dom-b"),
        _transition(
            "dom-c",
            label=("运行对应测试用例", "coding_action", ("pytest",)),
            domain="software_delivery",
        ),
        _transition(
            "dom-d",
            label=("推送代码", "coding_action", ("git",)),
            domain="software_delivery",
        ),
    )

    patterns = PredictionPatternLearner().learn(documents, learned_at=LEARNED_AT)

    states = _sequence_states(patterns)
    assert len(states) == 2
    by_domain = {item.fields["identity_material"]["target_domain"]: item for item in states}
    assert set(by_domain) == {"device_control", "software_delivery"}
    assert by_domain["device_control"].fields["support_count"] == 2
    assert by_domain["software_delivery"].fields["support_count"] == 2
    coding_branches = [
        item
        for item in patterns
        if item.kind is PredictionPatternKind.BRANCH
        and item.fields["state_key"] == by_domain["software_delivery"].address.state_key
    ]
    assert {item.fields["behavior"]["semantics"] for item in coding_branches} == {
        "运行对应测试用例",
        "推送代码",
    }


def test_probability_closure_holds_for_fractional_counts() -> None:
    documents = (
        *(_transition(f"frac-root-a{index}") for index in range(4)),
        *(_transition(f"frac-root-b{index}", label=_WATER) for index in range(4)),
        _transition(
            "frac-ctx-a",
            prefix=1,
            history=(_AC_STEP,),
            source_confidence=0.2,
        ),
        _transition(
            "frac-ctx-b",
            prefix=1,
            history=(_AC_STEP,),
            label=_WATER,
            source_confidence=0.2,
        ),
    )

    patterns = PredictionPatternLearner().learn(documents, learned_at=LEARNED_AT)

    states = _states(patterns)
    for state in states:
        total = sum(
            item.fields["conditional_probability"]
            for item in patterns
            if item.kind is PredictionPatternKind.BRANCH
            and item.fields["state_key"] == state.address.state_key
        )
        assert total < 1
    for item in patterns:
        if item.kind is PredictionPatternKind.BRANCH:
            assert 0 <= item.fields["conditional_probability"] <= 1


def test_temporal_states_capture_time_of_day_routines() -> None:
    documents = (
        *(
            _transition(
                f"morning-{index}",
                label=("出门晨跑", "daily_activity", ("公园",)),
                cutoff_offset_minutes=-210 + index,
            )
            for index in range(3)
        ),
        _transition("mid-a"),
        _transition("mid-b"),
    )

    patterns = PredictionPatternLearner().learn(documents, learned_at=LEARNED_AT)

    temporal = {
        item.fields["identity_material"]["hour_bucket"]: item
        for item in _states(patterns)
        if item.fields["identity_material"]["family"] == "temporal"
    }
    assert set(temporal) == {"06-09", "09-12"}
    morning = temporal["06-09"]
    assert morning.fields["support_count"] == 3
    morning_branches = [
        item
        for item in patterns
        if item.kind is PredictionPatternKind.BRANCH
        and item.fields["state_key"] == morning.address.state_key
    ]
    top = max(morning_branches, key=lambda item: item.fields["conditional_probability"])
    assert top.fields["behavior"]["semantics"] == "出门晨跑"
    assert top.fields["support_count"] == 3


def test_learner_is_deterministic_for_identical_inputs_and_versioned_by_learned_at() -> None:
    documents = (
        _transition("ev-a"),
        _transition("ev-b"),
        _transition("ev-d", label=_WATER),
    )

    first = PredictionPatternLearner().learn(documents, learned_at=LEARNED_AT)
    second = PredictionPatternLearner().learn(tuple(reversed(documents)), learned_at=LEARNED_AT)
    assert first == second

    shifted = PredictionPatternLearner().learn(documents, learned_at=LEARNED_AT + timedelta(days=1))
    assert (
        shifted[0].fields["pattern_generation"] != first[0].fields["pattern_generation"]
    )


def test_group_weight_cap_bounds_one_occurrence_group_influence() -> None:
    steps = (("步骤零", "daily_activity", ()), ("步骤一", "daily_activity", ()), ("步骤二", "daily_activity", ()))
    documents = (
        _transition("cap-a"),
        _transition("cap-a", prefix=1, history=steps[:1]),
        _transition("cap-a", prefix=2, history=steps[:2]),
        _transition("cap-a", prefix=3, history=steps[:3]),
        _transition("cap-b", label=_WATER),
    )

    def _root_probability(cap: float) -> float:
        learner = PredictionPatternLearner(config=PredictionLearningConfig(group_weight_cap=cap))
        patterns = learner.learn(documents, learned_at=LEARNED_AT)
        root = _root_state(patterns)
        branch = next(
            item
            for item in patterns
            if item.kind is PredictionPatternKind.BRANCH
            and item.fields["state_key"] == root.address.state_key
            and item.fields["behavior"]["semantics"] == "打开空调"
        )
        assert branch.fields["support_count"] == 4
        return branch.fields["conditional_probability"]

    assert _root_probability(1.0) < _root_probability(100.0)


def test_censored_samples_add_exposure_without_creating_branches() -> None:
    observed = (_transition("cen-a"), _transition("cen-b"))
    censored = _transition("cen-c", censored=True)

    learner = PredictionPatternLearner()
    with_censored = learner.learn((*observed, censored), learned_at=LEARNED_AT)
    without_censored = learner.learn(observed, learned_at=LEARNED_AT)

    root = _root_state(with_censored)
    assert root.fields["support_count"] == 3
    branches = [
        item
        for item in with_censored
        if item.kind is PredictionPatternKind.BRANCH
        and item.fields["state_key"] == root.address.state_key
    ]
    assert len(branches) == 1
    assert branches[0].fields["support_count"] == 2
    plain_root = _root_state(without_censored)
    plain_branches = [
        item
        for item in without_censored
        if item.kind is PredictionPatternKind.BRANCH
        and item.fields["state_key"] == plain_root.address.state_key
    ]
    assert branches[0].fields["conditional_probability"] < plain_branches[0].fields["conditional_probability"]


def test_learner_rejects_empty_batches_naive_clocks_and_mixed_projections() -> None:
    learner = PredictionPatternLearner()

    with pytest.raises(PredictionLearningError, match="at least one"):
        learner.learn((), learned_at=LEARNED_AT)

    with pytest.raises(PredictionLearningError, match="timezone-aware"):
        learner.learn((_transition("ev-a"),), learned_at=datetime(2026, 8, 9, 12, 0))

    mixed = (
        _transition("ev-a"),
        _transition("ev-b", projection_version="behavior-prediction-v3"),
    )
    with pytest.raises(PredictionLearningError, match="one projection version"):
        learner.learn(mixed, learned_at=LEARNED_AT)
