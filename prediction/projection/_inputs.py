"""构造锚点前模型可见的观测帧、行为前缀与决策空间。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from prediction.projection._anchors import (
    _anchor_observed_at,
)
from prediction.projection._behavior_source import BehaviorSourceRecord
from prediction.projection._refs import (
    _action_subjects,
    _event_subjects,
    _stable_subjects,
)
from prediction.projection._steps import (
    _active_projected_action,
    _active_projected_event,
    _projected_action,
    _projected_event,
)


def _scope(
    *,
    participants: Sequence[str],
    target_level: str,
    target_domain: str | None,
    prediction_mode: str,
) -> dict[str, Any]:
    return {
        "participants": tuple(participants),
        "target_level": target_level,
        "target_domain": target_domain,
        "prediction_mode": prediction_mode,
    }


def _event_input(
    fields: Mapping[str, Any],
    event_uri: str,
    anchor: Mapping[str, Any],
    *,
    completed_actions: Sequence[Mapping[str, Any]],
    active_actions: Sequence[Mapping[str, Any]] = (),
    include_final_facts: bool = False,
) -> dict[str, Any]:
    facts: list[dict[str, Any]] = []
    cutoff = _anchor_observed_at(anchor)
    if cutoff is None or fields["onset_available_at"] <= cutoff:
        facts.append(
            _fact(
                fact_id="event_onset",
                semantics=fields["onset_semantics"],
                observed_at=fields["started_at"],
                available_at=fields["onset_available_at"],
                confidence=fields["confidence"],
                evidence_refs=fields["evidence_refs"],
            )
        )
    if include_final_facts:
        facts.extend(
            _fact(
                fact_id=f"event_fact_{index:04d}",
                semantics=semantics,
                observed_at=fields["ended_at"] or fields["started_at"],
                confidence=fields["confidence"],
                evidence_refs=fields["evidence_refs"],
            )
            for index, semantics in enumerate(fields["semantic_facts"], start=1)
        )
    return {
        "observation_frame": _observation_frame(
            observed_at=_anchor_observed_at(anchor),
            subjects=_stable_subjects(
                fields["participants"],
                _action_subjects((*completed_actions, *active_actions)),
            ),
            facts=facts,
            active_goals=(),
            constraints=(),
        ),
        "behavior_history": _behavior_history(
            completed_actions=tuple(
                _projected_action(event_uri, action, expose_source=False) for action in completed_actions
            ),
            active_behaviors=tuple(
                _active_projected_action(event_uri, action, expose_source=False) for action in active_actions
            ),
        ),
        "decision_space": _unknown_decision_space(),
    }


def _completed_event_input(event: BehaviorSourceRecord, anchor: Mapping[str, Any]) -> dict[str, Any]:
    result = _event_input(
        event.fields,
        event.uri,
        anchor,
        completed_actions=event.fields["actions"],
        include_final_facts=True,
    )
    result["behavior_history"] = _behavior_history(
        completed_events=(_projected_event(event, expose_source=False),)
    )
    return result


def _episode_input(
    anchor: Mapping[str, Any],
    *,
    completed_events: Sequence[BehaviorSourceRecord] = (),
    active_events: Sequence[BehaviorSourceRecord] = (),
    behavior_history: Mapping[str, Any] | None = None,
    observed_subjects: Sequence[str] | None = None,
) -> dict[str, Any]:
    history = (
        _behavior_history(
            completed_events=tuple(_projected_event(event, expose_source=False) for event in completed_events),
            active_behaviors=tuple(_active_projected_event(event, expose_source=False) for event in active_events),
        )
        if behavior_history is None
        else dict(behavior_history)
    )
    return {
        "observation_frame": _observation_frame(
            observed_at=_anchor_observed_at(anchor),
            subjects=_event_subjects((*completed_events, *active_events))
            if observed_subjects is None
            else observed_subjects,
            facts=(),
            active_goals=(),
            constraints=(),
        ),
        "behavior_history": history,
        "decision_space": _unknown_decision_space(),
    }


def _observation_frame(
    *,
    observed_at: datetime | None,
    subjects: Sequence[str],
    facts: Sequence[Mapping[str, Any]],
    active_goals: Sequence[str],
    constraints: Sequence[str],
) -> dict[str, Any]:
    return {
        "observed_at": observed_at,
        "available_at": observed_at,
        "observer": None,
        "subjects": tuple(subjects),
        "facts": tuple(facts),
        "active_goals": tuple(active_goals),
        "constraints": tuple(constraints),
        "coverage": {
            "available_modalities": (),
            "missing_modalities": (),
            "blind_intervals": (),
            "coverage_score": None,
        },
    }


def _fact(
    *,
    fact_id: str,
    semantics: str,
    observed_at: datetime,
    available_at: datetime | None = None,
    confidence: float,
    evidence_refs: Sequence[str],
) -> dict[str, Any]:
    return {
        "fact_id": fact_id,
        "category": "semantic",
        "semantics": semantics,
        "subject_ref": None,
        "attribute": None,
        "value": semantics,
        "unit": None,
        "observed_at": observed_at,
        "available_at": available_at or observed_at,
        "valid_from": observed_at,
        "valid_to": None,
        "knowledge_state": "observed",
        "confidence": confidence,
        "evidence_refs": tuple(evidence_refs),
    }


def _behavior_history(
    *,
    completed_events: Sequence[Mapping[str, Any]] = (),
    completed_actions: Sequence[Mapping[str, Any]] = (),
    completed_phases: Sequence[Mapping[str, Any]] = (),
    active_behaviors: Sequence[Mapping[str, Any]] = (),
    parallel_behaviors: Sequence[Mapping[str, Any]] = (),
    interruptions: Sequence[Mapping[str, Any]] = (),
    resumptions: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "completed_events": tuple(completed_events),
        "completed_actions": tuple(completed_actions),
        "completed_phases": tuple(completed_phases),
        "active_behaviors": tuple(active_behaviors),
        "parallel_behaviors": tuple(parallel_behaviors),
        "interruptions": tuple(interruptions),
        "resumptions": tuple(resumptions),
    }


def _unknown_decision_space() -> dict[str, Any]:
    return {
        "known_available": (),
        "known_unavailable": (),
        "prohibited": (),
        "unknown": ("counterfactual_action_space",),
    }

