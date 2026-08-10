"""从真实行为边界构造严格的预测切点。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from prediction.projection._behavior_source import BehaviorSourceRecord
from prediction.projection._refs import (
    _opaque_container_ref,
    _opaque_step_ref,
)


def _action_anchor(
    event_uri: str,
    fields: Mapping[str, Any],
    target_index: int,
) -> dict[str, Any]:
    actions = fields["actions"]
    previous = actions[target_index - 1] if target_index else None
    if previous is None:
        timing = _exact_timing(fields["onset_available_at"])
        decision_basis = "container_observed"
    elif previous["available_at"] is not None:
        timing = _exact_timing(previous["available_at"])
        decision_basis = "previous_step_observed"
    else:
        timing = _bounded_or_order_timing(fields["started_at"], None)
        decision_basis = "previous_step_ordered"
    cutoff = timing["cutoff_at"] or timing["lower_bound_at"]
    visible_previous = tuple(
        action
        for action in actions[:target_index]
        if cutoff is not None and action["available_at"] <= cutoff
    )
    return {
        "anchor_type": "action",
        "container_ref": _opaque_container_ref(event_uri),
        "prefix_length": len(visible_previous),
        "previous_step_ref": None
        if previous is None
        else f"action:{previous['action_id']}",
        "decision_basis": decision_basis,
        **timing,
    }


def _action_treatment_anchor(
    event_uri: str,
    fields: Mapping[str, Any],
    action_index: int,
) -> dict[str, Any]:
    """在给定 Action 真正开始时预测其后果；Outcome 时间不参与切点选择。"""

    action = fields["actions"][action_index]
    if action["started_at"] is None:
        raise ValueError(
            "Action consequence projection requires an observable treatment start"
        )
    if action["ended_at"] is not None and action["ended_at"] <= action["available_at"]:
        raise ValueError(
            "Action consequence projection requires treatment identity before completion"
        )
    visible_previous = tuple(
        previous
        for previous in fields["actions"][:action_index]
        if previous["available_at"] <= action["available_at"]
    )
    previous = visible_previous[-1] if visible_previous else None
    return {
        "anchor_type": "action",
        "container_ref": _opaque_container_ref(event_uri),
        "prefix_length": len(visible_previous),
        "previous_step_ref": None if previous is None else f"action:{previous['action_id']}",
        "decision_basis": "treatment_observed",
        **_exact_timing(action["available_at"]),
    }


def _event_start_anchor(event_uri: str, fields: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "anchor_type": "event",
        "container_ref": _opaque_container_ref(event_uri),
        "prefix_length": 0,
        "previous_step_ref": None,
        "decision_basis": "treatment_observed",
        **_exact_timing(fields["onset_available_at"]),
    }


def _event_anchor(
    episode_uri: str,
    fields: Mapping[str, Any],
    events: Sequence[BehaviorSourceRecord],
    target_index: int,
) -> dict[str, Any]:
    previous = events[target_index - 1] if target_index else None
    if previous is None:
        timing = _exact_timing(fields["started_at"])
        decision_basis = "container_started"
    else:
        timing = _exact_timing(previous.fields["onset_available_at"])
        decision_basis = "previous_step_observed"
    cutoff = timing["cutoff_at"]
    visible_previous = tuple(
        event
        for event in events[:target_index]
        if cutoff is not None and event.fields["onset_available_at"] <= cutoff
    )
    return {
        "anchor_type": "event",
        "container_ref": _opaque_container_ref(episode_uri),
        "prefix_length": len(visible_previous),
        "previous_step_ref": None if previous is None else _opaque_step_ref("event", previous.uri),
        "decision_basis": decision_basis,
        **timing,
    }


def _phase_anchor(
    episode_uri: str,
    fields: Mapping[str, Any],
    target_index: int,
) -> dict[str, Any]:
    phases = fields["phases"]
    previous = phases[target_index - 1] if target_index else None
    if previous is None:
        timing = _exact_timing(fields["started_at"])
        decision_basis = "container_started"
    else:
        timing = _exact_timing(previous["started_at"])
        decision_basis = "previous_step_started"
    cutoff = timing["cutoff_at"]
    visible_previous = tuple(
        phase
        for phase in phases[:target_index]
        if cutoff is not None and phase["started_at"] <= cutoff
    )
    return {
        "anchor_type": "phase",
        "container_ref": _opaque_container_ref(episode_uri),
        "prefix_length": len(visible_previous),
        "previous_step_ref": None
        if previous is None
        else f"phase:{previous['phase_id']}",
        "decision_basis": decision_basis,
        **timing,
    }


def _episode_terminal_anchor(
    episode_uri: str,
    fields: Mapping[str, Any],
) -> dict[str, Any]:
    return _phase_anchor(episode_uri, fields, len(fields["phases"]))


def _exact_timing(value: datetime) -> dict[str, Any]:
    return {
        "cutoff_at": value,
        "precision": "exact",
        "lower_bound_at": None,
        "upper_bound_at": None,
    }


def _bounded_or_order_timing(
    lower_bound: datetime | None,
    upper_bound: datetime | None,
) -> dict[str, Any]:
    if lower_bound is None and upper_bound is None:
        return {
            "cutoff_at": None,
            "precision": "order_only",
            "lower_bound_at": None,
            "upper_bound_at": None,
        }
    return {
        "cutoff_at": None,
        "precision": "bounded",
        "lower_bound_at": lower_bound,
        "upper_bound_at": upper_bound,
    }


def _anchor_observed_at(anchor: Mapping[str, Any]) -> datetime | None:
    return anchor["cutoff_at"] or anchor["lower_bound_at"]


def _delay_from_anchor(anchor: Mapping[str, Any], target: datetime | None) -> float | None:
    cutoff = anchor["cutoff_at"]
    if cutoff is None or target is None or target < cutoff:
        return None
    return (target - cutoff).total_seconds()


def _target_started_before_anchor(
    anchor: Mapping[str, Any],
    target_started_at: datetime | None,
) -> bool:
    """跳过在可观察切点前已发生的目标，而不让一个坏前缀中止整批投影。"""

    if anchor["precision"] != "exact" or target_started_at is None:
        return False
    cutoff = anchor["cutoff_at"]
    return cutoff is not None and target_started_at < cutoff

