"""从真实后续行为构造监督标签。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from prediction.projection._anchors import (
    _delay_from_anchor,
)
from prediction.projection._behavior_source import BehaviorSourceRecord
from prediction.projection._partition import (
    _resumption_steps,
)
from prediction.projection._refs import (
    _event_targets,
    _member_ref,
    _single,
)
from prediction.projection._steps import (
    _projected_event,
    _projected_phase,
)


def _action_transition_label(
    event_uri: str,
    action: Mapping[str, Any],
    anchor: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "target_kind": "action",
        "source_ref": _member_ref(event_uri, "action", action["action_id"]),
        "actor": action["actor"],
        "behavior_type": action["action_type"],
        "semantics": action["semantics"],
        "target_refs": action["target_refs"],
        "parameters": action["parameters"],
        "started_at": action["started_at"],
        "delay_seconds": _delay_from_anchor(anchor, action["started_at"]),
        "relations": (),
        "terminal": None,
    }


def _event_transition_label(
    event: BehaviorSourceRecord,
    anchor: Mapping[str, Any],
    relations: Sequence[str],
) -> dict[str, Any]:
    fields = event.fields
    return {
        "target_kind": "event",
        "source_ref": event.uri,
        "actor": _single(fields["participants"]),
        "behavior_type": "event",
        "semantics": fields["semantic_summary"],
        "target_refs": _event_targets(fields),
        "parameters": {},
        "started_at": fields["started_at"],
        "delay_seconds": _delay_from_anchor(anchor, fields["started_at"]),
        "relations": tuple(relations),
        "terminal": None,
    }


def _terminal_transition_label(
    fields: Mapping[str, Any],
    anchor: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "target_kind": "terminal",
        "source_ref": None,
        "actor": None,
        "behavior_type": None,
        "semantics": fields["closure_reason"] or f"{fields['status']} terminal state",
        "target_refs": (),
        "parameters": {},
        "started_at": fields["ended_at"],
        "delay_seconds": _delay_from_anchor(anchor, fields["ended_at"]),
        "relations": (),
        "terminal": {"status": fields["status"], "reason": fields["closure_reason"]},
    }


def _continuation_trajectory_label(
    episode_uri: str,
    fields: Mapping[str, Any],
    future_phases: Sequence[Mapping[str, Any]],
    future_unphased: Sequence[Mapping[str, Any]],
    events_by_uri: Mapping[str, BehaviorSourceRecord],
    event_positions: Mapping[str, int],
    boundary: int,
) -> dict[str, Any]:
    first_phase = future_phases[0]
    future_by_role = {
        role: tuple(
            _projected_event(events_by_uri[item["event_uri"]])
            for item in future_unphased
            if item["role"] == role
        )
        for role in ("contextual", "parallel", "interruption", "uncertain")
    }
    return {
        "next_phase_ref": _member_ref(episode_uri, "phase", first_phase["phase_id"]),
        "mainline": tuple(_projected_phase(episode_uri, phase) for phase in future_phases),
        "remaining_events": tuple(
            _projected_event(events_by_uri[event_uri])
            for phase in future_phases
            for event_uri in phase["event_uris"]
        ),
        "parallel_branches": future_by_role["parallel"],
        "interruptions": future_by_role["interruption"],
        "resumptions": _resumption_steps(
            fields["transitions"], events_by_uri, boundary, event_positions, future=True
        ),
        "future_context": future_by_role["contextual"],
        "uncertain_events": future_by_role["uncertain"],
        "transition_edges": tuple(
            {
                "from_ref": edge["from_event_uri"],
                "to_ref": edge["to_event_uri"],
                "relation": edge["relation"],
            }
            for edge in fields["transitions"]
            if event_positions[edge["to_event_uri"]] > boundary
        ),
        "terminal": None,
    }


def _terminal_trajectory_label(fields: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "next_phase_ref": None,
        "mainline": (),
        "remaining_events": (),
        "parallel_branches": (),
        "interruptions": (),
        "resumptions": (),
        "future_context": (),
        "uncertain_events": (),
        "transition_edges": (),
        "terminal": {"status": fields["status"], "reason": fields["closure_reason"]},
    }
