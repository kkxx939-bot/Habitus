"""Episode Phase 前缀的长轨迹 TrajectorySample 构造。"""

from __future__ import annotations

from collections.abc import Sequence

from prediction.document import PredictionDocument
from prediction.factory import PredictionSampleFactory
from prediction.model import PredictionKind
from prediction.projection._anchors import (
    _anchor_observed_at,
    _episode_terminal_anchor,
    _phase_anchor,
    _target_started_before_anchor,
)
from prediction.projection._behavior_source import BehaviorProjectionSource, BehaviorSourceRecord
from prediction.projection._contract import PROJECTION_VERSION
from prediction.projection._inputs import (
    _behavior_history,
    _episode_input,
    _scope,
)
from prediction.projection._labels import (
    _continuation_trajectory_label,
    _terminal_trajectory_label,
)
from prediction.projection._metadata import (
    _lineage,
    _provenance,
    _quality,
    _supervision,
)
from prediction.projection._partition import (
    _episode_bindings,
    _partition_events,
)
from prediction.projection._refs import (
    _event_subjects,
    _events_inferred_ratio,
    _member_ref,
    _stable_subjects,
)
from prediction.projection._steps import (
    _active_projected_event,
    _active_projected_phase,
    _projected_event,
    _projected_phase,
)


def episode_trajectory_samples(source: BehaviorProjectionSource, factory: PredictionSampleFactory,episode: BehaviorSourceRecord,
    events: Sequence[BehaviorSourceRecord],
) -> tuple[PredictionDocument, ...]:
    fields = episode.fields
    phases = fields["phases"]
    events_by_uri = {event.uri: event for event in events}
    event_positions = {event.uri: index for index, event in enumerate(events)}
    bindings = _episode_bindings(episode, events)
    documents: list[PredictionDocument] = []
    for prefix_length in range(len(phases) + 1):
        anchor = _phase_anchor(episode.uri, fields, prefix_length)
        terminal = prefix_length == len(phases)
        if terminal:
            anchor = _episode_terminal_anchor(episode.uri, fields)
            target_started_at = fields["ended_at"]
        else:
            target_started_at = phases[prefix_length]["started_at"]
        if _target_started_before_anchor(anchor, target_started_at):
            continue
        cutoff = _anchor_observed_at(anchor)
        completed_phases = tuple(
            phase
            for phase in phases[:prefix_length]
            if cutoff is not None and phase["ended_at"] is not None and phase["ended_at"] <= cutoff
        )
        active_phases = tuple(
            phase
            for phase in phases[:prefix_length]
            if cutoff is not None
            and phase["started_at"] <= cutoff
            and (phase["ended_at"] is None or phase["ended_at"] > cutoff)
        )
        future_phase_event_uris = {
            event_uri for phase in phases[prefix_length:] for event_uri in phase["event_uris"]
        }
        input_candidates = (
            events
            if terminal
            else tuple(event for event in events if event.uri not in future_phase_event_uris)
        )
        completed_events, active_events = _partition_events(input_candidates, anchor)
        visible_event_uris = {event.uri for event in (*completed_events, *active_events)}
        future_unphased = tuple(
            item for item in fields["unphased_events"] if item["event_uri"] not in visible_event_uris
        )
        boundary = max((event_positions[uri] for uri in visible_event_uris), default=-1)
        observed_subjects = _stable_subjects(
            fields["participants"],
            _event_subjects((*completed_events, *active_events)),
        )
        parallel_uris = {
            item["event_uri"] for item in fields["unphased_events"] if item["role"] == "parallel"
        }
        interruption_uris = {
            item["event_uri"] for item in fields["unphased_events"] if item["role"] == "interruption"
        }
        history = _behavior_history(
            completed_events=tuple(
                _projected_event(event, expose_source=False) for event in completed_events
            ),
            completed_phases=tuple(
                _projected_phase(episode.uri, phase, expose_source=False)
                for phase in completed_phases
            ),
            active_behaviors=(
                *(
                    _active_projected_event(event, expose_source=False)
                    for event in active_events
                    if event.uri not in parallel_uris | interruption_uris
                ),
                *(
                    _active_projected_phase(episode.uri, phase, expose_source=False)
                    for phase in active_phases
                ),
            ),
            parallel_behaviors=tuple(
                _active_projected_event(event, expose_source=False)
                for event in active_events
                if event.uri in parallel_uris
            ),
            interruptions=tuple(
                _active_projected_event(event, expose_source=False)
                for event in active_events
                if event.uri in interruption_uris
            ),
            resumptions=(),
        )
        if terminal:
            label = _terminal_trajectory_label(fields)
            target_ref = "terminal"
            target_level = "terminal"
            mode = "termination"
        else:
            label = _continuation_trajectory_label(
                episode.uri,
                fields,
                phases[prefix_length:],
                future_unphased,
                events_by_uri,
                event_positions,
                boundary,
            )
            target_ref = _member_ref(episode.uri, "phase", phases[prefix_length]["phase_id"])
            target_level = "phase"
            mode = "continuation"
        partial = fields["status"] == "partial"
        document = factory.build(
            PredictionKind.TRAJECTORY,
            sample_date=fields["episode_date"],
            projection_version=PROJECTION_VERSION,
            identity={
                "container_uri": episode.uri,
                "anchor_type": "phase",
                "prefix_length": anchor["prefix_length"],
                "target_ref": target_ref,
            },
            fields={
                "prediction_scope": _scope(
                    participants=observed_subjects,
                    target_level=target_level,
                    target_domain="episode",
                    prediction_mode=mode,
                ),
                "anchor": anchor,
                "input": _episode_input(
                    anchor,
                    behavior_history=history,
                    observed_subjects=observed_subjects,
                ),
                "label": label,
                "supervision": _supervision(
                    label_status="terminal" if terminal else "observed",
                    started_at=_anchor_observed_at(anchor),
                    closed_at=fields["ended_at"],
                    censoring_reason=(fields["closure_reason"] or "partial_episode") if partial else None,
                ),
                "lineage": _lineage(
                    root_uri=episode.uri,
                    episode_uri=episode.uri,
                    occurrence_group_id=episode.uri,
                ),
                "provenance": _provenance(bindings),
                "quality": _quality(
                    confidence=fields["confidence"],
                    evidence_coverage=fields["evidence_coverage"],
                    conflicts=sum(len(event.fields["conflicts"]) for event in events),
                    inferred_ratio=_events_inferred_ratio((*completed_events, *active_events)),
                ),
            },
        )
        documents.append(document)
    return tuple(documents)
