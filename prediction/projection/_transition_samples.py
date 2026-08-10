"""Event 与 Episode 的下一步 TransitionSample 构造。"""

from __future__ import annotations

from collections.abc import Sequence

from behavior import BehaviorKind, BehaviorURI
from prediction.document import PredictionDocument
from prediction.factory import PredictionSampleFactory
from prediction.model import PredictionKind
from prediction.projection._anchors import (
    _action_anchor,
    _anchor_observed_at,
    _event_anchor,
    _target_started_before_anchor,
)
from prediction.projection._behavior_source import BehaviorProjectionSource, BehaviorSourceRecord
from prediction.projection._contract import PROJECTION_VERSION
from prediction.projection._inputs import (
    _episode_input,
    _event_input,
    _scope,
)
from prediction.projection._labels import (
    _action_transition_label,
    _event_transition_label,
    _terminal_transition_label,
)
from prediction.projection._metadata import (
    _lineage,
    _provenance,
    _quality,
    _supervision,
)
from prediction.projection._partition import (
    _episode_bindings,
    _partition_actions,
    _partition_events,
)
from prediction.projection._refs import (
    _action_subjects,
    _event_subjects,
    _events_inferred_ratio,
    _inferred_ratio,
    _member_ref,
    _stable_subjects,
)


def event_transition_samples(
    source: BehaviorProjectionSource,
    factory: PredictionSampleFactory,
    uri: BehaviorURI | str,
) -> tuple[PredictionDocument, ...]:
    """从一个 Event 生成每个 Action 前缀以及 Event 终止的 TransitionSample。"""

    event = source.read(uri, expected_kind=BehaviorKind.EVENT)
    fields = event.fields
    actions = fields["actions"]
    documents: list[PredictionDocument] = []
    for prefix_length, action in enumerate(actions):
        anchor = _action_anchor(event.uri, fields, prefix_length)
        if _target_started_before_anchor(anchor, action["started_at"]):
            continue
        completed_actions, active_actions = _partition_actions(actions[:prefix_length], anchor)
        observed_subjects = _stable_subjects(
            fields["participants"],
            _action_subjects((*completed_actions, *active_actions)),
        )
        document = factory.build(
            PredictionKind.TRANSITION,
            sample_date=fields["event_date"],
            projection_version=PROJECTION_VERSION,
            identity={
                "container_uri": event.uri,
                "anchor_type": "action",
                "prefix_length": anchor["prefix_length"],
                "target_ref": _member_ref(event.uri, "action", action["action_id"]),
            },
            fields={
                "prediction_scope": _scope(
                    participants=observed_subjects,
                    target_level="action",
                    target_domain=None,
                    prediction_mode="next_step",
                ),
                "anchor": anchor,
                "input": _event_input(
                    fields,
                    event.uri,
                    anchor,
                    completed_actions=completed_actions,
                    active_actions=active_actions,
                ),
                "label": _action_transition_label(event.uri, action, anchor),
                "supervision": _supervision(
                    label_status="observed",
                    started_at=_anchor_observed_at(anchor),
                    closed_at=action["started_at"] or action["ended_at"] or fields["ended_at"],
                ),
                "lineage": _lineage(
                    root_uri=event.uri,
                    event_uri=event.uri,
                    occurrence_group_id=event.uri,
                ),
                "provenance": _provenance((event.binding(member_type="action", member_id=action["action_id"]),)),
                "quality": _quality(
                    confidence=fields["confidence"],
                    conflicts=len(fields["conflicts"]),
                    inferred_ratio=_inferred_ratio((*completed_actions, *active_actions)),
                ),
            },
        )
        documents.append(document)

    terminal_anchor = _action_anchor(event.uri, fields, len(actions))
    if _target_started_before_anchor(terminal_anchor, fields["ended_at"]):
        return tuple(documents)
    completed_actions, active_actions = _partition_actions(actions, terminal_anchor)
    partial = fields["status"] == "partial"
    observed_subjects = _stable_subjects(fields["participants"], _action_subjects(actions))
    terminal = factory.build(
        PredictionKind.TRANSITION,
        sample_date=fields["event_date"],
        projection_version=PROJECTION_VERSION,
        identity={
            "container_uri": event.uri,
            "anchor_type": "action",
            "prefix_length": terminal_anchor["prefix_length"],
            "target_ref": "terminal",
        },
        fields={
            "prediction_scope": _scope(
                participants=observed_subjects,
                target_level="terminal",
                target_domain="event",
                prediction_mode="termination",
            ),
            "anchor": terminal_anchor,
            "input": _event_input(
                fields,
                event.uri,
                terminal_anchor,
                completed_actions=completed_actions,
                active_actions=active_actions,
            ),
            "label": _terminal_transition_label(fields, terminal_anchor),
            "supervision": _supervision(
                label_status="terminal",
                started_at=_anchor_observed_at(terminal_anchor),
                closed_at=fields["ended_at"],
                censoring_reason=(fields["closure_reason"] or "partial_event") if partial else None,
            ),
            "lineage": _lineage(
                root_uri=event.uri,
                event_uri=event.uri,
                occurrence_group_id=event.uri,
            ),
            "provenance": _provenance((event.binding(member_type="terminal"),)),
            "quality": _quality(
                confidence=fields["confidence"],
                conflicts=len(fields["conflicts"]),
                inferred_ratio=_inferred_ratio((*completed_actions, *active_actions)),
            ),
        },
    )
    documents.append(terminal)
    return tuple(documents)


def episode_transition_samples(source: BehaviorProjectionSource, factory: PredictionSampleFactory,episode: BehaviorSourceRecord,
    events: Sequence[BehaviorSourceRecord],
) -> tuple[PredictionDocument, ...]:
    fields = episode.fields
    bindings = _episode_bindings(episode, events)
    documents: list[PredictionDocument] = []
    for prefix_length, target in enumerate(events):
        anchor = _event_anchor(episode.uri, fields, events, prefix_length)
        if _target_started_before_anchor(anchor, target.fields["started_at"]):
            continue
        completed_events, active_events = _partition_events(events[:prefix_length], anchor)
        observed_subjects = _stable_subjects(
            fields["participants"],
            _event_subjects((*completed_events, *active_events)),
        )
        relations = tuple(
            f"{edge['relation']}:{edge['from_event_uri']}->{edge['to_event_uri']}"
            for edge in fields["transitions"]
            if edge["to_event_uri"] == target.uri
            and edge["from_event_uri"] in {event.uri for event in events[:prefix_length]}
        )
        document = factory.build(
            PredictionKind.TRANSITION,
            sample_date=fields["episode_date"],
            projection_version=PROJECTION_VERSION,
            identity={
                "container_uri": episode.uri,
                "anchor_type": "event",
                "prefix_length": anchor["prefix_length"],
                "target_ref": target.uri,
            },
            fields={
                "prediction_scope": _scope(
                    participants=observed_subjects,
                    target_level="event",
                    target_domain="event",
                    prediction_mode="next_step",
                ),
                "anchor": anchor,
                "input": _episode_input(
                    anchor,
                    completed_events=completed_events,
                    active_events=active_events,
                ),
                "label": _event_transition_label(target, anchor, relations),
                "supervision": _supervision(
                    label_status="observed",
                    started_at=_anchor_observed_at(anchor),
                    closed_at=target.fields["started_at"],
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
                    conflicts=sum(len(event.fields["conflicts"]) for event in events[: prefix_length + 1]),
                    inferred_ratio=_events_inferred_ratio((*completed_events, *active_events)),
                ),
            },
        )
        documents.append(document)

    anchor = _event_anchor(episode.uri, fields, events, len(events))
    if _target_started_before_anchor(anchor, fields["ended_at"]):
        return tuple(documents)
    completed_events, active_events = _partition_events(events, anchor)
    partial = fields["status"] == "partial"
    observed_subjects = _stable_subjects(fields["participants"], _event_subjects(events))
    terminal = factory.build(
        PredictionKind.TRANSITION,
        sample_date=fields["episode_date"],
        projection_version=PROJECTION_VERSION,
        identity={
            "container_uri": episode.uri,
            "anchor_type": "event",
            "prefix_length": anchor["prefix_length"],
            "target_ref": "terminal",
        },
        fields={
            "prediction_scope": _scope(
                participants=observed_subjects,
                target_level="terminal",
                target_domain="episode",
                prediction_mode="termination",
            ),
            "anchor": anchor,
            "input": _episode_input(
                anchor,
                completed_events=completed_events,
                active_events=active_events,
            ),
            "label": _terminal_transition_label(fields, anchor),
            "supervision": _supervision(
                label_status="terminal",
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
                inferred_ratio=_events_inferred_ratio(events),
            ),
        },
    )
    documents.append(terminal)
    return tuple(documents)
