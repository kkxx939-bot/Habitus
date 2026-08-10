"""Outcome 每条结果的 ConsequenceSample 构造。"""

from __future__ import annotations

from behavior import BehaviorKind, BehaviorURI
from foundation.integrity import canonical_digest
from prediction.document import PredictionDocument
from prediction.factory import PredictionSampleFactory
from prediction.model import PredictionKind
from prediction.projection._anchors import (
    _action_treatment_anchor,
    _anchor_observed_at,
    _event_start_anchor,
)
from prediction.projection._behavior_source import BehaviorProjectionSource
from prediction.projection._contract import PROJECTION_VERSION
from prediction.projection._inputs import (
    _event_input,
    _scope,
)
from prediction.projection._metadata import (
    _lineage,
    _provenance,
    _quality,
    _supervision,
)
from prediction.projection._partition import (
    _partition_actions,
)
from prediction.projection._refs import (
    _action_subjects,
    _inferred_ratio,
    _stable_subjects,
)
from prediction.projection._steps import (
    _active_projected_action,
    _active_projected_event,
)


def outcome_consequence_samples(
    source: BehaviorProjectionSource,
    factory: PredictionSampleFactory,
    uri: BehaviorURI | str,
) -> tuple[PredictionDocument, ...]:
    """从当前 Outcome revision 为每条结果生成一个版本绑定的 ConsequenceSample。"""

    outcome_source = source.read(uri, expected_kind=BehaviorKind.OUTCOME)
    event = source.read(outcome_source.fields["event_uri"], expected_kind=BehaviorKind.EVENT)
    event_fields = event.fields
    actions_by_id = {action["action_id"]: (index, action) for index, action in enumerate(event_fields["actions"])}
    documents: list[PredictionDocument] = []
    for outcome in outcome_source.fields["outcomes"]:
        if outcome["target_type"] == "action":
            target_action_id = outcome["target_action_id"]
            if target_action_id not in actions_by_id:
                raise ValueError("Outcome target Action is absent from its Event")
            action_index, action = actions_by_id[target_action_id]
            prefix_length = action_index
            anchor = _action_treatment_anchor(event.uri, event_fields, action_index)
            selected_behavior = _active_projected_action(event.uri, action)
            completed_actions, active_actions = _partition_actions(
                event_fields["actions"][:prefix_length],
                anchor,
            )
            prediction_input = _event_input(
                event_fields,
                event.uri,
                anchor,
                completed_actions=completed_actions,
                active_actions=active_actions,
            )
            behavior_time = action["started_at"] or event_fields["started_at"]
            target_key = f"action:{target_action_id}"
            observed_subjects = _stable_subjects(
                event_fields["participants"],
                _action_subjects((*completed_actions, *active_actions)),
            )
        else:
            prefix_length = 0
            anchor = _event_start_anchor(event.uri, event_fields)
            if (
                event_fields["ended_at"] is not None
                and event_fields["ended_at"] <= event_fields["onset_available_at"]
            ):
                raise ValueError(
                    "Event consequence projection requires onset identity before completion"
                )
            selected_behavior = _active_projected_event(event)
            prediction_input = _event_input(
                event_fields,
                event.uri,
                anchor,
                completed_actions=(),
            )
            behavior_time = event_fields["started_at"]
            target_key = "event"
            observed_subjects = tuple(event_fields["participants"])

        delay_seconds = max(0.0, (outcome["occurred_at"] - behavior_time).total_seconds())
        consequence_group_id = canonical_digest({"event_uri": event.uri, "target": target_key})
        document = factory.build(
            PredictionKind.CONSEQUENCE,
            sample_date=event_fields["event_date"],
            projection_version=PROJECTION_VERSION,
            materialization_context={
                "outcome_revision": outcome_source.document.metadata.revision,
            },
            identity={
                "outcome_uri": outcome_source.uri,
                "outcome_id": outcome["outcome_id"],
            },
            fields={
                "prediction_scope": _scope(
                    participants=observed_subjects,
                    target_level="outcome",
                    target_domain=None,
                    prediction_mode="consequence",
                ),
                "anchor": anchor,
                "input": prediction_input,
                "treatment": selected_behavior,
                "label": {
                    "outcome": {
                        "outcome_id": outcome["outcome_id"],
                        "occurred_at": outcome["occurred_at"],
                        "outcome_type": outcome["outcome_type"],
                        "semantics": outcome["semantics"],
                        "valence": outcome["valence"],
                        "knowledge_state": outcome["knowledge_state"],
                        "confidence": outcome["confidence"],
                        "delay_seconds": delay_seconds,
                    },
                    "attribution": "temporal_only",
                },
                "supervision": _supervision(
                    label_status="observed",
                    started_at=_anchor_observed_at(anchor),
                    closed_at=outcome["occurred_at"],
                ),
                "lineage": _lineage(
                    root_uri=event.uri,
                    event_uri=event.uri,
                    outcome_uri=outcome_source.uri,
                    occurrence_group_id=event.uri,
                    consequence_group_id=consequence_group_id,
                ),
                "provenance": _provenance(
                    (
                        event.binding(
                            member_type="action" if outcome["target_type"] == "action" else "event",
                            member_id=outcome["target_action_id"],
                        ),
                        outcome_source.binding(member_type="outcome", member_id=outcome["outcome_id"]),
                    )
                ),
                "quality": _quality(
                    confidence=min(event_fields["confidence"], outcome["confidence"]),
                    conflicts=len(event_fields["conflicts"]),
                    inferred_ratio=_inferred_ratio(
                        () if outcome["target_type"] == "event" else (*completed_actions, *active_actions)
                    ),
                ),
            },
        )
        documents.append(document)
    return tuple(documents)
