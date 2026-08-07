"""Behavior RecordKind 对应的严格 Payload 判别联合。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeAlias

from behavior._validation import (
    bounded_text,
    external_reference,
    identifier,
    json_snapshot,
    json_value_snapshot,
    optional_bounded_text,
    optional_identifier,
    require_fields,
    sha256_digest,
    strict_fields,
)
from behavior.config import BehaviorEvidenceConfig

if TYPE_CHECKING:
    from behavior.evidence.content import BehaviorModality, BehaviorRecordKind, BehaviorRole

_ABSOLUTE_CHARS = 1_000_000
_ABSOLUTE_ITEMS = 10_000
_ABSOLUTE_DEPTH = 32


class PhaseHint(str, Enum):
    STARTED = "STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    END_CANDIDATE = "END_CANDIDATE"
    INTERRUPTED_CANDIDATE = "INTERRUPTED_CANDIDATE"
    UNKNOWN = "UNKNOWN"


class InteractionMode(str, Enum):
    DIALOGUE = "DIALOGUE"
    AMBIENT = "AMBIENT"
    UNKNOWN = "UNKNOWN"


class CommunicationChannel(str, Enum):
    TEXT = "TEXT"
    VOICE = "VOICE"
    OTHER = "OTHER"


class ToolResultStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"
    CANCELLED = "CANCELLED"
    UNKNOWN = "UNKNOWN"


class CoverageStatus(str, Enum):
    COVERED = "COVERED"
    BLIND = "BLIND"
    UNKNOWN = "UNKNOWN"


class FeedbackPolarity(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    MIXED = "MIXED"


def _attributes(value: object, field_name: str) -> Mapping[str, Any]:
    return json_snapshot(
        value,
        field_name,
        maximum_chars=_ABSOLUTE_CHARS,
        maximum_items=_ABSOLUTE_ITEMS,
        maximum_depth=_ABSOLUTE_DEPTH,
    )


def _value(value: object, field_name: str) -> Any:
    return json_value_snapshot(
        value,
        field_name,
        maximum_chars=_ABSOLUTE_CHARS,
        maximum_items=_ABSOLUTE_ITEMS,
        maximum_depth=_ABSOLUTE_DEPTH,
    )


@dataclass(frozen=True)
class ActivitySegmentPayload:
    activity: str
    phase_hint: PhaseHint
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "activity", identifier(self.activity, "payload.activity"))
        object.__setattr__(self, "phase_hint", PhaseHint(self.phase_hint))
        object.__setattr__(self, "attributes", _attributes(self.attributes, "payload.attributes"))


@dataclass(frozen=True)
class UtteranceSegmentPayload:
    text: str
    language: str | None
    interaction_mode: InteractionMode
    communication_channel: CommunicationChannel

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", bounded_text(self.text, "payload.text", maximum=_ABSOLUTE_CHARS))
        object.__setattr__(self, "language", optional_identifier(self.language, "payload.language", maximum=64))
        object.__setattr__(self, "interaction_mode", InteractionMode(self.interaction_mode))
        object.__setattr__(self, "communication_channel", CommunicationChannel(self.communication_channel))


@dataclass(frozen=True)
class StateAssertionPayload:
    state_name: str
    value: Any
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_name", identifier(self.state_name, "payload.state_name"))
        object.__setattr__(self, "value", _value(self.value, "payload.value"))
        object.__setattr__(self, "attributes", _attributes(self.attributes, "payload.attributes"))


@dataclass(frozen=True)
class StateTransitionPayload:
    state_name: str
    before: Any
    after: Any
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_name", identifier(self.state_name, "payload.state_name"))
        object.__setattr__(self, "before", _value(self.before, "payload.before"))
        object.__setattr__(self, "after", _value(self.after, "payload.after"))
        object.__setattr__(self, "attributes", _attributes(self.attributes, "payload.attributes"))


@dataclass(frozen=True)
class InteractionSegmentPayload:
    interaction_type: str
    counterparty_role: BehaviorRole
    phase_hint: PhaseHint
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        from behavior.evidence.content import BehaviorRole

        object.__setattr__(self, "interaction_type", identifier(self.interaction_type, "payload.interaction_type"))
        object.__setattr__(self, "counterparty_role", BehaviorRole(self.counterparty_role))
        object.__setattr__(self, "phase_hint", PhaseHint(self.phase_hint))
        object.__setattr__(self, "attributes", _attributes(self.attributes, "payload.attributes"))


@dataclass(frozen=True)
class ActionEventPayload:
    action_name: str
    phase: str
    result: Any
    capability_ref: str | None
    target_ref: str | None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "action_name", identifier(self.action_name, "payload.action_name"))
        object.__setattr__(self, "phase", identifier(self.phase, "payload.phase"))
        object.__setattr__(self, "result", _value(self.result, "payload.result"))
        object.__setattr__(self, "capability_ref", optional_identifier(self.capability_ref, "payload.capability_ref"))
        object.__setattr__(self, "target_ref", optional_identifier(self.target_ref, "payload.target_ref"))
        object.__setattr__(self, "attributes", _attributes(self.attributes, "payload.attributes"))


@dataclass(frozen=True)
class ToolCallPayload:
    tool_name: str
    tool_call_id: str
    arguments_digest: str
    arguments_summary: str | None
    capability_ref: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_name", identifier(self.tool_name, "payload.tool_name"))
        object.__setattr__(self, "tool_call_id", identifier(self.tool_call_id, "payload.tool_call_id"))
        object.__setattr__(self, "arguments_digest", sha256_digest(self.arguments_digest, "payload.arguments_digest"))
        object.__setattr__(
            self,
            "arguments_summary",
            optional_bounded_text(self.arguments_summary, "payload.arguments_summary", maximum=_ABSOLUTE_CHARS),
        )
        object.__setattr__(self, "capability_ref", optional_identifier(self.capability_ref, "payload.capability_ref"))


@dataclass(frozen=True)
class ToolResultPayload:
    tool_name: str
    tool_call_id: str
    status: ToolResultStatus
    result_ref: str | None
    result_digest: str
    result_summary: str | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_name", identifier(self.tool_name, "payload.tool_name"))
        object.__setattr__(self, "tool_call_id", identifier(self.tool_call_id, "payload.tool_call_id"))
        object.__setattr__(self, "status", ToolResultStatus(self.status))
        object.__setattr__(
            self,
            "result_ref",
            None
            if self.result_ref is None
            else external_reference(self.result_ref, "payload.result_ref", maximum=2_048),
        )
        object.__setattr__(self, "result_digest", sha256_digest(self.result_digest, "payload.result_digest"))
        object.__setattr__(
            self,
            "result_summary",
            optional_bounded_text(self.result_summary, "payload.result_summary", maximum=_ABSOLUTE_CHARS),
        )


@dataclass(frozen=True)
class EnvironmentChangePayload:
    predicate: str
    before: Any
    after: Any
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "predicate", identifier(self.predicate, "payload.predicate"))
        object.__setattr__(self, "before", _value(self.before, "payload.before"))
        object.__setattr__(self, "after", _value(self.after, "payload.after"))
        object.__setattr__(self, "attributes", _attributes(self.attributes, "payload.attributes"))


@dataclass(frozen=True)
class CoverageIntervalPayload:
    modality: BehaviorModality
    coverage_status: CoverageStatus
    coverage_scope_ref: str | None
    reason: str | None

    def __post_init__(self) -> None:
        from behavior.evidence.content import BehaviorModality

        object.__setattr__(self, "modality", BehaviorModality(self.modality))
        object.__setattr__(self, "coverage_status", CoverageStatus(self.coverage_status))
        object.__setattr__(
            self,
            "coverage_scope_ref",
            optional_identifier(self.coverage_scope_ref, "payload.coverage_scope_ref"),
        )
        object.__setattr__(self, "reason", optional_bounded_text(self.reason, "payload.reason", maximum=2_048))


@dataclass(frozen=True)
class FeedbackPayload:
    feedback_kind: str
    target_ref: str | None
    polarity: FeedbackPolarity
    explicit_text_ref: str | None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "feedback_kind", identifier(self.feedback_kind, "payload.feedback_kind"))
        object.__setattr__(self, "target_ref", optional_identifier(self.target_ref, "payload.target_ref"))
        object.__setattr__(self, "polarity", FeedbackPolarity(self.polarity))
        object.__setattr__(
            self,
            "explicit_text_ref",
            None
            if self.explicit_text_ref is None
            else external_reference(self.explicit_text_ref, "payload.explicit_text_ref", maximum=2_048),
        )
        object.__setattr__(self, "attributes", _attributes(self.attributes, "payload.attributes"))


@dataclass(frozen=True)
class FreeTextSemanticPayload:
    text: str
    language: str | None
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "text", bounded_text(self.text, "payload.text", maximum=_ABSOLUTE_CHARS))
        object.__setattr__(self, "language", optional_identifier(self.language, "payload.language", maximum=64))
        if not isinstance(self.labels, tuple) or len(self.labels) > _ABSOLUTE_ITEMS:
            raise TypeError("payload.labels must be a bounded tuple")
        labels = tuple(identifier(item, f"payload.labels[{index}]") for index, item in enumerate(self.labels))
        if len(labels) != len(set(labels)):
            raise ValueError("payload.labels must not contain duplicates")
        object.__setattr__(self, "labels", labels)


BehaviorPayload: TypeAlias = (
    ActivitySegmentPayload
    | UtteranceSegmentPayload
    | StateAssertionPayload
    | StateTransitionPayload
    | InteractionSegmentPayload
    | ActionEventPayload
    | ToolCallPayload
    | ToolResultPayload
    | EnvironmentChangePayload
    | CoverageIntervalPayload
    | FeedbackPayload
    | FreeTextSemanticPayload
)


def payload_to_dict(payload: BehaviorPayload) -> dict[str, Any]:
    if isinstance(payload, ActivitySegmentPayload):
        return {"activity": payload.activity, "phase_hint": payload.phase_hint.value, "attributes": payload.attributes}
    if isinstance(payload, UtteranceSegmentPayload):
        return {
            "text": payload.text,
            "language": payload.language,
            "interaction_mode": payload.interaction_mode.value,
            "communication_channel": payload.communication_channel.value,
        }
    if isinstance(payload, StateAssertionPayload):
        return {"state_name": payload.state_name, "value": payload.value, "attributes": payload.attributes}
    if isinstance(payload, StateTransitionPayload):
        return {
            "state_name": payload.state_name,
            "before": payload.before,
            "after": payload.after,
            "attributes": payload.attributes,
        }
    if isinstance(payload, InteractionSegmentPayload):
        return {
            "interaction_type": payload.interaction_type,
            "counterparty_role": payload.counterparty_role.value,
            "phase_hint": payload.phase_hint.value,
            "attributes": payload.attributes,
        }
    if isinstance(payload, ActionEventPayload):
        return {
            "action_name": payload.action_name,
            "phase": payload.phase,
            "result": payload.result,
            "capability_ref": payload.capability_ref,
            "target_ref": payload.target_ref,
            "attributes": payload.attributes,
        }
    if isinstance(payload, ToolCallPayload):
        return {
            "tool_name": payload.tool_name,
            "tool_call_id": payload.tool_call_id,
            "arguments_digest": payload.arguments_digest,
            "arguments_summary": payload.arguments_summary,
            "capability_ref": payload.capability_ref,
        }
    if isinstance(payload, ToolResultPayload):
        return {
            "tool_name": payload.tool_name,
            "tool_call_id": payload.tool_call_id,
            "status": payload.status.value,
            "result_ref": payload.result_ref,
            "result_digest": payload.result_digest,
            "result_summary": payload.result_summary,
        }
    if isinstance(payload, EnvironmentChangePayload):
        return {"predicate": payload.predicate, "before": payload.before, "after": payload.after, "attributes": payload.attributes}
    if isinstance(payload, CoverageIntervalPayload):
        return {
            "modality": payload.modality.value,
            "coverage_status": payload.coverage_status.value,
            "coverage_scope_ref": payload.coverage_scope_ref,
            "reason": payload.reason,
        }
    if isinstance(payload, FeedbackPayload):
        return {
            "feedback_kind": payload.feedback_kind,
            "target_ref": payload.target_ref,
            "polarity": payload.polarity.value,
            "explicit_text_ref": payload.explicit_text_ref,
            "attributes": payload.attributes,
        }
    if isinstance(payload, FreeTextSemanticPayload):
        return {"text": payload.text, "language": payload.language, "labels": payload.labels}
    raise TypeError("unsupported Behavior payload")


def payload_from_dict(
    record_kind: BehaviorRecordKind,
    value: object,
) -> BehaviorPayload:
    from behavior.evidence.content import BehaviorModality, BehaviorRecordKind, BehaviorRole

    kind = BehaviorRecordKind(record_kind)
    fields: dict[BehaviorRecordKind, frozenset[str]] = {
        BehaviorRecordKind.ACTIVITY_SEGMENT: frozenset({"activity", "phase_hint", "attributes"}),
        BehaviorRecordKind.UTTERANCE_SEGMENT: frozenset(
            {"text", "language", "interaction_mode", "communication_channel"}
        ),
        BehaviorRecordKind.STATE_ASSERTION: frozenset({"state_name", "value", "attributes"}),
        BehaviorRecordKind.STATE_TRANSITION: frozenset({"state_name", "before", "after", "attributes"}),
        BehaviorRecordKind.INTERACTION_SEGMENT: frozenset(
            {"interaction_type", "counterparty_role", "phase_hint", "attributes"}
        ),
        BehaviorRecordKind.ACTION_EVENT: frozenset(
            {"action_name", "phase", "result", "capability_ref", "target_ref", "attributes"}
        ),
        BehaviorRecordKind.TOOL_CALL_EVENT: frozenset(
            {"tool_name", "tool_call_id", "arguments_digest", "arguments_summary", "capability_ref"}
        ),
        BehaviorRecordKind.TOOL_RESULT_EVENT: frozenset(
            {"tool_name", "tool_call_id", "status", "result_ref", "result_digest", "result_summary"}
        ),
        BehaviorRecordKind.ENVIRONMENT_CHANGE: frozenset({"predicate", "before", "after", "attributes"}),
        BehaviorRecordKind.COVERAGE_INTERVAL: frozenset(
            {"modality", "coverage_status", "coverage_scope_ref", "reason"}
        ),
        BehaviorRecordKind.FEEDBACK_EVENT: frozenset(
            {"feedback_kind", "target_ref", "polarity", "explicit_text_ref", "attributes"}
        ),
        BehaviorRecordKind.FREE_TEXT_SEMANTIC: frozenset({"text", "language", "labels"}),
    }
    data = strict_fields(value, "payload", fields[kind])
    require_fields(data, "payload", fields[kind])
    if kind is BehaviorRecordKind.ACTIVITY_SEGMENT:
        return ActivitySegmentPayload(data["activity"], PhaseHint(data["phase_hint"]), data["attributes"])
    if kind is BehaviorRecordKind.UTTERANCE_SEGMENT:
        return UtteranceSegmentPayload(
            data["text"],
            data["language"],
            InteractionMode(data["interaction_mode"]),
            CommunicationChannel(data["communication_channel"]),
        )
    if kind is BehaviorRecordKind.STATE_ASSERTION:
        return StateAssertionPayload(data["state_name"], data["value"], data["attributes"])
    if kind is BehaviorRecordKind.STATE_TRANSITION:
        return StateTransitionPayload(data["state_name"], data["before"], data["after"], data["attributes"])
    if kind is BehaviorRecordKind.INTERACTION_SEGMENT:
        return InteractionSegmentPayload(
            data["interaction_type"],
            BehaviorRole(data["counterparty_role"]),
            PhaseHint(data["phase_hint"]),
            data["attributes"],
        )
    if kind is BehaviorRecordKind.ACTION_EVENT:
        return ActionEventPayload(
            data["action_name"],
            data["phase"],
            data["result"],
            data["capability_ref"],
            data["target_ref"],
            data["attributes"],
        )
    if kind is BehaviorRecordKind.TOOL_CALL_EVENT:
        return ToolCallPayload(
            data["tool_name"],
            data["tool_call_id"],
            data["arguments_digest"],
            data["arguments_summary"],
            data["capability_ref"],
        )
    if kind is BehaviorRecordKind.TOOL_RESULT_EVENT:
        return ToolResultPayload(
            data["tool_name"],
            data["tool_call_id"],
            ToolResultStatus(data["status"]),
            data["result_ref"],
            data["result_digest"],
            data["result_summary"],
        )
    if kind is BehaviorRecordKind.ENVIRONMENT_CHANGE:
        return EnvironmentChangePayload(data["predicate"], data["before"], data["after"], data["attributes"])
    if kind is BehaviorRecordKind.COVERAGE_INTERVAL:
        return CoverageIntervalPayload(
            BehaviorModality(data["modality"]),
            CoverageStatus(data["coverage_status"]),
            data["coverage_scope_ref"],
            data["reason"],
        )
    if kind is BehaviorRecordKind.FEEDBACK_EVENT:
        return FeedbackPayload(
            data["feedback_kind"],
            data["target_ref"],
            FeedbackPolarity(data["polarity"]),
            data["explicit_text_ref"],
            data["attributes"],
        )
    return FreeTextSemanticPayload(data["text"], data["language"], tuple(data["labels"]))


def validate_payload_capacity(payload: BehaviorPayload, config: BehaviorEvidenceConfig) -> None:
    snapshot = payload_to_dict(payload)
    json_value_snapshot(
        snapshot,
        "payload",
        maximum_chars=config.max_payload_chars,
        maximum_items=config.max_payload_items,
        maximum_depth=config.max_payload_depth,
    )
    for name in ("text", "arguments_summary", "result_summary", "reason"):
        value = snapshot.get(name)
        if isinstance(value, str) and len(value) > config.max_text_chars:
            raise ValueError(f"payload.{name} exceeds the configured text boundary")


__all__ = [
    "ActionEventPayload",
    "ActivitySegmentPayload",
    "BehaviorPayload",
    "CommunicationChannel",
    "CoverageIntervalPayload",
    "CoverageStatus",
    "EnvironmentChangePayload",
    "FeedbackPayload",
    "FeedbackPolarity",
    "FreeTextSemanticPayload",
    "InteractionMode",
    "InteractionSegmentPayload",
    "PhaseHint",
    "StateAssertionPayload",
    "StateTransitionPayload",
    "ToolCallPayload",
    "ToolResultPayload",
    "ToolResultStatus",
    "UtteranceSegmentPayload",
    "payload_from_dict",
    "payload_to_dict",
    "validate_payload_capacity",
]
