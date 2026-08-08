"""Behavior RecordKind 对应的严格 Payload 判别联合。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeAlias

from behavior._validation import (
    bounded_text,
    external_reference,
    identifier,
    identifier_tuple,
    json_snapshot,
    json_value_snapshot,
    optional_bounded_text,
    optional_identifier,
    sha256_digest,
    strict_object,
)

if TYPE_CHECKING:
    from behavior.evidence.content import BehaviorModality, BehaviorRecordKind, BehaviorRole

_ABSOLUTE_CHARS = 1_000_000
_ABSOLUTE_ITEMS = 10_000
_ABSOLUTE_DEPTH = 32
_SEMANTIC_SYSTEM_FIELDS = frozenset(
    {
        "claim_id",
        "claim_kind",
        "claim_sequence",
        "content_digest",
        "evidence_record_id",
        "evidence_sequence",
        "normalizer_fingerprint",
        "processing_identity",
        "producer_fingerprint",
        "capability_digest",
        "source_trust",
        "semantic_digest",
        "policy_digest",
        "event_id",
        "episode_id",
        "pattern_id",
        "prediction_id",
        "storage_metadata",
        "owner_ref",
        "owner_binding",
        "owner_identity",
        "owner_identity_digest",
        "user_id",
        "tenant_id",
        "account_id",
    }
)


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
        forbidden_keys=_SEMANTIC_SYSTEM_FIELDS,
    )


def _value(value: object, field_name: str) -> Any:
    return json_value_snapshot(
        value,
        field_name,
        maximum_chars=_ABSOLUTE_CHARS,
        maximum_items=_ABSOLUTE_ITEMS,
        maximum_depth=_ABSOLUTE_DEPTH,
        forbidden_keys=_SEMANTIC_SYSTEM_FIELDS,
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
        object.__setattr__(
            self,
            "labels",
            identifier_tuple(self.labels, "payload.labels", maximum_items=_ABSOLUTE_ITEMS),
        )


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


PayloadDecoder = Callable[[object], object]


@dataclass(frozen=True, slots=True)
class PayloadCodec:
    payload_type: type[Any]
    decoders: tuple[tuple[str, PayloadDecoder], ...] = ()
    identifier_fields: frozenset[str] = frozenset()
    reference_fields: frozenset[str] = frozenset()
    text_fields: frozenset[str] = frozenset()

    def encode(self, payload: BehaviorPayload) -> dict[str, Any]:
        if not isinstance(payload, self.payload_type):
            raise TypeError("payload does not match its registered codec")
        return {
            item.name: _encode_value(getattr(payload, item.name))
            for item in fields(self.payload_type)
        }

    def decode(self, value: object) -> BehaviorPayload:
        field_names = frozenset(item.name for item in fields(self.payload_type))
        data = strict_object(value, "payload", field_names)
        converters = dict(self.decoders)
        arguments = {
            name: converters[name](item) if name in converters else item
            for name, item in data.items()
        }
        return self.payload_type(**arguments)


def _encode_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {key: _encode_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_encode_value(item) for item in value)
    return value


def payload_from_dict(
    record_kind: BehaviorRecordKind,
    value: object,
) -> BehaviorPayload:
    from behavior.evidence.specs import record_spec

    return record_spec(record_kind).payload_codec.decode(value)
