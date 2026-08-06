"""由 SemanticRecordKind 判别的不可变语义 Payload 联合。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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
    require_fields,
    strict_fields,
)
from behavior.config import IngressConfig
from behavior.errors import SemanticRecordError

if TYPE_CHECKING:
    from behavior.ingress.model import SemanticModality


class PhaseHint(str, Enum):
    STARTED = "STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    END_CANDIDATE = "END_CANDIDATE"
    INTERRUPTED_CANDIDATE = "INTERRUPTED_CANDIDATE"
    UNKNOWN = "UNKNOWN"


class UtteranceChannel(str, Enum):
    VOICE = "VOICE"
    TEXT = "TEXT"


class InteractionCounterpartyRole(str, Enum):
    ROBOT = "ROBOT"
    AGENT = "AGENT"
    OTHER_ANONYMOUS = "OTHER_ANONYMOUS"
    ENVIRONMENT = "ENVIRONMENT"
    TOOL = "TOOL"


class CoverageStatus(str, Enum):
    COVERED = "COVERED"
    BLIND = "BLIND"
    UNKNOWN = "UNKNOWN"


def _limits(config: IngressConfig | None) -> IngressConfig:
    if config is not None and not isinstance(config, IngressConfig):
        raise TypeError("config must be IngressConfig")
    return config or IngressConfig()


def _attributes(value: object, field_name: str, limits: IngressConfig) -> Mapping[str, Any]:
    return json_snapshot(
        value,
        field_name,
        maximum_chars=limits.max_payload_chars,
        maximum_items=limits.max_payload_items,
        maximum_depth=limits.max_payload_depth,
    )


def _value(value: object, field_name: str, limits: IngressConfig) -> Any:
    return json_value_snapshot(
        value,
        field_name,
        maximum_chars=limits.max_payload_chars,
        maximum_items=limits.max_payload_items,
        maximum_depth=limits.max_payload_depth,
    )


@dataclass(frozen=True, init=False)
class ActivitySegmentPayload:
    activity: str
    phase_hint: PhaseHint
    attributes: Mapping[str, Any]

    def __init__(
        self, activity: object, phase_hint: PhaseHint | str, attributes: object, *, config: IngressConfig | None = None
    ) -> None:
        limits = _limits(config)
        object.__setattr__(self, "activity", identifier(activity, "payload.activity"))
        object.__setattr__(self, "phase_hint", PhaseHint(phase_hint))
        object.__setattr__(self, "attributes", _attributes(attributes, "payload.attributes", limits))

    def to_dict(self) -> dict[str, object]:
        return {"activity": self.activity, "phase_hint": self.phase_hint.value, "attributes": self.attributes}


@dataclass(frozen=True, init=False)
class UtteranceSegmentPayload:
    text: str
    language: str
    channel: UtteranceChannel

    def __init__(
        self, text: object, language: object, channel: UtteranceChannel | str, *, config: IngressConfig | None = None
    ) -> None:
        limits = _limits(config)
        object.__setattr__(self, "text", bounded_text(text, "payload.text", maximum=limits.max_text_chars))
        object.__setattr__(self, "language", identifier(language, "payload.language", maximum=64))
        object.__setattr__(self, "channel", UtteranceChannel(channel))

    def to_dict(self) -> dict[str, object]:
        return {"text": self.text, "language": self.language, "channel": self.channel.value}


@dataclass(frozen=True, init=False)
class StateAssertionPayload:
    state_name: str
    value: Any

    def __init__(self, state_name: object, value: object, *, config: IngressConfig | None = None) -> None:
        limits = _limits(config)
        object.__setattr__(self, "state_name", identifier(state_name, "payload.state_name"))
        object.__setattr__(self, "value", _value(value, "payload.value", limits))

    def to_dict(self) -> dict[str, object]:
        return {"state_name": self.state_name, "value": self.value}


@dataclass(frozen=True, init=False)
class StateTransitionPayload:
    state_name: str
    before: Any
    after: Any

    def __init__(
        self, state_name: object, before: object, after: object, *, config: IngressConfig | None = None
    ) -> None:
        limits = _limits(config)
        object.__setattr__(self, "state_name", identifier(state_name, "payload.state_name"))
        object.__setattr__(self, "before", _value(before, "payload.before", limits))
        object.__setattr__(self, "after", _value(after, "payload.after", limits))

    def to_dict(self) -> dict[str, object]:
        return {"state_name": self.state_name, "before": self.before, "after": self.after}


@dataclass(frozen=True, init=False)
class InteractionSegmentPayload:
    interaction_type: str
    counterparty_role: InteractionCounterpartyRole
    phase_hint: PhaseHint
    attributes: Mapping[str, Any]

    def __init__(
        self,
        interaction_type: object,
        counterparty_role: InteractionCounterpartyRole | str,
        phase_hint: PhaseHint | str,
        attributes: object,
        *,
        config: IngressConfig | None = None,
    ) -> None:
        limits = _limits(config)
        object.__setattr__(self, "interaction_type", identifier(interaction_type, "payload.interaction_type"))
        object.__setattr__(self, "counterparty_role", InteractionCounterpartyRole(counterparty_role))
        object.__setattr__(self, "phase_hint", PhaseHint(phase_hint))
        object.__setattr__(self, "attributes", _attributes(attributes, "payload.attributes", limits))

    def to_dict(self) -> dict[str, object]:
        return {
            "interaction_type": self.interaction_type,
            "counterparty_role": self.counterparty_role.value,
            "phase_hint": self.phase_hint.value,
            "attributes": self.attributes,
        }


@dataclass(frozen=True, init=False)
class ActionEventPayload:
    action_name: str
    phase: str
    result: str | None
    attributes: Mapping[str, Any]

    def __init__(
        self,
        action_name: object,
        phase: object,
        result: object,
        attributes: object,
        *,
        config: IngressConfig | None = None,
    ) -> None:
        limits = _limits(config)
        object.__setattr__(self, "action_name", identifier(action_name, "payload.action_name"))
        object.__setattr__(self, "phase", identifier(phase, "payload.phase"))
        object.__setattr__(self, "result", optional_identifier(result, "payload.result"))
        object.__setattr__(self, "attributes", _attributes(attributes, "payload.attributes", limits))

    def to_dict(self) -> dict[str, object]:
        return {
            "action_name": self.action_name,
            "phase": self.phase,
            "result": self.result,
            "attributes": self.attributes,
        }


@dataclass(frozen=True, init=False)
class ToolResultPayload:
    tool_name: str
    tool_call_ref: str
    status: str
    result_ref: str
    result_summary: str

    def __init__(
        self,
        tool_name: object,
        tool_call_ref: object,
        status: object,
        result_ref: object,
        result_summary: object,
        *,
        config: IngressConfig | None = None,
    ) -> None:
        limits = _limits(config)
        object.__setattr__(self, "tool_name", identifier(tool_name, "payload.tool_name"))
        object.__setattr__(self, "tool_call_ref", identifier(tool_call_ref, "payload.tool_call_ref"))
        object.__setattr__(self, "status", identifier(status, "payload.status"))
        object.__setattr__(
            self, "result_ref", external_reference(result_ref, "payload.result_ref", maximum=limits.max_reference_chars)
        )
        object.__setattr__(
            self,
            "result_summary",
            bounded_text(result_summary, "payload.result_summary", maximum=limits.max_text_chars),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "tool_name": self.tool_name,
            "tool_call_ref": self.tool_call_ref,
            "status": self.status,
            "result_ref": self.result_ref,
            "result_summary": self.result_summary,
        }


@dataclass(frozen=True, init=False)
class SensorFactPayload:
    metric_name: str
    value: Any
    unit: str | None
    aggregation: str | None
    attributes: Mapping[str, Any]

    def __init__(
        self,
        metric_name: object,
        value: object,
        unit: object,
        aggregation: object,
        attributes: object,
        *,
        config: IngressConfig | None = None,
    ) -> None:
        limits = _limits(config)
        object.__setattr__(self, "metric_name", identifier(metric_name, "payload.metric_name"))
        object.__setattr__(self, "value", _value(value, "payload.value", limits))
        object.__setattr__(self, "unit", optional_identifier(unit, "payload.unit"))
        object.__setattr__(self, "aggregation", optional_identifier(aggregation, "payload.aggregation"))
        object.__setattr__(self, "attributes", _attributes(attributes, "payload.attributes", limits))

    def to_dict(self) -> dict[str, object]:
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "unit": self.unit,
            "aggregation": self.aggregation,
            "attributes": self.attributes,
        }


@dataclass(frozen=True, init=False)
class DeviceStatePayload:
    device_ref: str
    state_name: str
    value: Any

    def __init__(
        self, device_ref: object, state_name: object, value: object, *, config: IngressConfig | None = None
    ) -> None:
        limits = _limits(config)
        object.__setattr__(self, "device_ref", identifier(device_ref, "payload.device_ref"))
        object.__setattr__(self, "state_name", identifier(state_name, "payload.state_name"))
        object.__setattr__(self, "value", _value(value, "payload.value", limits))

    def to_dict(self) -> dict[str, object]:
        return {"device_ref": self.device_ref, "state_name": self.state_name, "value": self.value}


@dataclass(frozen=True, init=False)
class EnvironmentChangePayload:
    predicate: str
    before: Any
    after: Any
    attributes: Mapping[str, Any]

    def __init__(
        self,
        predicate: object,
        before: object,
        after: object,
        attributes: object,
        *,
        config: IngressConfig | None = None,
    ) -> None:
        limits = _limits(config)
        object.__setattr__(self, "predicate", identifier(predicate, "payload.predicate"))
        object.__setattr__(self, "before", _value(before, "payload.before", limits))
        object.__setattr__(self, "after", _value(after, "payload.after", limits))
        object.__setattr__(self, "attributes", _attributes(attributes, "payload.attributes", limits))

    def to_dict(self) -> dict[str, object]:
        return {"predicate": self.predicate, "before": self.before, "after": self.after, "attributes": self.attributes}


@dataclass(frozen=True, init=False)
class CoverageIntervalPayload:
    modality: SemanticModality
    coverage_status: CoverageStatus
    reason: str | None
    coverage_scope_ref: str | None

    def __init__(
        self,
        modality: object,
        coverage_status: CoverageStatus | str,
        reason: object,
        coverage_scope_ref: object = None,
        *,
        config: IngressConfig | None = None,
    ) -> None:
        limits = _limits(config)
        from behavior.ingress.model import SemanticModality

        object.__setattr__(self, "modality", SemanticModality(modality))
        object.__setattr__(
            self,
            "coverage_scope_ref",
            optional_identifier(coverage_scope_ref, "payload.coverage_scope_ref"),
        )
        object.__setattr__(self, "coverage_status", CoverageStatus(coverage_status))
        object.__setattr__(
            self, "reason", optional_bounded_text(reason, "payload.reason", maximum=limits.max_text_chars)
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "modality": self.modality.value,
            "coverage_scope_ref": self.coverage_scope_ref,
            "coverage_status": self.coverage_status.value,
            "reason": self.reason,
        }


@dataclass(frozen=True, init=False)
class FreeTextSemanticPayload:
    text: str
    language: str
    labels: tuple[str, ...]

    def __init__(self, text: object, language: object, labels: object, *, config: IngressConfig | None = None) -> None:
        limits = _limits(config)
        object.__setattr__(self, "text", bounded_text(text, "payload.text", maximum=limits.max_text_chars))
        object.__setattr__(self, "language", identifier(language, "payload.language", maximum=64))
        object.__setattr__(
            self, "labels", identifier_tuple(labels, "payload.labels", maximum_items=limits.max_payload_items)
        )

    def to_dict(self) -> dict[str, object]:
        return {"text": self.text, "language": self.language, "labels": self.labels}


SemanticPayload: TypeAlias = (
    ActivitySegmentPayload
    | UtteranceSegmentPayload
    | StateAssertionPayload
    | StateTransitionPayload
    | InteractionSegmentPayload
    | ActionEventPayload
    | ToolResultPayload
    | SensorFactPayload
    | DeviceStatePayload
    | EnvironmentChangePayload
    | CoverageIntervalPayload
    | FreeTextSemanticPayload
)


def _object(value: object, name: str, fields: frozenset[str]) -> dict[str, Any]:
    data = strict_fields(value, name, fields)
    require_fields(data, name, fields)
    return data


def payload_from_dict(record_kind: object, value: object, *, config: IngressConfig | None = None) -> SemanticPayload:
    """按记录种类严格构造唯一允许的 Payload 类型。"""

    from behavior.ingress.model import SemanticRecordKind

    limits = _limits(config)
    kind = SemanticRecordKind(record_kind)
    try:
        if kind is SemanticRecordKind.OWNER_ACTIVITY_SEGMENT:
            data = _object(value, "payload", frozenset({"activity", "phase_hint", "attributes"}))
            return ActivitySegmentPayload(**data, config=limits)
        if kind is SemanticRecordKind.OWNER_UTTERANCE_SEGMENT:
            data = _object(value, "payload", frozenset({"text", "language", "channel"}))
            return UtteranceSegmentPayload(**data, config=limits)
        if kind is SemanticRecordKind.OWNER_STATE_ASSERTION:
            data = _object(value, "payload", frozenset({"state_name", "value"}))
            return StateAssertionPayload(**data, config=limits)
        if kind is SemanticRecordKind.OWNER_STATE_TRANSITION:
            data = _object(value, "payload", frozenset({"state_name", "before", "after"}))
            return StateTransitionPayload(**data, config=limits)
        if kind is SemanticRecordKind.OWNER_INTERACTION_SEGMENT:
            data = _object(
                value, "payload", frozenset({"interaction_type", "counterparty_role", "phase_hint", "attributes"})
            )
            return InteractionSegmentPayload(**data, config=limits)
        if kind in {SemanticRecordKind.ROBOT_ACTION_EVENT, SemanticRecordKind.AGENT_ACTION_EVENT}:
            data = _object(value, "payload", frozenset({"action_name", "phase", "result", "attributes"}))
            return ActionEventPayload(**data, config=limits)
        if kind is SemanticRecordKind.TOOL_RESULT_EVENT:
            data = _object(
                value, "payload", frozenset({"tool_name", "tool_call_ref", "status", "result_ref", "result_summary"})
            )
            return ToolResultPayload(**data, config=limits)
        if kind in {SemanticRecordKind.OWNER_SENSOR_FACT, SemanticRecordKind.ENVIRONMENT_SENSOR_FACT}:
            data = _object(value, "payload", frozenset({"metric_name", "value", "unit", "aggregation", "attributes"}))
            return SensorFactPayload(**data, config=limits)
        if kind is SemanticRecordKind.DEVICE_STATE:
            data = _object(value, "payload", frozenset({"device_ref", "state_name", "value"}))
            return DeviceStatePayload(**data, config=limits)
        if kind is SemanticRecordKind.ENVIRONMENT_CHANGE:
            data = _object(value, "payload", frozenset({"predicate", "before", "after", "attributes"}))
            return EnvironmentChangePayload(**data, config=limits)
        if kind is SemanticRecordKind.COVERAGE_INTERVAL:
            data = _object(
                value,
                "payload",
                frozenset({"modality", "coverage_scope_ref", "coverage_status", "reason"}),
            )
            return CoverageIntervalPayload(**data, config=limits)
        if kind is SemanticRecordKind.FREE_TEXT_SEMANTIC:
            data = _object(value, "payload", frozenset({"text", "language", "labels"}))
            return FreeTextSemanticPayload(**data, config=limits)
    except (TypeError, ValueError) as exc:
        raise SemanticRecordError(str(exc)) from exc
    raise SemanticRecordError("record kind has no registered Payload contract")


def validate_payload(record_kind: object, payload: object, *, config: IngressConfig | None = None) -> SemanticPayload:
    if not isinstance(
        payload,
        ActivitySegmentPayload
        | UtteranceSegmentPayload
        | StateAssertionPayload
        | StateTransitionPayload
        | InteractionSegmentPayload
        | ActionEventPayload
        | ToolResultPayload
        | SensorFactPayload
        | DeviceStatePayload
        | EnvironmentChangePayload
        | CoverageIntervalPayload
        | FreeTextSemanticPayload,
    ):
        raise SemanticRecordError("payload must be a supported immutable semantic Payload")
    return payload_from_dict(record_kind, payload.to_dict(), config=config)


def _identifier_schema(*, nullable: bool = False) -> dict[str, object]:
    value: dict[str, object] = {
        "type": "string",
        "minLength": 1,
        "maxLength": 256,
        "pattern": r"^[A-Za-z0-9][A-Za-z0-9._:@+\-]{0,255}$",
    }
    return {"anyOf": [value, {"type": "null"}]} if nullable else value


def _strict_schema(properties: Mapping[str, object]) -> dict[str, object]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(properties),
        "properties": dict(properties),
    }


def payload_json_schema(
    record_kind: object,
    *,
    config: IngressConfig | None = None,
) -> dict[str, object]:
    from behavior.ingress.model import SemanticRecordKind

    limits = _limits(config)
    kind = SemanticRecordKind(record_kind)
    json_value: dict[str, object] = {}
    attributes = {"type": "object", "maxProperties": limits.max_payload_items}
    nullable_identifier = _identifier_schema(nullable=True)
    if kind is SemanticRecordKind.OWNER_ACTIVITY_SEGMENT:
        return _strict_schema(
            {
                "activity": _identifier_schema(),
                "phase_hint": {"type": "string", "enum": [item.value for item in PhaseHint]},
                "attributes": attributes,
            }
        )
    if kind is SemanticRecordKind.OWNER_UTTERANCE_SEGMENT:
        return _strict_schema(
            {
                "text": {"type": "string", "minLength": 1, "maxLength": limits.max_text_chars},
                "language": _identifier_schema(),
                "channel": {"type": "string", "enum": [item.value for item in UtteranceChannel]},
            }
        )
    if kind is SemanticRecordKind.OWNER_STATE_ASSERTION:
        return _strict_schema({"state_name": _identifier_schema(), "value": json_value})
    if kind is SemanticRecordKind.OWNER_STATE_TRANSITION:
        return _strict_schema({"state_name": _identifier_schema(), "before": json_value, "after": json_value})
    if kind is SemanticRecordKind.OWNER_INTERACTION_SEGMENT:
        return _strict_schema(
            {
                "interaction_type": _identifier_schema(),
                "counterparty_role": {"type": "string", "enum": [item.value for item in InteractionCounterpartyRole]},
                "phase_hint": {"type": "string", "enum": [item.value for item in PhaseHint]},
                "attributes": attributes,
            }
        )
    if kind in {SemanticRecordKind.ROBOT_ACTION_EVENT, SemanticRecordKind.AGENT_ACTION_EVENT}:
        return _strict_schema(
            {
                "action_name": _identifier_schema(),
                "phase": _identifier_schema(),
                "result": nullable_identifier,
                "attributes": attributes,
            }
        )
    if kind is SemanticRecordKind.TOOL_RESULT_EVENT:
        return _strict_schema(
            {
                "tool_name": _identifier_schema(),
                "tool_call_ref": _identifier_schema(),
                "status": _identifier_schema(),
                "result_ref": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": limits.max_reference_chars,
                },
                "result_summary": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": limits.max_text_chars,
                },
            }
        )
    if kind in {SemanticRecordKind.OWNER_SENSOR_FACT, SemanticRecordKind.ENVIRONMENT_SENSOR_FACT}:
        return _strict_schema(
            {
                "metric_name": _identifier_schema(),
                "value": json_value,
                "unit": nullable_identifier,
                "aggregation": nullable_identifier,
                "attributes": attributes,
            }
        )
    if kind is SemanticRecordKind.DEVICE_STATE:
        return _strict_schema(
            {"device_ref": _identifier_schema(), "state_name": _identifier_schema(), "value": json_value}
        )
    if kind is SemanticRecordKind.ENVIRONMENT_CHANGE:
        return _strict_schema(
            {"predicate": _identifier_schema(), "before": json_value, "after": json_value, "attributes": attributes}
        )
    if kind is SemanticRecordKind.COVERAGE_INTERVAL:
        from behavior.ingress.model import SemanticModality

        return _strict_schema(
            {
                "modality": {"type": "string", "enum": [item.value for item in SemanticModality]},
                "coverage_scope_ref": nullable_identifier,
                "coverage_status": {"type": "string", "enum": [item.value for item in CoverageStatus]},
                "reason": {
                    "anyOf": [
                        {"type": "string", "minLength": 1, "maxLength": limits.max_text_chars},
                        {"type": "null"},
                    ]
                },
            }
        )
    if kind is SemanticRecordKind.FREE_TEXT_SEMANTIC:
        return _strict_schema(
            {
                "text": {"type": "string", "minLength": 1, "maxLength": limits.max_text_chars},
                "language": _identifier_schema(),
                "labels": {
                    "type": "array",
                    "maxItems": limits.max_payload_items,
                    "uniqueItems": True,
                    "items": _identifier_schema(),
                },
            }
        )
    raise SemanticRecordError("record kind has no JSON Schema Payload contract")


__all__ = [
    "ActionEventPayload",
    "ActivitySegmentPayload",
    "CoverageIntervalPayload",
    "CoverageStatus",
    "DeviceStatePayload",
    "EnvironmentChangePayload",
    "FreeTextSemanticPayload",
    "InteractionCounterpartyRole",
    "InteractionSegmentPayload",
    "PhaseHint",
    "SemanticPayload",
    "SensorFactPayload",
    "StateAssertionPayload",
    "StateTransitionPayload",
    "ToolResultPayload",
    "UtteranceChannel",
    "UtteranceSegmentPayload",
    "payload_from_dict",
    "payload_json_schema",
    "validate_payload",
]
