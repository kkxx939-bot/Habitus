"""按声明的字段类型把原始载荷规范化为强类型领域值。"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from typing import Any

from behavior.model import behavior_local_timestamp
from behavior.schema.model import BehaviorFieldSchema, BehaviorFieldType, BehaviorSchemaError
from behavior.schema.vocabulary import (
    ACTION_STATUSES,
    KNOWLEDGE_STATES,
    OUTCOME_TARGET_TYPES,
    OUTCOME_TYPES,
    OUTCOME_VALENCES,
    PHASE_STATUSES,
    RECORD_ID,
    SHA256,
    TRANSITION_TYPES,
    UNPHASED_EVENT_ROLES,
)
from behavior.uri import BehaviorURI
from foundation.integrity import canonical_json, canonicalize


def strict_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise BehaviorSchemaError(f"{label} must be an object with string keys")
    return dict(value)


def require_keys(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    unknown = set(payload) - expected
    missing = expected - set(payload)
    if unknown:
        raise BehaviorSchemaError(f"{label} contains unsupported keys: {sorted(unknown)}")
    if missing:
        raise BehaviorSchemaError(f"{label} is missing keys: {sorted(missing)}")


def text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BehaviorSchemaError(f"{label} must be non-empty text")
    normalized = value.strip()
    if any(not character.isprintable() and character not in "\n\t" for character in normalized):
        raise BehaviorSchemaError(f"{label} contains control characters")
    return normalized


def optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return text(value, label)


def date_value(value: Any, label: str) -> date:
    if isinstance(value, datetime):
        raise BehaviorSchemaError(f"{label} must be a date without a time")
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        raise BehaviorSchemaError(f"{label} must be an ISO date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise BehaviorSchemaError(f"{label} must be an ISO date") from exc


def datetime_value(value: Any, label: str) -> datetime:
    parsed = value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise BehaviorSchemaError(f"{label} must be an ISO timestamp") from exc
    try:
        return behavior_local_timestamp(parsed, label)
    except (TypeError, ValueError) as exc:
        raise BehaviorSchemaError(f"{label} must include a whole-minute UTC offset") from exc


def optional_datetime(value: Any, label: str) -> datetime | None:
    if value is None:
        return None
    return datetime_value(value, label)


def confidence(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BehaviorSchemaError(f"{label} must be numeric")
    normalized = float(value)
    if not 0.0 <= normalized <= 1.0:
        raise BehaviorSchemaError(f"{label} must be between zero and one")
    return normalized


def _sequence(value: Any, label: str) -> tuple[Any, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise BehaviorSchemaError(f"{label} must be an array")
    return tuple(value)


def string_tuple(value: Any, label: str) -> tuple[str, ...]:
    values = tuple(text(item, f"{label} item") for item in _sequence(value, label))
    if len(values) != len(set(values)):
        raise BehaviorSchemaError(f"{label} must not contain duplicates")
    return values


def record_id(value: Any, label: str) -> str:
    resolved = text(value, label)
    if not RECORD_ID.fullmatch(resolved):
        raise BehaviorSchemaError(f"{label} must be a stable lowercase record identifier")
    return resolved


def positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BehaviorSchemaError(f"{label} must be a positive integer")
    return value


def enum(value: Any, allowed: frozenset[str], label: str) -> str:
    resolved = text(value, label)
    if resolved not in allowed:
        raise BehaviorSchemaError(f"{label} must be one of {sorted(allowed)}")
    return resolved


def uri_tuple(value: Any, label: str) -> tuple[str, ...]:
    uris = tuple(str(BehaviorURI.parse(text(item, f"{label} item"))) for item in _sequence(value, label))
    if len(uris) != len(set(uris)):
        raise BehaviorSchemaError(f"{label} must not contain duplicate URIs")
    return uris


def actions(value: Any, label: str) -> tuple[dict[str, Any], ...]:
    expected = {
        "action_id",
        "sequence",
        "actor",
        "action_type",
        "semantics",
        "target_refs",
        "method",
        "parameters",
        "started_at",
        "ended_at",
        "available_at",
        "status",
        "knowledge_state",
        "evidence_refs",
    }
    resolved: list[dict[str, Any]] = []
    for index, item in enumerate(_sequence(value, label), start=1):
        payload = strict_mapping(item, f"{label}[{index}]")
        require_keys(payload, expected, f"{label}[{index}]")
        parameters = strict_mapping(payload["parameters"], f"{label}[{index}].parameters")
        normalized_parameters = canonicalize(parameters)
        try:
            canonical_json(normalized_parameters).encode("utf-8")
        except UnicodeEncodeError as exc:
            raise BehaviorSchemaError("action parameters must be canonical UTF-8 JSON") from exc
        started_at = optional_datetime(payload["started_at"], "action started_at")
        ended_at = optional_datetime(payload["ended_at"], "action ended_at")
        available_at = datetime_value(payload["available_at"], "action available_at")
        if started_at is not None and ended_at is not None and ended_at < started_at:
            raise BehaviorSchemaError("action ended_at cannot precede started_at")
        if started_at is not None and available_at < started_at:
            raise BehaviorSchemaError("action available_at cannot precede started_at")
        resolved.append(
            {
                "action_id": record_id(payload["action_id"], "action_id"),
                "sequence": positive_integer(payload["sequence"], "action sequence"),
                "actor": text(payload["actor"], "action actor"),
                "action_type": text(payload["action_type"], "action type"),
                "semantics": text(payload["semantics"], "action semantics"),
                "target_refs": string_tuple(payload["target_refs"], "action target_refs"),
                "method": optional_text(payload["method"], "action method"),
                "parameters": normalized_parameters,
                "started_at": started_at,
                "ended_at": ended_at,
                "available_at": available_at,
                "status": enum(payload["status"], ACTION_STATUSES, "action status"),
                "knowledge_state": enum(payload["knowledge_state"], KNOWLEDGE_STATES, "action knowledge_state"),
                "evidence_refs": string_tuple(payload["evidence_refs"], "action evidence_refs"),
            }
        )
    return tuple(resolved)


def outcomes(value: Any, label: str) -> tuple[dict[str, Any], ...]:
    expected = {
        "outcome_id",
        "occurred_at",
        "outcome_type",
        "target_type",
        "target_action_id",
        "semantics",
        "valence",
        "knowledge_state",
        "confidence",
        "evidence_refs",
    }
    resolved: list[dict[str, Any]] = []
    for index, item in enumerate(_sequence(value, label), start=1):
        payload = strict_mapping(item, f"{label}[{index}]")
        require_keys(payload, expected, f"{label}[{index}]")
        target_type = enum(payload["target_type"], OUTCOME_TARGET_TYPES, "outcome target_type")
        target_action_id = (
            None
            if payload["target_action_id"] is None
            else record_id(payload["target_action_id"], "outcome target_action_id")
        )
        if (target_type == "action") != (target_action_id is not None):
            raise BehaviorSchemaError("action Outcome requires target_action_id; Event Outcome forbids it")
        resolved.append(
            {
                "outcome_id": record_id(payload["outcome_id"], "outcome_id"),
                "occurred_at": datetime_value(payload["occurred_at"], "outcome occurred_at"),
                "outcome_type": enum(payload["outcome_type"], OUTCOME_TYPES, "outcome type"),
                "target_type": target_type,
                "target_action_id": target_action_id,
                "semantics": text(payload["semantics"], "outcome semantics"),
                "valence": enum(payload["valence"], OUTCOME_VALENCES, "outcome valence"),
                "knowledge_state": enum(payload["knowledge_state"], KNOWLEDGE_STATES, "outcome knowledge_state"),
                "confidence": confidence(payload["confidence"], "outcome confidence"),
                "evidence_refs": string_tuple(payload["evidence_refs"], "outcome evidence_refs"),
            }
        )
    return tuple(resolved)


def phases(value: Any, label: str) -> tuple[dict[str, Any], ...]:
    expected = {
        "phase_id",
        "sequence",
        "semantics",
        "status",
        "event_uris",
        "started_at",
        "ended_at",
        "confidence",
    }
    resolved: list[dict[str, Any]] = []
    for index, item in enumerate(_sequence(value, label), start=1):
        payload = strict_mapping(item, f"{label}[{index}]")
        require_keys(payload, expected, f"{label}[{index}]")
        event_uris = uri_tuple(payload["event_uris"], "phase event_uris")
        if not event_uris:
            raise BehaviorSchemaError("episode Phase must contain at least one Event")
        resolved.append(
            {
                "phase_id": record_id(payload["phase_id"], "phase_id"),
                "sequence": positive_integer(payload["sequence"], "phase sequence"),
                "semantics": text(payload["semantics"], "phase semantics"),
                "status": enum(payload["status"], PHASE_STATUSES, "phase status"),
                "event_uris": event_uris,
                "started_at": datetime_value(payload["started_at"], "phase started_at"),
                "ended_at": optional_datetime(payload["ended_at"], "phase ended_at"),
                "confidence": confidence(payload["confidence"], "phase confidence"),
            }
        )
    return tuple(resolved)


def unphased_events(value: Any, label: str) -> tuple[dict[str, str], ...]:
    expected = {"event_uri", "role", "reason"}
    resolved: list[dict[str, str]] = []
    for index, item in enumerate(_sequence(value, label), start=1):
        payload = strict_mapping(item, f"{label}[{index}]")
        require_keys(payload, expected, f"{label}[{index}]")
        resolved.append(
            {
                "event_uri": str(BehaviorURI.parse(text(payload["event_uri"], "unphased event_uri"))),
                "role": enum(payload["role"], UNPHASED_EVENT_ROLES, "unphased Event role"),
                "reason": text(payload["reason"], "unphased Event reason"),
            }
        )
    uris = tuple(item["event_uri"] for item in resolved)
    if len(uris) != len(set(uris)):
        raise BehaviorSchemaError("episode unphased Events must not contain duplicates")
    return tuple(resolved)


def outcome_snapshots(value: Any, label: str) -> tuple[dict[str, Any], ...]:
    expected = {"uri", "revision", "digest"}
    resolved: list[dict[str, Any]] = []
    for index, item in enumerate(_sequence(value, label), start=1):
        payload = strict_mapping(item, f"{label}[{index}]")
        require_keys(payload, expected, f"{label}[{index}]")
        revision = positive_integer(payload["revision"], "Outcome snapshot revision")
        digest = text(payload["digest"], "Outcome snapshot digest")
        if not SHA256.fullmatch(digest):
            raise BehaviorSchemaError("Outcome snapshot digest must be a lowercase SHA-256 hex digest")
        resolved.append(
            {
                "uri": str(BehaviorURI.parse(text(payload["uri"], "Outcome snapshot URI"))),
                "revision": revision,
                "digest": digest,
            }
        )
    uris = tuple(snapshot["uri"] for snapshot in resolved)
    if len(uris) != len(set(uris)):
        raise BehaviorSchemaError("Outcome snapshots must not contain duplicate URIs")
    return tuple(resolved)


def transitions(value: Any, label: str) -> tuple[dict[str, str], ...]:
    expected = {"from_event_uri", "to_event_uri", "relation"}
    resolved: list[dict[str, str]] = []
    for index, item in enumerate(_sequence(value, label), start=1):
        payload = strict_mapping(item, f"{label}[{index}]")
        require_keys(payload, expected, f"{label}[{index}]")
        source = str(BehaviorURI.parse(text(payload["from_event_uri"], "transition from_event_uri")))
        target = str(BehaviorURI.parse(text(payload["to_event_uri"], "transition to_event_uri")))
        if source == target:
            raise BehaviorSchemaError("episode transition cannot reference one Event twice")
        resolved.append(
            {
                "from_event_uri": source,
                "to_event_uri": target,
                "relation": enum(payload["relation"], TRANSITION_TYPES, "transition relation"),
            }
        )
    return tuple(resolved)


_VALIDATORS: dict[BehaviorFieldType, Callable[[Any, str], Any]] = {
    BehaviorFieldType.STRING: text,
    BehaviorFieldType.OPTIONAL_STRING: optional_text,
    BehaviorFieldType.DATE: date_value,
    BehaviorFieldType.DATETIME: datetime_value,
    BehaviorFieldType.OPTIONAL_DATETIME: optional_datetime,
    BehaviorFieldType.NUMBER: confidence,
    BehaviorFieldType.STRING_LIST: string_tuple,
    BehaviorFieldType.ACTION_LIST: actions,
    BehaviorFieldType.OUTCOME_LIST: outcomes,
    BehaviorFieldType.URI_LIST: uri_tuple,
    BehaviorFieldType.PHASE_LIST: phases,
    BehaviorFieldType.UNPHASED_EVENT_LIST: unphased_events,
    BehaviorFieldType.OUTCOME_SNAPSHOT_LIST: outcome_snapshots,
    BehaviorFieldType.TRANSITION_LIST: transitions,
}


def validate_field(field: BehaviorFieldSchema, value: Any) -> Any:
    """按声明类型规范化单个字段；非 Schema 异常统一收敛为 BehaviorSchemaError。"""

    try:
        return _VALIDATORS[field.field_type](value, field.name)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, BehaviorSchemaError):
            raise
        raise BehaviorSchemaError(f"behavior field {field.name} is invalid") from exc


__all__ = [
    "require_keys",
    "strict_mapping",
    "text",
    "validate_field",
]
