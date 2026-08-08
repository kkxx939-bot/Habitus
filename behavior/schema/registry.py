"""加载声明式 Schema，并严格校验行为语义 L2 业务字段。"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping, Sequence
from datetime import date, datetime
from importlib import resources
from typing import Any

import yaml

from behavior.model import BehaviorAddress, BehaviorKind, behavior_local_timestamp, semantic_name
from behavior.schema.model import (
    BehaviorFieldRole,
    BehaviorFieldSchema,
    BehaviorFieldType,
    BehaviorMergeStrategy,
    BehaviorOperationMode,
    BehaviorSchemaError,
    BehaviorSchemaMaterialization,
    BehaviorTypeSchema,
)
from behavior.uri import BehaviorURI
from foundation.integrity import canonical_json, canonicalize

_SCHEMA_FILES = {
    BehaviorKind.EVENT: "events.yaml",
    BehaviorKind.OUTCOME: "outcomes.yaml",
    BehaviorKind.EPISODE: "episodes.yaml",
}
_TYPE_KEYS = {"behavior_type", "description", "path_template", "operation_mode", "fields"}
_FIELD_KEYS = {"name", "type", "role", "required", "merge", "description"}
_RECORD_ID = re.compile(r"^[a-z][a-z0-9_-]{0,127}$")
_EVENT_STATUSES = frozenset({"completed", "failed", "cancelled", "interrupted", "partial"})
_ACTION_STATUSES = frozenset({"observed", "completed", "failed", "cancelled", "interrupted"})
_KNOWLEDGE_STATES = frozenset({"observed", "reported", "inferred", "corrected"})
_OUTCOME_TYPES = frozenset(
    {
        "state_change",
        "task_result",
        "human_response",
        "human_feedback",
        "correction",
        "delayed_effect",
        "safety_result",
    }
)
_OUTCOME_TARGET_TYPES = frozenset({"event", "action"})
_OUTCOME_VALENCES = frozenset({"positive", "neutral", "negative", "mixed", "not_applicable"})
_EPISODE_STATUSES = frozenset({"completed", "failed", "cancelled", "partial"})
_PHASE_STATUSES = frozenset({"completed", "failed", "cancelled", "interrupted", "partial"})
_UNPHASED_EVENT_ROLES = frozenset({"contextual", "parallel", "interruption", "uncertain"})
_TRANSITION_TYPES = frozenset({"precedes", "enables", "causes", "interrupts", "resumes", "completes", "fails"})
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _behavior_kind(value: BehaviorKind | str) -> BehaviorKind:
    try:
        return BehaviorKind(value)
    except (TypeError, ValueError) as exc:
        raise BehaviorSchemaError(f"unknown behavior type: {value}") from exc


def _document_uris(
    values: Sequence[str],
    expected_kind: BehaviorKind,
    label: str,
) -> tuple[BehaviorURI, ...]:
    uris: list[BehaviorURI] = []
    for value in values:
        try:
            uri = BehaviorURI.parse(value)
            if uri.to_address().kind is not expected_kind:
                raise BehaviorSchemaError(f"{label} URI identifies the wrong document type")
        except ValueError as exc:
            if isinstance(exc, BehaviorSchemaError):
                raise
            raise BehaviorSchemaError(
                f"{label} document URI must identify an L2 {expected_kind.value} document"
            ) from exc
        uris.append(uri)
    return tuple(uris)


class BehaviorSchemaRegistry:
    """三类 L2 文档共用的唯一 Schema 注册表和规范渲染入口。"""

    def __init__(self, schemas: tuple[BehaviorTypeSchema, ...]) -> None:
        by_kind: dict[BehaviorKind, BehaviorTypeSchema] = {}
        for schema in schemas:
            if not isinstance(schema, BehaviorTypeSchema):
                raise TypeError("schemas must contain BehaviorTypeSchema values")
            if schema.kind in by_kind:
                raise BehaviorSchemaError(f"duplicate behavior schema: {schema.kind.value}")
            by_kind[schema.kind] = schema
        if set(by_kind) != set(BehaviorKind):
            missing = sorted(kind.value for kind in set(BehaviorKind) - set(by_kind))
            raise BehaviorSchemaError(f"behavior schema registry is incomplete: {missing}")
        self._schemas = tuple(by_kind[kind] for kind in BehaviorKind)
        self._by_kind = by_kind

    @classmethod
    def load_default(cls) -> BehaviorSchemaRegistry:
        definitions = resources.files("behavior.schema.definitions")
        return cls(
            tuple(
                _load_schema(definitions.joinpath(filename).read_text(encoding="utf-8"), filename)
                for _kind, filename in _SCHEMA_FILES.items()
            )
        )

    def all(self) -> tuple[BehaviorTypeSchema, ...]:
        return self._schemas

    def get(self, kind: BehaviorKind | str) -> BehaviorTypeSchema:
        return self._by_kind[_behavior_kind(kind)]

    def validate(self, kind: BehaviorKind | str, payload: Mapping[str, Any]) -> dict[str, Any]:
        normalized_kind = _behavior_kind(kind)
        schema = self.get(normalized_kind)
        source = _strict_mapping(payload, f"{normalized_kind.value} payload")
        unknown = set(source) - set(schema.field_map)
        if unknown:
            raise BehaviorSchemaError(f"behavior payload contains unknown fields: {sorted(unknown)}")
        normalized: dict[str, Any] = {}
        for field in schema.fields:
            if field.name not in source:
                if field.required:
                    raise BehaviorSchemaError(f"behavior payload is missing required field: {field.name}")
                continue
            normalized[field.name] = self._field_value(field, source[field.name])
        self._validate_kind(normalized_kind, normalized)
        return normalized

    def materialize(
        self,
        kind: BehaviorKind | str,
        payload: Mapping[str, Any],
    ) -> BehaviorSchemaMaterialization:
        """一次校验后生成规范地址、JSON-safe 字段和人类可读正文。"""

        normalized_kind = _behavior_kind(kind)
        normalized = self.validate(normalized_kind, payload)
        renderer = {
            BehaviorKind.EVENT: _render_event,
            BehaviorKind.OUTCOME: _render_outcome,
            BehaviorKind.EPISODE: _render_episode,
        }[normalized_kind]
        return BehaviorSchemaMaterialization(
            address=self._address_for_validated(normalized_kind, normalized),
            storage_fields=_storage_fields(normalized),
            markdown_body=renderer(normalized),
        )

    def address_for(self, kind: BehaviorKind | str, payload: Mapping[str, Any]) -> BehaviorAddress:
        normalized_kind = _behavior_kind(kind)
        normalized = self.validate(normalized_kind, payload)
        return self._address_for_validated(normalized_kind, normalized)

    @staticmethod
    def _address_for_validated(kind: BehaviorKind, normalized: Mapping[str, Any]) -> BehaviorAddress:
        """只接受本注册表已经规范化完成的领域字段。"""

        if kind is BehaviorKind.EVENT:
            return BehaviorAddress.event(
                normalized["event_date"],
                normalized["event_name"],
                normalized["started_at"],
            )
        if kind is BehaviorKind.OUTCOME:
            event_address = BehaviorURI.parse(normalized["event_uri"]).to_address()
            return BehaviorAddress.outcome(
                event_address.occurred_on,
                event_address.name,
                event_address.started_at,
            )
        return BehaviorAddress.episode(
            normalized["episode_date"],
            normalized["episode_name"],
            normalized["started_at"],
        )

    def render_markdown(self, kind: BehaviorKind | str, payload: Mapping[str, Any]) -> str:
        normalized_kind = _behavior_kind(kind)
        normalized = self.validate(normalized_kind, payload)
        renderer = {
            BehaviorKind.EVENT: _render_event,
            BehaviorKind.OUTCOME: _render_outcome,
            BehaviorKind.EPISODE: _render_episode,
        }[normalized_kind]
        return renderer(normalized)

    @staticmethod
    def _field_value(field: BehaviorFieldSchema, value: Any) -> Any:
        validators: dict[BehaviorFieldType, Callable[[Any, str], Any]] = {
            BehaviorFieldType.STRING: _text,
            BehaviorFieldType.OPTIONAL_STRING: _optional_text,
            BehaviorFieldType.DATE: _date_value,
            BehaviorFieldType.DATETIME: _datetime_value,
            BehaviorFieldType.OPTIONAL_DATETIME: _optional_datetime,
            BehaviorFieldType.NUMBER: _confidence,
            BehaviorFieldType.STRING_LIST: _string_tuple,
            BehaviorFieldType.ACTION_LIST: _actions,
            BehaviorFieldType.OUTCOME_LIST: _outcomes,
            BehaviorFieldType.URI_LIST: _uri_tuple,
            BehaviorFieldType.PHASE_LIST: _phases,
            BehaviorFieldType.UNPHASED_EVENT_LIST: _unphased_events,
            BehaviorFieldType.OUTCOME_SNAPSHOT_LIST: _outcome_snapshots,
            BehaviorFieldType.TRANSITION_LIST: _transitions,
        }
        try:
            return validators[field.field_type](value, field.name)
        except (TypeError, ValueError) as exc:
            if isinstance(exc, BehaviorSchemaError):
                raise
            raise BehaviorSchemaError(f"behavior field {field.name} is invalid") from exc

    @staticmethod
    def _validate_kind(kind: BehaviorKind, payload: dict[str, Any]) -> None:
        if kind is BehaviorKind.EVENT:
            _validate_event(payload)
        elif kind is BehaviorKind.OUTCOME:
            _validate_outcome(payload)
        else:
            _validate_episode(payload)


def _validate_event(payload: dict[str, Any]) -> None:
    semantic_name(payload["event_name"], "event name")
    if payload["event_date"] != payload["started_at"].date():
        raise BehaviorSchemaError("event_date must match the local started_at date")
    if payload["ended_at"] is not None and payload["ended_at"] < payload["started_at"]:
        raise BehaviorSchemaError("event ended_at cannot precede started_at")
    if payload["status"] not in _EVENT_STATUSES:
        raise BehaviorSchemaError(f"event status must be one of {sorted(_EVENT_STATUSES)}")
    if not payload["actions"]:
        raise BehaviorSchemaError("event must contain at least one observed action")
    sequences = tuple(action["sequence"] for action in payload["actions"])
    if sequences != tuple(range(1, len(sequences) + 1)):
        raise BehaviorSchemaError("event action sequences must be contiguous and start at one")
    action_ids = tuple(action["action_id"] for action in payload["actions"])
    if len(action_ids) != len(set(action_ids)):
        raise BehaviorSchemaError("event action IDs must be unique")
    event_start = payload["started_at"]
    event_end = payload["ended_at"]
    for action in payload["actions"]:
        for timestamp in (action["started_at"], action["ended_at"]):
            if timestamp is None:
                continue
            if timestamp < event_start or (event_end is not None and timestamp > event_end):
                raise BehaviorSchemaError("Action time must remain inside its Event time window")
    for index, earlier in enumerate(payload["actions"]):
        earlier_start = earlier["started_at"]
        if earlier_start is None:
            continue
        for later in payload["actions"][index + 1 :]:
            later_end = later["ended_at"]
            if later_end is not None and later_end <= earlier_start:
                raise BehaviorSchemaError("Action order contradicts known non-overlapping timestamps")


def _validate_outcome(payload: dict[str, Any]) -> None:
    semantic_name(payload["event_name"], "event name")
    try:
        actual_event = BehaviorURI.parse(payload["event_uri"])
        actual_address = actual_event.to_address()
        if actual_address.kind is not BehaviorKind.EVENT:
            raise BehaviorSchemaError("outcome event_uri must identify an Event document")
    except ValueError as exc:
        raise BehaviorSchemaError("outcome event_uri is not a valid Event URI") from exc
    expected_event = BehaviorURI.from_address(
        BehaviorAddress.event(
            payload["event_date"],
            payload["event_name"],
            actual_address.started_at,
        )
    )
    if actual_event != expected_event:
        raise BehaviorSchemaError("outcome address must mirror its target Event URI")
    payload["event_uri"] = str(actual_event)
    if not payload["outcomes"]:
        raise BehaviorSchemaError("outcome document must contain at least one result")
    identifiers = tuple(item["outcome_id"] for item in payload["outcomes"])
    if len(identifiers) != len(set(identifiers)):
        raise BehaviorSchemaError("outcome IDs must be unique")
    payload["outcomes"] = tuple(
        sorted(
            payload["outcomes"],
            key=lambda item: (item["occurred_at"], item["outcome_id"]),
        )
    )


def _validate_episode(payload: dict[str, Any]) -> None:
    semantic_name(payload["episode_name"], "episode name")
    if payload["episode_date"] != payload["started_at"].date():
        raise BehaviorSchemaError("episode_date must match the local started_at date")
    if payload["ended_at"] < payload["started_at"]:
        raise BehaviorSchemaError("episode ended_at cannot precede started_at")
    if payload["status"] not in _EPISODE_STATUSES:
        raise BehaviorSchemaError(f"episode status must be one of {sorted(_EPISODE_STATUSES)}")
    events = _document_uris(payload["ordered_event_uris"], BehaviorKind.EVENT, "Episode Event")
    if len(events) < 2:
        raise BehaviorSchemaError("episode must reference at least two Events")
    outcomes = _document_uris(payload["outcome_uris"], BehaviorKind.OUTCOME, "Episode Outcome")
    snapshot_outcomes = _document_uris(
        tuple(snapshot["uri"] for snapshot in payload["outcome_snapshots"]),
        BehaviorKind.OUTCOME,
        "Episode Outcome snapshot",
    )
    if tuple(str(uri) for uri in snapshot_outcomes) != tuple(str(uri) for uri in outcomes):
        raise BehaviorSchemaError("episode outcome snapshots must match outcome_uris in order")

    event_positions = {str(uri): index for index, uri in enumerate(events)}
    if not payload["phases"]:
        raise BehaviorSchemaError("episode must contain at least one semantic Phase")
    phase_sequences = tuple(phase["sequence"] for phase in payload["phases"])
    if phase_sequences != tuple(range(1, len(phase_sequences) + 1)):
        raise BehaviorSchemaError("episode phase sequences must be contiguous and start at one")
    phase_ids = tuple(phase["phase_id"] for phase in payload["phases"])
    if len(phase_ids) != len(set(phase_ids)):
        raise BehaviorSchemaError("episode phase IDs must be unique")
    assigned_events: set[str] = set()
    previous_phase_position = -1
    for phase in payload["phases"]:
        phase_events = phase["event_uris"]
        if any(uri not in event_positions for uri in phase_events):
            raise BehaviorSchemaError("episode phase references an Event outside ordered_event_uris")
        if any(uri in assigned_events for uri in phase_events):
            raise BehaviorSchemaError("one Event cannot belong to multiple Episode phases")
        positions = tuple(event_positions[uri] for uri in phase_events)
        if positions != tuple(sorted(positions)):
            raise BehaviorSchemaError("episode phase Event order must follow ordered_event_uris")
        if positions and positions[0] <= previous_phase_position:
            raise BehaviorSchemaError("Episode Phase order must follow ordered_event_uris without crossing")
        if positions:
            previous_phase_position = positions[-1]
        assigned_events.update(phase_events)

    unphased_events = tuple(item["event_uri"] for item in payload["unphased_events"])
    _document_uris(unphased_events, BehaviorKind.EVENT, "Episode unphased Event")
    unphased_positions = tuple(event_positions.get(uri, -1) for uri in unphased_events)
    if any(position < 0 for position in unphased_positions):
        raise BehaviorSchemaError("episode unphased Event lies outside ordered_event_uris")
    if unphased_positions != tuple(sorted(unphased_positions)):
        raise BehaviorSchemaError("episode unphased Events must follow ordered_event_uris")
    unphased_set = set(unphased_events)
    if assigned_events & unphased_set:
        raise BehaviorSchemaError("one Event cannot be both phased and unphased")
    if assigned_events | unphased_set != set(event_positions):
        raise BehaviorSchemaError("every Episode Event must be classified as phased or unphased")

    identities: set[tuple[str, str, str]] = set()
    for transition in payload["transitions"]:
        source = transition["from_event_uri"]
        target = transition["to_event_uri"]
        if source not in event_positions or target not in event_positions:
            raise BehaviorSchemaError("episode transition references an Event outside ordered_event_uris")
        if event_positions[source] >= event_positions[target]:
            raise BehaviorSchemaError("episode transition must follow the real Event order")
        identity = (source, target, transition["relation"])
        if identity in identities:
            raise BehaviorSchemaError("episode transitions must be unique")
        identities.add(identity)


def _load_schema(source: str, filename: str) -> BehaviorTypeSchema:
    try:
        raw = yaml.safe_load(source)
    except yaml.YAMLError as exc:
        raise BehaviorSchemaError(f"invalid YAML in {filename}") from exc
    payload = _strict_mapping(raw, f"schema {filename}")
    _require_keys(payload, _TYPE_KEYS, f"schema {filename}")
    raw_fields = payload["fields"]
    if not isinstance(raw_fields, list):
        raise BehaviorSchemaError(f"schema {filename} fields must be a list")
    fields: list[BehaviorFieldSchema] = []
    for raw_field in raw_fields:
        field = _strict_mapping(raw_field, f"field in {filename}")
        _require_keys(field, _FIELD_KEYS, f"field in {filename}")
        fields.append(
            BehaviorFieldSchema(
                name=_text(field["name"], "field name"),
                field_type=BehaviorFieldType(_text(field["type"], "field type")),
                role=BehaviorFieldRole(_text(field["role"], "field role")),
                required=field["required"],
                merge_strategy=BehaviorMergeStrategy(_text(field["merge"], "field merge")),
                description=_text(field["description"], "field description"),
            )
        )
    return BehaviorTypeSchema(
        kind=BehaviorKind(_text(payload["behavior_type"], "behavior type")),
        description=_text(payload["description"], "behavior description"),
        path_template=_text(payload["path_template"], "behavior path template"),
        operation_mode=BehaviorOperationMode(_text(payload["operation_mode"], "behavior operation mode")),
        fields=tuple(fields),
    )


def _strict_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise BehaviorSchemaError(f"{label} must be an object with string keys")
    return dict(value)


def _require_keys(payload: Mapping[str, Any], expected: set[str], label: str) -> None:
    unknown = set(payload) - expected
    missing = expected - set(payload)
    if unknown:
        raise BehaviorSchemaError(f"{label} contains unsupported keys: {sorted(unknown)}")
    if missing:
        raise BehaviorSchemaError(f"{label} is missing keys: {sorted(missing)}")


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BehaviorSchemaError(f"{label} must be non-empty text")
    normalized = value.strip()
    if any(not character.isprintable() and character not in "\n\t" for character in normalized):
        raise BehaviorSchemaError(f"{label} contains control characters")
    return normalized


def _optional_text(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label)


def _date_value(value: Any, label: str) -> date:
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


def _datetime_value(value: Any, label: str) -> datetime:
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


def _optional_datetime(value: Any, label: str) -> datetime | None:
    if value is None:
        return None
    return _datetime_value(value, label)


def _confidence(value: Any, label: str) -> float:
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


def _string_tuple(value: Any, label: str) -> tuple[str, ...]:
    values = tuple(_text(item, f"{label} item") for item in _sequence(value, label))
    if len(values) != len(set(values)):
        raise BehaviorSchemaError(f"{label} must not contain duplicates")
    return values


def _record_id(value: Any, label: str) -> str:
    resolved = _text(value, label)
    if not _RECORD_ID.fullmatch(resolved):
        raise BehaviorSchemaError(f"{label} must be a stable lowercase record identifier")
    return resolved


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BehaviorSchemaError(f"{label} must be a positive integer")
    return value


def _enum(value: Any, allowed: frozenset[str], label: str) -> str:
    resolved = _text(value, label)
    if resolved not in allowed:
        raise BehaviorSchemaError(f"{label} must be one of {sorted(allowed)}")
    return resolved


def _actions(value: Any, label: str) -> tuple[dict[str, Any], ...]:
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
        "status",
        "knowledge_state",
        "evidence_refs",
    }
    actions: list[dict[str, Any]] = []
    for index, item in enumerate(_sequence(value, label), start=1):
        payload = _strict_mapping(item, f"{label}[{index}]")
        _require_keys(payload, expected, f"{label}[{index}]")
        parameters = _strict_mapping(payload["parameters"], f"{label}[{index}].parameters")
        normalized_parameters = canonicalize(parameters)
        try:
            canonical_json(normalized_parameters).encode("utf-8")
        except UnicodeEncodeError as exc:
            raise BehaviorSchemaError("action parameters must be canonical UTF-8 JSON") from exc
        started_at = _optional_datetime(payload["started_at"], "action started_at")
        ended_at = _optional_datetime(payload["ended_at"], "action ended_at")
        if started_at is not None and ended_at is not None and ended_at < started_at:
            raise BehaviorSchemaError("action ended_at cannot precede started_at")
        actions.append(
            {
                "action_id": _record_id(payload["action_id"], "action_id"),
                "sequence": _positive_integer(payload["sequence"], "action sequence"),
                "actor": _text(payload["actor"], "action actor"),
                "action_type": _text(payload["action_type"], "action type"),
                "semantics": _text(payload["semantics"], "action semantics"),
                "target_refs": _string_tuple(payload["target_refs"], "action target_refs"),
                "method": _optional_text(payload["method"], "action method"),
                "parameters": normalized_parameters,
                "started_at": started_at,
                "ended_at": ended_at,
                "status": _enum(payload["status"], _ACTION_STATUSES, "action status"),
                "knowledge_state": _enum(payload["knowledge_state"], _KNOWLEDGE_STATES, "action knowledge_state"),
                "evidence_refs": _string_tuple(payload["evidence_refs"], "action evidence_refs"),
            }
        )
    return tuple(actions)


def _outcomes(value: Any, label: str) -> tuple[dict[str, Any], ...]:
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
    outcomes: list[dict[str, Any]] = []
    for index, item in enumerate(_sequence(value, label), start=1):
        payload = _strict_mapping(item, f"{label}[{index}]")
        _require_keys(payload, expected, f"{label}[{index}]")
        target_type = _enum(payload["target_type"], _OUTCOME_TARGET_TYPES, "outcome target_type")
        target_action_id = (
            None
            if payload["target_action_id"] is None
            else _record_id(payload["target_action_id"], "outcome target_action_id")
        )
        if (target_type == "action") != (target_action_id is not None):
            raise BehaviorSchemaError("action Outcome requires target_action_id; Event Outcome forbids it")
        outcomes.append(
            {
                "outcome_id": _record_id(payload["outcome_id"], "outcome_id"),
                "occurred_at": _datetime_value(payload["occurred_at"], "outcome occurred_at"),
                "outcome_type": _enum(payload["outcome_type"], _OUTCOME_TYPES, "outcome type"),
                "target_type": target_type,
                "target_action_id": target_action_id,
                "semantics": _text(payload["semantics"], "outcome semantics"),
                "valence": _enum(payload["valence"], _OUTCOME_VALENCES, "outcome valence"),
                "knowledge_state": _enum(payload["knowledge_state"], _KNOWLEDGE_STATES, "outcome knowledge_state"),
                "confidence": _confidence(payload["confidence"], "outcome confidence"),
                "evidence_refs": _string_tuple(payload["evidence_refs"], "outcome evidence_refs"),
            }
        )
    return tuple(outcomes)


def _uri_tuple(value: Any, label: str) -> tuple[str, ...]:
    uris = tuple(str(BehaviorURI.parse(_text(item, f"{label} item"))) for item in _sequence(value, label))
    if len(uris) != len(set(uris)):
        raise BehaviorSchemaError(f"{label} must not contain duplicate URIs")
    return uris


def _phases(value: Any, label: str) -> tuple[dict[str, Any], ...]:
    expected = {"phase_id", "sequence", "semantics", "status", "event_uris", "confidence"}
    phases: list[dict[str, Any]] = []
    for index, item in enumerate(_sequence(value, label), start=1):
        payload = _strict_mapping(item, f"{label}[{index}]")
        _require_keys(payload, expected, f"{label}[{index}]")
        event_uris = _uri_tuple(payload["event_uris"], "phase event_uris")
        if not event_uris:
            raise BehaviorSchemaError("episode Phase must contain at least one Event")
        phases.append(
            {
                "phase_id": _record_id(payload["phase_id"], "phase_id"),
                "sequence": _positive_integer(payload["sequence"], "phase sequence"),
                "semantics": _text(payload["semantics"], "phase semantics"),
                "status": _enum(payload["status"], _PHASE_STATUSES, "phase status"),
                "event_uris": event_uris,
                "confidence": _confidence(payload["confidence"], "phase confidence"),
            }
        )
    return tuple(phases)


def _unphased_events(value: Any, label: str) -> tuple[dict[str, str], ...]:
    expected = {"event_uri", "role", "reason"}
    events: list[dict[str, str]] = []
    for index, item in enumerate(_sequence(value, label), start=1):
        payload = _strict_mapping(item, f"{label}[{index}]")
        _require_keys(payload, expected, f"{label}[{index}]")
        events.append(
            {
                "event_uri": str(BehaviorURI.parse(_text(payload["event_uri"], "unphased event_uri"))),
                "role": _enum(payload["role"], _UNPHASED_EVENT_ROLES, "unphased Event role"),
                "reason": _text(payload["reason"], "unphased Event reason"),
            }
        )
    uris = tuple(item["event_uri"] for item in events)
    if len(uris) != len(set(uris)):
        raise BehaviorSchemaError("episode unphased Events must not contain duplicates")
    return tuple(events)


def _outcome_snapshots(value: Any, label: str) -> tuple[dict[str, Any], ...]:
    expected = {"uri", "revision", "digest"}
    snapshots: list[dict[str, Any]] = []
    for index, item in enumerate(_sequence(value, label), start=1):
        payload = _strict_mapping(item, f"{label}[{index}]")
        _require_keys(payload, expected, f"{label}[{index}]")
        revision = _positive_integer(payload["revision"], "Outcome snapshot revision")
        digest = _text(payload["digest"], "Outcome snapshot digest")
        if not _SHA256.fullmatch(digest):
            raise BehaviorSchemaError("Outcome snapshot digest must be a lowercase SHA-256 hex digest")
        snapshots.append(
            {
                "uri": str(BehaviorURI.parse(_text(payload["uri"], "Outcome snapshot URI"))),
                "revision": revision,
                "digest": digest,
            }
        )
    uris = tuple(snapshot["uri"] for snapshot in snapshots)
    if len(uris) != len(set(uris)):
        raise BehaviorSchemaError("Outcome snapshots must not contain duplicate URIs")
    return tuple(snapshots)


def _transitions(value: Any, label: str) -> tuple[dict[str, str], ...]:
    expected = {"from_event_uri", "to_event_uri", "relation"}
    transitions: list[dict[str, str]] = []
    for index, item in enumerate(_sequence(value, label), start=1):
        payload = _strict_mapping(item, f"{label}[{index}]")
        _require_keys(payload, expected, f"{label}[{index}]")
        source = str(BehaviorURI.parse(_text(payload["from_event_uri"], "transition from_event_uri")))
        target = str(BehaviorURI.parse(_text(payload["to_event_uri"], "transition to_event_uri")))
        if source == target:
            raise BehaviorSchemaError("episode transition cannot reference one Event twice")
        transitions.append(
            {
                "from_event_uri": source,
                "to_event_uri": target,
                "relation": _enum(payload["relation"], _TRANSITION_TYPES, "transition relation"),
            }
        )
    return tuple(transitions)


def _render_event(payload: Mapping[str, Any]) -> str:
    lines = [
        f"# {payload['event_name']}",
        "",
        f"**Date:** {payload['event_date'].isoformat()}",
        f"**Time:** {_time_range(payload['started_at'], payload['ended_at'])}",
        f"**Status:** {payload['status']}",
        "",
        "## Summary",
        str(payload["semantic_summary"]),
    ]
    _append_bullets(lines, "Facts", payload["semantic_facts"])
    _append_optional(lines, "Trigger", payload["trigger"])
    _append_optional(lines, "Goal", payload["goal"])
    _append_bullets(lines, "Constraints", payload["constraints"])
    _append_bullets(lines, "Participants", payload["participants"])
    lines.extend(["", "## Actions"])
    for action in payload["actions"]:
        lines.append(
            f"{action['sequence']}. **{action['actor']} · {action['action_type']}** — "
            f"{action['semantics']} ({action['status']})"
        )
    _append_optional(lines, "Closure", payload["closure_reason"])
    _append_bullets(lines, "Conflicts", payload["conflicts"])
    return "\n".join(lines).rstrip() + "\n"


def _render_outcome(payload: Mapping[str, Any]) -> str:
    lines = [
        f"# {payload['event_name']} — Outcomes",
        "",
        f"**Event:** {payload['event_uri']}",
        "",
        "## Results",
    ]
    for outcome in payload["outcomes"]:
        target = outcome["target_type"]
        if outcome["target_action_id"] is not None:
            target = f"{target}:{outcome['target_action_id']}"
        lines.append(
            f"- **{outcome['occurred_at'].isoformat()} · {outcome['outcome_type']} · {target}** — "
            f"{outcome['semantics']} ({outcome['valence']})"
        )
    return "\n".join(lines).rstrip() + "\n"


def _render_episode(payload: Mapping[str, Any]) -> str:
    lines = [
        f"# {payload['episode_name']}",
        "",
        f"**Time:** {_time_range(payload['started_at'], payload['ended_at'])}",
        f"**Status:** {payload['status']}",
        "",
        "## Summary",
        str(payload["semantic_summary"]),
    ]
    _append_optional(lines, "Goal", payload["goal"])
    _append_bullets(lines, "Participants", payload["participants"])
    lines.extend(["", "## Event Timeline"])
    lines.extend(f"{index}. {uri}" for index, uri in enumerate(payload["ordered_event_uris"], start=1))
    if payload["phases"]:
        lines.extend(["", "## Phases"])
        for phase in payload["phases"]:
            lines.append(
                f"{phase['sequence']}. **{phase['semantics']}** "
                f"({phase['status']}, confidence={phase['confidence']:.3f})"
            )
            lines.extend(f"   - {uri}" for uri in phase["event_uris"])
    if payload["unphased_events"]:
        lines.extend(["", "## Unphased Events"])
        for event in payload["unphased_events"]:
            lines.append(f"- {event['event_uri']} [{event['role']}] — {event['reason']}")
    if payload["transitions"]:
        lines.extend(["", "## Transitions"])
        for transition in payload["transitions"]:
            lines.append(f"- {transition['from_event_uri']} --{transition['relation']}--> {transition['to_event_uri']}")
    _append_bullets(lines, "Key Turning Points", payload["key_turning_points"])
    _append_bullets(lines, "Outcomes", payload["outcome_uris"])
    _append_optional(lines, "Closure", payload["closure_reason"])
    return "\n".join(lines).rstrip() + "\n"


def _append_optional(lines: list[str], title: str, value: Any) -> None:
    if value is not None:
        lines.extend(["", f"## {title}", str(value)])


def _append_bullets(lines: list[str], title: str, values: Sequence[Any]) -> None:
    if values:
        lines.extend(["", f"## {title}"])
        lines.extend(f"- {value}" for value in values)


def _time_range(started_at: datetime, ended_at: datetime | None) -> str:
    end = "unknown" if ended_at is None else ended_at.isoformat()
    return f"{started_at.isoformat()} — {end}"


def _storage_fields(fields: Mapping[str, Any]) -> dict[str, Any]:
    """把强类型领域字段转成保留行为本地 offset 的规范 JSON 值。"""

    stored = _storage_value(fields)
    if not isinstance(stored, dict):
        raise BehaviorSchemaError("behavior storage fields must be a JSON object")
    try:
        canonical_json(stored).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise BehaviorSchemaError("behavior storage fields must be canonical UTF-8 JSON") from exc
    return stored


def _storage_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat(timespec="microseconds")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {key: _storage_value(item) for key, item in sorted(value.items())}
    if isinstance(value, tuple | list):
        return [_storage_value(item) for item in value]
    return canonicalize(value)


__all__ = ["BehaviorSchemaRegistry"]
