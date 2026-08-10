"""Prediction Pattern Graph 的严格 Schema 和 Markdown 物化。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from importlib import resources
from typing import Any

import yaml

from foundation.integrity import canonical_digest, canonical_json, canonicalize
from prediction.model import PredictionPatternAddress, PredictionPatternKind, prediction_sample_id
from prediction.uri import PredictionURI, PredictionURINodeType

_LEVELS = frozenset({"action", "event", "phase"})
_TARGETS = frozenset({"action", "event", "phase", "termination"})
_OPERATORS = frozenset({"eq", "neq", "gt", "gte", "lt", "lte", "contains", "exists"})
_DEFINITIONS = {
    PredictionPatternKind.STATE: ("states.yaml", "patterns/states/{shard}/{state_key}.md"),
    PredictionPatternKind.BRANCH: ("branches.yaml", "patterns/branches/{shard}/{state_key}/{branch_key}.md"),
}


class PredictionPatternSchemaError(ValueError):
    """模式图节点不满足声明式合同。"""


class PredictionPatternSchemaRegistry:
    def __init__(self, definitions: Mapping[PredictionPatternKind, Mapping[str, Any]]) -> None:
        if set(definitions) != set(PredictionPatternKind):
            raise PredictionPatternSchemaError("prediction pattern schema registry is incomplete")
        self._definitions = dict(definitions)

    @classmethod
    def load_default(cls) -> PredictionPatternSchemaRegistry:
        root = resources.files("prediction.pattern.definitions")
        loaded: dict[PredictionPatternKind, Mapping[str, Any]] = {}
        for kind, (filename, path_template) in _DEFINITIONS.items():
            value = yaml.safe_load(root.joinpath(filename).read_text(encoding="utf-8"))
            if not isinstance(value, dict) or set(value) != {
                "pattern_type",
                "description",
                "path_template",
                "operation_mode",
                "fields",
            }:
                raise PredictionPatternSchemaError(f"{filename} has an invalid top-level shape")
            if value["pattern_type"] != kind.value or value["path_template"] != path_template:
                raise PredictionPatternSchemaError(f"{filename} does not match the confirmed pattern tree")
            if value["operation_mode"] != "add_only" or not isinstance(value["fields"], list):
                raise PredictionPatternSchemaError(f"{filename} has an invalid operation contract")
            loaded[kind] = value
        return cls(loaded)

    def validate(self, kind: PredictionPatternKind | str, payload: Mapping[str, Any]) -> dict[str, Any]:
        normalized_kind = PredictionPatternKind(kind)
        definition = self._definitions[normalized_kind]
        expected = set(definition["fields"])
        source = _exact_mapping(payload, expected, f"{normalized_kind.value} pattern")
        if normalized_kind is PredictionPatternKind.STATE:
            normalized = self._state(source)
        else:
            normalized = self._branch(source)
        try:
            canonical_json(normalized).encode("utf-8")
        except (TypeError, ValueError, UnicodeEncodeError) as exc:
            raise PredictionPatternSchemaError("prediction pattern must be canonical UTF-8 JSON") from exc
        return normalized

    def materialize(
        self,
        kind: PredictionPatternKind | str,
        payload: Mapping[str, Any],
    ) -> tuple[dict[str, Any], PredictionPatternAddress, str]:
        normalized_kind = PredictionPatternKind(kind)
        normalized = self.validate(normalized_kind, payload)
        address = PredictionPatternAddress(
            normalized_kind,
            normalized["state_key"],
            normalized.get("branch_key"),
        )
        storage = canonicalize(normalized)
        assert isinstance(storage, dict)
        return storage, address, _render(normalized_kind, normalized)

    @staticmethod
    def _state(source: Mapping[str, Any]) -> dict[str, Any]:
        identity = _identity(source["identity_material"], "state identity")
        logical_state_key = _digest(source["logical_state_key"], "logical_state_key")
        pattern_generation = _digest(source["pattern_generation"], "pattern_generation")
        state_key = _digest(source["state_key"], "state_key")
        expected_logical_key = canonical_digest(
            {"contract": "prediction-logical-state-pattern-v1", "identity": identity}
        )
        if logical_state_key != expected_logical_key:
            raise PredictionPatternSchemaError("StatePattern logical_state_key does not match identity_material")
        expected_key = canonical_digest(
            {
                "contract": "prediction-state-pattern-materialization-v1",
                "logical_state_key": logical_state_key,
                "pattern_generation": pattern_generation,
            }
        )
        if state_key != expected_key:
            raise PredictionPatternSchemaError("StatePattern state_key does not match identity_material")
        return {
            "state_key": state_key,
            "logical_state_key": logical_state_key,
            "pattern_generation": pattern_generation,
            "identity_material": identity,
            "semantic_summary": _text(source["semantic_summary"], "semantic_summary"),
            "state_level": _enum(source["state_level"], _LEVELS, "state_level"),
            "predicates": _records(source["predicates"], _predicate, "predicates"),
            "active_goals": _strings(source["active_goals"], "active_goals"),
            "constraints": _strings(source["constraints"], "constraints"),
            "support_count": _positive_int(source["support_count"], "support_count"),
            "sample_refs": _sample_refs(source["sample_refs"]),
            "statistics": _statistics(source["statistics"]),
            "projection_version": _text(source["projection_version"], "projection_version"),
        }

    @staticmethod
    def _branch(source: Mapping[str, Any]) -> dict[str, Any]:
        state_key = _digest(source["state_key"], "state_key")
        identity = _identity(source["identity_material"], "branch identity")
        logical_branch_key = _digest(source["logical_branch_key"], "logical_branch_key")
        pattern_generation = _digest(source["pattern_generation"], "pattern_generation")
        branch_key = _digest(source["branch_key"], "branch_key")
        logical_state_key = identity.get("logical_state_key")
        expected_state_key = canonical_digest(
            {
                "contract": "prediction-state-pattern-materialization-v1",
                "logical_state_key": logical_state_key,
                "pattern_generation": pattern_generation,
            }
        )
        if state_key != expected_state_key:
            raise PredictionPatternSchemaError(
                "BranchPattern state_key does not match its logical StatePattern generation"
            )
        expected_logical_key = canonical_digest(
            {
                "contract": "prediction-logical-branch-pattern-v1",
                "logical_state_key": logical_state_key,
                "branch_identity": identity.get("branch_identity"),
            }
        )
        if logical_branch_key != expected_logical_key:
            raise PredictionPatternSchemaError("BranchPattern logical_branch_key does not match identity_material")
        expected_key = canonical_digest(
            {
                "contract": "prediction-branch-pattern-materialization-v1",
                "logical_branch_key": logical_branch_key,
                "pattern_generation": pattern_generation,
            }
        )
        if branch_key != expected_key:
            raise PredictionPatternSchemaError("BranchPattern branch_key does not match identity_material")
        outcomes = _records(source["outcomes"], _outcome, "outcomes")
        if sum(outcome["probability"] for outcome in outcomes) > 1.000000001:
            raise PredictionPatternSchemaError("BranchPattern outcome probabilities cannot exceed one")
        return {
            "state_key": state_key,
            "branch_key": branch_key,
            "logical_branch_key": logical_branch_key,
            "pattern_generation": pattern_generation,
            "identity_material": identity,
            "target_kind": _enum(source["target_kind"], _TARGETS, "target_kind"),
            "behavior": _behavior(source["behavior"]),
            "conditions": _records(source["conditions"], _predicate, "conditions"),
            "support_count": _positive_int(source["support_count"], "support_count"),
            "conditional_probability": _probability(source["conditional_probability"], "conditional_probability"),
            "confidence": _probability(source["confidence"], "confidence"),
            "sample_refs": _sample_refs(source["sample_refs"]),
            "outcomes": outcomes,
            "projection_version": _text(source["projection_version"], "projection_version"),
        }


def _predicate(value: Any, label: str) -> dict[str, Any]:
    item = _exact_mapping(value, {"field", "operator", "value"}, label)
    normalized_value = canonicalize(item["value"])
    return {
        "field": _text(item["field"], f"{label}.field"),
        "operator": _enum(item["operator"], _OPERATORS, f"{label}.operator"),
        "value": normalized_value,
    }


def _behavior(value: Any) -> dict[str, Any]:
    item = _exact_mapping(
        value,
        {"actor_role", "behavior_type", "semantics", "target_refs", "parameters"},
        "branch behavior",
    )
    parameters = canonicalize(_mapping(item["parameters"], "behavior parameters"))
    return {
        "actor_role": _text(item["actor_role"], "behavior actor_role"),
        "behavior_type": _text(item["behavior_type"], "behavior behavior_type"),
        "semantics": _text(item["semantics"], "behavior semantics"),
        "target_refs": _strings(item["target_refs"], "behavior target_refs"),
        "parameters": parameters,
    }


def _outcome(value: Any, label: str) -> dict[str, Any]:
    item = _exact_mapping(
        value,
        {"semantics", "probability", "average_delay_seconds", "valence"},
        label,
    )
    delay = item["average_delay_seconds"]
    if delay is not None:
        delay = _finite(delay, f"{label}.average_delay_seconds")
        if delay < 0:
            raise PredictionPatternSchemaError(f"{label}.average_delay_seconds cannot be negative")
    return {
        "semantics": _text(item["semantics"], f"{label}.semantics"),
        "probability": _probability(item["probability"], f"{label}.probability"),
        "average_delay_seconds": delay,
        "valence": _text(item["valence"], f"{label}.valence"),
    }


def _statistics(value: Any) -> dict[str, Any]:
    item = _exact_mapping(value, {"first_observed_at", "last_observed_at", "source_count", "confidence"}, "statistics")
    first = _datetime(item["first_observed_at"], "statistics.first_observed_at")
    last = _datetime(item["last_observed_at"], "statistics.last_observed_at")
    if last < first:
        raise PredictionPatternSchemaError("pattern statistics cannot move backwards")
    return {
        "first_observed_at": first,
        "last_observed_at": last,
        "source_count": _positive_int(item["source_count"], "statistics.source_count"),
        "confidence": _probability(item["confidence"], "statistics.confidence"),
    }


def _sample_refs(value: Any) -> tuple[str, ...]:
    refs = _strings(value, "sample_refs")
    for ref in refs:
        try:
            uri = PredictionURI.parse(ref)
        except ValueError as exc:
            raise PredictionPatternSchemaError("sample_refs must contain canonical Prediction sample URIs") from exc
        if uri.node_type is not PredictionURINodeType.DOCUMENT or uri.is_pattern_document:
            raise PredictionPatternSchemaError("sample_refs must identify PredictionSample documents")
        if str(uri) != ref:
            raise PredictionPatternSchemaError("sample_refs must use canonical Prediction URIs")
    return refs


def _identity(value: Any, label: str) -> dict[str, Any]:
    result = canonicalize(_mapping(value, label))
    if not result:
        raise PredictionPatternSchemaError(f"{label} must not be empty")
    assert isinstance(result, dict)
    return result


def _records(value: Any, validator: Any, label: str) -> tuple[dict[str, Any], ...]:
    return tuple(validator(item, f"{label}[{index}]") for index, item in enumerate(_sequence(value, label)))


def _strings(value: Any, label: str) -> tuple[str, ...]:
    result = tuple(_text(item, f"{label} item") for item in _sequence(value, label))
    if len(result) != len(set(result)):
        raise PredictionPatternSchemaError(f"{label} must not contain duplicates")
    return result


def _exact_mapping(value: Any, expected: set[str], label: str) -> dict[str, Any]:
    result = _mapping(value, label)
    if set(result) != expected:
        raise PredictionPatternSchemaError(f"{label} must contain exactly {sorted(expected)}")
    return result


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise PredictionPatternSchemaError(f"{label} must be an object with string keys")
    return dict(value)


def _sequence(value: Any, label: str) -> tuple[Any, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PredictionPatternSchemaError(f"{label} must be an array")
    return tuple(value)


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PredictionPatternSchemaError(f"{label} must be non-empty text")
    normalized = value.strip()
    try:
        normalized.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise PredictionPatternSchemaError(f"{label} must be valid UTF-8") from exc
    return normalized


def _enum(value: Any, allowed: frozenset[str], label: str) -> str:
    normalized = _text(value, label)
    if normalized not in allowed:
        raise PredictionPatternSchemaError(f"{label} must be one of {sorted(allowed)}")
    return normalized


def _digest(value: Any, label: str) -> str:
    try:
        return prediction_sample_id(value)
    except ValueError as exc:
        raise PredictionPatternSchemaError(f"{label} must be a lowercase SHA-256 digest") from exc


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PredictionPatternSchemaError(f"{label} must be a positive integer")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PredictionPatternSchemaError(f"{label} must be numeric")
    try:
        normalized = float(value)
    except (OverflowError, ValueError) as exc:
        raise PredictionPatternSchemaError(f"{label} must be finite") from exc
    if not math.isfinite(normalized):
        raise PredictionPatternSchemaError(f"{label} must be finite")
    return normalized


def _probability(value: Any, label: str) -> float:
    normalized = _finite(value, label)
    if not 0 <= normalized <= 1:
        raise PredictionPatternSchemaError(f"{label} must be between zero and one")
    return normalized


def _datetime(value: Any, label: str) -> datetime:
    parsed = value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise PredictionPatternSchemaError(f"{label} must be an ISO timestamp") from exc
    if not isinstance(parsed, datetime) or parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PredictionPatternSchemaError(f"{label} must be timezone-aware")
    return parsed


def _render(kind: PredictionPatternKind, fields: Mapping[str, Any]) -> str:
    if kind is PredictionPatternKind.STATE:
        lines = [
            "# State Pattern",
            "",
            fields["semantic_summary"],
            "",
            f"- State key: {fields['state_key']}",
            f"- Level: {fields['state_level']}",
            f"- Support: {fields['support_count']}",
            f"- Projection: {fields['projection_version']}",
        ]
    else:
        behavior = fields["behavior"]
        lines = [
            "# Branch Pattern",
            "",
            behavior["semantics"],
            "",
            f"- From state: {fields['state_key']}",
            f"- Branch key: {fields['branch_key']}",
            f"- Target: {fields['target_kind']}",
            f"- Probability: {fields['conditional_probability']:.6f}",
            f"- Support: {fields['support_count']}",
        ]
    return "\n".join(lines) + "\n"


__all__ = ["PredictionPatternSchemaError", "PredictionPatternSchemaRegistry"]
