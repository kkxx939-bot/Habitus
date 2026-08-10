"""Prediction 上下文中 scope、anchor 与领域组合的共享合同。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from prediction.model import (
    PredictionAnchorType,
    PredictionDecisionBasis,
    PredictionKind,
    PredictionMode,
    PredictionTargetLevel,
    PredictionTimePrecision,
)

_SCOPE_FIELDS = frozenset(
    {"participants", "target_level", "target_domain", "prediction_mode"}
)
_ANCHOR_FIELDS = frozenset(
    {
        "anchor_type",
        "container_ref",
        "prefix_length",
        "previous_step_ref",
        "decision_basis",
        "cutoff_at",
        "precision",
        "lower_bound_at",
        "upper_bound_at",
    }
)
_HISTORY_COLLECTIONS = (
    "active_behaviors",
    "parallel_behaviors",
    "interruptions",
    "resumptions",
    "completed_events",
    "completed_actions",
    "completed_phases",
)


class PredictionContextContractError(ValueError):
    """线上上下文的结构或跨字段语义不闭合。"""


def normalize_prediction_scope(value: object) -> dict[str, Any]:
    """严格规范化模型可见的预测范围，不接受缺失或扩展字段。"""

    payload = _exact_mapping(value, _SCOPE_FIELDS, "prediction scope")
    participants = tuple(
        _text(item, "prediction scope participant")
        for item in _sequence(payload["participants"], "prediction scope participants")
    )
    if len(participants) != len(set(participants)):
        raise PredictionContextContractError(
            "prediction scope participants must not contain duplicates"
        )
    return {
        "participants": participants,
        "target_level": _enum_value(
            payload["target_level"],
            PredictionTargetLevel,
            "prediction scope target_level",
        ),
        "target_domain": _optional_text(
            payload["target_domain"], "prediction scope target_domain"
        ),
        "prediction_mode": _enum_value(
            payload["prediction_mode"],
            PredictionMode,
            "prediction scope prediction_mode",
        ),
    }


def normalize_prediction_anchor(value: object) -> dict[str, Any]:
    """严格规范化预测切点及其时间精度。"""

    payload = _exact_mapping(value, _ANCHOR_FIELDS, "prediction anchor")
    precision = _enum_value(
        payload["precision"], PredictionTimePrecision, "prediction anchor precision"
    )
    cutoff_at = _optional_datetime(payload["cutoff_at"], "prediction anchor cutoff_at")
    lower_bound = _optional_datetime(
        payload["lower_bound_at"], "prediction anchor lower_bound_at"
    )
    upper_bound = _optional_datetime(
        payload["upper_bound_at"], "prediction anchor upper_bound_at"
    )
    if precision == PredictionTimePrecision.EXACT.value:
        if cutoff_at is None or lower_bound is not None or upper_bound is not None:
            raise PredictionContextContractError(
                "exact prediction anchor requires only cutoff_at"
            )
    elif precision == PredictionTimePrecision.BOUNDED.value:
        if cutoff_at is not None or (lower_bound is None and upper_bound is None):
            raise PredictionContextContractError(
                "bounded prediction anchor requires at least one bound and no cutoff_at"
            )
        if lower_bound is not None and upper_bound is not None and upper_bound < lower_bound:
            raise PredictionContextContractError(
                "prediction anchor upper bound cannot precede lower bound"
            )
    elif cutoff_at is not None or lower_bound is not None or upper_bound is not None:
        raise PredictionContextContractError(
            "order-only prediction anchor cannot contain timestamps"
        )
    return {
        "anchor_type": _enum_value(
            payload["anchor_type"], PredictionAnchorType, "prediction anchor anchor_type"
        ),
        "container_ref": _text(payload["container_ref"], "prediction anchor container_ref"),
        "prefix_length": _non_negative_integer(
            payload["prefix_length"], "prediction anchor prefix_length"
        ),
        "previous_step_ref": _optional_text(
            payload["previous_step_ref"], "prediction anchor previous_step_ref"
        ),
        "decision_basis": _enum_value(
            payload["decision_basis"],
            PredictionDecisionBasis,
            "prediction anchor decision_basis",
        ),
        "cutoff_at": cutoff_at,
        "precision": precision,
        "lower_bound_at": lower_bound,
        "upper_bound_at": upper_bound,
    }


def validate_prediction_context_contract(
    kind: PredictionKind | str,
    scope: Mapping[str, Any],
    anchor: Mapping[str, Any],
    prediction_input: Mapping[str, Any],
    *,
    treatment: Mapping[str, Any] | None = None,
) -> None:
    """校验无 label 上下文仍可独立成立的结构、切点和种类约束。"""

    try:
        normalized_kind = PredictionKind(kind)
    except (TypeError, ValueError) as exc:
        raise PredictionContextContractError("prediction context kind is invalid") from exc
    normalized_scope = normalize_prediction_scope(scope)
    normalized_anchor = normalize_prediction_anchor(anchor)
    _validate_kind_scope_anchor(normalized_kind, normalized_scope, normalized_anchor)
    _validate_decision_basis(
        normalized_kind,
        normalized_anchor,
        prediction_input,
        treatment=treatment,
    )


def _validate_kind_scope_anchor(
    kind: PredictionKind,
    scope: Mapping[str, Any],
    anchor: Mapping[str, Any],
) -> None:
    target = PredictionTargetLevel(scope["target_level"])
    mode = PredictionMode(scope["prediction_mode"])
    anchor_type = PredictionAnchorType(anchor["anchor_type"])
    if kind is PredictionKind.TRANSITION:
        if anchor_type not in {PredictionAnchorType.ACTION, PredictionAnchorType.EVENT}:
            raise PredictionContextContractError(
                "Transition context requires an Action or Event anchor"
            )
        if target not in {
            PredictionTargetLevel.ACTION,
            PredictionTargetLevel.EVENT,
            PredictionTargetLevel.TERMINAL,
        }:
            raise PredictionContextContractError(
                "Transition context has an incompatible target level"
            )
        expected_mode = (
            PredictionMode.TERMINATION
            if target is PredictionTargetLevel.TERMINAL
            else PredictionMode.NEXT_STEP
        )
        if mode is not expected_mode:
            raise PredictionContextContractError(
                "Transition context target and prediction mode do not match"
            )
        return
    if kind is PredictionKind.TRAJECTORY:
        if anchor_type is not PredictionAnchorType.PHASE:
            raise PredictionContextContractError(
                "Trajectory context requires a Phase anchor"
            )
        if target not in {PredictionTargetLevel.PHASE, PredictionTargetLevel.TERMINAL}:
            raise PredictionContextContractError(
                "Trajectory context has an incompatible target level"
            )
        expected_mode = (
            PredictionMode.TERMINATION
            if target is PredictionTargetLevel.TERMINAL
            else PredictionMode.CONTINUATION
        )
        if mode is not expected_mode:
            raise PredictionContextContractError(
                "Trajectory context target and prediction mode do not match"
            )
        return
    if (
        anchor_type not in {PredictionAnchorType.ACTION, PredictionAnchorType.EVENT}
        or target is not PredictionTargetLevel.OUTCOME
        or mode is not PredictionMode.CONSEQUENCE
    ):
        raise PredictionContextContractError(
            "Consequence context requires Action/Event to Outcome consequence scope"
        )


def _validate_decision_basis(
    kind: PredictionKind,
    anchor: Mapping[str, Any],
    prediction_input: Mapping[str, Any],
    *,
    treatment: Mapping[str, Any] | None,
) -> None:
    basis = PredictionDecisionBasis(anchor["decision_basis"])
    precision = PredictionTimePrecision(anchor["precision"])
    prefix_length = anchor["prefix_length"]
    previous_ref = anchor["previous_step_ref"]
    if basis in {
        PredictionDecisionBasis.CONTAINER_STARTED,
        PredictionDecisionBasis.CONTAINER_OBSERVED,
    }:
        if prefix_length != 0 or previous_ref is not None:
            raise PredictionContextContractError(
                "container anchor requires an empty prefix"
            )
        if precision is not PredictionTimePrecision.EXACT:
            raise PredictionContextContractError(
                "container anchor requires exact precision"
            )
        frame = _mapping(
            prediction_input.get("observation_frame"),
            "prediction input observation_frame",
        )
        evidence_field = (
            "observed_at"
            if basis is PredictionDecisionBasis.CONTAINER_STARTED
            else "available_at"
        )
        evidence_at = _datetime(
            frame.get(evidence_field), f"container anchor frame {evidence_field}"
        )
        if evidence_at != anchor["cutoff_at"]:
            raise PredictionContextContractError(
                f"container anchor cutoff must equal frame {evidence_field}"
            )
        return
    if basis is PredictionDecisionBasis.TREATMENT_OBSERVED:
        if kind is not PredictionKind.CONSEQUENCE or treatment is None:
            raise PredictionContextContractError(
                "treatment-observed anchor is only valid for Consequence"
            )
        if precision is not PredictionTimePrecision.EXACT:
            raise PredictionContextContractError(
                "treatment-observed anchor requires exact precision"
            )
        treatment_available = _datetime(
            treatment.get("available_at"), "prediction treatment available_at"
        )
        if treatment_available != anchor["cutoff_at"]:
            raise PredictionContextContractError(
                "treatment-observed cutoff must equal treatment available_at"
            )
        treatment_kind = _text(
            treatment.get("step_kind"), "prediction treatment step_kind"
        )
        if treatment_kind != anchor["anchor_type"]:
            raise PredictionContextContractError(
                "treatment-observed anchor type must match treatment step_kind"
            )
        _validate_prefix_reference(anchor, prediction_input)
        return
    if prefix_length == 0 or previous_ref is None:
        raise PredictionContextContractError(
            "previous-step anchor requires a non-empty prefix"
        )
    previous_step = _validate_prefix_reference(anchor, prediction_input)
    assert previous_step is not None
    if basis is PredictionDecisionBasis.PREVIOUS_STEP_ORDERED:
        if precision is PredictionTimePrecision.EXACT:
            raise PredictionContextContractError(
                "ordered previous-step anchor cannot use exact precision"
            )
        return
    if precision is not PredictionTimePrecision.EXACT:
        raise PredictionContextContractError(
            "observed, started or completed previous-step anchor requires exact precision"
        )
    cutoff = anchor["cutoff_at"]
    if basis is PredictionDecisionBasis.PREVIOUS_STEP_OBSERVED:
        if _optional_datetime(
            previous_step.get("available_at"), "previous step available_at"
        ) != cutoff:
            raise PredictionContextContractError(
                "previous-step-observed anchor requires its referenced step at cutoff"
            )
        return
    if basis is PredictionDecisionBasis.PREVIOUS_STEP_STARTED:
        if _optional_datetime(
            previous_step.get("started_at"), "previous step started_at"
        ) != cutoff:
            raise PredictionContextContractError(
                "previous-step-started anchor requires its referenced step at cutoff"
            )
        return
    completed_refs = {
        _text(step.get("step_ref"), "completed step_ref")
        for step in _completed_history_steps(prediction_input)
    }
    if previous_ref not in completed_refs or _optional_datetime(
        previous_step.get("ended_at"), "previous step ended_at"
    ) != cutoff:
        raise PredictionContextContractError(
            "previous-step-completed anchor requires its referenced completed step at cutoff"
        )


def _validate_prefix_reference(
    anchor: Mapping[str, Any],
    prediction_input: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    anchor_type = anchor["anchor_type"]
    relevant_steps = tuple(
        step
        for step in _history_steps(prediction_input)
        if step.get("step_kind") == anchor_type
    )
    refs = tuple(_text(step.get("step_ref"), "history step_ref") for step in relevant_steps)
    if len(refs) != len(set(refs)):
        raise PredictionContextContractError(
            "prediction history contains duplicate step_ref values"
        )
    if anchor["prefix_length"] != len(relevant_steps):
        raise PredictionContextContractError(
            "prediction anchor prefix_length must equal its visible typed history"
        )
    previous_ref = anchor["previous_step_ref"]
    if not relevant_steps:
        if previous_ref is not None:
            raise PredictionContextContractError(
                "empty prediction prefix forbids previous_step_ref"
            )
        return None
    if previous_ref is None:
        raise PredictionContextContractError(
            "non-empty prediction prefix requires previous_step_ref"
        )
    matching = tuple(
        step for step in relevant_steps if step.get("step_ref") == previous_ref
    )
    if len(matching) != 1:
        raise PredictionContextContractError(
            "prediction previous_step_ref must identify one visible history step"
        )
    return matching[0]


def _history_steps(prediction_input: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    history = _mapping(
        prediction_input.get("behavior_history"), "prediction input behavior_history"
    )
    return tuple(
        step
        for name in _HISTORY_COLLECTIONS
        for step in _mapping_sequence(history.get(name), f"behavior_history.{name}")
    )


def _completed_history_steps(
    prediction_input: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    history = _mapping(
        prediction_input.get("behavior_history"), "prediction input behavior_history"
    )
    return tuple(
        step
        for name in ("completed_events", "completed_actions", "completed_phases")
        for step in _mapping_sequence(history.get(name), f"behavior_history.{name}")
    )


def _mapping_sequence(value: object, label: str) -> tuple[Mapping[str, Any], ...]:
    return tuple(_mapping(item, f"{label} item") for item in _sequence(value, label))


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise PredictionContextContractError(f"{label} must be an object with string keys")
    return value


def _exact_mapping(
    value: object,
    expected: frozenset[str],
    label: str,
) -> dict[str, Any]:
    payload = dict(_mapping(value, label))
    unknown = set(payload) - expected
    missing = expected - set(payload)
    if unknown:
        raise PredictionContextContractError(
            f"{label} contains unsupported keys: {sorted(unknown)}"
        )
    if missing:
        raise PredictionContextContractError(
            f"{label} is missing keys: {sorted(missing)}"
        )
    return payload


def _sequence(value: object, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PredictionContextContractError(f"{label} must be an array")
    return value


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PredictionContextContractError(f"{label} must be non-empty text")
    normalized = value.strip()
    if any(not character.isprintable() and character not in "\n\t" for character in normalized):
        raise PredictionContextContractError(f"{label} contains control characters")
    return normalized


def _optional_text(value: object, label: str) -> str | None:
    return None if value is None else _text(value, label)


def _enum_value(value: object, enum_type: type[Any], label: str) -> str:
    resolved = _text(value, label)
    try:
        return str(enum_type(resolved).value)
    except ValueError as exc:
        raise PredictionContextContractError(f"{label} is not an allowed value") from exc


def _non_negative_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PredictionContextContractError(f"{label} must be a non-negative integer")
    return value


def _optional_datetime(value: object, label: str) -> datetime | None:
    return None if value is None else _datetime(value, label)


def _datetime(value: object, label: str) -> datetime:
    parsed = value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise PredictionContextContractError(f"{label} must be an ISO timestamp") from exc
    if not isinstance(parsed, datetime) or parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PredictionContextContractError(f"{label} must be a timezone-aware datetime")
    return parsed


__all__ = [
    "PredictionContextContractError",
    "normalize_prediction_anchor",
    "normalize_prediction_scope",
    "validate_prediction_context_contract",
]
