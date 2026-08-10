"""预测截止时刻的统一信息可见性不变量。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any


class PredictionVisibilityError(ValueError):
    """预测输入包含截止时刻之后才可获得的信息。"""


def validate_context_visibility(anchor: Mapping[str, Any], prediction_input: Mapping[str, Any]) -> None:
    """时间锚点下，任何模型可见字段都不得越过可证明的最早 cutoff。"""

    precision = anchor.get("precision")
    if precision == "exact":
        cutoff = _datetime(anchor.get("cutoff_at"), "exact anchor cutoff")
    elif precision == "bounded" and anchor.get("lower_bound_at") is not None:
        cutoff = _datetime(anchor.get("lower_bound_at"), "bounded anchor lower bound")
    else:
        return
    frame = _mapping(prediction_input.get("observation_frame"), "observation frame")
    if frame.get("available_at") is None:
        raise PredictionVisibilityError("timestamped prediction context requires frame available_at")
    _not_after(frame.get("observed_at"), cutoff, "observation frame observed_at")
    _not_after(frame.get("available_at"), cutoff, "observation frame available_at")
    for fact in _sequence(frame.get("facts"), "observation facts"):
        fact_value = _mapping(fact, "observation fact")
        _not_after(fact_value.get("observed_at"), cutoff, "fact observed_at")
        _not_after(fact_value.get("available_at"), cutoff, "fact available_at")
    coverage = _mapping(frame.get("coverage"), "observation coverage")
    for interval in _sequence(coverage.get("blind_intervals"), "blind intervals"):
        interval_value = _mapping(interval, "blind interval")
        _not_after(interval_value.get("ended_at"), cutoff, "blind interval ended_at")

    history = _mapping(prediction_input.get("behavior_history"), "behavior history")
    completed_names = ("completed_events", "completed_actions", "completed_phases")
    for name in completed_names:
        for step in _sequence(history.get(name), name):
            step_value = _mapping(step, f"{name} step")
            ended_at = step_value.get("ended_at")
            if ended_at is None:
                raise PredictionVisibilityError(f"{name} cannot contain a step without ended_at")
            _not_after(ended_at, cutoff, f"{name} ended_at")
            _not_after(step_value.get("available_at"), cutoff, f"{name} available_at")
    for name in ("active_behaviors", "parallel_behaviors", "interruptions", "resumptions"):
        for step in _sequence(history.get(name), name):
            step_value = _mapping(step, f"{name} step")
            _not_after(step_value.get("started_at"), cutoff, f"{name} started_at")
            _not_after(step_value.get("available_at"), cutoff, f"{name} available_at")
            if step_value.get("ended_at") is not None:
                raise PredictionVisibilityError(f"{name} must hide post-cutoff ended_at")
            if step_value.get("status") != "active":
                raise PredictionVisibilityError(f"{name} must expose active status")


def validate_label_boundary(anchor: Mapping[str, Any], label: Mapping[str, Any]) -> None:
    """exact anchor 下，非终止标签不得在 cutoff 之前已经开始。"""

    if anchor.get("precision") != "exact":
        return
    cutoff = _datetime(anchor.get("cutoff_at"), "exact anchor cutoff")
    if "outcome" in label:
        outcome = _mapping(label["outcome"], "consequence outcome")
        occurred_at = _datetime(outcome.get("occurred_at"), "outcome occurred_at")
        if occurred_at <= cutoff:
            raise PredictionVisibilityError(
                "prediction outcome must occur after its exact cutoff"
            )
        return
    candidates: list[Any] = []
    if "target_kind" in label:
        candidates.append(label.get("started_at"))
    elif label.get("mainline"):
        first = _sequence(label["mainline"], "trajectory mainline")[0]
        candidates.append(_mapping(first, "trajectory first phase").get("started_at"))
    for value in candidates:
        if value is not None and _datetime(value, "label started_at") < cutoff:
            raise PredictionVisibilityError("prediction label started before its exact cutoff")


def validate_treatment_boundary(anchor: Mapping[str, Any], treatment: Mapping[str, Any]) -> None:
    """给定行为必须在预测切点已经开始，且不得携带结束后信息。"""

    if anchor.get("precision") != "exact":
        return
    cutoff = _datetime(anchor.get("cutoff_at"), "exact anchor cutoff")
    started_at = _datetime(treatment.get("started_at"), "treatment started_at")
    available_at = _datetime(treatment.get("available_at"), "treatment available_at")
    if started_at > cutoff:
        raise PredictionVisibilityError("prediction treatment has not started at its exact cutoff")
    if available_at > cutoff:
        raise PredictionVisibilityError("prediction treatment is not available at its exact cutoff")
    if treatment.get("ended_at") is not None:
        raise PredictionVisibilityError("prediction treatment must hide post-cutoff ended_at")
    if treatment.get("status") != "active":
        raise PredictionVisibilityError("prediction treatment must expose active status")


def _not_after(value: Any, cutoff: datetime, label: str) -> None:
    if value is not None and _datetime(value, label) > cutoff:
        raise PredictionVisibilityError(f"{label} is later than the prediction cutoff")


def _datetime(value: Any, label: str) -> datetime:
    parsed = value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise PredictionVisibilityError(f"{label} is not an ISO timestamp") from exc
    if not isinstance(parsed, datetime) or parsed.tzinfo is None or parsed.utcoffset() is None:
        raise PredictionVisibilityError(f"{label} must be timezone-aware")
    return parsed


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PredictionVisibilityError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PredictionVisibilityError(f"{label} must be an array")
    return value


__all__ = [
    "PredictionVisibilityError",
    "validate_context_visibility",
    "validate_label_boundary",
    "validate_treatment_boundary",
]
