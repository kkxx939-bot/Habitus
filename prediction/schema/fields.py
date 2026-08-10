"""预测样本各领域字段的规范化校验器。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from foundation.integrity import canonical_json, canonicalize
from prediction.context_contract import (
    PredictionContextContractError,
    normalize_prediction_anchor,
    normalize_prediction_scope,
)
from prediction.model import PredictionLabelStatus, PredictionTargetLevel, prediction_sample_id
from prediction.schema.model import PredictionFieldSchema, PredictionFieldType, PredictionSchemaError
from prediction.schema.primitives import (
    boolean,
    confidence,
    date_value,
    datetime_value,
    enum,
    enum_set,
    exact_mapping,
    non_negative_integer,
    optional_confidence,
    optional_datetime,
    optional_non_negative_number,
    optional_text,
    optional_uri,
    positive_integer,
    record_id,
    records,
    sha256,
    strict_mapping,
    string_tuple,
    text,
    uri_text,
)
from prediction.schema.vocabulary import ATTRIBUTIONS, FACT_CATEGORIES, KNOWLEDGE_STATES, STEP_KINDS


def _identity_material(value: Any, label: str) -> dict[str, Any]:
    payload = strict_mapping(value, label)
    try:
        normalized = canonicalize(payload)
        canonical_json(normalized).encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise PredictionSchemaError("prediction identity material must be canonical UTF-8 JSON") from exc
    assert isinstance(normalized, dict)
    return normalized


def _scope(value: Any, label: str) -> dict[str, Any]:
    del label
    try:
        return normalize_prediction_scope(value)
    except PredictionContextContractError as exc:
        raise PredictionSchemaError(str(exc)) from exc


def _anchor(value: Any, label: str) -> dict[str, Any]:
    del label
    try:
        return normalize_prediction_anchor(value)
    except PredictionContextContractError as exc:
        raise PredictionSchemaError(str(exc)) from exc


def _prediction_input(value: Any, label: str) -> dict[str, Any]:
    payload = exact_mapping(value, {"observation_frame", "behavior_history", "decision_space"}, label)
    return {
        "observation_frame": _observation_frame(payload["observation_frame"], f"{label}.observation_frame"),
        "behavior_history": _behavior_history(payload["behavior_history"], f"{label}.behavior_history"),
        "decision_space": _decision_space(payload["decision_space"], f"{label}.decision_space"),
    }


def _observation_frame(value: Any, label: str) -> dict[str, Any]:
    expected = {
        "observed_at",
        "available_at",
        "observer",
        "subjects",
        "facts",
        "active_goals",
        "constraints",
        "coverage",
    }
    payload = exact_mapping(value, expected, label)
    return {
        "observed_at": optional_datetime(payload["observed_at"], f"{label}.observed_at"),
        "available_at": optional_datetime(payload["available_at"], f"{label}.available_at"),
        "observer": optional_text(payload["observer"], f"{label}.observer"),
        "subjects": string_tuple(payload["subjects"], f"{label}.subjects"),
        "facts": records(payload["facts"], _observed_fact, f"{label}.facts", identity="fact_id"),
        "active_goals": string_tuple(payload["active_goals"], f"{label}.active_goals"),
        "constraints": string_tuple(payload["constraints"], f"{label}.constraints"),
        "coverage": _coverage(payload["coverage"], f"{label}.coverage"),
    }


def _observed_fact(value: Any, label: str) -> dict[str, Any]:
    expected = {
        "fact_id",
        "category",
        "semantics",
        "subject_ref",
        "attribute",
        "value",
        "unit",
        "observed_at",
        "available_at",
        "valid_from",
        "valid_to",
        "knowledge_state",
        "confidence",
        "evidence_refs",
    }
    payload = exact_mapping(value, expected, label)
    valid_from = optional_datetime(payload["valid_from"], f"{label}.valid_from")
    valid_to = optional_datetime(payload["valid_to"], f"{label}.valid_to")
    if valid_from is not None and valid_to is not None and valid_to < valid_from:
        raise PredictionSchemaError("ObservedFact valid_to cannot precede valid_from")
    normalized_value = canonicalize(payload["value"])
    try:
        canonical_json(normalized_value).encode("utf-8")
    except UnicodeEncodeError as exc:
        raise PredictionSchemaError("ObservedFact value must be canonical UTF-8 JSON") from exc
    return {
        "fact_id": record_id(payload["fact_id"], f"{label}.fact_id"),
        "category": enum_set(payload["category"], FACT_CATEGORIES, f"{label}.category"),
        "semantics": text(payload["semantics"], f"{label}.semantics"),
        "subject_ref": optional_text(payload["subject_ref"], f"{label}.subject_ref"),
        "attribute": optional_text(payload["attribute"], f"{label}.attribute"),
        "value": normalized_value,
        "unit": optional_text(payload["unit"], f"{label}.unit"),
        "observed_at": optional_datetime(payload["observed_at"], f"{label}.observed_at"),
        "available_at": datetime_value(payload["available_at"], f"{label}.available_at"),
        "valid_from": valid_from,
        "valid_to": valid_to,
        "knowledge_state": enum_set(payload["knowledge_state"], KNOWLEDGE_STATES, f"{label}.knowledge_state"),
        "confidence": optional_confidence(payload["confidence"], f"{label}.confidence"),
        "evidence_refs": string_tuple(payload["evidence_refs"], f"{label}.evidence_refs"),
    }


def _coverage(value: Any, label: str) -> dict[str, Any]:
    expected = {"available_modalities", "missing_modalities", "blind_intervals", "coverage_score"}
    payload = exact_mapping(value, expected, label)
    available = string_tuple(payload["available_modalities"], f"{label}.available_modalities")
    missing = string_tuple(payload["missing_modalities"], f"{label}.missing_modalities")
    if set(available) & set(missing):
        raise PredictionSchemaError("one observation modality cannot be both available and missing")
    return {
        "available_modalities": available,
        "missing_modalities": missing,
        "blind_intervals": records(payload["blind_intervals"], _blind_interval, f"{label}.blind_intervals"),
        "coverage_score": optional_confidence(payload["coverage_score"], f"{label}.coverage_score"),
    }


def _blind_interval(value: Any, label: str) -> dict[str, Any]:
    payload = exact_mapping(value, {"started_at", "ended_at", "reason"}, label)
    started_at = datetime_value(payload["started_at"], f"{label}.started_at")
    ended_at = datetime_value(payload["ended_at"], f"{label}.ended_at")
    if ended_at < started_at:
        raise PredictionSchemaError("observation blind interval cannot move backwards")
    return {
        "started_at": started_at,
        "ended_at": ended_at,
        "reason": text(payload["reason"], f"{label}.reason"),
    }


def _behavior_history(value: Any, label: str) -> dict[str, Any]:
    keys = {
        "completed_events",
        "completed_actions",
        "completed_phases",
        "active_behaviors",
        "parallel_behaviors",
        "interruptions",
        "resumptions",
    }
    payload = exact_mapping(value, keys, label)
    return {name: records(payload[name], _projected_step, f"{label}.{name}") for name in keys}


def _projected_step(value: Any, label: str) -> dict[str, Any]:
    expected = {
        "step_kind",
        "step_ref",
        "source_uri",
        "local_id",
        "sequence",
        "semantics",
        "actor",
        "behavior_type",
        "target_refs",
        "status",
        "started_at",
        "ended_at",
        "available_at",
    }
    payload = exact_mapping(value, expected, label)
    started_at = optional_datetime(payload["started_at"], f"{label}.started_at")
    ended_at = optional_datetime(payload["ended_at"], f"{label}.ended_at")
    if started_at is not None and ended_at is not None and ended_at < started_at:
        raise PredictionSchemaError("ProjectedStep ended_at cannot precede started_at")
    return {
        "step_kind": enum_set(payload["step_kind"], STEP_KINDS, f"{label}.step_kind"),
        "step_ref": text(payload["step_ref"], f"{label}.step_ref"),
        "source_uri": optional_uri(payload["source_uri"], f"{label}.source_uri"),
        "local_id": optional_text(payload["local_id"], f"{label}.local_id"),
        "sequence": positive_integer(payload["sequence"], f"{label}.sequence"),
        "semantics": text(payload["semantics"], f"{label}.semantics"),
        "actor": optional_text(payload["actor"], f"{label}.actor"),
        "behavior_type": optional_text(payload["behavior_type"], f"{label}.behavior_type"),
        "target_refs": string_tuple(payload["target_refs"], f"{label}.target_refs"),
        "status": text(payload["status"], f"{label}.status"),
        "started_at": started_at,
        "ended_at": ended_at,
        "available_at": datetime_value(payload["available_at"], f"{label}.available_at"),
    }


def _decision_space(value: Any, label: str) -> dict[str, Any]:
    keys = {"known_available", "known_unavailable", "prohibited", "unknown"}
    payload = exact_mapping(value, keys, label)
    return {
        "known_available": records(payload["known_available"], _decision_option, f"{label}.known_available"),
        "known_unavailable": records(
            payload["known_unavailable"], _decision_option, f"{label}.known_unavailable"
        ),
        "prohibited": records(payload["prohibited"], _decision_option, f"{label}.prohibited"),
        "unknown": string_tuple(payload["unknown"], f"{label}.unknown"),
    }


def _decision_option(value: Any, label: str) -> dict[str, Any]:
    expected = {"semantics", "behavior_type", "target_refs", "reason", "evidence_refs"}
    payload = exact_mapping(value, expected, label)
    return {
        "semantics": text(payload["semantics"], f"{label}.semantics"),
        "behavior_type": optional_text(payload["behavior_type"], f"{label}.behavior_type"),
        "target_refs": string_tuple(payload["target_refs"], f"{label}.target_refs"),
        "reason": optional_text(payload["reason"], f"{label}.reason"),
        "evidence_refs": string_tuple(payload["evidence_refs"], f"{label}.evidence_refs"),
    }


def _transition_label(value: Any, label: str) -> dict[str, Any]:
    expected = {
        "target_kind",
        "source_ref",
        "actor",
        "behavior_type",
        "semantics",
        "target_refs",
        "parameters",
        "started_at",
        "delay_seconds",
        "relations",
        "terminal",
    }
    payload = exact_mapping(value, expected, label)
    target_kind = enum_set(
        payload["target_kind"],
        {PredictionTargetLevel.ACTION.value, PredictionTargetLevel.EVENT.value, PredictionTargetLevel.TERMINAL.value},
        f"{label}.target_kind",
    )
    source_ref = optional_text(payload["source_ref"], f"{label}.source_ref")
    terminal = None if payload["terminal"] is None else _terminal(payload["terminal"], f"{label}.terminal")
    if target_kind == PredictionTargetLevel.TERMINAL.value:
        if source_ref is not None or terminal is None:
            raise PredictionSchemaError("terminal transition label forbids source_ref and requires terminal")
    elif source_ref is None or terminal is not None:
        raise PredictionSchemaError("non-terminal transition label requires source_ref and forbids terminal")
    return {
        "target_kind": target_kind,
        "source_ref": source_ref,
        "actor": optional_text(payload["actor"], f"{label}.actor"),
        "behavior_type": optional_text(payload["behavior_type"], f"{label}.behavior_type"),
        "semantics": text(payload["semantics"], f"{label}.semantics"),
        "target_refs": string_tuple(payload["target_refs"], f"{label}.target_refs"),
        "parameters": canonicalize(strict_mapping(payload["parameters"], f"{label}.parameters")),
        "started_at": optional_datetime(payload["started_at"], f"{label}.started_at"),
        "delay_seconds": optional_non_negative_number(payload["delay_seconds"], f"{label}.delay_seconds"),
        "relations": string_tuple(payload["relations"], f"{label}.relations"),
        "terminal": terminal,
    }


def _trajectory_label(value: Any, label: str) -> dict[str, Any]:
    expected = {
        "next_phase_ref",
        "mainline",
        "remaining_events",
        "parallel_branches",
        "interruptions",
        "resumptions",
        "future_context",
        "uncertain_events",
        "transition_edges",
        "terminal",
    }
    payload = exact_mapping(value, expected, label)
    mainline = records(payload["mainline"], _projected_step, f"{label}.mainline")
    if any(step["step_kind"] != "phase" for step in mainline):
        raise PredictionSchemaError("trajectory mainline must contain only Phase steps")
    remaining_events = records(payload["remaining_events"], _projected_step, f"{label}.remaining_events")
    parallel_branches = records(
        payload["parallel_branches"], _projected_step, f"{label}.parallel_branches"
    )
    interruptions = records(payload["interruptions"], _projected_step, f"{label}.interruptions")
    resumptions = records(payload["resumptions"], _projected_step, f"{label}.resumptions")
    future_context = records(payload["future_context"], _projected_step, f"{label}.future_context")
    uncertain_events = records(
        payload["uncertain_events"], _projected_step, f"{label}.uncertain_events"
    )
    labeled_events = (
        *remaining_events,
        *parallel_branches,
        *interruptions,
        *resumptions,
        *future_context,
        *uncertain_events,
    )
    if any(step["step_kind"] != "event" for step in labeled_events):
        raise PredictionSchemaError("trajectory non-mainline branches must contain only Event steps")
    if any(step["source_uri"] is None for step in (*mainline, *labeled_events)):
        raise PredictionSchemaError("trajectory label steps require source URIs")
    if any(step["local_id"] is None for step in mainline):
        raise PredictionSchemaError("trajectory mainline Phase steps require local IDs")
    mainline_identities = tuple((step["source_uri"], step["local_id"]) for step in mainline)
    if len(mainline_identities) != len(set(mainline_identities)):
        raise PredictionSchemaError("trajectory mainline must not repeat a Phase")
    mainline_sequences = tuple(step["sequence"] for step in mainline)
    if mainline_sequences != tuple(sorted(mainline_sequences)) or len(mainline_sequences) != len(set(mainline_sequences)):
        raise PredictionSchemaError("trajectory mainline Phase sequence must be unique and increasing")
    next_phase_ref = optional_text(payload["next_phase_ref"], f"{label}.next_phase_ref")
    terminal = None if payload["terminal"] is None else _terminal(payload["terminal"], f"{label}.terminal")
    transition_edges = records(payload["transition_edges"], _transition_edge, f"{label}.transition_edges")
    edge_identities = tuple((edge["from_ref"], edge["to_ref"], edge["relation"]) for edge in transition_edges)
    if len(edge_identities) != len(set(edge_identities)):
        raise PredictionSchemaError("trajectory transition edges must be unique")
    future_collections = (
        mainline,
        remaining_events,
        parallel_branches,
        interruptions,
        resumptions,
        future_context,
        uncertain_events,
        transition_edges,
    )
    if terminal is None and (next_phase_ref is None or not mainline):
        raise PredictionSchemaError("non-terminal trajectory requires next_phase_ref and a Phase mainline")
    if terminal is None:
        expected_ref = f"{mainline[0]['source_uri']}#phase:{mainline[0]['local_id']}"
        if next_phase_ref != expected_ref:
            raise PredictionSchemaError("trajectory next_phase_ref must identify the first mainline Phase")
    if terminal is not None and (next_phase_ref is not None or any(future_collections)):
        raise PredictionSchemaError("terminal trajectory forbids all future behavior labels")
    return {
        "next_phase_ref": next_phase_ref,
        "mainline": mainline,
        "remaining_events": remaining_events,
        "parallel_branches": parallel_branches,
        "interruptions": interruptions,
        "resumptions": resumptions,
        "future_context": future_context,
        "uncertain_events": uncertain_events,
        "transition_edges": transition_edges,
        "terminal": terminal,
    }


def _transition_edge(value: Any, label: str) -> dict[str, Any]:
    payload = exact_mapping(value, {"from_ref", "to_ref", "relation"}, label)
    source = text(payload["from_ref"], f"{label}.from_ref")
    target = text(payload["to_ref"], f"{label}.to_ref")
    if source == target:
        raise PredictionSchemaError("trajectory transition edge cannot be a self-edge")
    return {
        "from_ref": source,
        "to_ref": target,
        "relation": text(payload["relation"], f"{label}.relation"),
    }


def _treatment(value: Any, label: str) -> dict[str, Any]:
    treatment = _projected_step(value, label)
    if treatment["step_kind"] not in {"action", "event"}:
        raise PredictionSchemaError("ConsequenceSample treatment must be an Action or Event")
    if treatment["source_uri"] is None:
        raise PredictionSchemaError("ConsequenceSample treatment requires a source URI")
    return treatment


def _consequence_label(value: Any, label: str) -> dict[str, Any]:
    payload = exact_mapping(value, {"outcome", "attribution"}, label)
    return {
        "outcome": _outcome_label(payload["outcome"], f"{label}.outcome"),
        "attribution": enum_set(payload["attribution"], ATTRIBUTIONS, f"{label}.attribution"),
    }


def _outcome_label(value: Any, label: str) -> dict[str, Any]:
    expected = {
        "outcome_id",
        "occurred_at",
        "outcome_type",
        "semantics",
        "valence",
        "knowledge_state",
        "confidence",
        "delay_seconds",
    }
    payload = exact_mapping(value, expected, label)
    return {
        "outcome_id": record_id(payload["outcome_id"], f"{label}.outcome_id"),
        "occurred_at": datetime_value(payload["occurred_at"], f"{label}.occurred_at"),
        "outcome_type": text(payload["outcome_type"], f"{label}.outcome_type"),
        "semantics": text(payload["semantics"], f"{label}.semantics"),
        "valence": text(payload["valence"], f"{label}.valence"),
        "knowledge_state": enum_set(payload["knowledge_state"], KNOWLEDGE_STATES, f"{label}.knowledge_state"),
        "confidence": confidence(payload["confidence"], f"{label}.confidence"),
        "delay_seconds": optional_non_negative_number(payload["delay_seconds"], f"{label}.delay_seconds"),
    }


def _terminal(value: Any, label: str) -> dict[str, Any]:
    payload = exact_mapping(value, {"status", "reason"}, label)
    return {
        "status": text(payload["status"], f"{label}.status"),
        "reason": optional_text(payload["reason"], f"{label}.reason"),
    }


def _supervision(value: Any, label: str) -> dict[str, Any]:
    expected = {
        "label_status",
        "window_started_at",
        "window_closed_at",
        "censored",
        "censoring_reason",
    }
    payload = exact_mapping(value, expected, label)
    started_at = optional_datetime(payload["window_started_at"], f"{label}.window_started_at")
    closed_at = optional_datetime(payload["window_closed_at"], f"{label}.window_closed_at")
    if started_at is not None and closed_at is not None and closed_at < started_at:
        raise PredictionSchemaError("supervision window cannot move backwards")
    censored = boolean(payload["censored"], f"{label}.censored")
    reason = optional_text(payload["censoring_reason"], f"{label}.censoring_reason")
    if censored != (reason is not None):
        raise PredictionSchemaError("censored supervision requires one reason; uncensored supervision forbids it")
    return {
        "label_status": enum(payload["label_status"], PredictionLabelStatus, f"{label}.label_status"),
        "window_started_at": started_at,
        "window_closed_at": closed_at,
        "censored": censored,
        "censoring_reason": reason,
    }


def _lineage(value: Any, label: str) -> dict[str, Any]:
    expected = {
        "behavior_root_uri",
        "event_uri",
        "episode_uri",
        "outcome_uri",
        "occurrence_group_id",
        "consequence_group_id",
    }
    payload = exact_mapping(value, expected, label)
    return {
        "behavior_root_uri": uri_text(payload["behavior_root_uri"], f"{label}.behavior_root_uri"),
        "event_uri": optional_uri(payload["event_uri"], f"{label}.event_uri"),
        "episode_uri": optional_uri(payload["episode_uri"], f"{label}.episode_uri"),
        "outcome_uri": optional_uri(payload["outcome_uri"], f"{label}.outcome_uri"),
        "occurrence_group_id": text(payload["occurrence_group_id"], f"{label}.occurrence_group_id"),
        "consequence_group_id": optional_text(
            payload["consequence_group_id"], f"{label}.consequence_group_id"
        ),
    }


def _provenance(value: Any, label: str) -> dict[str, Any]:
    payload = exact_mapping(value, {"source_bindings", "projection_version", "projector_digest"}, label)
    bindings = records(payload["source_bindings"], _source_binding, f"{label}.source_bindings")
    if not bindings:
        raise PredictionSchemaError("prediction provenance requires at least one source binding")
    identities = tuple((binding["uri"], binding["member_type"], binding["member_id"]) for binding in bindings)
    if len(identities) != len(set(identities)):
        raise PredictionSchemaError("prediction source bindings must be unique")
    return {
        "source_bindings": bindings,
        "projection_version": text(payload["projection_version"], f"{label}.projection_version"),
        "projector_digest": sha256(payload["projector_digest"], f"{label}.projector_digest"),
    }


def _source_binding(value: Any, label: str) -> dict[str, Any]:
    payload = exact_mapping(value, {"uri", "revision", "digest", "member_type", "member_id"}, label)
    return {
        "uri": uri_text(payload["uri"], f"{label}.uri"),
        "revision": positive_integer(payload["revision"], f"{label}.revision"),
        "digest": sha256(payload["digest"], f"{label}.digest"),
        "member_type": text(payload["member_type"], f"{label}.member_type"),
        "member_id": optional_text(payload["member_id"], f"{label}.member_id"),
    }


def _quality(value: Any, label: str) -> dict[str, Any]:
    expected = {
        "source_confidence",
        "evidence_coverage",
        "context_completeness",
        "conflict_count",
        "inferred_fact_ratio",
    }
    payload = exact_mapping(value, expected, label)
    return {
        "source_confidence": confidence(payload["source_confidence"], f"{label}.source_confidence"),
        "evidence_coverage": optional_confidence(payload["evidence_coverage"], f"{label}.evidence_coverage"),
        "context_completeness": optional_confidence(
            payload["context_completeness"], f"{label}.context_completeness"
        ),
        "conflict_count": non_negative_integer(payload["conflict_count"], f"{label}.conflict_count"),
        "inferred_fact_ratio": confidence(payload["inferred_fact_ratio"], f"{label}.inferred_fact_ratio"),
    }


_VALIDATORS: dict[PredictionFieldType, Callable[[Any, str], Any]] = {
    PredictionFieldType.DATE: date_value,
    PredictionFieldType.SAMPLE_ID: lambda value, _label: prediction_sample_id(value),
    PredictionFieldType.IDENTITY_MATERIAL: _identity_material,
    PredictionFieldType.SCOPE: _scope,
    PredictionFieldType.ANCHOR: _anchor,
    PredictionFieldType.INPUT: _prediction_input,
    PredictionFieldType.TREATMENT: _treatment,
    PredictionFieldType.TRANSITION_LABEL: _transition_label,
    PredictionFieldType.TRAJECTORY_LABEL: _trajectory_label,
    PredictionFieldType.CONSEQUENCE_LABEL: _consequence_label,
    PredictionFieldType.SUPERVISION: _supervision,
    PredictionFieldType.LINEAGE: _lineage,
    PredictionFieldType.PROVENANCE: _provenance,
    PredictionFieldType.QUALITY: _quality,
}


def validate_field(field: PredictionFieldSchema, value: Any) -> Any:
    """按声明类型规范化单个字段；非 Schema 异常统一收敛为 PredictionSchemaError。"""

    try:
        return _VALIDATORS[field.field_type](value, field.name)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, PredictionSchemaError):
            raise
        raise PredictionSchemaError(f"prediction field {field.name} is invalid") from exc


__all__ = ["validate_field"]
