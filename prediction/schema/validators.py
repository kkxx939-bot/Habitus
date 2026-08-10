"""预测样本的跨字段不变量、身份一致性和来源闭合校验。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from foundation.integrity import canonical_digest
from prediction.identity import derive_sample_identity
from prediction.model import PredictionAnchorType, PredictionKind, PredictionTargetLevel
from prediction.schema.model import PredictionSchemaError
from prediction.schema.primitives import (
    enum,
    non_negative_integer,
    positive_integer,
    record_id,
    text,
    uri_text,
)


def validate_kind(kind: PredictionKind, payload: Mapping[str, Any]) -> None:
    scope = payload["prediction_scope"]
    label = payload["label"]
    if kind is PredictionKind.TRANSITION:
        if scope["target_level"] != label["target_kind"]:
            raise PredictionSchemaError("TransitionSample target level must match its label")
        terminal = label["target_kind"] == PredictionTargetLevel.TERMINAL.value
        if terminal != (scope["prediction_mode"] == "termination"):
            raise PredictionSchemaError(
                "TransitionSample termination mode must match a terminal label"
            )
    elif kind is PredictionKind.TRAJECTORY:
        expected_level = (
            PredictionTargetLevel.TERMINAL.value
            if label["terminal"] is not None
            else PredictionTargetLevel.PHASE.value
        )
        if scope["target_level"] != expected_level:
            raise PredictionSchemaError("TrajectorySample target level does not match its continuation")
        terminal = label["terminal"] is not None
        if terminal != (scope["prediction_mode"] == "termination"):
            raise PredictionSchemaError(
                "TrajectorySample termination mode must match a terminal label"
            )


def validate_identity(kind: PredictionKind, payload: Mapping[str, Any]) -> None:
    identity = payload["identity_material"]
    context = payload["materialization_context"]
    anchor = payload["anchor"]
    lineage = payload["lineage"]
    if anchor["prefix_length"] == 0:
        if anchor["previous_step_ref"] is not None:
            raise PredictionSchemaError("zero-length prediction prefix forbids previous_step_ref")
    elif anchor["previous_step_ref"] is None:
        raise PredictionSchemaError("non-empty prediction prefix requires previous_step_ref")
    expected_container_ref = "behavior-container:" + canonical_digest(
        {"uri": lineage["behavior_root_uri"]}
    )
    if anchor["container_ref"] != expected_container_ref:
        raise PredictionSchemaError("prediction anchor container must match behavior_root_uri")

    if kind in {PredictionKind.TRANSITION, PredictionKind.TRAJECTORY}:
        expected_keys = {"container_uri", "anchor_type", "prefix_length", "target_ref"}
        if set(identity) != expected_keys:
            raise PredictionSchemaError(
                f"{kind.value} identity material must contain exactly {sorted(expected_keys)}"
            )
        container_uri = uri_text(identity["container_uri"], "identity_material.container_uri")
        if container_uri != identity["container_uri"]:
            raise PredictionSchemaError("prediction identity container URI must be canonical")
        if container_uri != lineage["behavior_root_uri"]:
            raise PredictionSchemaError("prediction identity container must match behavior_root_uri")
        identity_anchor_type = enum(
            identity["anchor_type"],
            PredictionAnchorType,
            "identity_material.anchor_type",
        )
        identity_prefix_length = non_negative_integer(
            identity["prefix_length"],
            "identity_material.prefix_length",
        )
        identity_target = text(identity["target_ref"], "identity_material.target_ref")
        if identity_anchor_type != anchor["anchor_type"]:
            raise PredictionSchemaError("prediction identity anchor type must match anchor")
        if identity_prefix_length != anchor["prefix_length"]:
            raise PredictionSchemaError("prediction identity prefix length must match anchor")
        if context:
            raise PredictionSchemaError("Transition and Trajectory materialization context must be empty")
        label = payload["label"]
        if kind is PredictionKind.TRANSITION:
            expected_target = label["source_ref"] or "terminal"
        else:
            expected_target = label["next_phase_ref"] or "terminal"
        if identity_target != expected_target:
            raise PredictionSchemaError("prediction identity target must match its label")
    else:
        expected_keys = {"outcome_uri", "outcome_id"}
        if set(identity) != expected_keys:
            raise PredictionSchemaError(
                f"consequence identity material must contain exactly {sorted(expected_keys)}"
            )
        outcome_uri = uri_text(identity["outcome_uri"], "identity_material.outcome_uri")
        if outcome_uri != identity["outcome_uri"]:
            raise PredictionSchemaError("Consequence identity Outcome URI must be canonical")
        if outcome_uri != lineage["outcome_uri"]:
            raise PredictionSchemaError("prediction identity Outcome must match lineage outcome_uri")
        outcome_id = record_id(identity["outcome_id"], "identity_material.outcome_id")
        if outcome_id != payload["label"]["outcome"]["outcome_id"]:
            raise PredictionSchemaError("Consequence identity outcome_id must match its label")
        if set(context) != {"outcome_revision"}:
            raise PredictionSchemaError(
                "Consequence materialization context requires exactly outcome_revision"
            )
        revision = positive_integer(
            context["outcome_revision"],
            "materialization_context.outcome_revision",
        )
        matching_revisions = {
            binding["revision"]
            for binding in payload["provenance"]["source_bindings"]
            if binding["uri"] == outcome_uri
            and binding["member_type"] == "outcome"
            and binding["member_id"] == outcome_id
        }
        if matching_revisions != {revision}:
            raise PredictionSchemaError(
                "Consequence outcome_revision must match its Outcome provenance binding"
            )
    derived = derive_sample_identity(
        kind,
        identity,
        payload["provenance"]["projection_version"],
        context,
    )
    if payload["logical_sample_id"] != derived.logical_sample_id:
        raise PredictionSchemaError("prediction logical sample ID does not match its identity material")
    if payload["materialization_id"] != derived.materialization_id:
        raise PredictionSchemaError("prediction materialization ID does not match its projection version")


def validate_source_closure(kind: PredictionKind, payload: Mapping[str, Any]) -> None:
    bindings = payload["provenance"]["source_bindings"]
    binding_uris = {binding["uri"] for binding in bindings}
    lineage = payload["lineage"]
    lineage_uris = {
        uri
        for uri in (
            lineage["behavior_root_uri"],
            lineage["event_uri"],
            lineage["episode_uri"],
            lineage["outcome_uri"],
        )
        if uri is not None
    }
    if not lineage_uris <= binding_uris:
        raise PredictionSchemaError("prediction lineage must be closed over provenance source bindings")
    identity = payload["identity_material"]
    root_identity = identity.get("container_uri")
    if root_identity is not None and root_identity != lineage["behavior_root_uri"]:
        raise PredictionSchemaError("prediction identity container must match behavior_root_uri")
    outcome_identity = identity.get("outcome_uri")
    if outcome_identity is not None and outcome_identity != lineage["outcome_uri"]:
        raise PredictionSchemaError("prediction identity Outcome must match lineage outcome_uri")

    label = payload["label"]
    label_uris: set[str] = set()
    if kind is PredictionKind.TRANSITION:
        if label["source_ref"] is not None:
            label_uris.add(label["source_ref"].split("#", 1)[0])
    elif kind is PredictionKind.TRAJECTORY:
        collections = (
            label["mainline"],
            label["remaining_events"],
            label["parallel_branches"],
            label["interruptions"],
            label["resumptions"],
            label["future_context"],
            label["uncertain_events"],
        )
        label_uris.update(step["source_uri"] for collection in collections for step in collection)
    else:
        treatment = payload["treatment"]
        label_uris.add(treatment["source_uri"])
        expected_member_id = treatment["local_id"] if treatment["step_kind"] == "action" else None
        if not any(
            binding["uri"] == treatment["source_uri"]
            and binding["member_type"] == treatment["step_kind"]
            and binding["member_id"] == expected_member_id
            for binding in bindings
        ):
            raise PredictionSchemaError("Consequence treatment must match a provenance source member")
        outcome_ids = {
            binding["member_id"]
            for binding in bindings
            if binding["member_type"] == "outcome"
        }
        if label["outcome"]["outcome_id"] not in outcome_ids:
            raise PredictionSchemaError("Consequence outcome label must match its provenance member")
    if not label_uris <= binding_uris:
        raise PredictionSchemaError("prediction label sources must be closed over provenance bindings")


__all__ = ["validate_identity", "validate_kind", "validate_source_closure"]
