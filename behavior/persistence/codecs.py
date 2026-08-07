"""Behavior 耐久对象的规范 JSON 编解码与 Digest 回读校验。"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from behavior._validation import parse_utc, strict_object
from behavior.claim.model import (
    BehaviorClaim,
    DerivationClass,
    SourceEpistemicClass,
    claim_to_dict,
)
from behavior.claim.planner import NormalizationLane
from behavior.claim.proposal import ClaimKind
from behavior.claim.receipt import (
    AttemptStatus,
    ClaimNormalizationAttempt,
    ClaimNormalizationReceipt,
    ReceiptStatus,
    attempt_to_dict,
    normalization_receipt_to_dict,
)
from behavior.errors import BehaviorStoreError
from behavior.evidence.content import BehaviorRole, content_from_dict
from behavior.evidence.ingress import (
    BehaviorEvidenceIngressReceipt,
    IngressReceiptStatus,
    ingress_receipt_to_dict,
)
from behavior.evidence.provenance import provenance_from_dict
from behavior.evidence.record import BehaviorEvidenceRecord, record_to_dict
from behavior.evidence.trust import BehaviorSourceTrust
from foundation.integrity import canonical_json, text_digest

T = TypeVar("T")


def encode_value(value: object, encoder: Callable[[Any], Mapping[str, object]]) -> tuple[str, str]:
    text = canonical_json(encoder(value))
    return text, text_digest(text)


def verify_projection(
    row: sqlite3.Row,
    expected: Mapping[str, object],
    error_message: str,
) -> None:
    if any(row[name] != value for name, value in expected.items()):
        raise BehaviorStoreError(error_message)


def decode_evidence_record(text: str, encoded_digest: str) -> BehaviorEvidenceRecord:
    expected = frozenset(
        {
            "evidence_record_id",
            "semantic_content",
            "provenance",
            "source_trust",
            "ingested_at",
            "semantic_digest",
            "content_digest",
            "schema_version",
        }
    )
    data = _decode(text, encoded_digest, expected)
    record = BehaviorEvidenceRecord(
        semantic_content=content_from_dict(data["semantic_content"]),
        provenance=provenance_from_dict(data["provenance"]),
        source_trust=BehaviorSourceTrust(data["source_trust"]),
        ingested_at=parse_utc(data["ingested_at"], "evidence_record.ingested_at"),
    )
    _assert_equal(record_to_dict(record), data, "Evidence record")
    return record


def decode_ingress_receipt(text: str, encoded_digest: str) -> BehaviorEvidenceIngressReceipt:
    expected = frozenset(
        {
            "delivery_id",
            "request_digest",
            "adapter_name",
            "adapter_fingerprint",
            "capability_digest",
            "status",
            "reason_code",
            "rejected_item_indexes",
            "evidence_record_ids",
            "recorded_at",
            "content_digest",
        }
    )
    data = _decode(text, encoded_digest, expected)
    receipt = BehaviorEvidenceIngressReceipt(
        delivery_id=data["delivery_id"],
        request_digest=data["request_digest"],
        adapter_name=data["adapter_name"],
        adapter_fingerprint=data["adapter_fingerprint"],
        capability_digest=data["capability_digest"],
        status=IngressReceiptStatus(data["status"]),
        reason_code=data["reason_code"],
        rejected_item_indexes=tuple(_array(data["rejected_item_indexes"], "rejected indexes")),
        evidence_record_ids=tuple(_array(data["evidence_record_ids"], "Evidence record ids")),
        recorded_at=parse_utc(data["recorded_at"], "ingress_receipt.recorded_at"),
    )
    _assert_equal(ingress_receipt_to_dict(receipt), data, "Evidence ingress receipt")
    return receipt


def decode_claim(text: str, encoded_digest: str) -> BehaviorClaim:
    expected = frozenset(
        {
            "claim_id",
            "evidence_record_id",
            "evidence_record_digest",
            "subject_role",
            "actor_role",
            "time_start",
            "time_end",
            "time_uncertainty_ms",
            "claim_kind",
            "semantic_family",
            "predicate",
            "activity",
            "phase",
            "semantic_payload",
            "human_summary",
            "source_epistemic_class",
            "derivation_class",
            "source_confidence",
            "normalizer_confidence",
            "effective_confidence",
            "local_alternative_group_id",
            "alternative_group_key",
            "normalizer_fingerprint",
            "compatibility_policy_digest",
            "binding_policy_digest",
            "confidence_policy_digest",
            "semantic_fingerprint",
            "created_at",
            "content_digest",
            "schema_version",
        }
    )
    data = _decode(text, encoded_digest, expected)
    claim = BehaviorClaim(
        evidence_record_id=data["evidence_record_id"],
        evidence_record_digest=data["evidence_record_digest"],
        subject_role=BehaviorRole(data["subject_role"]),
        actor_role=None if data["actor_role"] is None else BehaviorRole(data["actor_role"]),
        time_start=parse_utc(data["time_start"], "claim.time_start"),
        time_end=parse_utc(data["time_end"], "claim.time_end"),
        time_uncertainty_ms=data["time_uncertainty_ms"],
        claim_kind=ClaimKind(data["claim_kind"]),
        semantic_family=data["semantic_family"],
        predicate=data["predicate"],
        activity=data["activity"],
        phase=data["phase"],
        semantic_payload=data["semantic_payload"],
        human_summary=data["human_summary"],
        source_epistemic_class=SourceEpistemicClass(data["source_epistemic_class"]),
        derivation_class=DerivationClass(data["derivation_class"]),
        source_confidence=data["source_confidence"],
        normalizer_confidence=data["normalizer_confidence"],
        effective_confidence=data["effective_confidence"],
        local_alternative_group_id=data["local_alternative_group_id"],
        alternative_group_key=data["alternative_group_key"],
        normalizer_fingerprint=data["normalizer_fingerprint"],
        compatibility_policy_digest=data["compatibility_policy_digest"],
        binding_policy_digest=data["binding_policy_digest"],
        confidence_policy_digest=data["confidence_policy_digest"],
        created_at=parse_utc(data["created_at"], "claim.created_at"),
    )
    _assert_equal(claim_to_dict(claim), data, "Claim")
    return claim


def decode_attempt(text: str, encoded_digest: str) -> ClaimNormalizationAttempt:
    expected = frozenset(
        {
            "attempt_id",
            "processing_identity",
            "evidence_record_id",
            "normalizer_name",
            "normalizer_fingerprint",
            "lane",
            "attempt_number",
            "status",
            "proposal_digest",
            "claim_count",
            "error_code",
            "retryable",
            "started_at",
            "completed_at",
            "content_digest",
        }
    )
    data = _decode(text, encoded_digest, expected)
    attempt = ClaimNormalizationAttempt(
        processing_identity=data["processing_identity"],
        evidence_record_id=data["evidence_record_id"],
        normalizer_name=data["normalizer_name"],
        normalizer_fingerprint=data["normalizer_fingerprint"],
        lane=NormalizationLane(data["lane"]),
        attempt_number=data["attempt_number"],
        status=AttemptStatus(data["status"]),
        proposal_digest=data["proposal_digest"],
        claim_count=data["claim_count"],
        error_code=data["error_code"],
        retryable=data["retryable"],
        started_at=parse_utc(data["started_at"], "attempt.started_at"),
        completed_at=parse_utc(data["completed_at"], "attempt.completed_at"),
    )
    _assert_equal(attempt_to_dict(attempt), data, "Normalization Attempt")
    return attempt


def decode_normalization_receipt(text: str, encoded_digest: str) -> ClaimNormalizationReceipt:
    expected = frozenset(
        {
            "processing_identity",
            "evidence_record_id",
            "lane",
            "normalizer_fingerprint",
            "planner_policy_digest",
            "compatibility_policy_digest",
            "binding_policy_digest",
            "confidence_policy_digest",
            "status",
            "attempt_ids",
            "claim_ids",
            "completed_at",
            "publication_recorded_at",
            "content_digest",
        }
    )
    data = _decode(text, encoded_digest, expected)
    receipt = ClaimNormalizationReceipt(
        processing_identity=data["processing_identity"],
        evidence_record_id=data["evidence_record_id"],
        lane=NormalizationLane(data["lane"]),
        normalizer_fingerprint=data["normalizer_fingerprint"],
        planner_policy_digest=data["planner_policy_digest"],
        compatibility_policy_digest=data["compatibility_policy_digest"],
        binding_policy_digest=data["binding_policy_digest"],
        confidence_policy_digest=data["confidence_policy_digest"],
        status=ReceiptStatus(data["status"]),
        attempt_ids=tuple(_array(data["attempt_ids"], "attempt ids")),
        claim_ids=tuple(_array(data["claim_ids"], "claim ids")),
        completed_at=parse_utc(data["completed_at"], "normalization_receipt.completed_at"),
        publication_recorded_at=parse_utc(
            data["publication_recorded_at"],
            "normalization_receipt.publication_recorded_at",
        ),
    )
    _assert_equal(normalization_receipt_to_dict(receipt), data, "Normalization Receipt")
    return receipt


def _decode(text: str, encoded_digest: str, expected: frozenset[str] | set[str]) -> dict[str, Any]:
    if not isinstance(text, str) or text_digest(text) != encoded_digest:
        raise BehaviorStoreError("Behavior encoded digest mismatch")
    try:
        value = json.loads(
            text,
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: _reject_constant(token),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise BehaviorStoreError("Behavior content JSON is invalid") from exc
    if canonical_json(value) != text:
        raise BehaviorStoreError("Behavior content JSON is not canonical")
    try:
        data = strict_object(value, "durable_value", frozenset(expected))
    except (TypeError, ValueError) as exc:
        raise BehaviorStoreError("Behavior durable value schema has drifted") from exc
    return data


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant: {token}")


def _assert_equal(expected: Mapping[str, object], actual: Mapping[str, object], label: str) -> None:
    if canonical_json(expected) != canonical_json(actual):
        raise BehaviorStoreError(f"{label} canonical read-back mismatch")


def _array(value: object, field_name: str) -> list[Any] | tuple[Any, ...]:
    if not isinstance(value, list | tuple):
        raise BehaviorStoreError(f"{field_name} is not an array")
    return value


__all__ = [
    "decode_attempt",
    "decode_claim",
    "decode_evidence_record",
    "decode_ingress_receipt",
    "decode_normalization_receipt",
    "encode_value",
    "verify_projection",
]
