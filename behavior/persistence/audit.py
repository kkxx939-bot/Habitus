"""Behavior SQLite 的显式深度完整性审计。"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any

from behavior.claim.model import DerivationClass, claim_to_dict, source_epistemic_class
from behavior.claim.planner import NormalizationLane
from behavior.claim.publication import ProcessingIdentity
from behavior.claim.receipt import attempt_to_dict, normalization_receipt_to_dict
from behavior.errors import BehaviorStoreError
from behavior.evidence.ingress import IngressReceiptStatus, ingress_receipt_to_dict
from behavior.evidence.record import record_to_dict
from behavior.persistence.codecs import (
    decode_attempt,
    decode_claim,
    decode_evidence_record,
    decode_ingress_receipt,
    decode_normalization_receipt,
    verify_projection,
)
from behavior.persistence.database import BehaviorDatabase
from behavior.persistence.schema import validate_schema

_EVIDENCE_COLUMNS = {
    "evidence_record_id": ("evidence_record_id",),
    "producer_fingerprint": ("provenance", "producer_fingerprint", "digest"),
    "capability_digest": ("provenance", "capability_digest"),
    "source_trust": ("source_trust",),
    "origin_kind": ("provenance", "descriptor", "origin_kind"),
    "source_event_namespace": ("provenance", "descriptor", "source_event_ref", "namespace"),
    "source_event_value": ("provenance", "descriptor", "source_event_ref", "value"),
    "source_item_index": ("provenance", "descriptor", "source_item_index"),
    "stream_namespace": ("provenance", "descriptor", "stream_ref", "namespace"),
    "stream_value": ("provenance", "descriptor", "stream_ref", "value"),
    "stream_generation": ("provenance", "descriptor", "stream_ref", "generation"),
    "source_sequence": ("provenance", "descriptor", "source_sequence"),
    "record_kind": ("semantic_content", "record_kind"),
    "subject_role": ("semantic_content", "subject_role"),
    "actor_role": ("semantic_content", "actor_role"),
    "event_time_start": ("semantic_content", "event_time_start"),
    "event_time_end": ("semantic_content", "event_time_end"),
    "ingested_at": ("ingested_at",),
    "semantic_digest": ("semantic_digest",),
    "content_digest": ("content_digest",),
}
_CLAIM_COLUMNS = (
    "claim_id", "evidence_record_id", "claim_kind", "subject_role", "actor_role",
    "semantic_family", "predicate", "time_start", "time_end", "derivation_class",
    "source_epistemic_class", "effective_confidence", "semantic_fingerprint", "created_at",
    "content_digest",
)
_ATTEMPT_COLUMNS = (
    "attempt_id", "processing_identity", "evidence_record_id", "normalizer_name",
    "normalizer_fingerprint", "lane", "attempt_number", "status", "retryable",
    "completed_at", "content_digest",
)
_INGRESS_COLUMNS = (
    "delivery_id", "request_digest", "adapter_name", "adapter_fingerprint",
    "capability_digest", "status", "recorded_at", "content_digest",
)
_RECEIPT_COLUMNS = (
    "processing_identity", "evidence_record_id", "lane", "normalizer_fingerprint",
    "planner_policy_digest", "compatibility_policy_digest", "binding_policy_digest",
    "confidence_policy_digest", "status", "completed_at", "publication_recorded_at",
    "content_digest",
)


@dataclass(frozen=True, slots=True)
class BehaviorAuditReport:
    evidence_count: int
    ingress_receipt_count: int
    claim_count: int
    attempt_count: int
    normalization_receipt_count: int


class BehaviorAuditService:
    """按需检查索引投影、成员表和跨对象摘要链。"""

    def __init__(self, database: BehaviorDatabase) -> None:
        self.database = database

    def deep_check(self) -> BehaviorAuditReport:
        with self.database.connection.read() as connection:
            validate_schema(connection)
            counts = (
                self._evidence(connection), self._ingress(connection), self._claims(connection),
                self._attempts(connection), self._receipts(connection),
            )
        return BehaviorAuditReport(*counts)

    @staticmethod
    def _evidence(connection: sqlite3.Connection) -> int:
        rows = connection.execute("SELECT * FROM behavior_evidence_records ORDER BY evidence_sequence").fetchall()
        for row in rows:
            record = decode_evidence_record(row["content_json"], row["encoded_digest"])
            data = record_to_dict(record)
            verify_projection(row, {key: _path(data, path) for key, path in _EVIDENCE_COLUMNS.items()}, "Evidence indexed column disagrees with canonical content")
            descriptor = record.provenance.descriptor
            _members(connection, "behavior_evidence_correlations", record.evidence_record_id,
                     ("namespace", "value", "root_value"),
                     [(item.namespace, item.value, item.root_value) for item in descriptor.correlation_refs],
                     "Evidence correlation members disagree with canonical content")
            _members(connection, "behavior_evidence_causal_refs", record.evidence_record_id,
                     ("kind", "reference", "reference_digest"),
                     [(item.kind.value, item.reference, item.reference_digest) for item in descriptor.causal_refs],
                     "Evidence causal members disagree with canonical content")
            _members(connection, "behavior_evidence_parent_sources", record.evidence_record_id,
                     ("namespace", "value", "identity_digest"),
                     [(item.namespace, item.value, item.identity_digest) for item in descriptor.parent_source_event_refs],
                     "Evidence parent members disagree with canonical content")
        return len(rows)

    @staticmethod
    def _ingress(connection: sqlite3.Connection) -> int:
        rows = connection.execute("SELECT * FROM behavior_evidence_ingress_receipts ORDER BY delivery_id").fetchall()
        for row in rows:
            receipt = decode_ingress_receipt(row["content_json"], row["encoded_digest"])
            _direct_projection(row, ingress_receipt_to_dict(receipt), _INGRESS_COLUMNS, "Evidence Receipt projection disagrees with canonical content")
            if receipt.status is IngressReceiptStatus.COMMITTED:
                found = sum(connection.execute("SELECT EXISTS(SELECT 1 FROM behavior_evidence_records WHERE evidence_record_id=?)", (record_id,)).fetchone()[0] for record_id in receipt.evidence_record_ids)
                if found != len(receipt.evidence_record_ids):
                    raise BehaviorStoreError("Evidence Receipt points to missing Evidence")
        return len(rows)

    @staticmethod
    def _claims(connection: sqlite3.Connection) -> int:
        rows = connection.execute("SELECT * FROM behavior_claims ORDER BY claim_sequence").fetchall()
        for row in rows:
            claim = decode_claim(row["content_json"], row["encoded_digest"])
            _direct_projection(row, claim_to_dict(claim), _CLAIM_COLUMNS, "Claim indexed column disagrees with canonical content")
            evidence = connection.execute("SELECT content_json, encoded_digest FROM behavior_evidence_records WHERE evidence_record_id=?", (claim.evidence_record_id,)).fetchone()
            if evidence is None:
                raise BehaviorStoreError("Claim points to missing Evidence")
            record = decode_evidence_record(evidence[0], evidence[1])
            content = record.semantic_content
            confidence = content.source_confidence if claim.derivation_class is DerivationClass.DETERMINISTIC else min(content.source_confidence, claim.normalizer_confidence)
            if (claim.evidence_record_digest != record.content_digest or claim.subject_role is not content.subject_role
                    or claim.actor_role is not content.actor_role or claim.time_start != content.event_time_start
                    or claim.time_end != content.event_time_end or claim.time_uncertainty_ms != content.event_time_uncertainty_ms
                    or claim.source_confidence != content.source_confidence or claim.effective_confidence != confidence
                    or claim.source_epistemic_class is not source_epistemic_class(record.source_trust)):
                raise BehaviorStoreError("Claim Evidence-bound fields disagree")
        return len(rows)

    @staticmethod
    def _attempts(connection: sqlite3.Connection) -> int:
        rows = connection.execute("SELECT * FROM claim_normalization_attempts ORDER BY processing_identity, attempt_number").fetchall()
        for row in rows:
            attempt = decode_attempt(row["content_json"], row["encoded_digest"])
            _direct_projection(row, attempt_to_dict(attempt), _ATTEMPT_COLUMNS, "Normalization Attempt index disagrees with canonical content")
        return len(rows)

    @staticmethod
    def _receipts(connection: sqlite3.Connection) -> int:
        rows = connection.execute("SELECT * FROM claim_normalization_receipts ORDER BY processing_identity").fetchall()
        for row in rows:
            receipt = decode_normalization_receipt(row["content_json"], row["encoded_digest"])
            _direct_projection(row, normalization_receipt_to_dict(receipt), _RECEIPT_COLUMNS, "Normalization Receipt index disagrees with canonical content")
            evidence = connection.execute("SELECT content_digest FROM behavior_evidence_records WHERE evidence_record_id=?", (receipt.evidence_record_id,)).fetchone()
            if evidence is None or ProcessingIdentity.create(
                evidence_record_digest=evidence[0], lane=receipt.lane,
                normalizer_fingerprint=receipt.normalizer_fingerprint,
                planner_policy_digest=receipt.planner_policy_digest,
                compatibility_policy_digest=receipt.compatibility_policy_digest,
                binding_policy_digest=receipt.binding_policy_digest,
                confidence_policy_digest=receipt.confidence_policy_digest,
            ).value != receipt.processing_identity:
                raise BehaviorStoreError("Normalization Processing Identity disagrees")
            _receipt_members(connection, receipt)
        return len(rows)


def _receipt_members(connection: sqlite3.Connection, receipt: Any) -> None:
    attempts = connection.execute("SELECT attempt.* FROM claim_receipt_members member JOIN claim_normalization_attempts attempt ON attempt.attempt_id=member.attempt_id WHERE member.processing_identity=? AND member.member_kind='ATTEMPT' ORDER BY member.member_order", (receipt.processing_identity,)).fetchall()
    claims = connection.execute("SELECT claim.* FROM claim_receipt_members member JOIN behavior_claims claim ON claim.claim_id=member.claim_id WHERE member.processing_identity=? AND member.member_kind='CLAIM' ORDER BY member.member_order", (receipt.processing_identity,)).fetchall()
    decoded_attempts = tuple(decode_attempt(row["content_json"], row["encoded_digest"]) for row in attempts)
    decoded_claims = tuple(decode_claim(row["content_json"], row["encoded_digest"]) for row in claims)
    if tuple(item.attempt_id for item in decoded_attempts) != receipt.attempt_ids:
        raise BehaviorStoreError("Normalization Receipt Attempt members disagree")
    if tuple(item.claim_id for item in decoded_claims) != receipt.claim_ids:
        raise BehaviorStoreError("Normalization Receipt Claim members disagree")
    if any(item.processing_identity != receipt.processing_identity or item.evidence_record_id != receipt.evidence_record_id or item.lane is not receipt.lane or item.normalizer_fingerprint != receipt.normalizer_fingerprint for item in decoded_attempts):
        raise BehaviorStoreError("Normalization Receipt Attempt binding disagrees")
    derivation = DerivationClass.DETERMINISTIC if receipt.lane is NormalizationLane.CORE else DerivationClass.MODEL
    if any(claim.evidence_record_id != receipt.evidence_record_id or claim.normalizer_fingerprint != receipt.normalizer_fingerprint or claim.compatibility_policy_digest != receipt.compatibility_policy_digest or claim.binding_policy_digest != receipt.binding_policy_digest or claim.confidence_policy_digest != receipt.confidence_policy_digest or claim.derivation_class is not derivation for claim in decoded_claims):
        raise BehaviorStoreError("Normalization Receipt Claim binding disagrees")


def _members(connection: sqlite3.Connection, table: str, record_id: str, columns: tuple[str, ...], expected: list[tuple[Any, ...]], error: str) -> None:
    rows = connection.execute(f"SELECT {', '.join(columns)} FROM {table} WHERE evidence_record_id=? ORDER BY member_order", (record_id,)).fetchall()
    if [tuple(row) for row in rows] != expected:
        raise BehaviorStoreError(error)


def _direct_projection(row: sqlite3.Row, data: dict[str, Any], columns: tuple[str, ...], error: str) -> None:
    expected = {name: int(data[name]) if name == "retryable" else data[name] for name in columns}
    verify_projection(row, expected, error)


def _path(data: dict[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = data
    for name in path:
        value = value[name]
    return value
