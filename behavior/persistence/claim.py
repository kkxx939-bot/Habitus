"""SQLite-backed append-only Claim Ledger 与原子处理发布。"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from datetime import datetime

from behavior._validation import (
    decode_cursor,
    encode_cursor,
    identifier,
    non_negative_int,
    sha256_digest,
    strict_utc,
    utc_text,
)
from behavior.claim.ledger import ClaimPage
from behavior.claim.model import (
    BehaviorClaim,
    BehaviorClaimLedgerEntry,
    DerivationClass,
    claim_to_dict,
    source_epistemic_class,
)
from behavior.claim.receipt import (
    AttemptStatus,
    ClaimNormalizationAttempt,
    ClaimNormalizationReceipt,
    ReceiptStatus,
    attempt_to_dict,
    normalization_receipt_to_dict,
)
from behavior.errors import (
    BehaviorClaimCapacityError,
    BehaviorClaimConflictError,
    BehaviorStoreError,
    ClaimNormalizationConflictError,
)
from behavior.persistence.codecs import (
    decode_attempt,
    decode_claim,
    decode_evidence_record,
    decode_normalization_receipt,
    encode_value,
)
from behavior.persistence.database import BehaviorDatabase
from foundation.integrity import canonical_digest


class SQLiteBehaviorClaimLedger:
    def __init__(self, database: BehaviorDatabase) -> None:
        if not isinstance(database, BehaviorDatabase):
            raise TypeError("database must be BehaviorDatabase")
        self.database = database
        self.config = database.config

    def publish_route(
        self,
        attempt: ClaimNormalizationAttempt | None,
        claims: tuple[BehaviorClaim, ...],
        receipt: ClaimNormalizationReceipt,
    ) -> tuple[ClaimNormalizationReceipt, bool]:
        if attempt is not None and not isinstance(attempt, ClaimNormalizationAttempt):
            raise TypeError("attempt must be ClaimNormalizationAttempt or None")
        if not isinstance(claims, tuple) or any(not isinstance(item, BehaviorClaim) for item in claims):
            raise TypeError("claims must contain BehaviorClaim values")
        if not isinstance(receipt, ClaimNormalizationReceipt):
            raise TypeError("receipt must be ClaimNormalizationReceipt")
        if attempt is None and receipt.attempt_ids:
            raise ValueError("Receipt cannot contain Attempts when none are published")
        if attempt is not None and (
            not receipt.attempt_ids or receipt.attempt_ids[-1] != attempt.attempt_id
        ):
            raise ValueError("Receipt must end with the published Attempt")
        if receipt.claim_ids != tuple(claim.claim_id for claim in claims):
            raise ValueError("Receipt Claim members disagree with publication")
        if attempt is None and receipt.status is not ReceiptStatus.NO_CORE_REQUIRED:
            raise ValueError("only NO_CORE_REQUIRED can publish without an Attempt")
        if attempt is not None:
            if attempt.processing_identity != receipt.processing_identity:
                raise ValueError("Attempt and Receipt processing identities disagree")
            if attempt.evidence_record_id != receipt.evidence_record_id:
                raise ValueError("Attempt and Receipt Evidence identities disagree")
            if attempt.lane is not receipt.lane:
                raise ValueError("Attempt and Receipt lanes disagree")
            if attempt.normalizer_fingerprint != receipt.normalizer_fingerprint:
                raise ValueError("Attempt and Receipt Normalizer fingerprints disagree")
            if attempt.status not in {AttemptStatus.COMPLETED, AttemptStatus.ABSTAINED}:
                raise ValueError("successful publication requires a completed or abstained Attempt")
            if attempt.claim_count != len(claims):
                raise ValueError("Attempt Claim count disagrees with publication")
            if (attempt.status is AttemptStatus.ABSTAINED) != (
                receipt.status is ReceiptStatus.ABSTAINED
            ):
                raise ValueError("Attempt and Receipt statuses disagree")
        try:
            with self.database.connection.write() as connection:
                prior = self._read_receipt_row(connection, receipt.processing_identity)
                if prior is not None:
                    existing = self._decode_receipt(connection, prior)
                    if existing.content_digest != receipt.content_digest:
                        raise ClaimNormalizationConflictError("processing Receipt replay changed content")
                    self._validate_route_replay(connection, attempt, claims)
                    return existing, True
                prior_attempts = connection.execute(
                    "SELECT attempt_id FROM claim_normalization_attempts "
                    "WHERE processing_identity=? ORDER BY attempt_number ASC",
                    (receipt.processing_identity,),
                ).fetchall()
                expected_attempt_ids = tuple(row[0] for row in prior_attempts) + (
                    () if attempt is None else (attempt.attempt_id,)
                )
                if receipt.attempt_ids != expected_attempt_ids:
                    raise ClaimNormalizationConflictError(
                        "Receipt Attempt history disagrees with durable Attempts"
                    )
                self._validate_evidence_binding(connection, receipt.evidence_record_id, claims)
                self._require_capacities(connection, len(claims), 0 if attempt is None else 1, 1)
                encoded_values: list[tuple[str, str]] = []
                encoded_attempt = None
                if attempt is not None:
                    encoded_attempt = encode_value(attempt, attempt_to_dict)
                    encoded_values.append(encoded_attempt)
                encoded_claims = [encode_value(claim, claim_to_dict) for claim in claims]
                encoded_values.extend(encoded_claims)
                encoded_receipt = encode_value(receipt, normalization_receipt_to_dict)
                encoded_values.append(encoded_receipt)
                self._require_encoded_sizes(encoded_values)
                expected_bytes = sum(len(text.encode("utf-8")) for text, _ in encoded_values)
                btree_writes = (
                    2
                    + (0 if attempt is None else 5)
                    + 7 * len(claims)
                    + 4 * (len(receipt.attempt_ids) + len(receipt.claim_ids))
                )
                if self.database.connection.projected_write_size(
                    connection,
                    encoded_bytes=expected_bytes,
                    btree_writes=btree_writes,
                ) > self.config.store.max_database_bytes:
                    raise BehaviorClaimCapacityError("Claim publication exceeds database capacity")
                if attempt is not None and encoded_attempt is not None:
                    self._insert_attempt(connection, attempt, encoded_attempt)
                for claim, encoded in zip(claims, encoded_claims, strict=True):
                    self._insert_claim(connection, claim, encoded)
                self._insert_receipt(connection, receipt, encoded_receipt)
                self._insert_members(connection, receipt)
                if self.database.connection.database_size(connection) > self.config.store.max_database_bytes:
                    raise BehaviorClaimCapacityError("Claim publication exceeds database capacity")
                return receipt, False
        except sqlite3.IntegrityError as exc:
            raise ClaimNormalizationConflictError("Claim route publication uniqueness conflict") from exc

    def publish_failed_attempt(
        self,
        attempt: ClaimNormalizationAttempt,
    ) -> ClaimNormalizationAttempt:
        if not isinstance(attempt, ClaimNormalizationAttempt):
            raise TypeError("attempt must be ClaimNormalizationAttempt")
        if attempt.status not in {
            AttemptStatus.FAILED_RETRYABLE,
            AttemptStatus.FAILED_NON_RETRYABLE,
            AttemptStatus.FAILED_POLICY,
        }:
            raise ValueError("publish_failed_attempt requires a failed Attempt")
        try:
            with self.database.connection.write() as connection:
                row = connection.execute(
                    "SELECT * FROM claim_normalization_attempts WHERE attempt_id=?",
                    (attempt.attempt_id,),
                ).fetchone()
                if row is not None:
                    existing = self._decode_attempt_row(row)
                    if existing.content_digest != attempt.content_digest:
                        raise ClaimNormalizationConflictError("failed Attempt replay changed content")
                    return existing
                self._validate_evidence_binding(connection, attempt.evidence_record_id, ())
                self._require_capacities(connection, 0, 1, 0)
                encoded = encode_value(attempt, attempt_to_dict)
                self._require_encoded_sizes([encoded])
                if self.database.connection.projected_write_size(
                    connection,
                    encoded_bytes=len(encoded[0].encode("utf-8")),
                    btree_writes=5,
                ) > self.config.store.max_database_bytes:
                    raise BehaviorClaimCapacityError("failed Attempt exceeds database capacity")
                self._insert_attempt(connection, attempt, encoded)
                if self.database.connection.database_size(connection) > self.config.store.max_database_bytes:
                    raise BehaviorClaimCapacityError("failed Attempt exceeds database capacity")
                return attempt
        except sqlite3.IntegrityError as exc:
            raise ClaimNormalizationConflictError("failed Attempt uniqueness conflict") from exc

    def read_claim(self, claim_id: str) -> BehaviorClaim | None:
        entry = self.read_claim_entry(claim_id)
        return None if entry is None else entry.claim

    def read_claim_entry(self, claim_id: str) -> BehaviorClaimLedgerEntry | None:
        resolved = identifier(claim_id, "claim_id")
        with self.database.connection.read() as connection:
            row = connection.execute(
                "SELECT * FROM behavior_claims WHERE claim_id=?",
                (resolved,),
            ).fetchone()
            return None if row is None else self._claim_entry(row)

    def list_after_sequence(self, sequence: int, limit: int) -> tuple[BehaviorClaimLedgerEntry, ...]:
        after = non_negative_int(sequence, "claim_sequence")
        bounded = self._limit(limit)
        with self.database.connection.read() as connection:
            rows = connection.execute(
                "SELECT * FROM behavior_claims WHERE claim_sequence>? ORDER BY claim_sequence ASC LIMIT ?",
                (after, bounded),
            ).fetchall()
            return tuple(self._claim_entry(row) for row in rows)

    def list_for_evidence(
        self,
        evidence_record_id: str,
        limit: int,
        cursor: str | None = None,
    ) -> ClaimPage:
        evidence_id = identifier(evidence_record_id, "evidence_record_id")
        query = {"evidence_record_id": evidence_id}
        return self._page(
            "evidence",
            query,
            "evidence_record_id=?",
            (evidence_id,),
            limit,
            cursor,
        )

    def list_by_event_time(
        self,
        start: datetime,
        end: datetime,
        limit: int,
        cursor: str | None = None,
    ) -> ClaimPage:
        start_value = strict_utc(start, "start")
        end_value = strict_utc(end, "end")
        if end_value < start_value:
            raise ValueError("end cannot precede start")
        query = {"end": utc_text(end_value), "start": utc_text(start_value)}
        return self._page(
            "event_time",
            query,
            "time_start<=? AND time_end>=?",
            (utc_text(end_value), utc_text(start_value)),
            limit,
            cursor,
        )

    def list_by_semantic_fingerprint(
        self,
        fingerprint: str,
        limit: int,
        cursor: str | None = None,
    ) -> ClaimPage:
        resolved = sha256_digest(fingerprint, "semantic_fingerprint")
        query = {"semantic_fingerprint": resolved}
        return self._page(
            "semantic_fingerprint",
            query,
            "semantic_fingerprint=?",
            (resolved,),
            limit,
            cursor,
        )

    def read_attempt(self, attempt_id: str) -> ClaimNormalizationAttempt | None:
        resolved = identifier(attempt_id, "attempt_id")
        with self.database.connection.read() as connection:
            row = connection.execute(
                "SELECT * FROM claim_normalization_attempts WHERE attempt_id=?",
                (resolved,),
            ).fetchone()
            return None if row is None else self._decode_attempt_row(row)

    def read_latest_attempt(self, processing_identity: str) -> ClaimNormalizationAttempt | None:
        identity = sha256_digest(processing_identity, "processing_identity")
        with self.database.connection.read() as connection:
            row = connection.execute(
                "SELECT * FROM claim_normalization_attempts WHERE processing_identity=? "
                "ORDER BY attempt_number DESC LIMIT 1",
                (identity,),
            ).fetchone()
            return None if row is None else self._decode_attempt_row(row)

    def read_receipt(self, processing_identity: str) -> ClaimNormalizationReceipt | None:
        identity = sha256_digest(processing_identity, "processing_identity")
        with self.database.connection.read() as connection:
            row = self._read_receipt_row(connection, identity)
            return None if row is None else self._decode_receipt(connection, row)

    @staticmethod
    def _validate_evidence_binding(
        connection: sqlite3.Connection,
        evidence_record_id: str,
        claims: tuple[BehaviorClaim, ...],
    ) -> None:
        row = connection.execute(
            "SELECT * FROM behavior_evidence_records WHERE evidence_record_id=?",
            (evidence_record_id,),
        ).fetchone()
        if row is None:
            raise BehaviorClaimConflictError("Claim publication references missing Evidence")
        record = decode_evidence_record(row["content_json"], row["encoded_digest"])
        if (
            row["evidence_record_id"] != record.evidence_record_id
            or row["content_digest"] != record.content_digest
        ):
            raise BehaviorStoreError("Evidence binding projection disagrees with canonical content")
        content = record.semantic_content
        for claim in claims:
            expected_effective = (
                content.source_confidence
                if claim.derivation_class is DerivationClass.DETERMINISTIC
                else min(content.source_confidence, claim.normalizer_confidence)
            )
            if (
                claim.evidence_record_id != evidence_record_id
                or claim.evidence_record_digest != record.content_digest
                or claim.subject_role is not content.subject_role
                or claim.actor_role is not content.actor_role
                or claim.time_start != content.event_time_start
                or claim.time_end != content.event_time_end
                or claim.time_uncertainty_ms != content.event_time_uncertainty_ms
                or claim.source_epistemic_class is not source_epistemic_class(record.source_trust)
                or claim.source_confidence != content.source_confidence
                or abs(claim.effective_confidence - expected_effective) > 1e-12
            ):
                raise BehaviorClaimConflictError("Claim is not bound to the published Evidence")

    def _insert_claim(
        self,
        connection: sqlite3.Connection,
        claim: BehaviorClaim,
        encoded: tuple[str, str],
    ) -> None:
        prior = connection.execute(
            "SELECT * FROM behavior_claims WHERE claim_id=?",
            (claim.claim_id,),
        ).fetchone()
        if prior is not None:
            existing = self._claim_entry(prior).claim
            if existing.content_digest != claim.content_digest:
                raise BehaviorClaimConflictError("Claim identity replay changed content")
            return
        connection.execute(
            """INSERT INTO behavior_claims(
                claim_id, evidence_record_id, claim_kind, subject_role, actor_role, semantic_family,
                predicate, time_start, time_end, derivation_class, source_epistemic_class,
                effective_confidence, semantic_fingerprint, created_at, content_digest,
                encoded_digest, content_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                claim.claim_id,
                claim.evidence_record_id,
                claim.claim_kind.value,
                claim.subject_role.value,
                None if claim.actor_role is None else claim.actor_role.value,
                claim.semantic_family,
                claim.predicate,
                utc_text(claim.time_start),
                utc_text(claim.time_end),
                claim.derivation_class.value,
                claim.source_epistemic_class.value,
                claim.effective_confidence,
                claim.semantic_fingerprint,
                utc_text(claim.created_at),
                claim.content_digest,
                encoded[1],
                encoded[0],
            ),
        )

    def _validate_route_replay(
        self,
        connection: sqlite3.Connection,
        attempt: ClaimNormalizationAttempt | None,
        claims: tuple[BehaviorClaim, ...],
    ) -> None:
        if attempt is not None:
            row = connection.execute(
                "SELECT * FROM claim_normalization_attempts WHERE attempt_id=?",
                (attempt.attempt_id,),
            ).fetchone()
            if row is None or self._decode_attempt_row(row).content_digest != attempt.content_digest:
                raise ClaimNormalizationConflictError("processing Attempt replay changed content")
        for claim in claims:
            row = connection.execute(
                "SELECT * FROM behavior_claims WHERE claim_id=?",
                (claim.claim_id,),
            ).fetchone()
            if row is None or self._claim_entry(row).claim.content_digest != claim.content_digest:
                raise BehaviorClaimConflictError("Claim identity replay changed content")

    @staticmethod
    def _insert_attempt(
        connection: sqlite3.Connection,
        attempt: ClaimNormalizationAttempt,
        encoded: tuple[str, str],
    ) -> None:
        connection.execute(
            """INSERT INTO claim_normalization_attempts(
                attempt_id, processing_identity, evidence_record_id, normalizer_name,
                normalizer_fingerprint, lane, attempt_number, status, retryable, completed_at,
                content_digest, encoded_digest, content_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                attempt.attempt_id,
                attempt.processing_identity,
                attempt.evidence_record_id,
                attempt.normalizer_name,
                attempt.normalizer_fingerprint,
                attempt.lane.value,
                attempt.attempt_number,
                attempt.status.value,
                int(attempt.retryable),
                utc_text(attempt.completed_at),
                attempt.content_digest,
                encoded[1],
                encoded[0],
            ),
        )

    @staticmethod
    def _insert_receipt(
        connection: sqlite3.Connection,
        receipt: ClaimNormalizationReceipt,
        encoded: tuple[str, str],
    ) -> None:
        connection.execute(
            """INSERT INTO claim_normalization_receipts(
                processing_identity, evidence_record_id, lane, normalizer_fingerprint,
                planner_policy_digest, compatibility_policy_digest, binding_policy_digest,
                confidence_policy_digest, status, completed_at, publication_recorded_at,
                content_digest, encoded_digest, content_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                receipt.processing_identity,
                receipt.evidence_record_id,
                receipt.lane.value,
                receipt.normalizer_fingerprint,
                receipt.planner_policy_digest,
                receipt.compatibility_policy_digest,
                receipt.binding_policy_digest,
                receipt.confidence_policy_digest,
                receipt.status.value,
                utc_text(receipt.completed_at),
                utc_text(receipt.publication_recorded_at),
                receipt.content_digest,
                encoded[1],
                encoded[0],
            ),
        )

    @staticmethod
    def _insert_members(connection: sqlite3.Connection, receipt: ClaimNormalizationReceipt) -> None:
        for order, attempt_id in enumerate(receipt.attempt_ids):
            connection.execute(
                "INSERT INTO claim_receipt_members VALUES (?, 'ATTEMPT', ?, ?, NULL)",
                (receipt.processing_identity, order, attempt_id),
            )
        for order, claim_id in enumerate(receipt.claim_ids):
            connection.execute(
                "INSERT INTO claim_receipt_members VALUES (?, 'CLAIM', ?, NULL, ?)",
                (receipt.processing_identity, order, claim_id),
            )

    def _claim_entry(self, row: sqlite3.Row) -> BehaviorClaimLedgerEntry:
        claim = decode_claim(row["content_json"], row["encoded_digest"])
        indexed = {
            "claim_id": claim.claim_id,
            "evidence_record_id": claim.evidence_record_id,
            "claim_kind": claim.claim_kind.value,
            "subject_role": claim.subject_role.value,
            "actor_role": None if claim.actor_role is None else claim.actor_role.value,
            "semantic_family": claim.semantic_family,
            "predicate": claim.predicate,
            "time_start": utc_text(claim.time_start),
            "time_end": utc_text(claim.time_end),
            "derivation_class": claim.derivation_class.value,
            "source_epistemic_class": claim.source_epistemic_class.value,
            "semantic_fingerprint": claim.semantic_fingerprint,
            "created_at": utc_text(claim.created_at),
            "content_digest": claim.content_digest,
        }
        for name, value in indexed.items():
            if row[name] != value:
                raise BehaviorStoreError("Claim indexed column disagrees with canonical content")
        if abs(float(row["effective_confidence"]) - claim.effective_confidence) > 1e-12:
            raise BehaviorStoreError("Claim confidence projection disagrees with canonical content")
        return BehaviorClaimLedgerEntry(int(row["claim_sequence"]), claim)

    @staticmethod
    def _decode_attempt_row(row: sqlite3.Row) -> ClaimNormalizationAttempt:
        attempt = decode_attempt(row["content_json"], row["encoded_digest"])
        indexed = {
            "attempt_id": attempt.attempt_id,
            "processing_identity": attempt.processing_identity,
            "evidence_record_id": attempt.evidence_record_id,
            "normalizer_name": attempt.normalizer_name,
            "normalizer_fingerprint": attempt.normalizer_fingerprint,
            "lane": attempt.lane.value,
            "attempt_number": attempt.attempt_number,
            "status": attempt.status.value,
            "retryable": int(attempt.retryable),
            "completed_at": utc_text(attempt.completed_at),
            "content_digest": attempt.content_digest,
        }
        for name, value in indexed.items():
            if row[name] != value:
                raise BehaviorStoreError("Normalization Attempt index disagrees with canonical content")
        return attempt

    def _decode_receipt(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row,
    ) -> ClaimNormalizationReceipt:
        receipt = decode_normalization_receipt(row["content_json"], row["encoded_digest"])
        indexed = {
            "processing_identity": receipt.processing_identity,
            "evidence_record_id": receipt.evidence_record_id,
            "lane": receipt.lane.value,
            "normalizer_fingerprint": receipt.normalizer_fingerprint,
            "planner_policy_digest": receipt.planner_policy_digest,
            "compatibility_policy_digest": receipt.compatibility_policy_digest,
            "binding_policy_digest": receipt.binding_policy_digest,
            "confidence_policy_digest": receipt.confidence_policy_digest,
            "status": receipt.status.value,
            "completed_at": utc_text(receipt.completed_at),
            "publication_recorded_at": utc_text(receipt.publication_recorded_at),
            "content_digest": receipt.content_digest,
        }
        for name, value in indexed.items():
            if row[name] != value:
                raise BehaviorStoreError("Normalization Receipt index disagrees with canonical content")
        attempt_rows = connection.execute(
            "SELECT attempt.* FROM claim_receipt_members AS member "
            "JOIN claim_normalization_attempts AS attempt ON attempt.attempt_id=member.attempt_id "
            "WHERE member.processing_identity=? AND member.member_kind='ATTEMPT' "
            "ORDER BY member.member_order",
            (receipt.processing_identity,),
        ).fetchall()
        claim_rows = connection.execute(
            "SELECT claim.* FROM claim_receipt_members AS member "
            "JOIN behavior_claims AS claim ON claim.claim_id=member.claim_id "
            "WHERE member.processing_identity=? AND member.member_kind='CLAIM' "
            "ORDER BY member.member_order",
            (receipt.processing_identity,),
        ).fetchall()
        attempts = tuple(self._decode_attempt_row(item) for item in attempt_rows)
        claims = tuple(self._claim_entry(item).claim for item in claim_rows)
        if tuple(item.attempt_id for item in attempts) != receipt.attempt_ids:
            raise BehaviorStoreError("Normalization Receipt Attempt members disagree")
        if tuple(item.claim_id for item in claims) != receipt.claim_ids:
            raise BehaviorStoreError("Normalization Receipt Claim members disagree")
        if any(
            item.processing_identity != receipt.processing_identity
            or item.evidence_record_id != receipt.evidence_record_id
            or item.lane is not receipt.lane
            or item.normalizer_fingerprint != receipt.normalizer_fingerprint
            for item in attempts
        ):
            raise BehaviorStoreError("Normalization Receipt Attempt binding disagrees")
        if any(item.evidence_record_id != receipt.evidence_record_id for item in claims):
            raise BehaviorStoreError("Normalization Receipt Claim binding disagrees")
        return receipt

    def _page(
        self,
        kind: str,
        query: Mapping[str, object],
        where: str,
        parameters: tuple[object, ...],
        limit: int,
        cursor: str | None,
    ) -> ClaimPage:
        bounded = self._limit(limit)
        sql = (
            f"SELECT * FROM behavior_claims WHERE {where} AND claim_sequence>? "
            "ORDER BY claim_sequence ASC LIMIT ?"
        )
        with self.database.connection.read() as connection:
            after = self._cursor(connection, kind, query, where, parameters, cursor)
            rows = connection.execute(sql, (*parameters, after, bounded + 1)).fetchall()
            visible = rows[:bounded]
            entries = tuple(self._claim_entry(row) for row in visible)
            next_cursor = None
            if len(rows) > bounded and entries:
                next_cursor = encode_cursor(
                    {
                        "kind": kind,
                        "query_digest": canonical_digest(query),
                        "sequence": entries[-1].sequence,
                    }
                )
            return entries, next_cursor

    @staticmethod
    def _cursor(
        connection: sqlite3.Connection,
        kind: str,
        query: Mapping[str, object],
        where: str,
        parameters: tuple[object, ...],
        cursor: str | None,
    ) -> int:
        if cursor is None:
            return 0
        data = decode_cursor(cursor)
        if data.get("kind") != kind or data.get("query_digest") != canonical_digest(query):
            raise ValueError("cursor does not belong to this Claim query")
        sequence = non_negative_int(data.get("sequence"), "cursor.sequence")
        matches = int(
            connection.execute(
                f"SELECT COUNT(*) FROM behavior_claims WHERE {where} AND claim_sequence=?",
                (*parameters, sequence),
            ).fetchone()[0]
        )
        if matches != 1:
            raise ValueError("cursor Claim entry is missing from this query")
        return sequence

    def _limit(self, limit: int) -> int:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= self.config.normalization.max_query_limit:
            raise ValueError("query limit exceeds the configured Claim boundary")
        return limit

    def _require_capacities(
        self,
        connection: sqlite3.Connection,
        claims: int,
        attempts: int,
        receipts: int,
    ) -> None:
        current_claims = int(connection.execute("SELECT COUNT(*) FROM behavior_claims").fetchone()[0])
        current_attempts = int(
            connection.execute("SELECT COUNT(*) FROM claim_normalization_attempts").fetchone()[0]
        )
        current_receipts = int(
            connection.execute("SELECT COUNT(*) FROM claim_normalization_receipts").fetchone()[0]
        )
        if current_claims + claims > self.config.store.max_claims:
            raise BehaviorClaimCapacityError("Claim capacity is exhausted")
        if current_attempts + attempts > self.config.store.max_normalization_attempts:
            raise BehaviorClaimCapacityError("Normalization Attempt capacity is exhausted")
        if current_receipts + receipts > self.config.store.max_normalization_receipts:
            raise BehaviorClaimCapacityError("Normalization Receipt capacity is exhausted")

    def _require_encoded_sizes(self, values: list[tuple[str, str]]) -> None:
        if any(len(text.encode("utf-8")) > self.config.store.max_json_bytes for text, _ in values):
            raise BehaviorClaimCapacityError("Claim durable JSON exceeds configured boundary")

    @staticmethod
    def _read_receipt_row(connection: sqlite3.Connection, identity: str) -> sqlite3.Row | None:
        return connection.execute(
            "SELECT * FROM claim_normalization_receipts WHERE processing_identity=?",
            (identity,),
        ).fetchone()


__all__ = ["SQLiteBehaviorClaimLedger"]
