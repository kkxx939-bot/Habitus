"""SQLite-backed append-only Claim Ledger 与原子处理发布。"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from datetime import datetime

from behavior._validation import (
    decode_sequence_cursor,
    encode_sequence_cursor,
    identifier,
    non_negative_int,
    sha256_digest,
    strict_utc,
    utc_text,
)
from behavior.claim.ledger import ClaimPage
from behavior.claim.model import BehaviorClaim, BehaviorClaimLedgerEntry, claim_to_dict
from behavior.claim.publication import ClaimPublication
from behavior.claim.receipt import (
    AttemptStatus,
    ClaimNormalizationAttempt,
    ClaimNormalizationReceipt,
    attempt_to_dict,
    normalization_receipt_to_dict,
)
from behavior.errors import (
    BehaviorClaimCapacityError,
    BehaviorClaimConflictError,
    ClaimNormalizationConflictError,
)
from behavior.evidence.record import BehaviorEvidenceRecord
from behavior.persistence.codecs import (
    decode_attempt,
    decode_claim,
    decode_evidence_record,
    decode_normalization_receipt,
    encode_value,
)
from behavior.persistence.database import BehaviorDatabase


class SQLiteBehaviorClaimLedger:
    def __init__(self, database: BehaviorDatabase) -> None:
        if not isinstance(database, BehaviorDatabase):
            raise TypeError("database must be BehaviorDatabase")
        self.database = database
        self.config = database.config

    def publish(
        self,
        publication: ClaimPublication,
    ) -> tuple[ClaimNormalizationReceipt, bool]:
        return self._publish(
            publication.attempt,
            publication.claims,
            publication.receipt,
        )

    def _publish(
        self,
        attempt: ClaimNormalizationAttempt | None,
        claims: tuple[BehaviorClaim, ...],
        receipt: ClaimNormalizationReceipt,
    ) -> tuple[ClaimNormalizationReceipt, bool]:
        try:
            with self.database.connection.write() as connection:
                prior = self._read_receipt_row(connection, receipt.processing_identity)
                if prior is not None:
                    existing = self._decode_receipt(prior)
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
                self._validate_publication_binding(connection, receipt, claims)
                new_claims = self._new_claims(connection, claims)
                self._require_capacities(connection, len(new_claims), 0 if attempt is None else 1, 1)
                encoded_values: list[tuple[str, str]] = []
                encoded_attempt = None
                if attempt is not None:
                    encoded_attempt = encode_value(attempt, attempt_to_dict)
                    encoded_values.append(encoded_attempt)
                encoded_claims = [
                    (claim, encode_value(claim, claim_to_dict))
                    for claim in new_claims
                ]
                encoded_values.extend(encoded for _, encoded in encoded_claims)
                encoded_receipt = encode_value(receipt, normalization_receipt_to_dict)
                encoded_values.append(encoded_receipt)
                self._require_encoded_sizes(encoded_values)
                expected_bytes = sum(len(text.encode("utf-8")) for text, _ in encoded_values)
                btree_writes = (
                    2
                    + (0 if attempt is None else 5)
                    + 7 * len(new_claims)
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
                for claim, encoded in encoded_claims:
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
                self._require_evidence(connection, attempt.evidence_record_id)
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
            return None if row is None else self._decode_receipt(row)

    @staticmethod
    def _validate_publication_binding(
        connection: sqlite3.Connection,
        receipt: ClaimNormalizationReceipt,
        claims: tuple[BehaviorClaim, ...],
    ) -> None:
        record = SQLiteBehaviorClaimLedger._require_evidence(
            connection,
            receipt.evidence_record_id,
        )
        for claim in claims:
            if (
                claim.evidence_record_id != receipt.evidence_record_id
                or claim.evidence_record_digest != record.content_digest
            ):
                raise BehaviorClaimConflictError(
                    "Claim Evidence binding disagrees with durable Evidence"
                )

    @staticmethod
    def _require_evidence(
        connection: sqlite3.Connection,
        evidence_record_id: str,
    ) -> BehaviorEvidenceRecord:
        row = connection.execute(
            "SELECT * FROM behavior_evidence_records WHERE evidence_record_id=?",
            (evidence_record_id,),
        ).fetchone()
        if row is None:
            raise BehaviorClaimConflictError("Claim publication references missing Evidence")
        return decode_evidence_record(row["content_json"], row["encoded_digest"])

    def _new_claims(
        self,
        connection: sqlite3.Connection,
        claims: tuple[BehaviorClaim, ...],
    ) -> tuple[BehaviorClaim, ...]:
        new: list[BehaviorClaim] = []
        for claim in claims:
            row = connection.execute(
                "SELECT * FROM behavior_claims WHERE claim_id=?",
                (claim.claim_id,),
            ).fetchone()
            if row is None:
                new.append(claim)
            elif self._claim_entry(row).claim.content_digest != claim.content_digest:
                raise BehaviorClaimConflictError("Claim identity replay changed content")
        return tuple(new)

    def _insert_claim(
        self,
        connection: sqlite3.Connection,
        claim: BehaviorClaim,
        encoded: tuple[str, str],
    ) -> None:
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
        return BehaviorClaimLedgerEntry(int(row["claim_sequence"]), claim)

    @staticmethod
    def _decode_attempt_row(row: sqlite3.Row) -> ClaimNormalizationAttempt:
        return decode_attempt(row["content_json"], row["encoded_digest"])

    @staticmethod
    def _decode_receipt(row: sqlite3.Row) -> ClaimNormalizationReceipt:
        return decode_normalization_receipt(row["content_json"], row["encoded_digest"])

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
                next_cursor = encode_sequence_cursor(kind, query, entries[-1].sequence)
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
        sequence = decode_sequence_cursor(
            cursor,
            kind=kind,
            query=query,
            subject="Claim",
        )
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
        current_claims, current_attempts, current_receipts = connection.execute(
            "SELECT (SELECT COUNT(*) FROM behavior_claims), "
            "(SELECT COUNT(*) FROM claim_normalization_attempts), "
            "(SELECT COUNT(*) FROM claim_normalization_receipts)"
        ).fetchone()
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
