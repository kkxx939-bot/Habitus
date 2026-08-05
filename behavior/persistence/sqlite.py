"""Evidence & Claim Layer 的单一 SQLite 耐久 Store。"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TypeVar

from behavior._validation import identifier, parse_utc, sha256_digest, strict_utc, utc_text
from behavior.claim.admission import ClaimAdmissionDecision, ClaimAdmissionStatus
from behavior.claim.model import Claim, ClaimBatch, ClaimProcessingReceipt, ClaimProducerRun
from behavior.config import BehaviorConfig
from behavior.errors import (
    BehaviorOwnerConflictError,
    ClaimProcessingConflictError,
    ClaimStoreError,
    EvidenceManifestError,
    EvidenceWindowStateError,
    SourceRecordConflictError,
)
from behavior.evidence.manifest import EvidenceManifest
from behavior.evidence.model import (
    EvidenceSealReason,
    EvidenceWindow,
    EvidenceWindowState,
    SourceIngestResult,
    SourceIngestStatus,
)
from behavior.evidence.window import EvidenceWindowAssembler
from behavior.owner import ConfirmedOwnerBinding
from behavior.source.model import SourceRecord
from foundation.integrity import canonical_json

_SCHEMA_VERSION = "1"
_DATABASE_NAME = "evidence_claims.sqlite3"
_T = TypeVar("_T")

_TABLE_COLUMNS = {
    "behavior_metadata": ("key", "value"),
    "source_records": (
        "source_record_id",
        "stream_id",
        "source_sequence",
        "owner_binding_digest",
        "event_time_start",
        "event_time_end",
        "content_digest",
        "content_json",
    ),
    "active_evidence_windows": (
        "window_id",
        "grouping_key",
        "generation",
        "owner_binding_digest",
        "watermark",
        "content_digest",
        "content_json",
    ),
    "evidence_watermarks": (
        "grouping_key",
        "owner_binding_digest",
        "max_event_time",
        "watermark",
    ),
    "evidence_window_members": ("window_id", "source_record_id"),
    "evidence_manifests": (
        "manifest_id",
        "window_id",
        "grouping_key",
        "generation",
        "owner_binding_digest",
        "started_at",
        "ended_at",
        "sealed_at",
        "manifest_digest",
        "content_json",
    ),
    "evidence_manifest_members": ("manifest_id", "source_record_id", "member_order"),
    "claim_producer_runs": (
        "run_id",
        "processing_identity",
        "manifest_id",
        "producer_fingerprint",
        "content_digest",
        "content_json",
    ),
    "claim_batches": (
        "claim_batch_id",
        "processing_identity",
        "manifest_id",
        "producer_fingerprint",
        "created_at",
        "content_digest",
        "content_json",
    ),
    "claims": (
        "claim_id",
        "claim_batch_id",
        "manifest_id",
        "owner_binding_digest",
        "semantic_fingerprint",
        "claim_kind",
        "time_start",
        "time_end",
        "created_at",
        "content_digest",
        "content_json",
    ),
    "claim_admission_decisions": (
        "decision_id",
        "processing_identity",
        "claim_id",
        "status",
        "decided_at",
        "content_digest",
        "content_json",
    ),
    "claim_processing_receipts": (
        "processing_identity",
        "manifest_id",
        "completed_at",
        "receipt_digest",
        "content_json",
    ),
}

_REQUIRED_INDEXES = frozenset(
    {
        "idx_source_stream_sequence",
        "idx_source_event_time",
        "idx_active_grouping",
        "idx_manifest_window",
        "idx_manifest_time",
        "idx_manifest_members_source",
        "idx_producer_runs_processing",
        "idx_batches_processing",
        "idx_claims_manifest",
        "idx_claims_created",
        "idx_claims_semantic_time",
        "idx_decisions_claim_status",
        "idx_receipts_completed",
    }
)

_SCHEMA_STATEMENTS = (
    "CREATE TABLE behavior_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    """CREATE TABLE source_records (
        source_record_id TEXT PRIMARY KEY,
        stream_id TEXT NOT NULL,
        source_sequence INTEGER NOT NULL CHECK(source_sequence >= 0),
        owner_binding_digest TEXT NOT NULL,
        event_time_start TEXT NOT NULL,
        event_time_end TEXT NOT NULL,
        content_digest TEXT NOT NULL,
        content_json TEXT NOT NULL
    )""",
    "CREATE UNIQUE INDEX idx_source_stream_sequence ON source_records(stream_id, source_sequence)",
    "CREATE INDEX idx_source_event_time ON source_records(event_time_start, source_record_id)",
    """CREATE TABLE active_evidence_windows (
        window_id TEXT PRIMARY KEY,
        grouping_key TEXT NOT NULL UNIQUE,
        generation INTEGER NOT NULL CHECK(generation >= 0),
        owner_binding_digest TEXT NOT NULL,
        watermark TEXT NOT NULL,
        content_digest TEXT NOT NULL,
        content_json TEXT NOT NULL
    )""",
    "CREATE UNIQUE INDEX idx_active_grouping ON active_evidence_windows(grouping_key)",
    """CREATE TABLE evidence_watermarks (
        grouping_key TEXT PRIMARY KEY,
        owner_binding_digest TEXT NOT NULL,
        max_event_time TEXT NOT NULL,
        watermark TEXT NOT NULL
    )""",
    """CREATE TABLE evidence_window_members (
        window_id TEXT NOT NULL REFERENCES active_evidence_windows(window_id) ON DELETE CASCADE,
        source_record_id TEXT NOT NULL REFERENCES source_records(source_record_id),
        PRIMARY KEY(window_id, source_record_id),
        UNIQUE(source_record_id)
    )""",
    """CREATE TABLE evidence_manifests (
        manifest_id TEXT PRIMARY KEY,
        window_id TEXT NOT NULL UNIQUE,
        grouping_key TEXT NOT NULL,
        generation INTEGER NOT NULL CHECK(generation >= 0),
        owner_binding_digest TEXT NOT NULL,
        started_at TEXT NOT NULL,
        ended_at TEXT NOT NULL,
        sealed_at TEXT NOT NULL,
        manifest_digest TEXT NOT NULL UNIQUE,
        content_json TEXT NOT NULL
    )""",
    "CREATE UNIQUE INDEX idx_manifest_window ON evidence_manifests(window_id)",
    "CREATE INDEX idx_manifest_time ON evidence_manifests(sealed_at, manifest_id)",
    """CREATE TABLE evidence_manifest_members (
        manifest_id TEXT NOT NULL REFERENCES evidence_manifests(manifest_id),
        source_record_id TEXT NOT NULL REFERENCES source_records(source_record_id),
        member_order INTEGER NOT NULL CHECK(member_order >= 0),
        PRIMARY KEY(manifest_id, source_record_id),
        UNIQUE(manifest_id, member_order)
    )""",
    "CREATE INDEX idx_manifest_members_source ON evidence_manifest_members(source_record_id, manifest_id)",
    """CREATE TABLE claim_producer_runs (
        run_id TEXT PRIMARY KEY,
        processing_identity TEXT NOT NULL,
        manifest_id TEXT NOT NULL REFERENCES evidence_manifests(manifest_id),
        producer_fingerprint TEXT NOT NULL,
        content_digest TEXT NOT NULL,
        content_json TEXT NOT NULL
    )""",
    "CREATE INDEX idx_producer_runs_processing ON claim_producer_runs(processing_identity, run_id)",
    """CREATE TABLE claim_batches (
        claim_batch_id TEXT PRIMARY KEY,
        processing_identity TEXT NOT NULL,
        manifest_id TEXT NOT NULL REFERENCES evidence_manifests(manifest_id),
        producer_fingerprint TEXT NOT NULL,
        created_at TEXT NOT NULL,
        content_digest TEXT NOT NULL,
        content_json TEXT NOT NULL
    )""",
    "CREATE INDEX idx_batches_processing ON claim_batches(processing_identity, claim_batch_id)",
    """CREATE TABLE claims (
        claim_id TEXT PRIMARY KEY,
        claim_batch_id TEXT NOT NULL REFERENCES claim_batches(claim_batch_id),
        manifest_id TEXT NOT NULL REFERENCES evidence_manifests(manifest_id),
        owner_binding_digest TEXT NOT NULL,
        semantic_fingerprint TEXT NOT NULL,
        claim_kind TEXT NOT NULL,
        time_start TEXT NOT NULL,
        time_end TEXT NOT NULL,
        created_at TEXT NOT NULL,
        content_digest TEXT NOT NULL,
        content_json TEXT NOT NULL
    )""",
    "CREATE INDEX idx_claims_manifest ON claims(manifest_id, claim_id)",
    "CREATE INDEX idx_claims_created ON claims(created_at, claim_id)",
    "CREATE INDEX idx_claims_semantic_time ON claims(semantic_fingerprint, time_end, claim_id)",
    """CREATE TABLE claim_admission_decisions (
        decision_id TEXT PRIMARY KEY,
        processing_identity TEXT NOT NULL,
        claim_id TEXT NOT NULL,
        status TEXT NOT NULL,
        decided_at TEXT NOT NULL,
        content_digest TEXT NOT NULL,
        content_json TEXT NOT NULL
    )""",
    "CREATE INDEX idx_decisions_claim_status ON claim_admission_decisions(claim_id, status, decided_at)",
    """CREATE TABLE claim_processing_receipts (
        processing_identity TEXT PRIMARY KEY,
        manifest_id TEXT NOT NULL REFERENCES evidence_manifests(manifest_id),
        completed_at TEXT NOT NULL,
        receipt_digest TEXT NOT NULL UNIQUE,
        content_json TEXT NOT NULL
    )""",
    "CREATE INDEX idx_receipts_completed ON claim_processing_receipts(completed_at, processing_identity)",
)


class SQLiteBehaviorEvidenceClaimStore:
    """单文件、append-only Claim 与不可变 Manifest 的 SQLite 实现。"""

    def __init__(
        self,
        behavior_root: str | Path,
        *,
        config: BehaviorConfig,
        initialize: bool = False,
    ) -> None:
        if not isinstance(config, BehaviorConfig):
            raise TypeError("config must be BehaviorConfig")
        try:
            requested_root = Path(behavior_root).expanduser().absolute()
        except TypeError as exc:
            raise ClaimStoreError("behavior_root must be a filesystem path") from exc
        self._reject_existing_symlink_components(requested_root)
        root = requested_root.resolve(strict=False)
        self.root = root
        self.path = root / _DATABASE_NAME
        self.config = config
        self.max_claim_capacity = config.store.max_claims
        self.initialized = False
        self._reject_symlink_targets()
        if initialize:
            self.initialize()

    def initialize(self) -> None:
        if self.initialized:
            return
        self._ensure_root()
        self._reject_symlink_targets()
        try:
            with closing(self._connect()) as connection:
                connection.execute("PRAGMA journal_mode=WAL")
                connection.execute("BEGIN IMMEDIATE")
                try:
                    tables = {
                        row[0]
                        for row in connection.execute(
                            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                        )
                    }
                    if not tables:
                        for statement in _SCHEMA_STATEMENTS:
                            connection.execute(statement)
                        connection.execute(
                            "INSERT INTO behavior_metadata(key, value) VALUES('schema_version', ?)",
                            (_SCHEMA_VERSION,),
                        )
                    else:
                        if "behavior_metadata" not in tables:
                            raise ClaimStoreError("existing Behavior database has no schema metadata")
                        row = connection.execute(
                            "SELECT value FROM behavior_metadata WHERE key='schema_version'"
                        ).fetchone()
                        if row is None or row[0] != _SCHEMA_VERSION:
                            raise ClaimStoreError("Behavior database schema version mismatch")
                    self._validate_schema(connection)
                    connection.execute("COMMIT")
                except BaseException:
                    connection.execute("ROLLBACK")
                    raise
        except ClaimStoreError:
            raise
        except (OSError, sqlite3.Error) as exc:
            raise ClaimStoreError("failed to initialize Behavior SQLite Store") from exc
        try:
            os.chmod(self.path, 0o600)
        except OSError as exc:
            raise ClaimStoreError("failed to set Behavior database permissions") from exc
        self.initialized = True

    def readiness(self) -> tuple[bool, str]:
        if not self.initialized:
            return False, "not_initialized"
        try:
            with closing(self._connect()) as connection:
                row = connection.execute(
                    "SELECT value FROM behavior_metadata WHERE key='schema_version'"
                ).fetchone()
                self._validate_schema(connection)
            if row is None or row[0] != _SCHEMA_VERSION:
                return False, "schema_version_mismatch"
            return True, f"schema={_SCHEMA_VERSION}"
        except Exception as exc:
            return False, type(exc).__name__

    def owner_binding_digest(self) -> str | None:
        self._require_initialized()
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT value FROM behavior_metadata WHERE key='owner_binding_digest'"
            ).fetchone()
        return None if row is None else sha256_digest(row[0], "owner_binding_digest")

    def ingest_source(
        self,
        record: SourceRecord,
        assembler: EvidenceWindowAssembler,
    ) -> SourceIngestResult:
        self._require_initialized()
        if not isinstance(record, SourceRecord):
            raise TypeError("record must be SourceRecord")
        if not isinstance(assembler, EvidenceWindowAssembler):
            raise TypeError("assembler must be EvidenceWindowAssembler")
        record = SourceRecord.from_dict(record.to_dict(), config=self.config.source)
        grouping_key = assembler.grouping_key(record)
        try:
            with self._transaction() as connection:
                self._bind_owner(connection, record.owner_binding)
                existing_row = connection.execute(
                    "SELECT content_json, content_digest FROM source_records WHERE source_record_id=?",
                    (record.source_record_id,),
                ).fetchone()
                if existing_row is not None:
                    existing = self._decode_durable(
                        existing_row[0],
                        existing_row[1],
                        lambda value: SourceRecord.from_dict(value, config=self.config.source),
                        "SourceRecord",
                    )
                    if canonical_json(existing.to_dict()) != canonical_json(record.to_dict()):
                        raise SourceRecordConflictError("SourceRecord identity conflicts with durable content")
                    active = self._active_for_source(connection, record.source_record_id)
                    return SourceIngestResult(
                        SourceIngestStatus.REPLAYED,
                        record.source_record_id,
                        active,
                    )
                active = self._active_by_group(connection, grouping_key)
                committed_max, committed_watermark = self._group_watermark(
                    connection,
                    grouping_key,
                )
                if assembler.is_late(
                    record,
                    active,
                    committed_watermark=committed_watermark,
                ):
                    return SourceIngestResult(
                        SourceIngestStatus.LATE_REJECTED,
                        record.source_record_id,
                        active,
                        reason_code="event_time_before_committed_watermark",
                    )
                current_records = () if active is None else self._window_records(connection, active.window_id)
                records = (*current_records, record)
                partitions = assembler.partition(records)
                if active is None and any(partition.seal_reason is None for partition in partitions):
                    active_count = connection.execute(
                        "SELECT COUNT(*) FROM active_evidence_windows"
                    ).fetchone()[0]
                    if active_count >= assembler.config.max_active_windows:
                        raise EvidenceWindowStateError(
                            "active EvidenceWindow capacity has been reached"
                        )
                self._insert_source(connection, record)
                minimum_watermark: datetime | None
                if active is not None:
                    connection.execute(
                        "DELETE FROM active_evidence_windows WHERE window_id=?",
                        (active.window_id,),
                    )
                    base_generation = active.generation
                    max_event_time = max(
                        active.max_event_time,
                        committed_max or record.event_time_end,
                        record.event_time_end,
                    )
                    minimum_watermark = max(
                        active.watermark,
                        committed_watermark or active.watermark,
                    )
                else:
                    base_generation = self._next_generation(connection, grouping_key)
                    max_event_time = max(record.event_time_end, committed_max or record.event_time_end)
                    minimum_watermark = committed_watermark
                manifests: list[EvidenceManifest] = []
                next_active: EvidenceWindow | None = None
                for offset, partition in enumerate(partitions):
                    window = assembler.materialize(
                        partition,
                        grouping_key=grouping_key,
                        generation=base_generation + offset,
                        max_event_time=max_event_time,
                        minimum_watermark=minimum_watermark,
                    )
                    if partition.seal_reason is None:
                        self._insert_active_window(connection, window)
                        next_active = window
                    else:
                        manifest = EvidenceManifest.seal(
                            window,
                            partition.records,
                            reason=partition.seal_reason,
                            max_blind_intervals=assembler.config.max_blind_intervals,
                        )
                        self._insert_manifest(
                            connection,
                            manifest,
                            grouping_key=grouping_key,
                            generation=window.generation,
                        )
                        manifests.append(manifest)
                committed_window = next_active
                if committed_window is None and partitions:
                    last_partition = partitions[-1]
                    committed_window = assembler.materialize(
                        last_partition,
                        grouping_key=grouping_key,
                        generation=base_generation + len(partitions) - 1,
                        max_event_time=max_event_time,
                        minimum_watermark=minimum_watermark,
                    )
                if committed_window is not None:
                    self._commit_watermark(connection, committed_window)
                return SourceIngestResult(
                    SourceIngestStatus.ACCEPTED,
                    record.source_record_id,
                    next_active,
                    tuple(manifest.manifest_id for manifest in manifests),
                    window_opened=(
                        active is None
                        or (next_active is not None and next_active.window_id != active.window_id)
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise SourceRecordConflictError("SourceRecord stream sequence or identity conflicts") from exc
        except (SourceRecordConflictError, BehaviorOwnerConflictError):
            raise
        except sqlite3.Error as exc:
            raise ClaimStoreError("failed to ingest SourceRecord atomically") from exc

    def read_source(self, source_record_id: str) -> SourceRecord | None:
        self._require_initialized()
        resolved = identifier(source_record_id, "source_record_id")
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT content_json, content_digest FROM source_records WHERE source_record_id=?",
                (resolved,),
            ).fetchone()
        return (
            None
            if row is None
            else self._decode_durable(
                row[0],
                row[1],
                lambda value: SourceRecord.from_dict(value, config=self.config.source),
                "SourceRecord",
            )
        )

    def read_active_window(self, window_id: str) -> EvidenceWindow | None:
        self._require_initialized()
        resolved = identifier(window_id, "window_id")
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT content_json, content_digest FROM active_evidence_windows WHERE window_id=?",
                (resolved,),
            ).fetchone()
        return (
            None
            if row is None
            else self._decode_durable(
                row[0],
                row[1],
                EvidenceWindow.from_dict,
                "EvidenceWindow",
            )
        )

    def seal_window(
        self,
        window_id: str,
        *,
        reason: EvidenceSealReason,
        assembler: EvidenceWindowAssembler,
    ) -> EvidenceManifest | None:
        self._require_initialized()
        resolved = identifier(window_id, "window_id")
        reason = EvidenceSealReason(reason)
        with self._transaction() as connection:
            existing = self._manifest_for_window(connection, resolved)
            if existing is not None:
                return existing
            row = connection.execute(
                "SELECT content_json, content_digest FROM active_evidence_windows WHERE window_id=?",
                (resolved,),
            ).fetchone()
            if row is None:
                return None
            window = self._decode_durable(
                row[0],
                row[1],
                EvidenceWindow.from_dict,
                "EvidenceWindow",
            )
            records = self._window_records(connection, resolved)
            if not records:
                return None
            sealed_window = replace(window, state=EvidenceWindowState.SEALED)
            manifest = EvidenceManifest.seal(
                sealed_window,
                records,
                reason=reason,
                max_blind_intervals=assembler.config.max_blind_intervals,
            )
            self._insert_manifest(
                connection,
                manifest,
                grouping_key=window.grouping_key,
                generation=window.generation,
            )
            connection.execute("DELETE FROM active_evidence_windows WHERE window_id=?", (resolved,))
            return manifest

    def read_manifest(self, manifest_id: str) -> EvidenceManifest | None:
        self._require_initialized()
        resolved = identifier(manifest_id, "manifest_id")
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT content_json, manifest_digest FROM evidence_manifests WHERE manifest_id=?",
                (resolved,),
            ).fetchone()
        return None if row is None else self._decode_manifest(row[0], row[1])

    def read_manifest_for_window(self, window_id: str) -> EvidenceManifest | None:
        self._require_initialized()
        resolved = identifier(window_id, "window_id")
        with closing(self._connect()) as connection:
            return self._manifest_for_window(connection, resolved)

    def list_manifests(
        self,
        *,
        start: datetime,
        end: datetime,
        limit: int,
        cursor: str | None = None,
    ) -> tuple[EvidenceManifest, ...]:
        self._require_initialized()
        start_utc, end_utc = self._query_bounds(start, end, limit, self.config.evidence.max_query_limit)
        with closing(self._connect()) as connection:
            cursor_time, cursor_id = self._cursor(connection, "evidence_manifests", "manifest_id", "sealed_at", cursor)
            rows = connection.execute(
                """SELECT content_json, manifest_digest FROM evidence_manifests
                   WHERE sealed_at>=? AND sealed_at<=?
                     AND (? IS NULL OR sealed_at>? OR (sealed_at=? AND manifest_id>?))
                   ORDER BY sealed_at, manifest_id LIMIT ?""",
                (
                    utc_text(start_utc),
                    utc_text(end_utc),
                    cursor_id,
                    cursor_time,
                    cursor_time,
                    cursor_id,
                    limit,
                ),
            ).fetchall()
        return tuple(self._decode_manifest(row[0], row[1]) for row in rows)

    def read_claim(self, claim_id: str) -> Claim | None:
        self._require_initialized()
        resolved = identifier(claim_id, "claim_id")
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT content_json, content_digest FROM claims WHERE claim_id=?",
                (resolved,),
            ).fetchone()
        return (
            None
            if row is None
            else self._decode_durable(row[0], row[1], Claim.from_dict, "Claim")
        )

    def list_claims(
        self,
        *,
        start: datetime,
        end: datetime,
        limit: int,
        cursor: str | None = None,
    ) -> tuple[Claim, ...]:
        self._require_initialized()
        start_utc, end_utc = self._query_bounds(start, end, limit, self.config.claim.max_query_limit)
        with closing(self._connect()) as connection:
            cursor_time, cursor_id = self._cursor(connection, "claims", "claim_id", "created_at", cursor)
            rows = connection.execute(
                """SELECT content_json, content_digest FROM claims
                   WHERE created_at>=? AND created_at<=?
                     AND (? IS NULL OR created_at>? OR (created_at=? AND claim_id>?))
                   ORDER BY created_at, claim_id LIMIT ?""",
                (
                    utc_text(start_utc),
                    utc_text(end_utc),
                    cursor_id,
                    cursor_time,
                    cursor_time,
                    cursor_id,
                    limit,
                ),
            ).fetchall()
        return tuple(
            self._decode_durable(row[0], row[1], Claim.from_dict, "Claim")
            for row in rows
        )

    def find_recent_accepted_claim(
        self,
        *,
        semantic_fingerprint: str,
        since: datetime,
        until: datetime,
    ) -> Claim | None:
        self._require_initialized()
        fingerprint = sha256_digest(semantic_fingerprint, "semantic_fingerprint")
        since_utc = strict_utc(since, "since")
        until_utc = strict_utc(until, "until")
        if until_utc < since_utc:
            raise ValueError("until cannot precede since")
        with closing(self._connect()) as connection:
            row = connection.execute(
                """SELECT c.content_json, c.content_digest FROM claims c
                   JOIN claim_admission_decisions d ON d.claim_id=c.claim_id
                   WHERE c.semantic_fingerprint=? AND c.time_end>=? AND c.time_start<=?
                     AND d.status=?
                   ORDER BY c.time_end DESC, c.claim_id DESC LIMIT 1""",
                (
                    fingerprint,
                    utc_text(since_utc),
                    utc_text(until_utc),
                    ClaimAdmissionStatus.ACCEPTED.value,
                ),
            ).fetchone()
        return (
            None
            if row is None
            else self._decode_durable(row[0], row[1], Claim.from_dict, "Claim")
        )

    def claim_count(self) -> int:
        self._require_initialized()
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT COUNT(*) FROM claims").fetchone()
        return int(row[0])

    def read_receipt(self, processing_identity: str) -> ClaimProcessingReceipt | None:
        self._require_initialized()
        resolved = identifier(processing_identity, "processing_identity")
        with closing(self._connect()) as connection:
            return self._read_receipt(connection, resolved, validate_complete=True)

    def read_producer_runs(self, processing_identity: str) -> tuple[ClaimProducerRun, ...]:
        self._require_initialized()
        resolved = identifier(processing_identity, "processing_identity")
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT content_json, content_digest FROM claim_producer_runs
                   WHERE processing_identity=? ORDER BY run_id""",
                (resolved,),
            ).fetchall()
        return tuple(
            self._decode_durable(
                row[0],
                row[1],
                ClaimProducerRun.from_dict,
                "ClaimProducerRun",
            )
            for row in rows
        )

    def read_decisions(self, processing_identity: str) -> tuple[ClaimAdmissionDecision, ...]:
        self._require_initialized()
        resolved = identifier(processing_identity, "processing_identity")
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT content_json, content_digest FROM claim_admission_decisions
                   WHERE processing_identity=? ORDER BY decision_id""",
                (resolved,),
            ).fetchall()
        return tuple(
            self._decode_durable(
                row[0],
                row[1],
                ClaimAdmissionDecision.from_dict,
                "ClaimAdmissionDecision",
            )
            for row in rows
        )

    def publish_processing(
        self,
        *,
        receipt: ClaimProcessingReceipt,
        producer_runs: tuple[ClaimProducerRun, ...],
        batches: tuple[ClaimBatch, ...],
        accepted_claims: tuple[Claim, ...],
        decisions: tuple[ClaimAdmissionDecision, ...],
    ) -> tuple[ClaimProcessingReceipt, bool]:
        self._require_initialized()
        with self._transaction() as connection:
            existing = self._read_receipt(connection, receipt.processing_identity, validate_complete=True)
            if existing is not None:
                if (
                    existing.manifest_id != receipt.manifest_id
                    or existing.manifest_digest != receipt.manifest_digest
                    or existing.producer_fingerprints != receipt.producer_fingerprints
                    or existing.claim_batch_ids != receipt.claim_batch_ids
                    or existing.schema_version != receipt.schema_version
                ):
                    raise ClaimProcessingConflictError("processing identity conflicts with another request")
                return existing, True
            partial_count = 0
            for table in ("claim_producer_runs", "claim_batches", "claim_admission_decisions"):
                partial_count += connection.execute(
                    f"SELECT COUNT(*) FROM {table} WHERE processing_identity=?",
                    (receipt.processing_identity,),
                ).fetchone()[0]
            if partial_count:
                raise ClaimProcessingConflictError("processing identity has partial durable results without Receipt")
            receipt_count = connection.execute(
                "SELECT COUNT(*) FROM claim_processing_receipts"
            ).fetchone()[0]
            if receipt_count >= self.config.store.max_receipts:
                raise ClaimStoreError("ClaimProcessingReceipt capacity has been reached")
            current_claim_count = connection.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
            new_claim_ids = {
                claim.claim_id
                for claim in accepted_claims
                if connection.execute(
                    "SELECT 1 FROM claims WHERE claim_id=?",
                    (claim.claim_id,),
                ).fetchone()
                is None
            }
            if current_claim_count + len(new_claim_ids) > self.config.store.max_claims:
                raise ClaimStoreError("Claim capacity has been reached during atomic publication")
            if tuple(run.producer_fingerprint for run in producer_runs) != receipt.producer_fingerprints:
                raise ClaimProcessingConflictError("Receipt producer fingerprints do not match producer runs")
            if tuple(batch.claim_batch_id for batch in batches) != receipt.claim_batch_ids:
                raise ClaimProcessingConflictError("Receipt ClaimBatch identities do not match publication")
            if tuple(claim.claim_id for claim in accepted_claims) != receipt.accepted_claim_ids:
                raise ClaimProcessingConflictError("Receipt accepted Claim identities do not match publication")
            if tuple(decision.decision_id for decision in decisions) != receipt.decision_ids:
                raise ClaimProcessingConflictError("Receipt decision identities do not match publication")
            for run in producer_runs:
                self._insert_run(connection, run)
            for batch in batches:
                self._insert_batch(connection, receipt.processing_identity, batch)
            for claim in accepted_claims:
                self._insert_claim(connection, claim)
            for decision in decisions:
                self._insert_decision(connection, receipt.processing_identity, decision)
            payload = self._encode(receipt.to_dict())
            connection.execute(
                """INSERT INTO claim_processing_receipts(
                       processing_identity, manifest_id, completed_at, receipt_digest, content_json
                   ) VALUES(?, ?, ?, ?, ?)""",
                (
                    receipt.processing_identity,
                    receipt.manifest_id,
                    utc_text(receipt.completed_at),
                    receipt.receipt_digest,
                    payload,
                ),
            )
            return receipt, False

    def _insert_source(self, connection: sqlite3.Connection, record: SourceRecord) -> None:
        payload = self._encode(record.to_dict())
        connection.execute(
            """INSERT INTO source_records(
                   source_record_id, stream_id, source_sequence, owner_binding_digest,
                   event_time_start, event_time_end, content_digest, content_json
               ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.source_record_id,
                record.stream_id,
                record.source_sequence,
                record.owner_binding.binding_digest,
                utc_text(record.event_time_start),
                utc_text(record.event_time_end),
                record.canonical_digest,
                payload,
            ),
        )

    def _insert_active_window(self, connection: sqlite3.Connection, window: EvidenceWindow) -> None:
        payload = self._encode(window.to_dict())
        connection.execute(
            """INSERT INTO active_evidence_windows(
                   window_id, grouping_key, generation, owner_binding_digest, watermark,
                   content_digest, content_json
               ) VALUES(?, ?, ?, ?, ?, ?, ?)""",
            (
                window.window_id,
                window.grouping_key,
                window.generation,
                window.owner_binding_digest,
                utc_text(window.watermark),
                self._content_digest(payload),
                payload,
            ),
        )
        connection.executemany(
            "INSERT INTO evidence_window_members(window_id, source_record_id) VALUES(?, ?)",
            ((window.window_id, source_id) for source_id in window.ordered_source_record_ids),
        )

    def _insert_manifest(
        self,
        connection: sqlite3.Connection,
        manifest: EvidenceManifest,
        *,
        grouping_key: str,
        generation: int,
    ) -> None:
        payload = self._encode(manifest.to_dict())
        row = connection.execute(
            "SELECT content_json FROM evidence_manifests WHERE manifest_id=? OR window_id=?",
            (manifest.manifest_id, manifest.window_id),
        ).fetchone()
        if row is not None:
            if row[0] != payload:
                raise EvidenceManifestError("EvidenceManifest identity conflicts with durable content")
            return
        connection.execute(
            """INSERT INTO evidence_manifests(
                   manifest_id, window_id, grouping_key, generation, owner_binding_digest,
                   started_at, ended_at, sealed_at, manifest_digest, content_json
               ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                manifest.manifest_id,
                manifest.window_id,
                grouping_key,
                generation,
                manifest.owner_binding_digest,
                utc_text(manifest.started_at),
                utc_text(manifest.ended_at),
                utc_text(manifest.sealed_at),
                manifest.manifest_digest,
                payload,
            ),
        )
        connection.executemany(
            """INSERT INTO evidence_manifest_members(manifest_id, source_record_id, member_order)
               VALUES(?, ?, ?)""",
            (
                (manifest.manifest_id, source.source_record_id, index)
                for index, source in enumerate(manifest.ordered_source_records)
            ),
        )

    def _insert_run(self, connection: sqlite3.Connection, run: ClaimProducerRun) -> None:
        self._insert_immutable(
            connection,
            table="claim_producer_runs",
            identity_column="run_id",
            identity=run.run_id,
            content=run.to_dict(),
            columns=("processing_identity", "manifest_id", "producer_fingerprint"),
            values=(run.processing_identity, run.manifest_id, run.producer_fingerprint),
        )

    def _insert_batch(
        self,
        connection: sqlite3.Connection,
        processing_identity: str,
        batch: ClaimBatch,
    ) -> None:
        self._insert_immutable(
            connection,
            table="claim_batches",
            identity_column="claim_batch_id",
            identity=batch.claim_batch_id,
            content=batch.to_dict(),
            columns=("processing_identity", "manifest_id", "producer_fingerprint", "created_at"),
            values=(processing_identity, batch.manifest_id, batch.producer_fingerprint, utc_text(batch.created_at)),
        )

    def _insert_claim(self, connection: sqlite3.Connection, claim: Claim) -> None:
        self._insert_immutable(
            connection,
            table="claims",
            identity_column="claim_id",
            identity=claim.claim_id,
            content=claim.to_dict(),
            columns=(
                "claim_batch_id",
                "manifest_id",
                "owner_binding_digest",
                "semantic_fingerprint",
                "claim_kind",
                "time_start",
                "time_end",
                "created_at",
            ),
            values=(
                claim.claim_batch_id,
                claim.evidence_manifest_id,
                claim.owner_binding_digest,
                claim.semantic_fingerprint,
                claim.proposal.claim_kind.value,
                utc_text(claim.proposal.time_start),
                utc_text(claim.proposal.time_end),
                utc_text(claim.created_at),
            ),
        )

    def _insert_decision(
        self,
        connection: sqlite3.Connection,
        processing_identity: str,
        decision: ClaimAdmissionDecision,
    ) -> None:
        if decision.processing_identity != processing_identity:
            raise ClaimProcessingConflictError("AdmissionDecision is bound to another processing identity")
        self._insert_immutable(
            connection,
            table="claim_admission_decisions",
            identity_column="decision_id",
            identity=decision.decision_id,
            content=decision.to_dict(),
            columns=("processing_identity", "claim_id", "status", "decided_at"),
            values=(processing_identity, decision.claim_id, decision.status.value, utc_text(decision.decided_at)),
        )

    def _insert_immutable(
        self,
        connection: sqlite3.Connection,
        *,
        table: str,
        identity_column: str,
        identity: str,
        content: dict[str, object],
        columns: tuple[str, ...],
        values: tuple[object, ...],
    ) -> None:
        payload = self._encode(content)
        row = connection.execute(
            f"SELECT content_json FROM {table} WHERE {identity_column}=?",
            (identity,),
        ).fetchone()
        if row is not None:
            if row[0] != payload:
                raise ClaimProcessingConflictError(f"{table} identity conflicts with durable content")
            return
        column_text = ", ".join((identity_column, *columns, "content_digest", "content_json"))
        placeholders = ", ".join("?" for _ in range(len(columns) + 3))
        connection.execute(
            f"INSERT INTO {table}({column_text}) VALUES({placeholders})",
            (identity, *values, self._content_digest(payload), payload),
        )

    def _read_receipt(
        self,
        connection: sqlite3.Connection,
        processing_identity: str,
        *,
        validate_complete: bool,
    ) -> ClaimProcessingReceipt | None:
        row = connection.execute(
            "SELECT content_json, receipt_digest FROM claim_processing_receipts WHERE processing_identity=?",
            (processing_identity,),
        ).fetchone()
        if row is None:
            return None
        receipt = self._decode(row[0], ClaimProcessingReceipt.from_dict, "ClaimProcessingReceipt")
        try:
            stored_receipt_digest = sha256_digest(row[1], "receipt_digest")
        except ValueError as exc:
            raise ClaimProcessingConflictError("ProcessingReceipt has an invalid digest column") from exc
        if receipt.receipt_digest != stored_receipt_digest:
            raise ClaimProcessingConflictError("ProcessingReceipt digest column does not match canonical content")
        if validate_complete:
            manifest_row = connection.execute(
                "SELECT manifest_digest FROM evidence_manifests WHERE manifest_id=?",
                (receipt.manifest_id,),
            ).fetchone()
            if manifest_row is None or manifest_row[0] != receipt.manifest_digest:
                raise ClaimProcessingConflictError("ProcessingReceipt references an invalid EvidenceManifest")
            batch_placeholders = ",".join("?" for _ in receipt.claim_batch_ids)
            batch_count = connection.execute(
                f"""SELECT COUNT(*) FROM claim_batches
                    WHERE processing_identity=? AND manifest_id=?
                      AND claim_batch_id IN ({batch_placeholders})""",
                (
                    receipt.processing_identity,
                    receipt.manifest_id,
                    *receipt.claim_batch_ids,
                ),
            ).fetchone()[0]
            if batch_count != len(receipt.claim_batch_ids):
                raise ClaimProcessingConflictError("ProcessingReceipt references incomplete ClaimBatches")
            if receipt.accepted_claim_ids:
                claim_placeholders = ",".join("?" for _ in receipt.accepted_claim_ids)
                claim_count = connection.execute(
                    f"""SELECT COUNT(*) FROM claims c
                        JOIN claim_batches b ON b.claim_batch_id=c.claim_batch_id
                        WHERE b.processing_identity=? AND c.manifest_id=?
                          AND c.claim_id IN ({claim_placeholders})""",
                    (
                        receipt.processing_identity,
                        receipt.manifest_id,
                        *receipt.accepted_claim_ids,
                    ),
                ).fetchone()[0]
                if claim_count != len(receipt.accepted_claim_ids):
                    raise ClaimProcessingConflictError("ProcessingReceipt references incomplete Claims")
            if receipt.decision_ids:
                decision_placeholders = ",".join("?" for _ in receipt.decision_ids)
                decision_count = connection.execute(
                    f"""SELECT COUNT(*) FROM claim_admission_decisions
                        WHERE processing_identity=?
                          AND decision_id IN ({decision_placeholders})""",
                    (receipt.processing_identity, *receipt.decision_ids),
                ).fetchone()[0]
                if decision_count != len(receipt.decision_ids):
                    raise ClaimProcessingConflictError("ProcessingReceipt references incomplete decisions")
            run_rows = connection.execute(
                """SELECT producer_fingerprint FROM claim_producer_runs
                   WHERE processing_identity=? ORDER BY run_id""",
                (processing_identity,),
            ).fetchall()
            if sorted(row[0] for row in run_rows) != sorted(receipt.producer_fingerprints):
                raise ClaimProcessingConflictError("ProcessingReceipt producer runs are incomplete")
        return receipt

    def _bind_owner(self, connection: sqlite3.Connection, binding: ConfirmedOwnerBinding) -> None:
        digest = binding.binding_digest
        row = connection.execute(
            "SELECT value FROM behavior_metadata WHERE key='owner_binding_digest'"
        ).fetchone()
        if row is not None:
            if row[0] != digest:
                raise BehaviorOwnerConflictError("Behavior Store is already bound to another Owner")
            return
        connection.execute(
            "INSERT INTO behavior_metadata(key, value) VALUES('owner_binding_digest', ?)",
            (digest,),
        )
        connection.execute(
            "INSERT INTO behavior_metadata(key, value) VALUES('owner_binding', ?)",
            (self._encode(binding.to_dict()),),
        )

    def _group_watermark(
        self,
        connection: sqlite3.Connection,
        grouping_key: str,
    ) -> tuple[datetime | None, datetime | None]:
        row = connection.execute(
            "SELECT max_event_time, watermark FROM evidence_watermarks WHERE grouping_key=?",
            (grouping_key,),
        ).fetchone()
        if row is None:
            return None, None
        try:
            return parse_utc(row[0], "max_event_time"), parse_utc(row[1], "watermark")
        except (TypeError, ValueError) as exc:
            raise ClaimStoreError("durable Evidence watermark is invalid") from exc

    @staticmethod
    def _commit_watermark(
        connection: sqlite3.Connection,
        window: EvidenceWindow,
    ) -> None:
        row = connection.execute(
            "SELECT owner_binding_digest, max_event_time, watermark FROM evidence_watermarks WHERE grouping_key=?",
            (window.grouping_key,),
        ).fetchone()
        if row is not None:
            if row[0] != window.owner_binding_digest:
                raise BehaviorOwnerConflictError("Evidence watermark belongs to another Owner")
            try:
                previous_max = parse_utc(row[1], "max_event_time")
                previous_watermark = parse_utc(row[2], "watermark")
            except (TypeError, ValueError) as exc:
                raise ClaimStoreError("durable Evidence watermark is invalid") from exc
            max_event_time = max(previous_max, window.max_event_time)
            watermark = max(previous_watermark, window.watermark)
            connection.execute(
                """UPDATE evidence_watermarks SET max_event_time=?, watermark=?
                   WHERE grouping_key=?""",
                (utc_text(max_event_time), utc_text(watermark), window.grouping_key),
            )
            return
        connection.execute(
            """INSERT INTO evidence_watermarks(
                   grouping_key, owner_binding_digest, max_event_time, watermark
               ) VALUES(?, ?, ?, ?)""",
            (
                window.grouping_key,
                window.owner_binding_digest,
                utc_text(window.max_event_time),
                utc_text(window.watermark),
            ),
        )

    def _active_by_group(
        self,
        connection: sqlite3.Connection,
        grouping_key: str,
    ) -> EvidenceWindow | None:
        row = connection.execute(
            "SELECT content_json, content_digest FROM active_evidence_windows WHERE grouping_key=?",
            (grouping_key,),
        ).fetchone()
        return (
            None
            if row is None
            else self._decode_durable(
                row[0],
                row[1],
                EvidenceWindow.from_dict,
                "EvidenceWindow",
            )
        )

    def _active_for_source(
        self,
        connection: sqlite3.Connection,
        source_record_id: str,
    ) -> EvidenceWindow | None:
        row = connection.execute(
            """SELECT w.content_json, w.content_digest FROM active_evidence_windows w
               JOIN evidence_window_members m ON m.window_id=w.window_id
               WHERE m.source_record_id=?""",
            (source_record_id,),
        ).fetchone()
        return (
            None
            if row is None
            else self._decode_durable(
                row[0],
                row[1],
                EvidenceWindow.from_dict,
                "EvidenceWindow",
            )
        )

    def _window_records(
        self,
        connection: sqlite3.Connection,
        window_id: str,
    ) -> tuple[SourceRecord, ...]:
        rows = connection.execute(
            """SELECT s.content_json, s.content_digest FROM evidence_window_members m
               JOIN source_records s ON s.source_record_id=m.source_record_id
               WHERE m.window_id=? ORDER BY s.event_time_start, s.event_time_end,
               s.stream_id, s.source_sequence, s.source_record_id""",
            (window_id,),
        ).fetchall()
        return tuple(
            self._decode_durable(
                row[0],
                row[1],
                lambda value: SourceRecord.from_dict(value, config=self.config.source),
                "SourceRecord",
            )
            for row in rows
        )

    def _manifest_for_window(
        self,
        connection: sqlite3.Connection,
        window_id: str,
    ) -> EvidenceManifest | None:
        row = connection.execute(
            "SELECT content_json, manifest_digest FROM evidence_manifests WHERE window_id=?",
            (window_id,),
        ).fetchone()
        return None if row is None else self._decode_manifest(row[0], row[1])

    @staticmethod
    def _next_generation(connection: sqlite3.Connection, grouping_key: str) -> int:
        active = connection.execute(
            "SELECT MAX(generation) FROM active_evidence_windows WHERE grouping_key=?",
            (grouping_key,),
        ).fetchone()[0]
        sealed = connection.execute(
            "SELECT MAX(generation) FROM evidence_manifests WHERE grouping_key=?",
            (grouping_key,),
        ).fetchone()[0]
        values = [value for value in (active, sealed) if value is not None]
        return 0 if not values else max(values) + 1

    def _validate_schema(self, connection: sqlite3.Connection) -> None:
        for table, expected in _TABLE_COLUMNS.items():
            actual = tuple(row[1] for row in connection.execute(f"PRAGMA table_info({table})"))
            if actual != expected:
                raise ClaimStoreError(f"Behavior database table shape mismatch: {table}")
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND sql IS NOT NULL"
            )
        }
        missing = sorted(_REQUIRED_INDEXES - indexes)
        if missing:
            raise ClaimStoreError(f"Behavior database is missing required indexes: {missing}")
        foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_keys:
            raise ClaimStoreError("Behavior database contains foreign-key violations")

    def _ensure_root(self) -> None:
        current = self.root
        missing: list[Path] = []
        while not current.exists():
            if current.is_symlink():
                raise ClaimStoreError("Behavior Store path cannot traverse a symbolic link")
            missing.append(current)
            parent = current.parent
            if parent == current:
                raise ClaimStoreError("Behavior Store has no existing directory ancestor")
            current = parent
        if current.is_symlink() or not current.is_dir():
            raise ClaimStoreError("Behavior Store ancestor must be a real directory")
        for directory in reversed(missing):
            directory.mkdir(mode=0o700)
            if directory.is_symlink() or not directory.is_dir():
                raise ClaimStoreError("Behavior Store root must be a real directory")
        try:
            os.chmod(self.root, 0o700)
        except OSError as exc:
            raise ClaimStoreError("failed to set Behavior Store directory permissions") from exc

    def _reject_symlink_targets(self) -> None:
        if self.root.is_symlink() or self.path.is_symlink():
            raise ClaimStoreError("Behavior Store root and database cannot be symbolic links")

    @staticmethod
    def _reject_existing_symlink_components(path: Path) -> None:
        for component in (path, *path.parents):
            if component.is_symlink():
                raise ClaimStoreError("Behavior Store path cannot traverse a symbolic link")

    def _connect(self) -> sqlite3.Connection:
        self._reject_symlink_targets()
        connection = sqlite3.connect(
            self.path,
            timeout=self.config.store.sqlite_timeout_seconds,
            isolation_level=None,
        )
        if self.path.is_symlink():
            connection.close()
            raise ClaimStoreError("Behavior database cannot be a symbolic link")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.row_factory = sqlite3.Row
        return connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def _require_initialized(self) -> None:
        if not self.initialized:
            raise ClaimStoreError("Behavior SQLite Store is not initialized")

    def _encode(self, value: object) -> str:
        payload = canonical_json(value)
        if len(payload.encode("utf-8")) > self.config.store.max_json_bytes:
            raise ClaimStoreError("canonical JSON exceeds the configured Store boundary")
        return payload

    def _decode(
        self,
        payload: str,
        factory: Callable[[object], _T],
        name: str,
    ) -> _T:
        if not isinstance(payload, str) or len(payload.encode("utf-8")) > self.config.store.max_json_bytes:
            raise ClaimStoreError(f"durable {name} exceeds the configured Store boundary")
        try:
            raw = json.loads(payload)
            result = factory(raw)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ClaimStoreError(f"durable {name} failed strict canonical validation") from exc
        to_dict = getattr(result, "to_dict", None)
        if not callable(to_dict) or canonical_json(to_dict()) != payload:
            raise ClaimStoreError(f"durable {name} is not canonically encoded")
        return result

    def _decode_durable(
        self,
        payload: str,
        stored_digest: str,
        factory: Callable[[object], _T],
        name: str,
    ) -> _T:
        try:
            expected = sha256_digest(stored_digest, f"{name} content_digest")
        except ValueError as exc:
            raise ClaimStoreError(f"durable {name} has an invalid content digest") from exc
        if self._content_digest(payload) != expected:
            raise ClaimStoreError(f"durable {name} content digest mismatch")
        return self._decode(payload, factory, name)

    def _decode_manifest(self, payload: str, stored_digest: str) -> EvidenceManifest:
        manifest = self._decode(payload, EvidenceManifest.from_dict, "EvidenceManifest")
        try:
            expected = sha256_digest(stored_digest, "manifest_digest")
        except ValueError as exc:
            raise ClaimStoreError("durable EvidenceManifest has an invalid digest column") from exc
        if manifest.manifest_digest != expected:
            raise ClaimStoreError("durable EvidenceManifest digest column mismatch")
        return manifest

    @staticmethod
    def _content_digest(payload: str) -> str:
        import hashlib

        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @staticmethod
    def _query_bounds(
        start: datetime,
        end: datetime,
        limit: int,
        maximum: int,
    ) -> tuple[datetime, datetime]:
        start_utc = strict_utc(start, "start")
        end_utc = strict_utc(end, "end")
        if end_utc < start_utc:
            raise ValueError("end cannot precede start")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= maximum:
            raise ValueError(f"limit must be between one and {maximum}")
        return start_utc, end_utc

    @staticmethod
    def _cursor(
        connection: sqlite3.Connection,
        table: str,
        identity_column: str,
        time_column: str,
        cursor: str | None,
    ) -> tuple[str | None, str | None]:
        if cursor is None:
            return None, None
        resolved = identifier(cursor, "cursor")
        row = connection.execute(
            f"SELECT {time_column} FROM {table} WHERE {identity_column}=?",
            (resolved,),
        ).fetchone()
        if row is None:
            raise ValueError("cursor does not identify an existing record")
        return row[0], resolved


__all__ = ["SQLiteBehaviorEvidenceClaimStore"]
