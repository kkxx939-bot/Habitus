"""Behavior Evidence & Claim Layer 的 SQLite Schema V2 耐久实现。"""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing, contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol, TypeVar

from behavior._validation import identifier, parse_utc, sha256_digest, strict_utc, utc_text
from behavior.claim.admission import (
    ClaimAdmissionDecision,
    ClaimAdmissionPolicy,
    ClaimAdmissionStatus,
    StaticAdmissionResult,
)
from behavior.claim.model import Claim, ClaimBatch, ClaimNormalizerRun, ClaimProcessingReceipt
from behavior.claim.proposal import ClaimKind
from behavior.config import BehaviorConfig
from behavior.errors import (
    BehaviorOwnerConflictError,
    ClaimProcessingConflictError,
    ClaimStoreCapacityError,
    ClaimStoreError,
    EvidenceBundleStateError,
    EvidenceManifestError,
    SemanticRecordConflictError,
)
from behavior.evidence.bundle import (
    EvidenceBundleState,
    EvidenceSealReason,
    SemanticBundlePartition,
    SemanticEvidenceBundle,
    SemanticEvidenceBundleAssembler,
    SemanticIngestResult,
    SemanticIngestStatus,
)
from behavior.evidence.manifest import EvidenceManifest
from behavior.ingress.model import (
    IngressDecision,
    IngressDecisionStatus,
    OwnerScopedSemanticRecord,
)
from foundation.integrity import canonical_digest, canonical_json

_SCHEMA_VERSION = "2"
_DATABASE_NAME = "evidence_claims.sqlite3"


class _DurableValue(Protocol):
    def to_dict(self) -> dict[str, object]: ...


_T = TypeVar("_T", bound=_DurableValue)

_TABLE_COLUMNS = {
    "behavior_metadata": ("key", "value"),
    "semantic_records": (
        "semantic_record_id",
        "producer_fingerprint",
        "stream_id",
        "source_sequence",
        "owner_identity_digest",
        "event_time_start",
        "event_time_end",
        "canonical_digest",
        "content_json",
    ),
    "semantic_ingress_decisions": (
        "decision_id",
        "semantic_record_id",
        "status",
        "decided_at",
        "content_digest",
        "content_json",
    ),
    "active_evidence_bundles": (
        "bundle_id",
        "grouping_key",
        "generation",
        "owner_identity_digest",
        "state",
        "watermark",
        "content_digest",
        "content_json",
    ),
    "evidence_bundle_members": ("bundle_id", "semantic_record_id", "member_order"),
    "evidence_watermarks": (
        "grouping_key",
        "owner_identity_digest",
        "max_event_time",
        "watermark",
        "latest_generation",
    ),
    "evidence_manifests": (
        "manifest_id",
        "bundle_id",
        "owner_identity_digest",
        "started_at",
        "ended_at",
        "sealed_at",
        "content_digest",
        "content_json",
    ),
    "evidence_manifest_members": ("manifest_id", "semantic_record_id", "member_order"),
    "claim_normalizer_runs": (
        "run_id",
        "processing_identity",
        "manifest_id",
        "semantic_record_id",
        "normalizer_fingerprint",
        "content_digest",
        "content_json",
    ),
    "claim_batches": (
        "claim_batch_id",
        "processing_identity",
        "manifest_id",
        "semantic_record_id",
        "normalizer_fingerprint",
        "created_at",
        "content_digest",
        "content_json",
    ),
    "claims": (
        "claim_id",
        "claim_batch_id",
        "manifest_id",
        "semantic_record_id",
        "owner_identity_digest",
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
        "published_at",
        "receipt_digest",
        "content_json",
    ),
}

_REQUIRED_INDEXES = frozenset(
    {
        "idx_semantic_stream_sequence",
        "idx_semantic_event_time",
        "idx_ingress_status_time",
        "idx_bundle_group_state",
        "idx_bundle_members_record",
        "idx_manifest_bundle",
        "idx_manifest_time",
        "idx_manifest_members_record",
        "idx_normalizer_runs_processing",
        "idx_batches_processing",
        "idx_claims_manifest",
        "idx_claims_record",
        "idx_claims_created",
        "idx_claims_semantic_time",
        "idx_decisions_processing",
        "idx_decisions_claim_status",
        "idx_receipts_completed",
    }
)

_SCHEMA_STATEMENTS = (
    "CREATE TABLE behavior_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    """CREATE TABLE semantic_records (
        semantic_record_id TEXT PRIMARY KEY, producer_fingerprint TEXT NOT NULL, stream_id TEXT NOT NULL,
        source_sequence INTEGER NOT NULL CHECK(source_sequence >= 0), owner_identity_digest TEXT NOT NULL,
        event_time_start TEXT NOT NULL, event_time_end TEXT NOT NULL, canonical_digest TEXT NOT NULL,
        content_json TEXT NOT NULL
    )""",
    "CREATE UNIQUE INDEX idx_semantic_stream_sequence ON semantic_records(producer_fingerprint, stream_id, source_sequence)",
    "CREATE INDEX idx_semantic_event_time ON semantic_records(event_time_start, semantic_record_id)",
    """CREATE TABLE semantic_ingress_decisions (
        decision_id TEXT PRIMARY KEY, semantic_record_id TEXT NOT NULL, status TEXT NOT NULL,
        decided_at TEXT NOT NULL, content_digest TEXT NOT NULL, content_json TEXT NOT NULL
    )""",
    "CREATE INDEX idx_ingress_status_time ON semantic_ingress_decisions(status, decided_at, decision_id)",
    """CREATE TABLE active_evidence_bundles (
        bundle_id TEXT PRIMARY KEY, grouping_key TEXT NOT NULL, generation INTEGER NOT NULL CHECK(generation >= 0),
        owner_identity_digest TEXT NOT NULL, state TEXT NOT NULL, watermark TEXT, content_digest TEXT NOT NULL,
        content_json TEXT NOT NULL, UNIQUE(grouping_key, generation)
    )""",
    "CREATE INDEX idx_bundle_group_state ON active_evidence_bundles(grouping_key, state, generation)",
    """CREATE TABLE evidence_bundle_members (
        bundle_id TEXT NOT NULL REFERENCES active_evidence_bundles(bundle_id),
        semantic_record_id TEXT NOT NULL REFERENCES semantic_records(semantic_record_id),
        member_order INTEGER NOT NULL CHECK(member_order >= 0), PRIMARY KEY(bundle_id, semantic_record_id),
        UNIQUE(bundle_id, member_order)
    )""",
    "CREATE INDEX idx_bundle_members_record ON evidence_bundle_members(semantic_record_id, bundle_id)",
    """CREATE TABLE evidence_watermarks (
        grouping_key TEXT PRIMARY KEY, owner_identity_digest TEXT NOT NULL, max_event_time TEXT,
        watermark TEXT, latest_generation INTEGER NOT NULL CHECK(latest_generation >= 0)
    )""",
    """CREATE TABLE evidence_manifests (
        manifest_id TEXT PRIMARY KEY, bundle_id TEXT NOT NULL UNIQUE, owner_identity_digest TEXT NOT NULL,
        started_at TEXT NOT NULL, ended_at TEXT NOT NULL, sealed_at TEXT NOT NULL,
        content_digest TEXT NOT NULL UNIQUE, content_json TEXT NOT NULL
    )""",
    "CREATE UNIQUE INDEX idx_manifest_bundle ON evidence_manifests(bundle_id)",
    "CREATE INDEX idx_manifest_time ON evidence_manifests(started_at, manifest_id)",
    """CREATE TABLE evidence_manifest_members (
        manifest_id TEXT NOT NULL REFERENCES evidence_manifests(manifest_id),
        semantic_record_id TEXT NOT NULL REFERENCES semantic_records(semantic_record_id),
        member_order INTEGER NOT NULL CHECK(member_order >= 0), PRIMARY KEY(manifest_id, semantic_record_id),
        UNIQUE(manifest_id, member_order)
    )""",
    "CREATE INDEX idx_manifest_members_record ON evidence_manifest_members(semantic_record_id, manifest_id)",
    """CREATE TABLE claim_normalizer_runs (
        run_id TEXT PRIMARY KEY, processing_identity TEXT NOT NULL,
        manifest_id TEXT NOT NULL REFERENCES evidence_manifests(manifest_id),
        semantic_record_id TEXT NOT NULL REFERENCES semantic_records(semantic_record_id),
        normalizer_fingerprint TEXT NOT NULL, content_digest TEXT NOT NULL, content_json TEXT NOT NULL
    )""",
    "CREATE INDEX idx_normalizer_runs_processing ON claim_normalizer_runs(processing_identity, run_id)",
    """CREATE TABLE claim_batches (
        claim_batch_id TEXT PRIMARY KEY, processing_identity TEXT NOT NULL,
        manifest_id TEXT NOT NULL REFERENCES evidence_manifests(manifest_id),
        semantic_record_id TEXT NOT NULL REFERENCES semantic_records(semantic_record_id),
        normalizer_fingerprint TEXT NOT NULL, created_at TEXT NOT NULL, content_digest TEXT NOT NULL,
        content_json TEXT NOT NULL
    )""",
    "CREATE INDEX idx_batches_processing ON claim_batches(processing_identity, claim_batch_id)",
    """CREATE TABLE claims (
        claim_id TEXT PRIMARY KEY, claim_batch_id TEXT NOT NULL REFERENCES claim_batches(claim_batch_id),
        manifest_id TEXT NOT NULL REFERENCES evidence_manifests(manifest_id),
        semantic_record_id TEXT NOT NULL REFERENCES semantic_records(semantic_record_id),
        owner_identity_digest TEXT NOT NULL, semantic_fingerprint TEXT NOT NULL, claim_kind TEXT NOT NULL,
        time_start TEXT NOT NULL, time_end TEXT NOT NULL, created_at TEXT NOT NULL,
        content_digest TEXT NOT NULL, content_json TEXT NOT NULL
    )""",
    "CREATE INDEX idx_claims_manifest ON claims(manifest_id, claim_id)",
    "CREATE INDEX idx_claims_record ON claims(semantic_record_id, claim_id)",
    "CREATE INDEX idx_claims_created ON claims(created_at, claim_id)",
    "CREATE INDEX idx_claims_semantic_time ON claims(semantic_fingerprint, time_end, claim_id)",
    """CREATE TABLE claim_admission_decisions (
        decision_id TEXT PRIMARY KEY, processing_identity TEXT NOT NULL,
        claim_id TEXT NOT NULL REFERENCES claims(claim_id), status TEXT NOT NULL, decided_at TEXT NOT NULL,
        content_digest TEXT NOT NULL, content_json TEXT NOT NULL
    )""",
    "CREATE INDEX idx_decisions_processing ON claim_admission_decisions(processing_identity, decision_id)",
    "CREATE INDEX idx_decisions_claim_status ON claim_admission_decisions(claim_id, status, decided_at, decision_id)",
    """CREATE TABLE claim_processing_receipts (
        processing_identity TEXT PRIMARY KEY, manifest_id TEXT NOT NULL REFERENCES evidence_manifests(manifest_id),
        completed_at TEXT NOT NULL, published_at TEXT NOT NULL, receipt_digest TEXT NOT NULL UNIQUE,
        content_json TEXT NOT NULL
    )""",
    "CREATE INDEX idx_receipts_completed ON claim_processing_receipts(completed_at, processing_identity)",
)


class SQLiteBehaviorEvidenceClaimStore:
    """单 Owner、单文件、显式容量受控的 Behavior Store。"""

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
            requested = Path(behavior_root).expanduser().absolute()
        except TypeError as exc:
            raise ClaimStoreError("behavior_root must be a filesystem path") from exc
        if requested.exists() and requested.is_symlink():
            raise ClaimStoreError("behavior_root cannot be a symlink")
        requested = requested.resolve(strict=False)
        self._reject_existing_symlink_components(requested)
        self.root = requested.resolve(strict=False)
        self.path = self.root / _DATABASE_NAME
        self.config = config
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
                            raise ClaimStoreError(
                                "Behavior database schema version mismatch; migration is not supported"
                            )
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
                self._validate_schema(connection)
                row = connection.execute("SELECT value FROM behavior_metadata WHERE key='schema_version'").fetchone()
            if row is None or row[0] != _SCHEMA_VERSION:
                return False, "schema_version_mismatch"
            return True, f"schema={_SCHEMA_VERSION}"
        except Exception as exc:
            return False, type(exc).__name__

    def health_snapshot(self) -> dict[str, int | str | bool]:
        self._require_initialized()
        with closing(self._connect()) as connection:
            counts = {
                "semantic_record_count": self._count(connection, "semantic_records"),
                "active_bundle_count": connection.execute(
                    "SELECT COUNT(*) FROM active_evidence_bundles WHERE state='OPEN'"
                ).fetchone()[0],
                "manifest_count": self._count(connection, "evidence_manifests"),
                "claim_count": self._count(connection, "claims"),
            }
        size = self._database_bytes()
        return {
            "schema_version": _SCHEMA_VERSION,
            **counts,
            "database_bytes": size,
            "database_size_warning": size >= self.config.store.max_database_bytes,
        }

    def owner_identity_digest(self) -> str | None:
        self._require_initialized()
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT value FROM behavior_metadata WHERE key='owner_identity_digest'").fetchone()
        return None if row is None else sha256_digest(row[0], "owner_identity_digest")

    def record_ingress_decision(
        self,
        decision: IngressDecision,
        *,
        record: OwnerScopedSemanticRecord,
    ) -> IngressDecision:
        self._require_initialized()
        if not isinstance(decision, IngressDecision) or not isinstance(record, OwnerScopedSemanticRecord):
            raise TypeError("decision and record must be Behavior ingress values")
        self._require_decision_binding(decision, record)
        if decision.status not in {
            IngressDecisionStatus.CLOCK_SKEW_REJECTED,
            IngressDecisionStatus.EVENT_TOO_OLD_REJECTED,
            IngressDecisionStatus.CAPACITY_REJECTED,
        }:
            raise SemanticRecordConflictError("only pre-persistence ingress rejections may be recorded directly")
        try:
            with self._transaction() as connection:
                owner = self._owner_digest(connection)
                if owner is not None and owner != record.owner_identity_digest:
                    raise BehaviorOwnerConflictError("Behavior Store is permanently bound to another Owner identity")
                return self._insert_ingress_decision(connection, decision)
        except sqlite3.Error as exc:
            raise ClaimStoreError("failed to persist ingress decision") from exc

    def ingest_semantic_record(
        self,
        record: OwnerScopedSemanticRecord,
        decision: IngressDecision,
        assembler: SemanticEvidenceBundleAssembler,
        *,
        sealed_at: datetime,
    ) -> SemanticIngestResult:
        self._require_initialized()
        if not isinstance(record, OwnerScopedSemanticRecord):
            raise TypeError("record must be OwnerScopedSemanticRecord")
        if not isinstance(decision, IngressDecision):
            raise TypeError("decision must be IngressDecision")
        if not isinstance(assembler, SemanticEvidenceBundleAssembler):
            raise TypeError("assembler must be SemanticEvidenceBundleAssembler")
        self._require_decision_binding(decision, record)
        if decision.status is not IngressDecisionStatus.ACCEPTED:
            raise SemanticRecordConflictError("a new semantic record requires an ACCEPTED ingress decision")
        seal_time = strict_utc(sealed_at, "sealed_at")
        durable_record = OwnerScopedSemanticRecord.from_dict(record.to_dict(), config=self.config.ingress)
        grouping_key = assembler.grouping_key(durable_record)
        try:
            with self._transaction() as connection:
                self._bind_owner(connection, durable_record.owner_identity_digest)
                existing = self._record_by_id(connection, durable_record.semantic_record_id)
                if existing is not None:
                    if existing.canonical_digest != durable_record.canonical_digest:
                        raise SemanticRecordConflictError("semantic record identity conflicts with durable content")
                    replay_decision = IngressDecision(
                        status=IngressDecisionStatus.REPLAYED,
                        reason_code="semantic_record_replayed",
                        record=durable_record,
                        decided_at=decision.decided_at,
                    )
                    stored_decision = self._insert_ingress_decision(connection, replay_decision)
                    return SemanticIngestResult(
                        SemanticIngestStatus.REPLAYED,
                        existing.semantic_record_id,
                        stored_decision,
                        self._active_for_record(connection, existing.semantic_record_id),
                    )
                sequence_row = connection.execute(
                    """SELECT semantic_record_id FROM semantic_records
                       WHERE producer_fingerprint=? AND stream_id=? AND source_sequence=?""",
                    (
                        durable_record.producer_fingerprint.digest,
                        durable_record.semantic_input.stream_id,
                        durable_record.semantic_input.source_sequence,
                    ),
                ).fetchone()
                if sequence_row is not None:
                    raise SemanticRecordConflictError("producer stream sequence conflicts with another record")
                _, committed_watermark, latest_generation = self._group_watermark(connection, grouping_key)
                if assembler.is_late(durable_record, committed_watermark=committed_watermark):
                    late = IngressDecision(
                        status=IngressDecisionStatus.LATE_REJECTED,
                        reason_code="event_time_before_committed_watermark",
                        record=durable_record,
                        decided_at=decision.decided_at,
                    )
                    late = self._insert_ingress_decision(connection, late)
                    return SemanticIngestResult(
                        SemanticIngestStatus.LATE_REJECTED,
                        durable_record.semantic_record_id,
                        late,
                        self._active_by_group(connection, grouping_key),
                    )
                self._require_table_capacity(connection, "semantic_records", self.config.store.max_semantic_records)
                self._require_database_capacity()
                self._insert_record(connection, durable_record)
                stored_decision = self._insert_ingress_decision(connection, decision)
                active = self._active_by_group(connection, grouping_key)
                records = () if active is None else self._bundle_records(connection, active.bundle_id)
                partitions = assembler.partition((*records, durable_record))
                if not partitions:
                    raise EvidenceBundleStateError("accepted semantic record did not form a Bundle partition")
                base_generation = active.generation if active is not None else latest_generation + 1
                if active is not None:
                    connection.execute(
                        "DELETE FROM evidence_bundle_members WHERE bundle_id=?",
                        (active.bundle_id,),
                    )
                manifests: list[EvidenceManifest] = []
                next_active: SemanticEvidenceBundle | None = None
                previous_max, previous_watermark, _ = self._group_watermark(connection, grouping_key)
                for offset, partition in enumerate(partitions):
                    bundle = assembler.materialize(
                        partition,
                        grouping_key=grouping_key,
                        generation=base_generation + offset,
                        previous_max_event_time=previous_max,
                        previous_watermark=previous_watermark,
                    )
                    if bundle.state is EvidenceBundleState.OPEN:
                        if active is None or bundle.bundle_id != active.bundle_id:
                            self._require_active_bundle_capacity(connection)
                        self._upsert_bundle(connection, bundle)
                        next_active = bundle
                    else:
                        self._upsert_bundle(connection, bundle)
                        self._require_table_capacity(connection, "evidence_manifests", self.config.store.max_manifests)
                        manifest = EvidenceManifest.seal(
                            bundle,
                            partition.records,
                            reason=bundle.seal_reason or EvidenceSealReason.EXPLICIT,
                            sealed_at=seal_time,
                            max_coverage_intervals=assembler.config.max_coverage_intervals,
                        )
                        self._insert_manifest(connection, manifest)
                        self._retire_bundle_row(connection, bundle.bundle_id)
                        manifests.append(manifest)
                    if bundle.max_event_time is not None:
                        previous_max = max(
                            (item for item in (previous_max, bundle.max_event_time) if item is not None),
                        )
                    if bundle.watermark is not None:
                        previous_watermark = max(
                            (item for item in (previous_watermark, bundle.watermark) if item is not None),
                        )
                self._commit_watermark(
                    connection,
                    grouping_key,
                    durable_record.owner_identity_digest,
                    previous_max,
                    previous_watermark,
                    base_generation + len(partitions) - 1,
                )
                return SemanticIngestResult(
                    SemanticIngestStatus.ACCEPTED,
                    durable_record.semantic_record_id,
                    stored_decision,
                    next_active,
                    tuple(item.manifest_id for item in manifests),
                )
        except (BehaviorOwnerConflictError, EvidenceBundleStateError, SemanticRecordConflictError):
            raise
        except sqlite3.IntegrityError as exc:
            raise SemanticRecordConflictError("semantic record or Bundle identity conflicts") from exc
        except sqlite3.Error as exc:
            raise ClaimStoreError("failed to ingest semantic record atomically") from exc

    def read_semantic_record(self, semantic_record_id: str) -> OwnerScopedSemanticRecord | None:
        self._require_initialized()
        resolved = identifier(semantic_record_id, "semantic_record_id")
        with closing(self._connect()) as connection:
            return self._record_by_id(connection, resolved)

    def read_active_bundle(self, bundle_id: str) -> SemanticEvidenceBundle | None:
        self._require_initialized()
        resolved = identifier(bundle_id, "bundle_id")
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT content_json, content_digest FROM active_evidence_bundles WHERE bundle_id=? AND state='OPEN'",
                (resolved,),
            ).fetchone()
        return (
            None
            if row is None
            else self._decode(row[0], row[1], SemanticEvidenceBundle.from_dict, "SemanticEvidenceBundle")
        )

    def seal_bundle(
        self,
        bundle_id: str,
        *,
        reason: EvidenceSealReason,
        assembler: SemanticEvidenceBundleAssembler,
        sealed_at: datetime,
    ) -> EvidenceManifest | None:
        self._require_initialized()
        resolved = identifier(bundle_id, "bundle_id")
        seal_reason = EvidenceSealReason(reason)
        seal_time = strict_utc(sealed_at, "sealed_at")
        with self._transaction() as connection:
            existing = self._manifest_for_bundle(connection, resolved)
            if existing is not None:
                return existing
            row = connection.execute(
                "SELECT content_json, content_digest FROM active_evidence_bundles WHERE bundle_id=?",
                (resolved,),
            ).fetchone()
            if row is None:
                return None
            bundle = self._decode(row[0], row[1], SemanticEvidenceBundle.from_dict, "SemanticEvidenceBundle")
            if bundle.state is not EvidenceBundleState.OPEN:
                raise EvidenceBundleStateError("sealed Bundle has no durable Manifest")
            records = self._bundle_records(connection, bundle.bundle_id)
            if not records:
                return None
            sealed = assembler.materialize(
                SemanticBundlePartition(records, seal_reason),
                grouping_key=bundle.grouping_key,
                generation=bundle.generation,
                previous_max_event_time=bundle.max_event_time,
                previous_watermark=bundle.watermark,
            )
            self._upsert_bundle(connection, sealed)
            self._require_table_capacity(connection, "evidence_manifests", self.config.store.max_manifests)
            manifest = EvidenceManifest.seal(
                sealed,
                records,
                reason=seal_reason,
                sealed_at=seal_time,
                max_coverage_intervals=assembler.config.max_coverage_intervals,
            )
            self._insert_manifest(connection, manifest)
            self._retire_bundle_row(connection, sealed.bundle_id)
            return manifest

    def read_manifest(self, manifest_id: str) -> EvidenceManifest | None:
        self._require_initialized()
        resolved = identifier(manifest_id, "manifest_id")
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT content_json, content_digest FROM evidence_manifests WHERE manifest_id=?",
                (resolved,),
            ).fetchone()
        return (
            None
            if row is None
            else self._decode(
                row[0],
                row[1],
                EvidenceManifest.from_dict,
                "EvidenceManifest",
                digest_attribute="content_digest",
            )
        )

    def read_manifest_for_bundle(self, bundle_id: str) -> EvidenceManifest | None:
        self._require_initialized()
        with closing(self._connect()) as connection:
            return self._manifest_for_bundle(connection, identifier(bundle_id, "bundle_id"))

    def list_manifests(
        self,
        *,
        start: datetime,
        end: datetime,
        limit: int,
        cursor: str | None = None,
    ) -> tuple[EvidenceManifest, ...]:
        start_time, end_time = self._bounded_range(start, end)
        size = self._bounded_limit(limit, self.config.evidence.max_query_limit)
        with closing(self._connect()) as connection:
            if cursor is None:
                rows = connection.execute(
                    """SELECT content_json, content_digest FROM evidence_manifests
                       WHERE started_at>=? AND started_at<=? ORDER BY started_at, manifest_id LIMIT ?""",
                    (utc_text(start_time), utc_text(end_time), size),
                ).fetchall()
            else:
                cursor_value = identifier(cursor, "cursor")
                cursor_row = connection.execute(
                    "SELECT started_at FROM evidence_manifests WHERE manifest_id=?",
                    (cursor_value,),
                ).fetchone()
                if cursor_row is None:
                    raise ValueError("Manifest query cursor does not exist")
                rows = connection.execute(
                    """SELECT content_json, content_digest FROM evidence_manifests
                       WHERE started_at>=? AND started_at<=?
                       AND (started_at>? OR (started_at=? AND manifest_id>?))
                       ORDER BY started_at, manifest_id LIMIT ?""",
                    (
                        utc_text(start_time),
                        utc_text(end_time),
                        cursor_row[0],
                        cursor_row[0],
                        cursor_value,
                        size,
                    ),
                ).fetchall()
        return tuple(
            self._decode(
                row[0],
                row[1],
                EvidenceManifest.from_dict,
                "EvidenceManifest",
                digest_attribute="content_digest",
            )
            for row in rows
        )

    def read_claim(self, claim_id: str) -> Claim | None:
        self._require_initialized()
        resolved = identifier(claim_id, "claim_id")
        with closing(self._connect()) as connection:
            return self._claim_by_id(connection, resolved)

    def list_claims(
        self,
        *,
        start: datetime,
        end: datetime,
        limit: int,
        cursor: str | None = None,
    ) -> tuple[Claim, ...]:
        return self._list_claim_query(start=start, end=end, limit=limit, cursor=cursor, accepted_only=False)

    def list_accepted_claims(
        self,
        *,
        start: datetime,
        end: datetime,
        limit: int,
        cursor: str | None = None,
    ) -> tuple[Claim, ...]:
        return self._list_claim_query(start=start, end=end, limit=limit, cursor=cursor, accepted_only=True)

    def list_claims_by_processing(
        self,
        processing_identity: str,
        *,
        limit: int,
        cursor: str | None = None,
    ) -> tuple[Claim, ...]:
        self._require_initialized()
        processing = identifier(processing_identity, "processing_identity")
        size = self._bounded_limit(limit, self.config.claim.max_query_limit)
        cursor_value = "" if cursor is None else identifier(cursor, "cursor")
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT c.content_json, c.content_digest FROM claims c
                   JOIN claim_admission_decisions d ON d.claim_id=c.claim_id
                   WHERE d.processing_identity=? AND c.claim_id>? ORDER BY c.claim_id LIMIT ?""",
                (processing, cursor_value, size),
            ).fetchall()
        return tuple(self._decode(row[0], row[1], Claim.from_dict, "Claim") for row in rows)

    def read_claim_decision(
        self,
        claim_id: str,
        *,
        processing_identity: str | None = None,
    ) -> ClaimAdmissionDecision | None:
        self._require_initialized()
        claim = identifier(claim_id, "claim_id")
        with closing(self._connect()) as connection:
            if processing_identity is None:
                row = connection.execute(
                    """SELECT content_json, content_digest FROM claim_admission_decisions
                       WHERE claim_id=? ORDER BY decided_at DESC, decision_id DESC LIMIT 1""",
                    (claim,),
                ).fetchone()
            else:
                row = connection.execute(
                    """SELECT content_json, content_digest FROM claim_admission_decisions
                       WHERE claim_id=? AND processing_identity=? ORDER BY decision_id LIMIT 1""",
                    (claim, identifier(processing_identity, "processing_identity")),
                ).fetchone()
        return (
            None
            if row is None
            else self._decode(row[0], row[1], ClaimAdmissionDecision.from_dict, "ClaimAdmissionDecision")
        )

    def read_receipt(self, processing_identity: str) -> ClaimProcessingReceipt | None:
        self._require_initialized()
        processing = identifier(processing_identity, "processing_identity")
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT content_json, receipt_digest FROM claim_processing_receipts WHERE processing_identity=?",
                (processing,),
            ).fetchone()
        return (
            None
            if row is None
            else self._decode(
                row[0],
                row[1],
                ClaimProcessingReceipt.from_dict,
                "ClaimProcessingReceipt",
                digest_attribute="receipt_digest",
            )
        )

    def read_normalizer_runs(self, processing_identity: str) -> tuple[ClaimNormalizerRun, ...]:
        return self._read_processing_values(
            "claim_normalizer_runs", processing_identity, ClaimNormalizerRun.from_dict, "ClaimNormalizerRun"
        )

    def read_decisions(self, processing_identity: str) -> tuple[ClaimAdmissionDecision, ...]:
        return self._read_processing_values(
            "claim_admission_decisions", processing_identity, ClaimAdmissionDecision.from_dict, "ClaimAdmissionDecision"
        )

    def publish_processing(
        self,
        *,
        processing_identity: str,
        manifest: EvidenceManifest,
        normalizer_fingerprints: tuple[str, ...],
        normalizer_runs: tuple[ClaimNormalizerRun, ...],
        batches: tuple[ClaimBatch, ...],
        claims: tuple[Claim, ...],
        static_results: tuple[StaticAdmissionResult, ...],
        admission_policy: ClaimAdmissionPolicy,
        decided_at: datetime,
        published_at: datetime,
        completed_at: datetime,
    ) -> tuple[ClaimProcessingReceipt, bool]:
        self._require_initialized()
        processing = identifier(processing_identity, "processing_identity")
        decision_time = strict_utc(decided_at, "decided_at")
        publish_time = strict_utc(published_at, "published_at")
        completion_time = strict_utc(completed_at, "completed_at")
        if len(claims) != len(static_results) or any(
            claim.claim_id != result.claim_id for claim, result in zip(claims, static_results, strict=True)
        ):
            raise ClaimProcessingConflictError("static Admission results do not align with validated Claims")
        try:
            with self._transaction() as connection:
                receipt = self._receipt_by_id(connection, processing)
                if receipt is not None:
                    return receipt, True
                self._assert_no_partial_processing(connection, processing)
                durable_manifest = self._manifest_for_id(connection, manifest.manifest_id)
                if durable_manifest is None or durable_manifest.content_digest != manifest.content_digest:
                    raise ClaimProcessingConflictError("processing Manifest is absent or has conflicting content")
                if self._owner_digest(connection) != manifest.owner_identity_digest:
                    raise ClaimProcessingConflictError("processing Owner scope conflicts with the Store")
                self._require_publish_capacities(connection, normalizer_runs, batches, claims)
                for run in normalizer_runs:
                    self._insert_json_row(
                        connection,
                        "claim_normalizer_runs",
                        "run_id",
                        run.run_id,
                        run.to_dict(),
                        ("processing_identity", "manifest_id", "semantic_record_id", "normalizer_fingerprint"),
                        (processing, run.manifest_id, run.semantic_record_id, run.normalizer_fingerprint),
                    )
                for batch in batches:
                    self._insert_or_reuse_batch(connection, batch)
                decisions: list[ClaimAdmissionDecision] = []
                accepted: list[str] = []
                accepted_count = self._accepted_claim_count(connection)
                processing_semantics: dict[str, str] = {}
                for claim, static in zip(claims, static_results, strict=True):
                    existing_claim = self._claim_by_id(connection, claim.claim_id)
                    if existing_claim is not None:
                        existing_payload = existing_claim.to_dict()
                        candidate_payload = claim.to_dict()
                        existing_payload.pop("created_at")
                        candidate_payload.pop("created_at")
                        if canonical_json(existing_payload) != canonical_json(candidate_payload):
                            raise ClaimProcessingConflictError("Claim identity conflicts with durable content")
                    if existing_claim is None:
                        self._insert_claim(connection, claim)
                    if static.rejection_status is not None:
                        status, reason, related = static.rejection_status, static.reason_code, None
                    else:
                        recent = self._recent_state_claim(
                            connection,
                            claim,
                            seconds=admission_policy.config.repeat_state_suppression_seconds,
                        )
                        status, reason, related = admission_policy.evaluate_dynamic(
                            claim,
                            exact_claim_id=claim.claim_id if existing_claim is not None else None,
                            same_batch_claim_id=processing_semantics.get(claim.semantic_fingerprint),
                            recent_state_claim_id=recent,
                            capacity_reached=accepted_count >= self.config.store.max_claims,
                        )
                    decision = ClaimAdmissionDecision.create(
                        claim,
                        status,
                        reason,
                        processing_identity=processing,
                        decided_at=decision_time,
                        existing_claim_id=related,
                    )
                    self._insert_decision(connection, decision)
                    decisions.append(decision)
                    processing_semantics.setdefault(claim.semantic_fingerprint, claim.claim_id)
                    if status is ClaimAdmissionStatus.ACCEPTED:
                        accepted.append(claim.claim_id)
                        accepted_count += 1
                receipt = self._make_receipt(
                    processing=processing,
                    manifest=manifest,
                    normalizer_fingerprints=normalizer_fingerprints,
                    runs=normalizer_runs,
                    batches=batches,
                    claims=claims,
                    decisions=tuple(decisions),
                    accepted=tuple(accepted),
                    published_at=publish_time,
                    completed_at=completion_time,
                )
                self._insert_receipt(connection, receipt)
                return receipt, False
        except ClaimProcessingConflictError:
            raise
        except sqlite3.IntegrityError as exc:
            raise ClaimProcessingConflictError("atomic Claim publication conflicted") from exc
        except sqlite3.Error as exc:
            raise ClaimStoreError("failed to publish Claim processing atomically") from exc

    def _insert_record(self, connection: sqlite3.Connection, record: OwnerScopedSemanticRecord) -> None:
        payload = self._encode(record.to_dict())
        connection.execute(
            "INSERT INTO semantic_records VALUES(?,?,?,?,?,?,?,?,?)",
            (
                record.semantic_record_id,
                record.producer_fingerprint.digest,
                record.semantic_input.stream_id,
                record.semantic_input.source_sequence,
                record.owner_identity_digest,
                utc_text(record.semantic_input.event_time_start),
                utc_text(record.semantic_input.event_time_end),
                record.canonical_digest,
                payload,
            ),
        )

    @staticmethod
    def _require_decision_binding(
        decision: IngressDecision,
        record: OwnerScopedSemanticRecord,
    ) -> None:
        expected = (
            record.semantic_record_id,
            record.owner_identity_digest,
            record.producer_fingerprint.digest,
            record.semantic_input.stream_id,
            record.semantic_input.source_sequence,
            record.semantic_input.record_kind,
        )
        actual = (
            decision.semantic_record_id,
            decision.owner_identity_digest,
            decision.producer_fingerprint,
            decision.stream_id,
            decision.source_sequence,
            decision.record_kind,
        )
        if actual != expected:
            raise SemanticRecordConflictError("IngressDecision does not bind the supplied semantic record")

    def _insert_ingress_decision(self, connection: sqlite3.Connection, decision: IngressDecision) -> IngressDecision:
        row = connection.execute(
            "SELECT content_json, content_digest FROM semantic_ingress_decisions WHERE decision_id=?",
            (decision.decision_id,),
        ).fetchone()
        if row is not None:
            existing = self._decode(
                row[0],
                row[1],
                IngressDecision.from_dict,
                "IngressDecision",
                digest_attribute="content_digest",
            )
            if existing.content_digest != decision.content_digest:
                raise SemanticRecordConflictError("IngressDecision identity conflicts")
            return existing
        self._require_table_capacity(connection, "semantic_ingress_decisions", self.config.store.max_ingress_decisions)
        self._require_database_capacity()
        payload = self._encode(decision.to_dict())
        connection.execute(
            "INSERT INTO semantic_ingress_decisions VALUES(?,?,?,?,?,?)",
            (
                decision.decision_id,
                decision.semantic_record_id,
                decision.status.value,
                utc_text(decision.decided_at),
                decision.content_digest,
                payload,
            ),
        )
        return decision

    def _upsert_bundle(self, connection: sqlite3.Connection, bundle: SemanticEvidenceBundle) -> None:
        payload = self._encode(bundle.to_dict())
        digest = canonical_digest(bundle.to_dict())
        row = connection.execute(
            "SELECT state, content_json, content_digest FROM active_evidence_bundles WHERE bundle_id=?",
            (bundle.bundle_id,),
        ).fetchone()
        if row is not None and row[0] != EvidenceBundleState.OPEN.value:
            existing = self._decode(row[1], row[2], SemanticEvidenceBundle.from_dict, "SemanticEvidenceBundle")
            if canonical_json(existing.to_dict()) != payload:
                raise EvidenceBundleStateError("sealed Bundle is immutable")
            return
        connection.execute(
            """INSERT INTO active_evidence_bundles VALUES(?,?,?,?,?,?,?,?)
               ON CONFLICT(bundle_id) DO UPDATE SET state=excluded.state, watermark=excluded.watermark,
               content_digest=excluded.content_digest, content_json=excluded.content_json""",
            (
                bundle.bundle_id,
                bundle.grouping_key,
                bundle.generation,
                bundle.owner_identity_digest,
                bundle.state.value,
                None if bundle.watermark is None else utc_text(bundle.watermark),
                digest,
                payload,
            ),
        )
        connection.execute("DELETE FROM evidence_bundle_members WHERE bundle_id=?", (bundle.bundle_id,))
        connection.executemany(
            "INSERT INTO evidence_bundle_members VALUES(?,?,?)",
            tuple(
                (bundle.bundle_id, record_id, order)
                for order, record_id in enumerate(bundle.ordered_semantic_record_ids)
            ),
        )

    def _insert_manifest(self, connection: sqlite3.Connection, manifest: EvidenceManifest) -> None:
        existing = self._manifest_for_id(connection, manifest.manifest_id)
        if existing is not None:
            if canonical_json(existing.to_dict()) != canonical_json(manifest.to_dict()):
                raise EvidenceManifestError("Manifest identity conflicts with durable content")
            return
        payload = self._encode(manifest.to_dict())
        connection.execute(
            "INSERT INTO evidence_manifests VALUES(?,?,?,?,?,?,?,?)",
            (
                manifest.manifest_id,
                manifest.bundle_id,
                manifest.owner_identity_digest,
                utc_text(manifest.started_at),
                utc_text(manifest.ended_at),
                utc_text(manifest.sealed_at),
                manifest.content_digest,
                payload,
            ),
        )
        connection.executemany(
            "INSERT INTO evidence_manifest_members VALUES(?,?,?)",
            tuple(
                (manifest.manifest_id, snapshot.semantic_record_id, order)
                for order, snapshot in enumerate(manifest.ordered_record_snapshots)
            ),
        )

    @staticmethod
    def _retire_bundle_row(connection: sqlite3.Connection, bundle_id: str) -> None:
        connection.execute(
            "DELETE FROM evidence_bundle_members WHERE bundle_id=?",
            (bundle_id,),
        )
        connection.execute(
            "DELETE FROM active_evidence_bundles WHERE bundle_id=?",
            (bundle_id,),
        )

    def _insert_claim(self, connection: sqlite3.Connection, claim: Claim) -> None:
        payload = self._encode(claim.to_dict())
        connection.execute(
            "INSERT INTO claims VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                claim.claim_id,
                claim.claim_batch_id,
                claim.manifest_id,
                claim.semantic_record_id,
                claim.owner_identity_digest,
                claim.semantic_fingerprint,
                claim.proposal.claim_kind.value,
                utc_text(claim.time_start),
                utc_text(claim.time_end),
                utc_text(claim.created_at),
                canonical_digest(claim.to_dict()),
                payload,
            ),
        )

    def _insert_or_reuse_batch(self, connection: sqlite3.Connection, batch: ClaimBatch) -> None:
        row = connection.execute(
            "SELECT content_json, content_digest FROM claim_batches WHERE claim_batch_id=?",
            (batch.claim_batch_id,),
        ).fetchone()
        if row is not None:
            existing = self._decode(row[0], row[1], ClaimBatch.from_dict, "ClaimBatch")
            existing_payload = existing.to_dict()
            candidate_payload = batch.to_dict()
            for payload in (existing_payload, candidate_payload):
                payload.pop("processing_identity")
                payload.pop("created_at")
            if canonical_json(existing_payload) != canonical_json(candidate_payload):
                raise ClaimProcessingConflictError("ClaimBatch identity conflicts with durable content")
            return
        self._insert_json_row(
            connection,
            "claim_batches",
            "claim_batch_id",
            batch.claim_batch_id,
            batch.to_dict(),
            (
                "processing_identity",
                "manifest_id",
                "semantic_record_id",
                "normalizer_fingerprint",
                "created_at",
            ),
            (
                batch.processing_identity,
                batch.manifest_id,
                batch.semantic_record_id,
                batch.normalizer_fingerprint,
                utc_text(batch.created_at),
            ),
        )

    def _insert_decision(self, connection: sqlite3.Connection, decision: ClaimAdmissionDecision) -> None:
        self._require_table_capacity(connection, "claim_admission_decisions", self.config.store.max_admission_decisions)
        payload = self._encode(decision.to_dict())
        connection.execute(
            "INSERT INTO claim_admission_decisions VALUES(?,?,?,?,?,?,?)",
            (
                decision.decision_id,
                decision.processing_identity,
                decision.claim_id,
                decision.status.value,
                utc_text(decision.decided_at),
                canonical_digest(decision.to_dict()),
                payload,
            ),
        )

    def _insert_receipt(self, connection: sqlite3.Connection, receipt: ClaimProcessingReceipt) -> None:
        payload = self._encode(receipt.to_dict())
        connection.execute(
            "INSERT INTO claim_processing_receipts VALUES(?,?,?,?,?,?)",
            (
                receipt.processing_identity,
                receipt.manifest_id,
                utc_text(receipt.completed_at),
                utc_text(receipt.published_at),
                receipt.receipt_digest,
                payload,
            ),
        )

    def _make_receipt(
        self,
        *,
        processing: str,
        manifest: EvidenceManifest,
        normalizer_fingerprints: tuple[str, ...],
        runs: tuple[ClaimNormalizerRun, ...],
        batches: tuple[ClaimBatch, ...],
        claims: tuple[Claim, ...],
        decisions: tuple[ClaimAdmissionDecision, ...],
        accepted: tuple[str, ...],
        published_at: datetime,
        completed_at: datetime,
    ) -> ClaimProcessingReceipt:
        normalizer_run_ids = tuple(item.run_id for item in runs)
        claim_batch_ids = tuple(item.claim_batch_id for item in batches)
        claim_ids = tuple(item.claim_id for item in claims)
        decision_ids = tuple(item.decision_id for item in decisions)
        digest_payload = {
            "processing_identity": processing,
            "manifest_id": manifest.manifest_id,
            "manifest_digest": manifest.content_digest,
            "normalizer_fingerprints": normalizer_fingerprints,
            "normalizer_run_ids": normalizer_run_ids,
            "claim_batch_ids": claim_batch_ids,
            "claim_ids": claim_ids,
            "accepted_claim_ids": accepted,
            "decision_ids": decision_ids,
            "completed_at": utc_text(completed_at),
            "published_at": utc_text(published_at),
            "schema_version": "2",
        }
        return ClaimProcessingReceipt(
            processing_identity=processing,
            manifest_id=manifest.manifest_id,
            manifest_digest=manifest.content_digest,
            normalizer_fingerprints=normalizer_fingerprints,
            normalizer_run_ids=normalizer_run_ids,
            claim_batch_ids=claim_batch_ids,
            claim_ids=claim_ids,
            accepted_claim_ids=accepted,
            decision_ids=decision_ids,
            completed_at=completed_at,
            published_at=published_at,
            receipt_digest=canonical_digest(digest_payload),
            schema_version="2",
        )

    def _insert_json_row(
        self,
        connection: sqlite3.Connection,
        table: str,
        key_name: str,
        key: str,
        value: dict[str, object],
        extra_names: tuple[str, ...],
        extra_values: tuple[object, ...],
    ) -> None:
        payload = self._encode(value)
        digest = canonical_digest(value)
        columns = (key_name, *extra_names, "content_digest", "content_json")
        placeholders = ",".join("?" for _ in columns)
        connection.execute(
            f"INSERT INTO {table}({','.join(columns)}) VALUES({placeholders})",
            (key, *extra_values, digest, payload),
        )

    def _record_by_id(self, connection: sqlite3.Connection, record_id: str) -> OwnerScopedSemanticRecord | None:
        row = connection.execute(
            "SELECT content_json, canonical_digest FROM semantic_records WHERE semantic_record_id=?", (record_id,)
        ).fetchone()
        return (
            None
            if row is None
            else self._decode(
                row[0],
                row[1],
                lambda value: OwnerScopedSemanticRecord.from_dict(value, config=self.config.ingress),
                "OwnerScopedSemanticRecord",
                digest_attribute="canonical_digest",
            )
        )

    def _claim_by_id(self, connection: sqlite3.Connection, claim_id: str) -> Claim | None:
        row = connection.execute(
            "SELECT content_json, content_digest FROM claims WHERE claim_id=?", (claim_id,)
        ).fetchone()
        return None if row is None else self._decode(row[0], row[1], Claim.from_dict, "Claim")

    def _receipt_by_id(self, connection: sqlite3.Connection, processing: str) -> ClaimProcessingReceipt | None:
        row = connection.execute(
            "SELECT content_json, receipt_digest FROM claim_processing_receipts WHERE processing_identity=?",
            (processing,),
        ).fetchone()
        return (
            None
            if row is None
            else self._decode(
                row[0],
                row[1],
                ClaimProcessingReceipt.from_dict,
                "ClaimProcessingReceipt",
                digest_attribute="receipt_digest",
            )
        )

    def _manifest_for_id(self, connection: sqlite3.Connection, manifest_id: str) -> EvidenceManifest | None:
        row = connection.execute(
            "SELECT content_json, content_digest FROM evidence_manifests WHERE manifest_id=?", (manifest_id,)
        ).fetchone()
        return (
            None
            if row is None
            else self._decode(
                row[0], row[1], EvidenceManifest.from_dict, "EvidenceManifest", digest_attribute="content_digest"
            )
        )

    def _manifest_for_bundle(self, connection: sqlite3.Connection, bundle_id: str) -> EvidenceManifest | None:
        row = connection.execute(
            "SELECT content_json, content_digest FROM evidence_manifests WHERE bundle_id=?", (bundle_id,)
        ).fetchone()
        return (
            None
            if row is None
            else self._decode(
                row[0], row[1], EvidenceManifest.from_dict, "EvidenceManifest", digest_attribute="content_digest"
            )
        )

    def _active_by_group(self, connection: sqlite3.Connection, grouping_key: str) -> SemanticEvidenceBundle | None:
        row = connection.execute(
            """SELECT content_json, content_digest FROM active_evidence_bundles
               WHERE grouping_key=? AND state='OPEN' ORDER BY generation DESC LIMIT 1""",
            (grouping_key,),
        ).fetchone()
        return (
            None
            if row is None
            else self._decode(row[0], row[1], SemanticEvidenceBundle.from_dict, "SemanticEvidenceBundle")
        )

    def _active_for_record(self, connection: sqlite3.Connection, record_id: str) -> SemanticEvidenceBundle | None:
        row = connection.execute(
            """SELECT b.content_json, b.content_digest FROM active_evidence_bundles b
               JOIN evidence_bundle_members m ON m.bundle_id=b.bundle_id
               WHERE m.semantic_record_id=? AND b.state='OPEN' ORDER BY b.generation DESC LIMIT 1""",
            (record_id,),
        ).fetchone()
        return (
            None
            if row is None
            else self._decode(row[0], row[1], SemanticEvidenceBundle.from_dict, "SemanticEvidenceBundle")
        )

    def _bundle_records(self, connection: sqlite3.Connection, bundle_id: str) -> tuple[OwnerScopedSemanticRecord, ...]:
        rows = connection.execute(
            """SELECT r.content_json, r.canonical_digest FROM semantic_records r
               JOIN evidence_bundle_members m ON m.semantic_record_id=r.semantic_record_id
               WHERE m.bundle_id=? ORDER BY m.member_order""",
            (bundle_id,),
        ).fetchall()
        return tuple(
            self._decode(
                row[0],
                row[1],
                lambda value: OwnerScopedSemanticRecord.from_dict(value, config=self.config.ingress),
                "OwnerScopedSemanticRecord",
                digest_attribute="canonical_digest",
            )
            for row in rows
        )

    def _group_watermark(
        self, connection: sqlite3.Connection, grouping_key: str
    ) -> tuple[datetime | None, datetime | None, int]:
        row = connection.execute(
            "SELECT max_event_time, watermark, latest_generation FROM evidence_watermarks WHERE grouping_key=?",
            (grouping_key,),
        ).fetchone()
        if row is None:
            return None, None, -1
        return (
            None if row[0] is None else parse_utc(row[0], "max_event_time"),
            None if row[1] is None else parse_utc(row[1], "watermark"),
            int(row[2]),
        )

    def _commit_watermark(
        self,
        connection: sqlite3.Connection,
        grouping_key: str,
        owner_digest: str,
        max_event_time: datetime | None,
        watermark: datetime | None,
        generation: int,
    ) -> None:
        connection.execute(
            """INSERT INTO evidence_watermarks VALUES(?,?,?,?,?)
               ON CONFLICT(grouping_key) DO UPDATE SET max_event_time=excluded.max_event_time,
               watermark=excluded.watermark, latest_generation=excluded.latest_generation""",
            (
                grouping_key,
                owner_digest,
                None if max_event_time is None else utc_text(max_event_time),
                None if watermark is None else utc_text(watermark),
                generation,
            ),
        )

    def _recent_state_claim(self, connection: sqlite3.Connection, claim: Claim, *, seconds: float) -> str | None:
        if claim.proposal.claim_kind is not ClaimKind.STATE_ASSERTION:
            return None
        lower = claim.time_start - timedelta(seconds=seconds)
        upper = claim.time_end + timedelta(seconds=seconds)
        row = connection.execute(
            """SELECT c.claim_id FROM claims c JOIN claim_admission_decisions d ON d.claim_id=c.claim_id
               WHERE c.semantic_fingerprint=? AND c.time_end>=? AND c.time_end<=? AND c.claim_id<>?
               AND d.status='ACCEPTED' ORDER BY c.time_end DESC, c.claim_id DESC LIMIT 1""",
            (claim.semantic_fingerprint, utc_text(lower), utc_text(upper), claim.claim_id),
        ).fetchone()
        return None if row is None else str(row[0])

    def _list_claim_query(
        self,
        *,
        start: datetime,
        end: datetime,
        limit: int,
        cursor: str | None,
        accepted_only: bool,
    ) -> tuple[Claim, ...]:
        self._require_initialized()
        start_time, end_time = self._bounded_range(start, end)
        size = self._bounded_limit(limit, self.config.claim.max_query_limit)
        with closing(self._connect()) as connection:
            cursor_time: str | None = None
            cursor_value: str | None = None
            if cursor is not None:
                cursor_value = identifier(cursor, "cursor")
                cursor_row = connection.execute(
                    "SELECT created_at FROM claims WHERE claim_id=?",
                    (cursor_value,),
                ).fetchone()
                if cursor_row is None:
                    raise ValueError("Claim query cursor does not exist")
                cursor_time = str(cursor_row[0])
            if accepted_only:
                if cursor_value is None or cursor_time is None:
                    rows = connection.execute(
                        """SELECT DISTINCT c.content_json, c.content_digest, c.created_at, c.claim_id
                           FROM claims c JOIN claim_admission_decisions d ON d.claim_id=c.claim_id
                           WHERE c.created_at>=? AND c.created_at<=? AND d.status='ACCEPTED'
                           ORDER BY c.created_at, c.claim_id LIMIT ?""",
                        (utc_text(start_time), utc_text(end_time), size),
                    ).fetchall()
                else:
                    rows = connection.execute(
                        """SELECT DISTINCT c.content_json, c.content_digest, c.created_at, c.claim_id
                           FROM claims c JOIN claim_admission_decisions d ON d.claim_id=c.claim_id
                           WHERE c.created_at>=? AND c.created_at<=? AND d.status='ACCEPTED'
                           AND (c.created_at>? OR (c.created_at=? AND c.claim_id>?))
                           ORDER BY c.created_at, c.claim_id LIMIT ?""",
                        (
                            utc_text(start_time),
                            utc_text(end_time),
                            cursor_time,
                            cursor_time,
                            cursor_value,
                            size,
                        ),
                    ).fetchall()
            else:
                if cursor_value is None or cursor_time is None:
                    rows = connection.execute(
                        """SELECT content_json, content_digest, created_at, claim_id FROM claims
                           WHERE created_at>=? AND created_at<=? ORDER BY created_at, claim_id LIMIT ?""",
                        (utc_text(start_time), utc_text(end_time), size),
                    ).fetchall()
                else:
                    rows = connection.execute(
                        """SELECT content_json, content_digest, created_at, claim_id FROM claims
                           WHERE created_at>=? AND created_at<=?
                           AND (created_at>? OR (created_at=? AND claim_id>?))
                           ORDER BY created_at, claim_id LIMIT ?""",
                        (
                            utc_text(start_time),
                            utc_text(end_time),
                            cursor_time,
                            cursor_time,
                            cursor_value,
                            size,
                        ),
                    ).fetchall()
        return tuple(self._decode(row[0], row[1], Claim.from_dict, "Claim") for row in rows)

    def _read_processing_values(
        self,
        table: str,
        processing_identity: str,
        factory: Callable[[object], _T],
        label: str,
    ) -> tuple[_T, ...]:
        self._require_initialized()
        processing = identifier(processing_identity, "processing_identity")
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"SELECT content_json, content_digest FROM {table} WHERE processing_identity=? ORDER BY 1",
                (processing,),
            ).fetchall()
        return tuple(self._decode(row[0], row[1], factory, label) for row in rows)

    def _assert_no_partial_processing(self, connection: sqlite3.Connection, processing: str) -> None:
        for table in ("claim_normalizer_runs", "claim_batches", "claim_admission_decisions"):
            if (
                connection.execute(
                    f"SELECT 1 FROM {table} WHERE processing_identity=? LIMIT 1", (processing,)
                ).fetchone()
                is not None
            ):
                raise ClaimProcessingConflictError("processing identity has partial durable state")

    def _require_publish_capacities(
        self,
        connection: sqlite3.Connection,
        runs: tuple[ClaimNormalizerRun, ...],
        batches: tuple[ClaimBatch, ...],
        claims: tuple[Claim, ...],
    ) -> None:
        capacities = (
            (
                "claim_normalizer_runs",
                self.config.store.max_normalizer_runs,
                self._missing_rows(connection, "claim_normalizer_runs", "run_id", tuple(item.run_id for item in runs)),
            ),
            (
                "claim_batches",
                self.config.store.max_claim_batches,
                self._missing_rows(
                    connection,
                    "claim_batches",
                    "claim_batch_id",
                    tuple(item.claim_batch_id for item in batches),
                ),
            ),
            (
                "claims",
                self.config.store.max_admission_decisions,
                self._missing_rows(connection, "claims", "claim_id", tuple(item.claim_id for item in claims)),
            ),
            ("claim_admission_decisions", self.config.store.max_admission_decisions, len(claims)),
            ("claim_processing_receipts", self.config.store.max_receipts, 1),
        )
        for table, maximum, increment in capacities:
            if self._count(connection, table) + increment > maximum:
                raise ClaimStoreCapacityError(f"{table} capacity has been reached")
        self._require_database_capacity()

    @staticmethod
    def _missing_rows(
        connection: sqlite3.Connection,
        table: str,
        key_name: str,
        keys: tuple[str, ...],
    ) -> int:
        return sum(
            connection.execute(
                f"SELECT 1 FROM {table} WHERE {key_name}=? LIMIT 1",
                (key,),
            ).fetchone()
            is None
            for key in keys
        )

    def _require_active_bundle_capacity(self, connection: sqlite3.Connection) -> None:
        count = connection.execute("SELECT COUNT(*) FROM active_evidence_bundles WHERE state='OPEN'").fetchone()[0]
        maximum = min(self.config.evidence.max_active_bundles, self.config.store.max_active_bundles)
        if count >= maximum:
            raise ClaimStoreCapacityError("active Bundle capacity has been reached")

    def _require_table_capacity(self, connection: sqlite3.Connection, table: str, maximum: int) -> None:
        if self._count(connection, table) >= maximum:
            raise ClaimStoreCapacityError(f"{table} capacity has been reached")

    def _require_database_capacity(self) -> None:
        if self._database_bytes() >= self.config.store.max_database_bytes:
            raise ClaimStoreCapacityError("Behavior database byte capacity has been reached")

    def _database_bytes(self) -> int:
        return sum(
            candidate.stat().st_size
            for candidate in (
                self.path,
                Path(str(self.path) + "-wal"),
                Path(str(self.path) + "-shm"),
            )
            if candidate.exists()
        )

    @staticmethod
    def _count(connection: sqlite3.Connection, table: str) -> int:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    @staticmethod
    def _accepted_claim_count(connection: sqlite3.Connection) -> int:
        return int(
            connection.execute(
                "SELECT COUNT(DISTINCT claim_id) FROM claim_admission_decisions WHERE status='ACCEPTED'"
            ).fetchone()[0]
        )

    def _bind_owner(self, connection: sqlite3.Connection, owner_digest: str) -> None:
        resolved = sha256_digest(owner_digest, "owner_identity_digest")
        current = self._owner_digest(connection)
        if current is None:
            connection.execute(
                "INSERT INTO behavior_metadata(key, value) VALUES('owner_identity_digest', ?)",
                (resolved,),
            )
        elif current != resolved:
            raise BehaviorOwnerConflictError("Behavior Store is permanently bound to another Owner identity")

    @staticmethod
    def _owner_digest(connection: sqlite3.Connection) -> str | None:
        row = connection.execute("SELECT value FROM behavior_metadata WHERE key='owner_identity_digest'").fetchone()
        return None if row is None else sha256_digest(row[0], "owner_identity_digest")

    def _encode(self, value: object) -> str:
        payload = canonical_json(value)
        if len(payload.encode("utf-8")) > self.config.store.max_json_bytes:
            raise ClaimStoreCapacityError("canonical JSON value exceeds the Store boundary")
        return payload

    @staticmethod
    def _decode(
        payload: str,
        stored_digest: str,
        factory: Callable[[object], _T],
        label: str,
        *,
        digest_attribute: str | None = None,
    ) -> _T:
        try:
            decoded = json.loads(payload)
            if canonical_json(decoded) != payload:
                raise ClaimStoreError(f"{label} is not stored in canonical JSON")
            result = factory(decoded)
            actual = getattr(result, digest_attribute) if digest_attribute is not None else canonical_digest(decoded)
            if actual != stored_digest:
                raise ClaimStoreError(f"{label} durable digest mismatch")
            if canonical_json(result.to_dict()) != payload:
                raise ClaimStoreError(f"{label} canonical read-back mismatch")
            return result
        except ClaimStoreError:
            raise
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ClaimStoreError(f"failed strict read-back for {label}") from exc

    def _bounded_range(self, start: datetime, end: datetime) -> tuple[datetime, datetime]:
        self._require_initialized()
        begin = strict_utc(start, "start")
        finish = strict_utc(end, "end")
        if finish < begin:
            raise ValueError("query end cannot precede start")
        return begin, finish

    @staticmethod
    def _bounded_limit(limit: int, maximum: int) -> int:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= maximum:
            raise ValueError("query limit is outside its configured boundary")
        return limit

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        self._require_initialized()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    def _connect(self) -> sqlite3.Connection:
        self._reject_symlink_targets()
        connection = sqlite3.connect(
            self.path,
            timeout=self.config.store.sqlite_timeout_seconds,
            isolation_level=None,
        )
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(f"PRAGMA busy_timeout={int(self.config.store.sqlite_timeout_seconds * 1000)}")
        return connection

    def _validate_schema(self, connection: sqlite3.Connection) -> None:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        }
        if tables != set(_TABLE_COLUMNS):
            raise ClaimStoreError("Behavior SQLite table set does not match Schema V2")
        for table, expected in _TABLE_COLUMNS.items():
            actual = tuple(row[1] for row in connection.execute(f"PRAGMA table_info({table})"))
            if actual != expected:
                raise ClaimStoreError(f"Behavior SQLite columns do not match Schema V2 for {table}")
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
            )
        }
        if not _REQUIRED_INDEXES.issubset(indexes):
            raise ClaimStoreError("Behavior SQLite indexes do not match Schema V2")
        with closing(sqlite3.connect(":memory:")) as reference:
            for statement in _SCHEMA_STATEMENTS:
                reference.execute(statement)
            expected_tables = self._schema_sql(reference, "table")
            expected_indexes = self._schema_sql(reference, "index")
        actual_tables = self._schema_sql(connection, "table")
        actual_indexes = self._schema_sql(connection, "index")
        for table in _TABLE_COLUMNS:
            if actual_tables.get(table) != expected_tables.get(table):
                raise ClaimStoreError(f"Behavior SQLite table definition does not match Schema V2 for {table}")
        for index in _REQUIRED_INDEXES:
            if actual_indexes.get(index) != expected_indexes.get(index):
                raise ClaimStoreError(f"Behavior SQLite index definition does not match Schema V2 for {index}")

    @staticmethod
    def _schema_sql(connection: sqlite3.Connection, object_type: str) -> dict[str, str]:
        return {
            str(name): " ".join(str(sql).split())
            for name, sql in connection.execute(
                """SELECT name, sql FROM sqlite_master
                   WHERE type=? AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL""",
                (object_type,),
            )
        }

    def _ensure_root(self) -> None:
        self._reject_existing_symlink_components(self.root)
        try:
            self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
            if self.root.is_symlink() or not self.root.is_dir():
                raise ClaimStoreError("behavior_root must be a non-symlink directory")
            os.chmod(self.root, 0o700)
        except OSError as exc:
            raise ClaimStoreError("failed to prepare behavior_root") from exc

    def _reject_symlink_targets(self) -> None:
        self._reject_existing_symlink_components(self.root)
        if self.root.exists() and self.root.is_symlink():
            raise ClaimStoreError("behavior_root cannot be a symlink")
        for candidate in (self.path, Path(str(self.path) + "-wal"), Path(str(self.path) + "-shm")):
            if candidate.exists() and candidate.is_symlink():
                raise ClaimStoreError("Behavior database files cannot be symlinks")

    @staticmethod
    def _reject_existing_symlink_components(path: Path) -> None:
        current = Path(path.anchor)
        for part in path.parts[1:]:
            current = current / part
            if current.exists() and current.is_symlink():
                raise ClaimStoreError("Behavior storage path cannot contain symlinks")

    def _require_initialized(self) -> None:
        if not self.initialized:
            raise ClaimStoreError("Behavior Store is not initialized")


__all__ = ["SQLiteBehaviorEvidenceClaimStore"]
