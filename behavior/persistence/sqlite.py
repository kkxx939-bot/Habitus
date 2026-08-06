"""Single-owner SQLite Schema V3 for semantic evidence and Claim publication."""

from __future__ import annotations

import json
import os
import sqlite3
from collections.abc import Callable, Iterator, Sequence
from contextlib import closing, contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

from behavior._validation import identifier, parse_utc, sha256_digest, strict_utc, utc_text
from behavior.claim.admission import (
    ClaimAdmissionDecision,
    ClaimAdmissionPolicy,
    ClaimAdmissionStatus,
    StaticAdmissionResult,
)
from behavior.claim.model import (
    Claim,
    ClaimBatch,
    ClaimNormalizerAttempt,
    ClaimNormalizerAttemptStatus,
    ClaimProcessingReceipt,
)
from behavior.claim.policy import ClaimDerivationClass, ClaimProcessingLane
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
from behavior.ingress.model import IngressDecision, IngressDecisionStatus, OwnerScopedSemanticRecord
from behavior.ingress.service import AcceptedSemanticIngress, Clock, SystemClock
from foundation.integrity import canonical_digest, canonical_json

_SCHEMA_VERSION = "3"
_DATABASE_NAME = "evidence_claims.sqlite3"
_SQLITE_ID_CHUNK = 400


class _DurableValue(Protocol):
    @property
    def content_digest(self) -> str: ...

    def to_dict(self) -> dict[str, object]: ...


_T = TypeVar("_T")

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
        "semantic_digest",
        "content_digest",
        "encoded_digest",
        "content_json",
    ),
    "semantic_ingress_decisions": (
        "decision_id",
        "semantic_record_id",
        "owner_identity_digest",
        "status",
        "decided_at",
        "decision_identity_digest",
        "content_digest",
        "encoded_digest",
        "content_json",
    ),
    "active_evidence_bundles": (
        "bundle_id",
        "grouping_key",
        "generation",
        "owner_identity_digest",
        "state",
        "watermark",
        "encoded_digest",
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
        "manifest_semantic_digest",
        "content_digest",
        "encoded_digest",
        "content_json",
    ),
    "evidence_manifest_members": ("manifest_id", "semantic_record_id", "member_order"),
    "claim_normalizer_attempts": (
        "attempt_id",
        "processing_identity",
        "processing_lane",
        "manifest_id",
        "semantic_record_id",
        "normalizer_fingerprint",
        "attempt_number",
        "status",
        "content_digest",
        "encoded_digest",
        "content_json",
    ),
    "claim_batches": (
        "claim_batch_id",
        "processing_identity",
        "processing_lane",
        "manifest_id",
        "semantic_record_id",
        "normalizer_fingerprint",
        "created_at",
        "content_digest",
        "encoded_digest",
        "content_json",
    ),
    "claim_batch_members": ("claim_batch_id", "claim_id", "member_order"),
    "claims": (
        "claim_id",
        "manifest_id",
        "semantic_record_id",
        "owner_identity_digest",
        "semantic_fingerprint",
        "derivation_class",
        "claim_kind",
        "time_start",
        "time_end",
        "created_at",
        "content_digest",
        "encoded_digest",
        "content_json",
    ),
    "claim_admission_decisions": (
        "decision_id",
        "processing_identity",
        "claim_id",
        "admission_policy_digest",
        "status",
        "decided_at",
        "content_digest",
        "encoded_digest",
        "content_json",
    ),
    "claim_processing_receipts": (
        "processing_identity",
        "processing_lane",
        "scope_semantic_record_id",
        "manifest_id",
        "admission_policy_digest",
        "completed_at",
        "publication_recorded_at",
        "content_digest",
        "encoded_digest",
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
        "idx_attempts_processing",
        "idx_attempts_route_number",
        "idx_batches_processing",
        "idx_batch_members_claim",
        "idx_claims_manifest",
        "idx_claims_record",
        "idx_claims_created",
        "idx_claims_semantic_time",
        "idx_claims_semantic_derivation_time",
        "idx_decisions_processing",
        "idx_decisions_claim_policy",
        "idx_decisions_policy_status",
        "idx_receipts_completed",
        "idx_receipts_scope",
    }
)

_SCHEMA_STATEMENTS = (
    "CREATE TABLE behavior_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
    """CREATE TABLE semantic_records (
        semantic_record_id TEXT PRIMARY KEY, producer_fingerprint TEXT NOT NULL, stream_id TEXT NOT NULL,
        source_sequence INTEGER NOT NULL CHECK(source_sequence >= 0), owner_identity_digest TEXT NOT NULL,
        event_time_start TEXT NOT NULL, event_time_end TEXT NOT NULL, semantic_digest TEXT NOT NULL,
        content_digest TEXT NOT NULL, encoded_digest TEXT NOT NULL, content_json TEXT NOT NULL
    )""",
    "CREATE UNIQUE INDEX idx_semantic_stream_sequence ON semantic_records(producer_fingerprint, stream_id, source_sequence)",
    "CREATE INDEX idx_semantic_event_time ON semantic_records(event_time_start, semantic_record_id)",
    """CREATE TABLE semantic_ingress_decisions (
        decision_id TEXT PRIMARY KEY, semantic_record_id TEXT NOT NULL, owner_identity_digest TEXT NOT NULL,
        status TEXT NOT NULL, decided_at TEXT NOT NULL, decision_identity_digest TEXT NOT NULL,
        content_digest TEXT NOT NULL, encoded_digest TEXT NOT NULL, content_json TEXT NOT NULL
    )""",
    "CREATE INDEX idx_ingress_status_time ON semantic_ingress_decisions(status, decided_at, decision_id)",
    """CREATE TABLE active_evidence_bundles (
        bundle_id TEXT PRIMARY KEY, grouping_key TEXT NOT NULL, generation INTEGER NOT NULL CHECK(generation >= 0),
        owner_identity_digest TEXT NOT NULL, state TEXT NOT NULL, watermark TEXT, encoded_digest TEXT NOT NULL,
        content_json TEXT NOT NULL, UNIQUE(grouping_key, generation)
    )""",
    "CREATE INDEX idx_bundle_group_state ON active_evidence_bundles(grouping_key, state, generation)",
    """CREATE TABLE evidence_bundle_members (
        bundle_id TEXT NOT NULL REFERENCES active_evidence_bundles(bundle_id) ON DELETE CASCADE,
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
        manifest_semantic_digest TEXT NOT NULL, content_digest TEXT NOT NULL, encoded_digest TEXT NOT NULL,
        content_json TEXT NOT NULL
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
    """CREATE TABLE claim_normalizer_attempts (
        attempt_id TEXT PRIMARY KEY, processing_identity TEXT NOT NULL, processing_lane TEXT NOT NULL,
        manifest_id TEXT NOT NULL REFERENCES evidence_manifests(manifest_id),
        semantic_record_id TEXT NOT NULL REFERENCES semantic_records(semantic_record_id),
        normalizer_fingerprint TEXT NOT NULL, attempt_number INTEGER NOT NULL CHECK(attempt_number > 0),
        status TEXT NOT NULL, content_digest TEXT NOT NULL, encoded_digest TEXT NOT NULL, content_json TEXT NOT NULL,
        UNIQUE(processing_identity, semantic_record_id, normalizer_fingerprint, attempt_number)
    )""",
    "CREATE INDEX idx_attempts_processing ON claim_normalizer_attempts(processing_identity, attempt_id)",
    "CREATE INDEX idx_attempts_route_number ON claim_normalizer_attempts(processing_identity, semantic_record_id, normalizer_fingerprint, attempt_number)",
    """CREATE TABLE claim_batches (
        claim_batch_id TEXT PRIMARY KEY, processing_identity TEXT NOT NULL, processing_lane TEXT NOT NULL,
        manifest_id TEXT NOT NULL REFERENCES evidence_manifests(manifest_id),
        semantic_record_id TEXT NOT NULL REFERENCES semantic_records(semantic_record_id),
        normalizer_fingerprint TEXT NOT NULL, created_at TEXT NOT NULL, content_digest TEXT NOT NULL,
        encoded_digest TEXT NOT NULL, content_json TEXT NOT NULL
    )""",
    "CREATE INDEX idx_batches_processing ON claim_batches(processing_identity, claim_batch_id)",
    """CREATE TABLE claims (
        claim_id TEXT PRIMARY KEY, manifest_id TEXT NOT NULL REFERENCES evidence_manifests(manifest_id),
        semantic_record_id TEXT NOT NULL REFERENCES semantic_records(semantic_record_id),
        owner_identity_digest TEXT NOT NULL, semantic_fingerprint TEXT NOT NULL,
        derivation_class TEXT NOT NULL, claim_kind TEXT NOT NULL,
        time_start TEXT NOT NULL, time_end TEXT NOT NULL, created_at TEXT NOT NULL,
        content_digest TEXT NOT NULL, encoded_digest TEXT NOT NULL, content_json TEXT NOT NULL
    )""",
    "CREATE INDEX idx_claims_manifest ON claims(manifest_id, claim_id)",
    "CREATE INDEX idx_claims_record ON claims(semantic_record_id, claim_id)",
    "CREATE INDEX idx_claims_created ON claims(created_at, claim_id)",
    "CREATE INDEX idx_claims_semantic_time ON claims(semantic_fingerprint, time_end, claim_id)",
    "CREATE INDEX idx_claims_semantic_derivation_time ON claims(semantic_fingerprint, derivation_class, time_end, claim_id)",
    """CREATE TABLE claim_batch_members (
        claim_batch_id TEXT NOT NULL REFERENCES claim_batches(claim_batch_id),
        claim_id TEXT NOT NULL REFERENCES claims(claim_id), member_order INTEGER NOT NULL CHECK(member_order >= 0),
        PRIMARY KEY(claim_batch_id, claim_id), UNIQUE(claim_batch_id, member_order)
    )""",
    "CREATE INDEX idx_batch_members_claim ON claim_batch_members(claim_id, claim_batch_id)",
    """CREATE TABLE claim_admission_decisions (
        decision_id TEXT PRIMARY KEY, processing_identity TEXT NOT NULL,
        claim_id TEXT NOT NULL REFERENCES claims(claim_id), admission_policy_digest TEXT NOT NULL,
        status TEXT NOT NULL, decided_at TEXT NOT NULL, content_digest TEXT NOT NULL,
        encoded_digest TEXT NOT NULL, content_json TEXT NOT NULL,
        UNIQUE(processing_identity, claim_id, admission_policy_digest)
    )""",
    "CREATE INDEX idx_decisions_processing ON claim_admission_decisions(processing_identity, decision_id)",
    "CREATE INDEX idx_decisions_claim_policy ON claim_admission_decisions(claim_id, admission_policy_digest, decided_at, decision_id)",
    "CREATE INDEX idx_decisions_policy_status ON claim_admission_decisions(admission_policy_digest, status, decided_at, claim_id)",
    """CREATE TABLE claim_processing_receipts (
        processing_identity TEXT PRIMARY KEY, processing_lane TEXT NOT NULL, scope_semantic_record_id TEXT,
        manifest_id TEXT NOT NULL REFERENCES evidence_manifests(manifest_id), admission_policy_digest TEXT NOT NULL,
        completed_at TEXT NOT NULL, publication_recorded_at TEXT NOT NULL, content_digest TEXT NOT NULL,
        encoded_digest TEXT NOT NULL, content_json TEXT NOT NULL
    )""",
    "CREATE INDEX idx_receipts_completed ON claim_processing_receipts(completed_at, processing_identity)",
    "CREATE INDEX idx_receipts_scope ON claim_processing_receipts(manifest_id, processing_lane, scope_semantic_record_id)",
)


class SQLiteBehaviorEvidenceClaimStore:
    def __init__(
        self,
        behavior_root: str | Path,
        *,
        config: BehaviorConfig,
        clock: Clock | None = None,
        initialize: bool = False,
    ) -> None:
        if not isinstance(config, BehaviorConfig):
            raise TypeError("config must be BehaviorConfig")
        requested = Path(behavior_root).expanduser().absolute()
        if requested.exists() and requested.is_symlink():
            raise ClaimStoreError("behavior_root cannot be a symlink")
        self._reject_existing_symlink_components(requested)
        self.root = requested.resolve(strict=False)
        self.path = self.root / _DATABASE_NAME
        self.config = config
        self.clock = clock or SystemClock()
        if not isinstance(self.clock, Clock):
            raise TypeError("clock must implement Clock")
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
                                "Behavior database schema version mismatch; V1/V2 migration is not supported"
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
            return True, f"schema={_SCHEMA_VERSION}"
        except Exception as exc:
            return False, type(exc).__name__

    def health_snapshot(self, *, admission_policy_digest: str | None = None) -> dict[str, int | str | bool]:
        self._require_initialized()
        policy = None if admission_policy_digest is None else sha256_digest(
            admission_policy_digest, "admission_policy_digest"
        )
        with closing(self._connect()) as connection:
            accepted = 0 if policy is None else self._accepted_claim_count(connection, policy)
            snapshot = {
                "schema_version": _SCHEMA_VERSION,
                "semantic_record_count": self._count(connection, "semantic_records"),
                "active_bundle_count": connection.execute(
                    "SELECT COUNT(*) FROM active_evidence_bundles WHERE state='OPEN'"
                ).fetchone()[0],
                "manifest_count": self._count(connection, "evidence_manifests"),
                "validated_claim_count": self._count(connection, "claims"),
                "current_policy_accepted_claim_count": accepted,
                "admission_decision_count": self._count(connection, "claim_admission_decisions"),
                "normalizer_attempt_count": self._count(connection, "claim_normalizer_attempts"),
                "processing_receipt_count": self._count(connection, "claim_processing_receipts"),
            }
        size = self._database_bytes()
        return {
            **snapshot,
            "database_bytes": size,
            "database_size_warning": size >= self.config.store.max_database_bytes,
        }

    def owner_identity_digest(self) -> str | None:
        self._require_initialized()
        with closing(self._connect()) as connection:
            return self._owner_digest(connection)

    def record_ingress_decision(
        self,
        decision: IngressDecision,
        *,
        record: OwnerScopedSemanticRecord,
    ) -> IngressDecision:
        self._require_initialized()
        self._require_decision_binding(decision, record)
        if decision.status not in {
            IngressDecisionStatus.CLOCK_SKEW_REJECTED,
            IngressDecisionStatus.EVENT_TOO_OLD_REJECTED,
            IngressDecisionStatus.CAPACITY_REJECTED,
        }:
            raise SemanticRecordConflictError("only pre-persistence ingress rejections may be recorded directly")
        with self._transaction() as connection:
            self._bind_owner(connection, record.owner_identity_digest)
            return self._insert_ingress_decision(connection, decision)

    def ingest_semantic_record(
        self,
        accepted: AcceptedSemanticIngress,
        assembler: SemanticEvidenceBundleAssembler,
        *,
        sealed_at: datetime,
    ) -> SemanticIngestResult:
        self._require_initialized()
        if not isinstance(accepted, AcceptedSemanticIngress):
            raise TypeError("accepted must be AcceptedSemanticIngress")
        if not isinstance(assembler, SemanticEvidenceBundleAssembler):
            raise TypeError("assembler must be SemanticEvidenceBundleAssembler")
        record = OwnerScopedSemanticRecord.from_dict(
            accepted.record.to_dict(), config=self.config.ingress
        )
        decision = accepted.decision
        self._require_decision_binding(decision, record)
        registered_adapter = accepted.adapter_registry.get(accepted.adapter_name)
        if accepted.adapter_fingerprint != record.producer_fingerprint.digest:
            raise SemanticRecordConflictError("accepted ingress Adapter fingerprint mismatch")
        if accepted.capability_digest != accepted.capability.digest:
            raise SemanticRecordConflictError("accepted ingress capability digest mismatch")
        if (
            accepted.capability.trust_class is not record.ingress_trust_class
            or record.semantic_input.record_kind not in accepted.capability.allowed_record_kinds
        ):
            raise SemanticRecordConflictError("accepted ingress capability does not authorize the record")
        if (
            registered_adapter.fingerprint.digest != accepted.adapter_fingerprint
            or registered_adapter.capabilities.digest != accepted.capability_digest
        ):
            raise SemanticRecordConflictError("accepted ingress does not match its registered Adapter")
        if accepted.ingress_policy_digest != canonical_digest(self.config.ingress.__dict__):
            raise SemanticRecordConflictError("accepted ingress policy digest differs from the Store")
        seal_time = strict_utc(sealed_at, "sealed_at")
        grouping_key = assembler.grouping_key(record)
        try:
            with self._transaction() as connection:
                self._bind_owner(connection, record.owner_identity_digest)
                existing = self._record_by_id(connection, record.semantic_record_id)
                if existing is not None:
                    if existing.semantic_digest != record.semantic_digest:
                        raise SemanticRecordConflictError("semantic record identity conflicts with durable content")
                    replay = IngressDecision(
                        status=IngressDecisionStatus.REPLAYED,
                        reason_code="semantic_record_replayed",
                        record=existing,
                        decided_at=decision.decided_at,
                    )
                    replay = self._insert_ingress_decision(connection, replay)
                    return SemanticIngestResult(
                        SemanticIngestStatus.REPLAYED,
                        existing.semantic_record_id,
                        replay,
                        self._active_for_record(connection, existing.semantic_record_id),
                    )
                sequence = connection.execute(
                    """SELECT semantic_record_id FROM semantic_records
                       WHERE producer_fingerprint=? AND stream_id=? AND source_sequence=?""",
                    (
                        record.producer_fingerprint.digest,
                        record.semantic_input.stream_id,
                        record.semantic_input.source_sequence,
                    ),
                ).fetchone()
                if sequence is not None:
                    raise SemanticRecordConflictError("producer stream sequence conflicts with another record")
                _, watermark, latest_generation = self._group_watermark(connection, grouping_key)
                if assembler.is_late(record, committed_watermark=watermark):
                    late = IngressDecision(
                        status=IngressDecisionStatus.LATE_REJECTED,
                        reason_code="event_time_before_committed_watermark",
                        record=record,
                        decided_at=decision.decided_at,
                    )
                    late = self._insert_ingress_decision(connection, late)
                    return SemanticIngestResult(
                        SemanticIngestStatus.LATE_REJECTED,
                        record.semantic_record_id,
                        late,
                        self._active_by_group(connection, grouping_key),
                    )
                active = self._active_by_group(connection, grouping_key)
                records = () if active is None else self._bundle_records(connection, active.bundle_id)
                partitions = assembler.partition((*records, record))
                if not partitions:
                    raise EvidenceBundleStateError("accepted semantic record did not form a Bundle partition")
                capacity_reason = self._ingress_capacity_reason(connection, active, partitions, record, decision)
                if capacity_reason is not None:
                    rejected = IngressDecision(
                        status=IngressDecisionStatus.CAPACITY_REJECTED,
                        reason_code=capacity_reason,
                        record=record,
                        decided_at=decision.decided_at,
                    )
                    rejected = self._insert_ingress_decision(connection, rejected)
                    return SemanticIngestResult(
                        SemanticIngestStatus.CAPACITY_REJECTED,
                        record.semantic_record_id,
                        rejected,
                        active,
                    )
                self._insert_record(connection, record)
                stored_decision = self._insert_ingress_decision(connection, decision)
                base_generation = active.generation if active is not None else latest_generation + 1
                if active is not None:
                    connection.execute("DELETE FROM evidence_bundle_members WHERE bundle_id=?", (active.bundle_id,))
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
                    self._upsert_bundle(connection, bundle)
                    if bundle.state is EvidenceBundleState.OPEN:
                        next_active = bundle
                    else:
                        manifest = EvidenceManifest.seal(
                            bundle,
                            partition.records,
                            reason=bundle.seal_reason or EvidenceSealReason.EXPLICIT,
                            sealed_at=seal_time,
                            max_coverage_intervals=assembler.config.max_coverage_intervals,
                            max_manifest_encoded_bytes=assembler.config.max_manifest_encoded_bytes,
                        )
                        self._insert_manifest(connection, manifest)
                        self._retire_bundle_row(connection, bundle.bundle_id)
                        manifests.append(manifest)
                    if bundle.max_event_time is not None:
                        previous_max = bundle.max_event_time if previous_max is None else max(previous_max, bundle.max_event_time)
                    if bundle.watermark is not None:
                        previous_watermark = (
                            bundle.watermark
                            if previous_watermark is None
                            else max(previous_watermark, bundle.watermark)
                        )
                self._commit_watermark(
                    connection,
                    grouping_key,
                    record.owner_identity_digest,
                    previous_max,
                    previous_watermark,
                    base_generation + len(partitions) - 1,
                )
                return SemanticIngestResult(
                    SemanticIngestStatus.ACCEPTED,
                    record.semantic_record_id,
                    stored_decision,
                    next_active,
                    tuple(item.manifest_id for item in manifests),
                )
        except (BehaviorOwnerConflictError, ClaimStoreCapacityError, EvidenceBundleStateError, SemanticRecordConflictError):
            raise
        except sqlite3.IntegrityError as exc:
            raise SemanticRecordConflictError("semantic record or Bundle identity conflicts") from exc
        except sqlite3.Error as exc:
            raise ClaimStoreError("failed to ingest semantic record atomically") from exc

    def read_ingress_decision(self, decision_id: str) -> IngressDecision | None:
        self._require_initialized()
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT content_json, encoded_digest FROM semantic_ingress_decisions WHERE decision_id=?",
                (identifier(decision_id, "decision_id"),),
            ).fetchone()
        return None if row is None else self._decode_durable(
            row,
            IngressDecision.from_dict,
            "IngressDecision",
        )

    def read_semantic_record(self, semantic_record_id: str) -> OwnerScopedSemanticRecord | None:
        self._require_initialized()
        with closing(self._connect()) as connection:
            return self._record_by_id(connection, identifier(semantic_record_id, "semantic_record_id"))

    def read_active_bundle(self, bundle_id: str) -> SemanticEvidenceBundle | None:
        self._require_initialized()
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT content_json, encoded_digest FROM active_evidence_bundles WHERE bundle_id=? AND state='OPEN'",
                (identifier(bundle_id, "bundle_id"),),
            ).fetchone()
        return None if row is None else self._decode_plain(row, SemanticEvidenceBundle.from_dict, "SemanticEvidenceBundle")

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
        with self._transaction() as connection:
            existing = self._manifest_for_bundle(connection, resolved)
            if existing is not None:
                return existing
            row = connection.execute(
                "SELECT content_json, encoded_digest FROM active_evidence_bundles WHERE bundle_id=?",
                (resolved,),
            ).fetchone()
            if row is None:
                return None
            bundle = self._decode_plain(row, SemanticEvidenceBundle.from_dict, "SemanticEvidenceBundle")
            records = self._bundle_records(connection, bundle.bundle_id)
            if not records:
                return None
            sealed = assembler.materialize(
                SemanticBundlePartition(records, EvidenceSealReason(reason)),
                grouping_key=bundle.grouping_key,
                generation=bundle.generation,
                previous_max_event_time=bundle.max_event_time,
                previous_watermark=bundle.watermark,
            )
            manifest = EvidenceManifest.seal(
                sealed,
                records,
                reason=EvidenceSealReason(reason),
                sealed_at=strict_utc(sealed_at, "sealed_at"),
                max_coverage_intervals=assembler.config.max_coverage_intervals,
                max_manifest_encoded_bytes=assembler.config.max_manifest_encoded_bytes,
            )
            self._require_table_capacity(connection, "evidence_manifests", self.config.store.max_manifests)
            self._upsert_bundle(connection, sealed)
            self._insert_manifest(connection, manifest)
            self._retire_bundle_row(connection, sealed.bundle_id)
            return manifest

    def read_manifest(self, manifest_id: str) -> EvidenceManifest | None:
        self._require_initialized()
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT content_json, encoded_digest FROM evidence_manifests WHERE manifest_id=?",
                (identifier(manifest_id, "manifest_id"),),
            ).fetchone()
        return None if row is None else self._decode_manifest(row)

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
        begin, finish = self._bounded_range(start, end)
        size = self._bounded_limit(limit, self.config.evidence.max_query_limit)
        cursor_id = None if cursor is None else identifier(cursor, "cursor")
        with closing(self._connect()) as connection:
            cursor_time: str | None = None
            if cursor_id is not None:
                cursor_row = connection.execute(
                    "SELECT started_at FROM evidence_manifests WHERE manifest_id=?", (cursor_id,)
                ).fetchone()
                if cursor_row is None:
                    raise ValueError("Manifest query cursor does not exist")
                cursor_time = str(cursor_row[0])
            rows = connection.execute(
                """SELECT content_json, encoded_digest FROM evidence_manifests
                   WHERE started_at>=? AND started_at<=?
                   AND (? IS NULL OR started_at>? OR (started_at=? AND manifest_id>?))
                   ORDER BY started_at, manifest_id LIMIT ?""",
                (
                    utc_text(begin),
                    utc_text(finish),
                    cursor_id,
                    cursor_time,
                    cursor_time,
                    cursor_id,
                    size,
                ),
            ).fetchall()
        return tuple(self._decode_manifest(row) for row in rows)

    def read_claim(self, claim_id: str) -> Claim | None:
        self._require_initialized()
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT content_json, encoded_digest FROM claims WHERE claim_id=?",
                (identifier(claim_id, "claim_id"),),
            ).fetchone()
        return None if row is None else self._decode_claim(row)

    def list_claims(
        self,
        *,
        start: datetime,
        end: datetime,
        limit: int,
        cursor: str | None = None,
    ) -> tuple[Claim, ...]:
        return self._list_claim_query(start=start, end=end, limit=limit, cursor=cursor, policy=None)

    def list_accepted_claims(
        self,
        *,
        start: datetime,
        end: datetime,
        limit: int,
        admission_policy_digest: str,
        cursor: str | None = None,
    ) -> tuple[Claim, ...]:
        return self._list_claim_query(
            start=start,
            end=end,
            limit=limit,
            cursor=cursor,
            policy=sha256_digest(admission_policy_digest, "admission_policy_digest"),
        )

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
        cursor_id = None if cursor is None else identifier(cursor, "cursor")
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """SELECT c.content_json, c.encoded_digest FROM claims c
                   JOIN claim_admission_decisions d ON d.claim_id=c.claim_id
                   WHERE d.processing_identity=? AND (? IS NULL OR c.claim_id>?)
                   ORDER BY c.claim_id LIMIT ?""",
                (processing, cursor_id, cursor_id, size),
            ).fetchall()
        return tuple(self._decode_claim(row) for row in rows)

    def read_claim_decision(
        self,
        claim_id: str,
        *,
        processing_identity: str | None = None,
        admission_policy_digest: str | None = None,
    ) -> ClaimAdmissionDecision | None:
        self._require_initialized()
        claim = identifier(claim_id, "claim_id")
        processing = None if processing_identity is None else identifier(processing_identity, "processing_identity")
        policy = None if admission_policy_digest is None else sha256_digest(
            admission_policy_digest, "admission_policy_digest"
        )
        with closing(self._connect()) as connection:
            row = connection.execute(
                """SELECT content_json, encoded_digest FROM claim_admission_decisions
                   WHERE claim_id=? AND (? IS NULL OR processing_identity=?)
                   AND (? IS NULL OR admission_policy_digest=?)
                   ORDER BY decided_at DESC, decision_id DESC LIMIT 1""",
                (claim, processing, processing, policy, policy),
            ).fetchone()
        return None if row is None else self._decode_durable(
            row, ClaimAdmissionDecision.from_dict, "ClaimAdmissionDecision"
        )

    def read_receipt(self, processing_identity: str) -> ClaimProcessingReceipt | None:
        self._require_initialized()
        with closing(self._connect()) as connection:
            receipt = self._receipt_by_id(connection, identifier(processing_identity, "processing_identity"))
            if receipt is not None:
                self._validate_receipt_graph(connection, receipt)
            return receipt

    def read_claims_by_ids(self, claim_ids: tuple[str, ...]) -> tuple[Claim, ...]:
        return self._read_by_ids(
            "claims",
            "claim_id",
            claim_ids,
            self._decode_claim,
            "Claim",
            maximum=self.config.claim.max_claims_per_processing,
        )

    def read_decisions_by_ids(self, decision_ids: tuple[str, ...]) -> tuple[ClaimAdmissionDecision, ...]:
        return self._read_by_ids(
            "claim_admission_decisions",
            "decision_id",
            decision_ids,
            lambda row: self._decode_durable(row, ClaimAdmissionDecision.from_dict, "ClaimAdmissionDecision"),
            "AdmissionDecision",
            maximum=self.config.claim.max_claims_per_processing,
        )

    def read_attempts_by_ids(self, attempt_ids: tuple[str, ...]) -> tuple[ClaimNormalizerAttempt, ...]:
        return self._read_by_ids(
            "claim_normalizer_attempts",
            "attempt_id",
            attempt_ids,
            lambda row: self._decode_durable(row, ClaimNormalizerAttempt.from_dict, "ClaimNormalizerAttempt"),
            "NormalizerAttempt",
            maximum=self.config.claim.max_normalizers_per_processing,
        )

    def read_batches_by_ids(self, batch_ids: tuple[str, ...]) -> tuple[ClaimBatch, ...]:
        return self._read_by_ids(
            "claim_batches",
            "claim_batch_id",
            batch_ids,
            lambda row: self._decode_durable(row, ClaimBatch.from_dict, "ClaimBatch"),
            "ClaimBatch",
            maximum=self.config.claim.max_normalizers_per_processing,
        )

    def read_latest_attempt(
        self,
        processing_identity: str,
        semantic_record_id: str,
        normalizer_fingerprint: str,
    ) -> ClaimNormalizerAttempt | None:
        self._require_initialized()
        with closing(self._connect()) as connection:
            row = connection.execute(
                """SELECT content_json, encoded_digest FROM claim_normalizer_attempts
                   WHERE processing_identity=? AND semantic_record_id=? AND normalizer_fingerprint=?
                   ORDER BY attempt_number DESC LIMIT 1""",
                (
                    identifier(processing_identity, "processing_identity"),
                    identifier(semantic_record_id, "semantic_record_id"),
                    sha256_digest(normalizer_fingerprint, "normalizer_fingerprint"),
                ),
            ).fetchone()
        return None if row is None else self._decode_durable(
            row, ClaimNormalizerAttempt.from_dict, "ClaimNormalizerAttempt"
        )

    def next_attempt_number(
        self,
        processing_identity: str,
        semantic_record_id: str,
        normalizer_fingerprint: str,
    ) -> int:
        self._require_initialized()
        with closing(self._connect()) as connection:
            return self._next_attempt_number_in_connection(
                connection,
                identifier(processing_identity, "processing_identity"),
                identifier(semantic_record_id, "semantic_record_id"),
                sha256_digest(normalizer_fingerprint, "normalizer_fingerprint"),
            )

    @staticmethod
    def _next_attempt_number_in_connection(
        connection: sqlite3.Connection,
        processing_identity: str,
        semantic_record_id: str,
        normalizer_fingerprint: str,
    ) -> int:
        row = connection.execute(
            """SELECT MAX(attempt_number) FROM claim_normalizer_attempts
               WHERE processing_identity=? AND semantic_record_id=? AND normalizer_fingerprint=?""",
            (processing_identity, semantic_record_id, normalizer_fingerprint),
        ).fetchone()
        return 1 if row is None or row[0] is None else int(row[0]) + 1

    def record_failed_attempt(self, attempt: ClaimNormalizerAttempt) -> ClaimNormalizerAttempt:
        self._require_initialized()
        if attempt.status.value in {"COMPLETED", "ABSTAINED"}:
            raise ClaimProcessingConflictError("successful attempt requires atomic lane publication")
        with self._transaction() as connection:
            if self._receipt_by_id(connection, attempt.processing_identity) is not None:
                raise ClaimProcessingConflictError("successful Receipt already exists for this enhancement")
            row = connection.execute(
                "SELECT content_json, encoded_digest FROM claim_normalizer_attempts WHERE attempt_id=?",
                (attempt.attempt_id,),
            ).fetchone()
            if row is not None:
                existing = self._decode_durable(
                    row,
                    ClaimNormalizerAttempt.from_dict,
                    "ClaimNormalizerAttempt",
                )
                if existing.status in {
                    ClaimNormalizerAttemptStatus.COMPLETED,
                    ClaimNormalizerAttemptStatus.ABSTAINED,
                }:
                    raise ClaimProcessingConflictError("failed attempt identity belongs to a successful Attempt")
                return existing
            self._require_table_capacity(
                connection,
                "claim_normalizer_attempts",
                self.config.store.max_normalizer_attempts,
            )
            self._require_database_capacity(
                len(self._encode(attempt.to_dict()).encode("utf-8")) + 512
            )
            self._insert_or_reuse_attempt(connection, attempt)
            return attempt

    def publish_lane(
        self,
        *,
        processing_identity: str,
        processing_lane: ClaimProcessingLane,
        scope_semantic_record_id: str | None,
        manifest: EvidenceManifest,
        routing_policy_digest: str,
        binding_policy_digest: str,
        confidence_policy_digest: str,
        attempts: tuple[ClaimNormalizerAttempt, ...],
        batches: tuple[ClaimBatch, ...],
        batch_claim_ids: tuple[tuple[str, ...], ...],
        claims: tuple[Claim, ...],
        static_results: tuple[StaticAdmissionResult, ...],
        admission_policy: ClaimAdmissionPolicy,
        processing_completed_at: datetime,
    ) -> tuple[ClaimProcessingReceipt, bool]:
        self._require_initialized()
        processing = identifier(processing_identity, "processing_identity")
        lane = ClaimProcessingLane(processing_lane)
        if len(claims) > self.config.claim.max_claims_per_processing:
            raise ClaimStoreCapacityError("one processing exceeds max_claims_per_processing")
        if len(batches) != len(batch_claim_ids):
            raise ClaimProcessingConflictError("Batch membership input is incomplete")
        with self._transaction() as connection:
            existing = self._receipt_by_id(connection, processing)
            if existing is not None:
                self._validate_receipt_graph(connection, existing)
                return existing, True
            self._assert_no_partial_publication(connection, processing)
            attempts = self._resolve_success_attempt_numbers(connection, attempts)
            durable_manifest = self._manifest_by_id(connection, manifest.manifest_id)
            if durable_manifest is None or durable_manifest.content_digest != manifest.content_digest:
                raise ClaimProcessingConflictError("Manifest durable content differs during publication")
            unique_claims: dict[str, Claim] = {}
            duplicate_ids: set[str] = set()
            for claim in claims:
                if claim.claim_id in unique_claims:
                    duplicate_ids.add(claim.claim_id)
                    if unique_claims[claim.claim_id].content_digest != claim.content_digest:
                        raise ClaimProcessingConflictError("duplicate Claim identity has conflicting content")
                else:
                    unique_claims[claim.claim_id] = claim
            for batch, members in zip(batches, batch_claim_ids, strict=True):
                if batch.claim_count != len(members):
                    raise ClaimProcessingConflictError("ClaimBatch count differs from its membership")
                if len(set(members)) != len(members):
                    raise ClaimProcessingConflictError("ClaimBatch membership contains a duplicate Claim")
                for claim_id in members:
                    if claim_id not in unique_claims:
                        raise ClaimProcessingConflictError("ClaimBatch references a Claim outside the publication")
                    claim = unique_claims[claim_id]
                    if (
                        claim.semantic_record_id != batch.semantic_record_id
                        or claim.normalizer_fingerprint != batch.normalizer_fingerprint
                        or claim.manifest_id != batch.manifest_id
                        or claim.manifest_digest != batch.manifest_digest
                    ):
                        raise ClaimProcessingConflictError("ClaimBatch member belongs to another normalization route")
            static_by_id = {item.claim_id: item for item in static_results}
            if set(static_by_id) != set(unique_claims):
                raise ClaimProcessingConflictError("Static Admission results do not cover all Claims")
            decisions = self._admit(
                connection,
                processing,
                tuple(unique_claims.values()),
                static_by_id,
                duplicate_ids,
                admission_policy,
            )
            publication_time = strict_utc(self.clock.now(), "clock.now")
            receipt = ClaimProcessingReceipt.create(
                processing_identity=processing,
                processing_lane=lane,
                scope_semantic_record_id=scope_semantic_record_id,
                manifest_id=manifest.manifest_id,
                manifest_digest=manifest.manifest_semantic_digest,
                routing_policy_digest=routing_policy_digest,
                binding_policy_digest=binding_policy_digest,
                confidence_policy_digest=confidence_policy_digest,
                admission_policy_digest=admission_policy.digest,
                normalizer_attempt_ids=tuple(item.attempt_id for item in attempts),
                claim_batch_ids=tuple(item.claim_batch_id for item in batches),
                claim_ids=tuple(unique_claims),
                accepted_claim_ids=tuple(
                    item.claim_id for item in decisions if item.status is ClaimAdmissionStatus.ACCEPTED
                ),
                decision_ids=tuple(item.decision_id for item in decisions),
                publication_recorded_at=publication_time,
                processing_completed_at=strict_utc(processing_completed_at, "processing_completed_at"),
            )
            self._require_publish_capacities(
                connection,
                attempts,
                batches,
                tuple(unique_claims.values()),
                decisions,
                receipt,
                batch_claim_ids,
            )
            for attempt in attempts:
                self._insert_or_reuse_attempt(connection, attempt)
            for batch in batches:
                self._insert_or_reuse_batch(connection, batch)
            for claim in unique_claims.values():
                self._insert_or_reuse_claim(connection, claim)
            for batch, members in zip(batches, batch_claim_ids, strict=True):
                for order, claim_id in enumerate(members):
                    connection.execute(
                        "INSERT INTO claim_batch_members VALUES(?,?,?)",
                        (batch.claim_batch_id, claim_id, order),
                    )
            for decision in decisions:
                self._insert_decision(connection, decision)
            self._insert_receipt(connection, receipt)
            self._validate_receipt_graph(connection, receipt)
            return receipt, False

    def _resolve_success_attempt_numbers(
        self,
        connection: sqlite3.Connection,
        attempts: tuple[ClaimNormalizerAttempt, ...],
    ) -> tuple[ClaimNormalizerAttempt, ...]:
        resolved: list[ClaimNormalizerAttempt] = []
        for attempt in attempts:
            if attempt.status not in {
                ClaimNormalizerAttemptStatus.COMPLETED,
                ClaimNormalizerAttemptStatus.ABSTAINED,
            }:
                raise ClaimProcessingConflictError("successful publication contains a failed Attempt")
            row = connection.execute(
                "SELECT content_json, encoded_digest FROM claim_normalizer_attempts WHERE attempt_id=?",
                (attempt.attempt_id,),
            ).fetchone()
            if row is None:
                resolved.append(attempt)
                continue
            existing = self._decode_durable(
                row,
                ClaimNormalizerAttempt.from_dict,
                "ClaimNormalizerAttempt",
            )
            if existing.status in {
                ClaimNormalizerAttemptStatus.COMPLETED,
                ClaimNormalizerAttemptStatus.ABSTAINED,
            }:
                raise ClaimProcessingConflictError(
                    "successful Attempt exists without its atomic Receipt"
                )
            next_number = self._next_attempt_number_in_connection(
                connection,
                attempt.processing_identity,
                attempt.semantic_record_id,
                attempt.normalizer_fingerprint,
            )
            resolved.append(
                ClaimNormalizerAttempt.create(
                    processing_identity=attempt.processing_identity,
                    processing_lane=attempt.processing_lane,
                    manifest_id=attempt.manifest_id,
                    semantic_record_id=attempt.semantic_record_id,
                    normalizer_name=attempt.normalizer_name,
                    normalizer_fingerprint=attempt.normalizer_fingerprint,
                    attempt_number=next_number,
                    status=attempt.status,
                    proposal_digest=attempt.proposal_digest,
                    claim_count=attempt.claim_count,
                    error_code=None,
                    retryable=False,
                    normalization_started_at=attempt.normalization_started_at,
                    normalization_completed_at=attempt.normalization_completed_at,
                )
            )
        return tuple(resolved)

    def _admit(
        self,
        connection: sqlite3.Connection,
        processing: str,
        claims: tuple[Claim, ...],
        static: dict[str, StaticAdmissionResult],
        duplicate_ids: set[str],
        policy: ClaimAdmissionPolicy,
    ) -> tuple[ClaimAdmissionDecision, ...]:
        passed = [item for item in claims if static[item.claim_id].rejection_status is None]
        winners: dict[str, Claim] = {}
        for claim in sorted(
            passed,
            key=lambda item: (
                -item.effective_confidence,
                -item.source_confidence,
                -item.normalizer_confidence,
                item.claim_id,
            ),
        ):
            winners.setdefault(claim.semantic_fingerprint, claim)
        decided_at = strict_utc(self.clock.now(), "clock.now")
        accepted_count = self._accepted_claim_count(connection, policy.digest)
        decisions: list[ClaimAdmissionDecision] = []
        for claim in claims:
            result = static[claim.claim_id]
            existing_claim_id: str | None = None
            if result.rejection_status is not None:
                status = result.rejection_status
                reason = result.reason_code
            elif claim.claim_id in duplicate_ids:
                status = ClaimAdmissionStatus.EXACT_DUPLICATE
                reason = "same_processing_claim_identity_duplicate"
                existing_claim_id = claim.claim_id
            elif winners[claim.semantic_fingerprint].claim_id != claim.claim_id:
                status = ClaimAdmissionStatus.NO_INFORMATION_GAIN
                reason = "same_processing_semantic_duplicate"
                existing_claim_id = winners[claim.semantic_fingerprint].claim_id
            else:
                deterministic_existing = (
                    self._accepted_semantic_claim(
                        connection,
                        claim.semantic_fingerprint,
                        policy.digest,
                        semantic_record_id=None,
                        derivation_class=ClaimDerivationClass.DETERMINISTIC,
                    )
                    if claim.derivation_class is ClaimDerivationClass.MODEL
                    else None
                )
                same_record_existing = self._accepted_semantic_claim(
                    connection,
                    claim.semantic_fingerprint,
                    policy.digest,
                    semantic_record_id=claim.semantic_record_id,
                    derivation_class=None,
                )
                if (
                    claim.derivation_class is ClaimDerivationClass.MODEL
                    and deterministic_existing is not None
                ):
                    status = ClaimAdmissionStatus.NO_INFORMATION_GAIN
                    reason = "core_semantic_claim_already_accepted_under_policy"
                    existing_claim_id = deterministic_existing
                elif same_record_existing is not None:
                    status = ClaimAdmissionStatus.NO_INFORMATION_GAIN
                    reason = "semantic_claim_already_accepted_under_policy"
                    existing_claim_id = same_record_existing
                else:
                    recent = self._recent_state_claim(
                        connection,
                        claim,
                        policy.digest,
                        policy.config.repeat_state_suppression_seconds,
                    )
                    if recent is not None:
                        status = ClaimAdmissionStatus.REPEATED_STATE_SUPPRESSED
                        reason = "state_repeated_within_configured_window"
                        existing_claim_id = recent
                    elif accepted_count >= policy.max_accepted_claims:
                        status = ClaimAdmissionStatus.CAPACITY_REJECTED
                        reason = "accepted_claim_capacity_reached"
                    else:
                        status = ClaimAdmissionStatus.ACCEPTED
                        reason = "claim_passed_admission"
                        accepted_count += 1
            decisions.append(
                ClaimAdmissionDecision.create(
                    claim,
                    status,
                    reason,
                    processing_identity=processing,
                    admission_policy_digest=policy.digest,
                    required_threshold=result.required_threshold,
                    evaluated_confidence=result.evaluated_confidence,
                    admission_decided_at=decided_at,
                    existing_claim_id=existing_claim_id,
                )
            )
        return tuple(decisions)

    def _ingress_capacity_reason(
        self,
        connection: sqlite3.Connection,
        active: SemanticEvidenceBundle | None,
        partitions: tuple[SemanticBundlePartition, ...],
        record: OwnerScopedSemanticRecord,
        decision: IngressDecision,
    ) -> str | None:
        if self._count(connection, "semantic_ingress_decisions") >= self.config.store.max_ingress_decisions:
            raise ClaimStoreCapacityError("IngressDecision audit capacity has been reached")
        if self._count(connection, "semantic_records") >= self.config.store.max_semantic_records:
            return "semantic_record_capacity_reached"
        new_manifests = sum(item.seal_reason is not None for item in partitions)
        if self._count(connection, "evidence_manifests") + new_manifests > self.config.store.max_manifests:
            return "manifest_capacity_reached"
        open_count = int(
            connection.execute("SELECT COUNT(*) FROM active_evidence_bundles WHERE state='OPEN'").fetchone()[0]
        )
        has_open = any(item.seal_reason is None for item in partitions)
        projected_open = open_count - (1 if active is not None else 0) + (1 if has_open else 0)
        maximum = min(self.config.evidence.max_active_bundles, self.config.store.max_active_bundles)
        if projected_open > maximum:
            return "active_bundle_capacity_reached"
        record_bytes = len(canonical_json(record.to_dict()).encode("utf-8"))
        if record_bytes > self.config.store.max_json_bytes:
            return "semantic_record_json_capacity_reached"
        estimated = record_bytes + len(canonical_json(decision.to_dict()).encode("utf-8"))
        estimated += sum(
            len(canonical_json(item.to_dict()).encode("utf-8"))
            for partition in partitions
            for item in partition.records
        )
        estimated += 8_192 * len(partitions)
        if self._database_bytes() + estimated > self.config.store.max_database_bytes:
            return "database_byte_capacity_reached"
        return None

    def _insert_record(self, connection: sqlite3.Connection, record: OwnerScopedSemanticRecord) -> None:
        payload, encoded = self._encoded(record)
        connection.execute(
            "INSERT INTO semantic_records VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (
                record.semantic_record_id,
                record.producer_fingerprint.digest,
                record.semantic_input.stream_id,
                record.semantic_input.source_sequence,
                record.owner_identity_digest,
                utc_text(record.semantic_input.event_time_start),
                utc_text(record.semantic_input.event_time_end),
                record.semantic_digest,
                record.content_digest,
                encoded,
                payload,
            ),
        )

    def _insert_ingress_decision(
        self,
        connection: sqlite3.Connection,
        decision: IngressDecision,
    ) -> IngressDecision:
        row = connection.execute(
            "SELECT content_json, encoded_digest FROM semantic_ingress_decisions WHERE decision_id=?",
            (decision.decision_id,),
        ).fetchone()
        if row is not None:
            existing = self._decode_durable(row, IngressDecision.from_dict, "IngressDecision")
            if existing.content_digest != decision.content_digest:
                raise SemanticRecordConflictError("IngressDecision identity conflicts with durable content")
            return existing
        self._require_table_capacity(
            connection,
            "semantic_ingress_decisions",
            self.config.store.max_ingress_decisions,
        )
        self._require_database_capacity(len(self._encode(decision.to_dict()).encode("utf-8")) + 512)
        payload, encoded = self._encoded(decision)
        connection.execute(
            "INSERT INTO semantic_ingress_decisions VALUES(?,?,?,?,?,?,?,?,?)",
            (
                decision.decision_id,
                decision.semantic_record_id,
                decision.owner_identity_digest,
                decision.status.value,
                utc_text(decision.decided_at),
                decision.decision_identity_digest,
                decision.content_digest,
                encoded,
                payload,
            ),
        )
        return decision

    def _upsert_bundle(self, connection: sqlite3.Connection, bundle: SemanticEvidenceBundle) -> None:
        payload = self._encode(bundle.to_dict())
        encoded = canonical_digest(bundle.to_dict())
        connection.execute(
            """INSERT INTO active_evidence_bundles VALUES(?,?,?,?,?,?,?,?)
               ON CONFLICT(bundle_id) DO UPDATE SET state=excluded.state, watermark=excluded.watermark,
               encoded_digest=excluded.encoded_digest, content_json=excluded.content_json""",
            (
                bundle.bundle_id,
                bundle.grouping_key,
                bundle.generation,
                bundle.owner_identity_digest,
                bundle.state.value,
                None if bundle.watermark is None else utc_text(bundle.watermark),
                encoded,
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
        row = connection.execute(
            "SELECT content_json, encoded_digest FROM evidence_manifests WHERE manifest_id=?",
            (manifest.manifest_id,),
        ).fetchone()
        if row is not None:
            existing = self._decode_manifest(row)
            if existing.content_digest != manifest.content_digest:
                raise EvidenceManifestError("Manifest identity conflicts with durable content")
            return
        payload, encoded = self._encoded(manifest)
        connection.execute(
            "INSERT INTO evidence_manifests VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                manifest.manifest_id,
                manifest.bundle_id,
                manifest.owner_identity_digest,
                utc_text(manifest.started_at),
                utc_text(manifest.ended_at),
                utc_text(manifest.sealed_at),
                manifest.manifest_semantic_digest,
                manifest.content_digest,
                encoded,
                payload,
            ),
        )
        connection.executemany(
            "INSERT INTO evidence_manifest_members VALUES(?,?,?)",
            tuple(
                (manifest.manifest_id, item.semantic_record_id, order)
                for order, item in enumerate(manifest.ordered_record_snapshots)
            ),
        )

    @staticmethod
    def _retire_bundle_row(connection: sqlite3.Connection, bundle_id: str) -> None:
        connection.execute("DELETE FROM active_evidence_bundles WHERE bundle_id=?", (bundle_id,))

    def _insert_or_reuse_attempt(self, connection: sqlite3.Connection, attempt: ClaimNormalizerAttempt) -> None:
        self._insert_or_compare(
            connection,
            "claim_normalizer_attempts",
            "attempt_id",
            attempt.attempt_id,
            attempt,
            (
                attempt.attempt_id,
                attempt.processing_identity,
                attempt.processing_lane.value,
                attempt.manifest_id,
                attempt.semantic_record_id,
                attempt.normalizer_fingerprint,
                attempt.attempt_number,
                attempt.status.value,
                attempt.content_digest,
            ),
        )

    def _insert_or_reuse_batch(self, connection: sqlite3.Connection, batch: ClaimBatch) -> None:
        self._insert_or_compare(
            connection,
            "claim_batches",
            "claim_batch_id",
            batch.claim_batch_id,
            batch,
            (
                batch.claim_batch_id,
                batch.processing_identity,
                batch.processing_lane.value,
                batch.manifest_id,
                batch.semantic_record_id,
                batch.normalizer_fingerprint,
                utc_text(batch.created_at),
                batch.content_digest,
            ),
        )

    def _insert_or_reuse_claim(self, connection: sqlite3.Connection, claim: Claim) -> None:
        row = connection.execute(
            "SELECT content_json, encoded_digest FROM claims WHERE claim_id=?",
            (claim.claim_id,),
        ).fetchone()
        if row is not None:
            existing = self._decode_claim(row)
            expected = claim.to_dict()
            durable = existing.to_dict()
            for audit_field in ("created_at", "content_digest"):
                expected.pop(audit_field)
                durable.pop(audit_field)
            if expected != durable:
                raise ClaimProcessingConflictError("Claim identity conflicts with durable semantic content")
            return
        payload, encoded = self._encoded(claim)
        connection.execute(
            "INSERT INTO claims VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                claim.claim_id,
                claim.manifest_id,
                claim.semantic_record_id,
                claim.owner_identity_digest,
                claim.semantic_fingerprint,
                claim.derivation_class.value,
                claim.proposal.claim_kind.value,
                utc_text(claim.time_start),
                utc_text(claim.time_end),
                utc_text(claim.created_at),
                claim.content_digest,
                encoded,
                payload,
            ),
        )

    def _insert_or_compare(
        self,
        connection: sqlite3.Connection,
        table: str,
        key_column: str,
        key: str,
        value: _DurableValue,
        prefix: tuple[object, ...],
    ) -> None:
        row = connection.execute(
            f"SELECT content_json, encoded_digest FROM {table} WHERE {key_column}=?",
            (key,),
        ).fetchone()
        if row is not None:
            if canonical_json(value.to_dict()) != row[0] or canonical_digest(value.to_dict()) != row[1]:
                raise ClaimProcessingConflictError(f"{table} identity conflicts with durable content")
            return
        payload, encoded = self._encoded(value)
        placeholders = ",".join("?" for _ in range(len(prefix) + 2))
        connection.execute(f"INSERT INTO {table} VALUES({placeholders})", (*prefix, encoded, payload))

    def _insert_decision(self, connection: sqlite3.Connection, decision: ClaimAdmissionDecision) -> None:
        payload, encoded = self._encoded(decision)
        connection.execute(
            "INSERT INTO claim_admission_decisions VALUES(?,?,?,?,?,?,?,?,?)",
            (
                decision.decision_id,
                decision.processing_identity,
                decision.claim_id,
                decision.admission_policy_digest,
                decision.status.value,
                utc_text(decision.admission_decided_at),
                decision.content_digest,
                encoded,
                payload,
            ),
        )

    def _insert_receipt(self, connection: sqlite3.Connection, receipt: ClaimProcessingReceipt) -> None:
        payload, encoded = self._encoded(receipt)
        connection.execute(
            "INSERT INTO claim_processing_receipts VALUES(?,?,?,?,?,?,?,?,?,?)",
            (
                receipt.processing_identity,
                receipt.processing_lane.value,
                receipt.scope_semantic_record_id,
                receipt.manifest_id,
                receipt.admission_policy_digest,
                utc_text(receipt.processing_completed_at),
                utc_text(receipt.publication_recorded_at),
                receipt.content_digest,
                encoded,
                payload,
            ),
        )

    def _record_by_id(
        self,
        connection: sqlite3.Connection,
        record_id: str,
    ) -> OwnerScopedSemanticRecord | None:
        row = connection.execute(
            "SELECT content_json, encoded_digest FROM semantic_records WHERE semantic_record_id=?",
            (record_id,),
        ).fetchone()
        return None if row is None else self._decode_durable(
            row,
            lambda value: OwnerScopedSemanticRecord.from_dict(value, config=self.config.ingress),
            "OwnerScopedSemanticRecord",
        )

    def _manifest_by_id(self, connection: sqlite3.Connection, manifest_id: str) -> EvidenceManifest | None:
        row = connection.execute(
            "SELECT content_json, encoded_digest FROM evidence_manifests WHERE manifest_id=?",
            (manifest_id,),
        ).fetchone()
        return None if row is None else self._decode_manifest(row)

    def _manifest_for_bundle(self, connection: sqlite3.Connection, bundle_id: str) -> EvidenceManifest | None:
        row = connection.execute(
            "SELECT content_json, encoded_digest FROM evidence_manifests WHERE bundle_id=?",
            (bundle_id,),
        ).fetchone()
        return None if row is None else self._decode_manifest(row)

    def _receipt_by_id(
        self,
        connection: sqlite3.Connection,
        processing_identity: str,
    ) -> ClaimProcessingReceipt | None:
        row = connection.execute(
            "SELECT content_json, encoded_digest FROM claim_processing_receipts WHERE processing_identity=?",
            (processing_identity,),
        ).fetchone()
        return None if row is None else self._decode_durable(
            row, ClaimProcessingReceipt.from_dict, "ClaimProcessingReceipt"
        )

    def _active_by_group(
        self,
        connection: sqlite3.Connection,
        grouping_key: str,
    ) -> SemanticEvidenceBundle | None:
        row = connection.execute(
            """SELECT content_json, encoded_digest FROM active_evidence_bundles
               WHERE grouping_key=? AND state='OPEN' ORDER BY generation DESC LIMIT 1""",
            (grouping_key,),
        ).fetchone()
        return None if row is None else self._decode_plain(row, SemanticEvidenceBundle.from_dict, "SemanticEvidenceBundle")

    def _active_for_record(
        self,
        connection: sqlite3.Connection,
        record_id: str,
    ) -> SemanticEvidenceBundle | None:
        row = connection.execute(
            """SELECT b.content_json, b.encoded_digest FROM active_evidence_bundles b
               JOIN evidence_bundle_members m ON m.bundle_id=b.bundle_id
               WHERE m.semantic_record_id=? AND b.state='OPEN' ORDER BY b.generation DESC LIMIT 1""",
            (record_id,),
        ).fetchone()
        return None if row is None else self._decode_plain(row, SemanticEvidenceBundle.from_dict, "SemanticEvidenceBundle")

    def _bundle_records(
        self,
        connection: sqlite3.Connection,
        bundle_id: str,
    ) -> tuple[OwnerScopedSemanticRecord, ...]:
        rows = connection.execute(
            """SELECT r.content_json, r.encoded_digest FROM semantic_records r
               JOIN evidence_bundle_members m ON m.semantic_record_id=r.semantic_record_id
               WHERE m.bundle_id=? ORDER BY m.member_order""",
            (bundle_id,),
        ).fetchall()
        return tuple(
            self._decode_durable(
                row,
                lambda value: OwnerScopedSemanticRecord.from_dict(value, config=self.config.ingress),
                "OwnerScopedSemanticRecord",
            )
            for row in rows
        )

    @staticmethod
    def _group_watermark(
        connection: sqlite3.Connection,
        grouping_key: str,
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

    @staticmethod
    def _commit_watermark(
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

    def _accepted_semantic_claim(
        self,
        connection: sqlite3.Connection,
        semantic_fingerprint: str,
        policy_digest: str,
        *,
        semantic_record_id: str | None,
        derivation_class: ClaimDerivationClass | None,
    ) -> str | None:
        row = connection.execute(
            """SELECT c.claim_id FROM claims c
               JOIN claim_admission_decisions d ON d.claim_id=c.claim_id
               WHERE c.semantic_fingerprint=? AND d.admission_policy_digest=? AND d.status='ACCEPTED'
               AND json_extract(d.content_json, '$.status')='ACCEPTED'
               AND json_extract(d.content_json, '$.admission_policy_digest')=d.admission_policy_digest
               AND json_extract(d.content_json, '$.claim_id')=d.claim_id
               AND (? IS NULL OR c.semantic_record_id=?)
               AND (? IS NULL OR c.derivation_class=?)
               ORDER BY d.decided_at DESC, c.claim_id LIMIT 1""",
            (
                semantic_fingerprint,
                policy_digest,
                semantic_record_id,
                semantic_record_id,
                None if derivation_class is None else derivation_class.value,
                None if derivation_class is None else derivation_class.value,
            ),
        ).fetchone()
        return None if row is None else str(row[0])

    def _recent_state_claim(
        self,
        connection: sqlite3.Connection,
        claim: Claim,
        policy_digest: str,
        seconds: float,
    ) -> str | None:
        if claim.proposal.claim_kind is not ClaimKind.STATE_ASSERTION:
            return None
        lower = claim.time_start - timedelta(seconds=seconds)
        upper = claim.time_end + timedelta(seconds=seconds)
        row = connection.execute(
            """SELECT c.claim_id FROM claims c JOIN claim_admission_decisions d ON d.claim_id=c.claim_id
               WHERE c.semantic_fingerprint=? AND c.time_end>=? AND c.time_end<=? AND c.claim_id<>?
               AND d.admission_policy_digest=? AND d.status='ACCEPTED'
               AND json_extract(d.content_json, '$.status')='ACCEPTED'
               AND json_extract(d.content_json, '$.admission_policy_digest')=d.admission_policy_digest
               AND json_extract(d.content_json, '$.claim_id')=d.claim_id
               ORDER BY c.time_end DESC, c.claim_id DESC LIMIT 1""",
            (
                claim.semantic_fingerprint,
                utc_text(lower),
                utc_text(upper),
                claim.claim_id,
                policy_digest,
            ),
        ).fetchone()
        return None if row is None else str(row[0])

    def _list_claim_query(
        self,
        *,
        start: datetime,
        end: datetime,
        limit: int,
        cursor: str | None,
        policy: str | None,
    ) -> tuple[Claim, ...]:
        begin, finish = self._bounded_range(start, end)
        size = self._bounded_limit(limit, self.config.claim.max_query_limit)
        cursor_id = None if cursor is None else identifier(cursor, "cursor")
        with closing(self._connect()) as connection:
            cursor_time: str | None = None
            if cursor_id is not None:
                cursor_row = connection.execute(
                    "SELECT created_at FROM claims WHERE claim_id=?",
                    (cursor_id,),
                ).fetchone()
                if cursor_row is None:
                    raise ValueError("Claim query cursor does not exist")
                cursor_time = str(cursor_row[0])
            if policy is None:
                rows = connection.execute(
                    """SELECT content_json, encoded_digest FROM claims
                       WHERE created_at>=? AND created_at<=?
                       AND (? IS NULL OR created_at>? OR (created_at=? AND claim_id>?))
                       ORDER BY created_at, claim_id LIMIT ?""",
                    (
                        utc_text(begin),
                        utc_text(finish),
                        cursor_id,
                        cursor_time,
                        cursor_time,
                        cursor_id,
                        size,
                    ),
                ).fetchall()
            else:
                rows = connection.execute(
                    """SELECT DISTINCT c.content_json, c.encoded_digest FROM claims c
                       JOIN claim_admission_decisions d ON d.claim_id=c.claim_id
                       WHERE c.created_at>=? AND c.created_at<=? AND d.status='ACCEPTED'
                       AND json_extract(d.content_json, '$.status')='ACCEPTED'
                       AND json_extract(d.content_json, '$.admission_policy_digest')=d.admission_policy_digest
                       AND json_extract(d.content_json, '$.claim_id')=d.claim_id
                       AND d.admission_policy_digest=?
                       AND (? IS NULL OR c.created_at>? OR (c.created_at=? AND c.claim_id>?))
                       ORDER BY c.created_at, c.claim_id LIMIT ?""",
                    (
                        utc_text(begin),
                        utc_text(finish),
                        policy,
                        cursor_id,
                        cursor_time,
                        cursor_time,
                        cursor_id,
                        size,
                    ),
                ).fetchall()
        return tuple(self._decode_claim(row) for row in rows)

    def _read_by_ids(
        self,
        table: str,
        key_column: str,
        ids: tuple[str, ...],
        decoder: Callable[[tuple[str, str]], _T],
        label: str,
        *,
        maximum: int,
    ) -> tuple[_T, ...]:
        self._require_initialized()
        resolved = tuple(identifier(item, f"{label}_id") for item in ids)
        if len(set(resolved)) != len(resolved):
            raise ClaimProcessingConflictError(f"{label} identity tuple contains duplicates")
        if len(resolved) > maximum:
            raise ClaimStoreCapacityError(f"{label} internal identity read exceeds its bound")
        found: dict[str, _T] = {}
        with closing(self._connect()) as connection:
            for offset in range(0, len(resolved), _SQLITE_ID_CHUNK):
                chunk = resolved[offset : offset + _SQLITE_ID_CHUNK]
                placeholders = ",".join("?" for _ in chunk)
                rows = connection.execute(
                    f"SELECT {key_column}, content_json, encoded_digest FROM {table} WHERE {key_column} IN ({placeholders})",
                    chunk,
                ).fetchall()
                for row in rows:
                    found[str(row[0])] = decoder((row[1], row[2]))
        missing = tuple(item for item in resolved if item not in found)
        if missing:
            raise ClaimProcessingConflictError(f"{label} Receipt reference is missing")
        return tuple(found[item] for item in resolved)

    def _validate_receipt_graph(self, connection: sqlite3.Connection, receipt: ClaimProcessingReceipt) -> None:
        attempt_rows = self._rows_by_ids_in_connection(
            connection, "claim_normalizer_attempts", "attempt_id", receipt.normalizer_attempt_ids
        )
        batch_rows = self._rows_by_ids_in_connection(
            connection, "claim_batches", "claim_batch_id", receipt.claim_batch_ids
        )
        decision_rows = self._rows_by_ids_in_connection(
            connection, "claim_admission_decisions", "decision_id", receipt.decision_ids
        )
        claim_rows = self._rows_by_ids_in_connection(connection, "claims", "claim_id", receipt.claim_ids)
        attempts_by_id = {
            str(row[0]): self._decode_durable(
                (row[-1], row[-2]),
                ClaimNormalizerAttempt.from_dict,
                "ClaimNormalizerAttempt",
            )
            for row in attempt_rows
        }
        batches_by_id = {
            str(row[0]): self._decode_durable(
                (row[-1], row[-2]),
                ClaimBatch.from_dict,
                "ClaimBatch",
            )
            for row in batch_rows
        }
        decisions_by_id = {
            str(row[0]): self._decode_durable(
                (row[-1], row[-2]),
                ClaimAdmissionDecision.from_dict,
                "ClaimAdmissionDecision",
            )
            for row in decision_rows
        }
        claims_by_id = {str(row[0]): self._decode_claim((str(row[-1]), str(row[-2]))) for row in claim_rows}
        attempts = tuple(attempts_by_id[item] for item in receipt.normalizer_attempt_ids)
        batches = tuple(batches_by_id[item] for item in receipt.claim_batch_ids)
        decisions = tuple(decisions_by_id[item] for item in receipt.decision_ids)
        claims = tuple(claims_by_id[item] for item in receipt.claim_ids)
        if (
            any(item.processing_identity != receipt.processing_identity for item in attempts)
            or any(item.processing_identity != receipt.processing_identity for item in batches)
            or any(item.processing_identity != receipt.processing_identity for item in decisions)
        ):
            raise ClaimProcessingConflictError("Receipt references an object from another Processing")
        if any(item.processing_lane is not receipt.processing_lane for item in attempts) or any(
            item.processing_lane is not receipt.processing_lane for item in batches
        ):
            raise ClaimProcessingConflictError("Receipt references an object from another processing lane")
        if (
            any(item.manifest_id != receipt.manifest_id for item in attempts)
            or any(item.manifest_id != receipt.manifest_id for item in batches)
            or any(item.manifest_id != receipt.manifest_id for item in claims)
        ):
            raise ClaimProcessingConflictError("Receipt references an object from another Manifest")
        if any(item.admission_policy_digest != receipt.admission_policy_digest for item in decisions):
            raise ClaimProcessingConflictError("Receipt Decision policy digest mismatch")
        if tuple(item.claim_id for item in decisions) != receipt.claim_ids:
            raise ClaimProcessingConflictError("Receipt Decision membership is incomplete or misordered")
        if any(
            item.binding_policy_digest != receipt.binding_policy_digest
            or item.confidence_policy_digest != receipt.confidence_policy_digest
            or item.manifest_digest != receipt.manifest_digest
            for item in claims
        ):
            raise ClaimProcessingConflictError("Receipt Claim policy or Manifest digest mismatch")
        if len(attempts) != len(batches):
            raise ClaimProcessingConflictError("Receipt Normalizer Attempts and Batches are incomplete")
        for attempt, batch in zip(attempts, batches, strict=True):
            if attempt.status not in {
                ClaimNormalizerAttemptStatus.COMPLETED,
                ClaimNormalizerAttemptStatus.ABSTAINED,
            }:
                raise ClaimProcessingConflictError("successful Receipt references a failed Attempt")
            if (
                attempt.semantic_record_id != batch.semantic_record_id
                or attempt.normalizer_fingerprint != batch.normalizer_fingerprint
                or attempt.manifest_id != batch.manifest_id
            ):
                raise ClaimProcessingConflictError("Receipt Attempt and Batch route identities differ")
        if receipt.processing_lane is ClaimProcessingLane.ENHANCEMENT:
            if (
                any(item.semantic_record_id != receipt.scope_semantic_record_id for item in attempts)
                or any(item.semantic_record_id != receipt.scope_semantic_record_id for item in batches)
                or any(item.semantic_record_id != receipt.scope_semantic_record_id for item in claims)
            ):
                raise ClaimProcessingConflictError(
                    "Enhancement Receipt contains an object outside its record scope"
                )
        claim_set = set(receipt.claim_ids)
        member_claim_ids: set[str] = set()
        for batch in batches:
            members = connection.execute(
                "SELECT claim_id, member_order FROM claim_batch_members WHERE claim_batch_id=? ORDER BY member_order",
                (batch.claim_batch_id,),
            ).fetchall()
            if len(members) != batch.claim_count or tuple(int(row[1]) for row in members) != tuple(
                range(batch.claim_count)
            ):
                raise ClaimProcessingConflictError("Receipt BatchMember count or ordering is incomplete")
            batch_claim_ids = {str(row[0]) for row in members}
            if not batch_claim_ids.issubset(claim_set):
                raise ClaimProcessingConflictError("Receipt BatchMember references a Claim outside the Receipt")
            for claim_id in batch_claim_ids:
                claim = claims_by_id[claim_id]
                if (
                    claim.semantic_record_id != batch.semantic_record_id
                    or claim.normalizer_fingerprint != batch.normalizer_fingerprint
                    or claim.manifest_id != batch.manifest_id
                    or claim.manifest_digest != batch.manifest_digest
                ):
                    raise ClaimProcessingConflictError(
                        "Receipt BatchMember belongs to another normalization route"
                    )
            member_claim_ids.update(batch_claim_ids)
        if member_claim_ids != claim_set:
            raise ClaimProcessingConflictError("Receipt Claim membership is incomplete")

    @staticmethod
    def _rows_by_ids_in_connection(
        connection: sqlite3.Connection,
        table: str,
        key: str,
        ids: tuple[str, ...],
    ) -> list[sqlite3.Row | tuple[object, ...]]:
        if not ids:
            return []
        result: list[sqlite3.Row | tuple[object, ...]] = []
        for offset in range(0, len(ids), _SQLITE_ID_CHUNK):
            chunk = ids[offset : offset + _SQLITE_ID_CHUNK]
            placeholders = ",".join("?" for _ in chunk)
            result.extend(connection.execute(f"SELECT * FROM {table} WHERE {key} IN ({placeholders})", chunk).fetchall())
        if len(result) != len(ids):
            raise ClaimProcessingConflictError("Receipt graph contains a missing durable object")
        return result

    def _assert_no_partial_publication(self, connection: sqlite3.Connection, processing: str) -> None:
        for table in ("claim_batches", "claim_admission_decisions"):
            if connection.execute(
                f"SELECT 1 FROM {table} WHERE processing_identity=? LIMIT 1", (processing,)
            ).fetchone() is not None:
                raise ClaimProcessingConflictError("processing identity has partial durable publication")

    def _require_publish_capacities(
        self,
        connection: sqlite3.Connection,
        attempts: tuple[ClaimNormalizerAttempt, ...],
        batches: tuple[ClaimBatch, ...],
        claims: tuple[Claim, ...],
        decisions: tuple[ClaimAdmissionDecision, ...],
        receipt: ClaimProcessingReceipt,
        batch_claim_ids: tuple[tuple[str, ...], ...],
    ) -> None:
        increments = (
            ("claim_normalizer_attempts", self.config.store.max_normalizer_attempts, len(attempts)),
            ("claim_batches", self.config.store.max_claim_batches, len(batches)),
            (
                "claims",
                self.config.store.max_validated_claims,
                self._missing_rows(connection, "claims", "claim_id", tuple(item.claim_id for item in claims)),
            ),
            ("claim_admission_decisions", self.config.store.max_admission_decisions, len({item.claim_id for item in claims})),
            ("claim_processing_receipts", self.config.store.max_processing_receipts, 1),
        )
        for table, maximum, increment in increments:
            if self._count(connection, table) + increment > maximum:
                raise ClaimStoreCapacityError(f"{table} capacity has been reached")
        projected_values = cast(
            tuple[_DurableValue, ...],
            (*attempts, *batches, *claims, *decisions, receipt),
        )
        projected = sum(len(self._encode(item.to_dict()).encode("utf-8")) for item in projected_values)
        projected += sum(
            len(
                canonical_json(
                    {
                        "claim_batch_id": batch.claim_batch_id,
                        "claim_id": claim_id,
                        "member_order": order,
                    }
                ).encode("utf-8")
            )
            for batch, members in zip(batches, batch_claim_ids, strict=True)
            for order, claim_id in enumerate(members)
        )
        projected += 512 * (
            len(attempts)
            + len(batches)
            + len(claims)
            + len(decisions)
            + sum(len(item) for item in batch_claim_ids)
            + 1
        )
        self._require_database_capacity(projected)

    @staticmethod
    def _missing_rows(
        connection: sqlite3.Connection,
        table: str,
        key_name: str,
        keys: tuple[str, ...],
    ) -> int:
        return sum(
            connection.execute(f"SELECT 1 FROM {table} WHERE {key_name}=? LIMIT 1", (key,)).fetchone() is None
            for key in set(keys)
        )

    @staticmethod
    def _require_decision_binding(decision: IngressDecision, record: OwnerScopedSemanticRecord) -> None:
        if decision.semantic_record_id != record.semantic_record_id or decision.owner_identity_digest != record.owner_identity_digest:
            raise SemanticRecordConflictError("IngressDecision does not bind the semantic record")

    def _require_table_capacity(self, connection: sqlite3.Connection, table: str, maximum: int) -> None:
        if self._count(connection, table) >= maximum:
            raise ClaimStoreCapacityError(f"{table} capacity has been reached")

    def _require_database_capacity(self, additional_bytes: int = 0) -> None:
        if self._database_bytes() + additional_bytes > self.config.store.max_database_bytes:
            raise ClaimStoreCapacityError("Behavior database byte capacity has been reached")

    def _database_bytes(self) -> int:
        return sum(
            candidate.stat().st_size
            for candidate in (self.path, Path(str(self.path) + "-wal"), Path(str(self.path) + "-shm"))
            if candidate.exists()
        )

    @staticmethod
    def _count(connection: sqlite3.Connection, table: str) -> int:
        return int(connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])

    @staticmethod
    def _accepted_claim_count(connection: sqlite3.Connection, policy_digest: str) -> int:
        return int(
            connection.execute(
                """SELECT COUNT(DISTINCT claim_id) FROM claim_admission_decisions
                   WHERE admission_policy_digest=? AND status='ACCEPTED'
                   AND json_extract(content_json, '$.status')='ACCEPTED'
                   AND json_extract(content_json, '$.admission_policy_digest')=admission_policy_digest
                   AND json_extract(content_json, '$.claim_id')=claim_id""",
                (policy_digest,),
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

    def _encoded(self, value: _DurableValue) -> tuple[str, str]:
        payload = self._encode(value.to_dict())
        return payload, canonical_digest(value.to_dict())

    def _decode_claim(self, row: tuple[str, str]) -> Claim:
        return self._decode_durable(
            row,
            lambda value: Claim.from_dict(value, config=self.config.claim),
            "Claim",
        )

    def _decode_manifest(self, row: Sequence[object]) -> EvidenceManifest:
        return self._decode_durable(
            row,
            lambda value: EvidenceManifest.from_dict(value, ingress_config=self.config.ingress),
            "EvidenceManifest",
        )

    @staticmethod
    def _decode_plain(
        row: Sequence[object],
        factory: Callable[[object], _T],
        label: str,
    ) -> _T:
        payload, encoded_digest = str(row[0]), str(row[1])
        try:
            decoded = json.loads(payload)
            if canonical_json(decoded) != payload or canonical_digest(decoded) != encoded_digest:
                raise ClaimStoreError(f"{label} durable encoding mismatch")
            result = factory(decoded)
            if canonical_json(cast(Any, result).to_dict()) != payload:
                raise ClaimStoreError(f"{label} canonical read-back mismatch")
            return result
        except ClaimStoreError:
            raise
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ClaimStoreError(f"failed strict read-back for {label}") from exc

    @classmethod
    def _decode_durable(
        cls,
        row: Sequence[object],
        factory: Callable[[object], _T],
        label: str,
    ) -> _T:
        result = cls._decode_plain(row, factory, label)
        durable = cast(Any, result)
        if canonical_digest(
            {key: value for key, value in durable.to_dict().items() if key != "content_digest"}
        ) != durable.content_digest:
            raise ClaimStoreError(f"{label} object content digest mismatch")
        return result

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
            raise ClaimStoreError("Behavior SQLite table set does not match Schema V3")
        for table, expected in _TABLE_COLUMNS.items():
            actual = tuple(row[1] for row in connection.execute(f"PRAGMA table_info({table})"))
            if actual != expected:
                raise ClaimStoreError(f"Behavior SQLite columns do not match Schema V3 for {table}")
        indexes = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='index' AND name NOT LIKE 'sqlite_%'"
            )
        }
        if not _REQUIRED_INDEXES.issubset(indexes):
            raise ClaimStoreError("Behavior SQLite indexes do not match Schema V3")
        foreign_key_rows = connection.execute("PRAGMA foreign_key_check").fetchall()
        if foreign_key_rows:
            tables_with_errors = ",".join(sorted({str(row[0]) for row in foreign_key_rows}))
            raise ClaimStoreError(f"Behavior SQLite foreign key integrity failed for tables: {tables_with_errors}")
        self._validate_materialized_columns(connection)
        with closing(sqlite3.connect(":memory:")) as reference:
            reference.execute("PRAGMA foreign_keys=ON")
            for statement in _SCHEMA_STATEMENTS:
                reference.execute(statement)
            expected_tables = self._schema_sql(reference, "table")
            expected_indexes = self._schema_sql(reference, "index")
        actual_tables = self._schema_sql(connection, "table")
        actual_indexes = self._schema_sql(connection, "index")
        for table in _TABLE_COLUMNS:
            if actual_tables.get(table) != expected_tables.get(table):
                raise ClaimStoreError(f"Behavior SQLite table definition does not match Schema V3 for {table}")
        for index in _REQUIRED_INDEXES:
            if actual_indexes.get(index) != expected_indexes.get(index):
                raise ClaimStoreError(f"Behavior SQLite index definition does not match Schema V3 for {index}")

    def _validate_materialized_columns(self, connection: sqlite3.Connection) -> None:
        def verify(actual: Sequence[object], expected: tuple[object, ...], label: str) -> None:
            if tuple(actual) != expected:
                raise ClaimStoreError(f"{label} indexed columns differ from canonical content")

        item: Any
        for row in connection.execute(
            """SELECT semantic_record_id, producer_fingerprint, stream_id, source_sequence,
               owner_identity_digest, event_time_start, event_time_end, semantic_digest,
               content_digest, content_json, encoded_digest FROM semantic_records"""
        ):
            item = self._decode_durable(
                row[-2:],
                lambda value: OwnerScopedSemanticRecord.from_dict(value, config=self.config.ingress),
                "OwnerScopedSemanticRecord",
            )
            verify(
                row[:-2],
                (
                    item.semantic_record_id,
                    item.producer_fingerprint.digest,
                    item.semantic_input.stream_id,
                    item.semantic_input.source_sequence,
                    item.owner_identity_digest,
                    utc_text(item.semantic_input.event_time_start),
                    utc_text(item.semantic_input.event_time_end),
                    item.semantic_digest,
                    item.content_digest,
                ),
                "semantic_records",
            )
        for row in connection.execute(
            """SELECT decision_id, semantic_record_id, owner_identity_digest, status, decided_at,
               decision_identity_digest, content_digest, content_json, encoded_digest
               FROM semantic_ingress_decisions"""
        ):
            item = self._decode_durable(row[-2:], IngressDecision.from_dict, "IngressDecision")
            verify(
                row[:-2],
                (
                    item.decision_id,
                    item.semantic_record_id,
                    item.owner_identity_digest,
                    item.status.value,
                    utc_text(item.decided_at),
                    item.decision_identity_digest,
                    item.content_digest,
                ),
                "semantic_ingress_decisions",
            )
        for row in connection.execute(
            """SELECT bundle_id, grouping_key, generation, owner_identity_digest, state, watermark,
               content_json, encoded_digest FROM active_evidence_bundles"""
        ):
            item = self._decode_plain(row[-2:], SemanticEvidenceBundle.from_dict, "SemanticEvidenceBundle")
            verify(
                row[:-2],
                (
                    item.bundle_id,
                    item.grouping_key,
                    item.generation,
                    item.owner_identity_digest,
                    item.state.value,
                    None if item.watermark is None else utc_text(item.watermark),
                ),
                "active_evidence_bundles",
            )
        for row in connection.execute(
            """SELECT manifest_id, bundle_id, owner_identity_digest, started_at, ended_at, sealed_at,
               manifest_semantic_digest, content_digest, content_json, encoded_digest
               FROM evidence_manifests"""
        ):
            item = self._decode_manifest(row[-2:])
            verify(
                row[:-2],
                (
                    item.manifest_id,
                    item.bundle_id,
                    item.owner_identity_digest,
                    utc_text(item.started_at),
                    utc_text(item.ended_at),
                    utc_text(item.sealed_at),
                    item.manifest_semantic_digest,
                    item.content_digest,
                ),
                "evidence_manifests",
            )
        for row in connection.execute(
            """SELECT attempt_id, processing_identity, processing_lane, manifest_id,
               semantic_record_id, normalizer_fingerprint, attempt_number, status, content_digest,
               content_json, encoded_digest FROM claim_normalizer_attempts"""
        ):
            item = self._decode_durable(row[-2:], ClaimNormalizerAttempt.from_dict, "ClaimNormalizerAttempt")
            verify(
                row[:-2],
                (
                    item.attempt_id,
                    item.processing_identity,
                    item.processing_lane.value,
                    item.manifest_id,
                    item.semantic_record_id,
                    item.normalizer_fingerprint,
                    item.attempt_number,
                    item.status.value,
                    item.content_digest,
                ),
                "claim_normalizer_attempts",
            )
        for row in connection.execute(
            """SELECT claim_batch_id, processing_identity, processing_lane, manifest_id,
               semantic_record_id, normalizer_fingerprint, created_at, content_digest,
               content_json, encoded_digest FROM claim_batches"""
        ):
            item = self._decode_durable(row[-2:], ClaimBatch.from_dict, "ClaimBatch")
            verify(
                row[:-2],
                (
                    item.claim_batch_id,
                    item.processing_identity,
                    item.processing_lane.value,
                    item.manifest_id,
                    item.semantic_record_id,
                    item.normalizer_fingerprint,
                    utc_text(item.created_at),
                    item.content_digest,
                ),
                "claim_batches",
            )
        for row in connection.execute(
            """SELECT claim_id, manifest_id, semantic_record_id, owner_identity_digest,
               semantic_fingerprint, derivation_class, claim_kind, time_start, time_end, created_at, content_digest,
               content_json, encoded_digest FROM claims"""
        ):
            item = self._decode_claim(row[-2:])
            verify(
                row[:-2],
                (
                    item.claim_id,
                    item.manifest_id,
                    item.semantic_record_id,
                    item.owner_identity_digest,
                    item.semantic_fingerprint,
                    item.derivation_class.value,
                    item.proposal.claim_kind.value,
                    utc_text(item.time_start),
                    utc_text(item.time_end),
                    utc_text(item.created_at),
                    item.content_digest,
                ),
                "claims",
            )
        for row in connection.execute(
            """SELECT decision_id, processing_identity, claim_id, admission_policy_digest,
               status, decided_at, content_digest, content_json, encoded_digest
               FROM claim_admission_decisions"""
        ):
            item = self._decode_durable(row[-2:], ClaimAdmissionDecision.from_dict, "ClaimAdmissionDecision")
            verify(
                row[:-2],
                (
                    item.decision_id,
                    item.processing_identity,
                    item.claim_id,
                    item.admission_policy_digest,
                    item.status.value,
                    utc_text(item.admission_decided_at),
                    item.content_digest,
                ),
                "claim_admission_decisions",
            )
        for row in connection.execute(
            """SELECT processing_identity, processing_lane, scope_semantic_record_id, manifest_id,
               admission_policy_digest, completed_at, publication_recorded_at, content_digest,
               content_json, encoded_digest FROM claim_processing_receipts"""
        ):
            item = self._decode_durable(row[-2:], ClaimProcessingReceipt.from_dict, "ClaimProcessingReceipt")
            verify(
                row[:-2],
                (
                    item.processing_identity,
                    item.processing_lane.value,
                    item.scope_semantic_record_id,
                    item.manifest_id,
                    item.admission_policy_digest,
                    utc_text(item.processing_completed_at),
                    utc_text(item.publication_recorded_at),
                    item.content_digest,
                ),
                "claim_processing_receipts",
            )

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
