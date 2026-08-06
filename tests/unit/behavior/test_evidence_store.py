from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import timedelta

import pytest

from behavior.config import BehaviorConfig
from behavior.errors import ClaimStoreError, SemanticRecordConflictError
from behavior.evidence import (
    CoverageSummary,
    EvidenceSealReason,
    SemanticIngestStatus,
)
from behavior.evidence.service import EvidenceService
from behavior.ingress import (
    BoundarySignal,
    CoverageIntervalPayload,
    CoverageStatus,
    EvidenceKind,
    EvidenceReference,
    SemanticModality,
    SemanticRecordKind,
)
from behavior.persistence.sqlite import SQLiteBehaviorEvidenceClaimStore
from foundation.observability import NullObserver
from tests.unit.behavior.conftest import (
    BASE_TIME,
    accepted_ingress,
    bind_record,
    digest,
    make_input,
)


def ingest(service: EvidenceService, record):
    return service.ingest(
        accepted_ingress(record, ingress_config=service.store.config.ingress)
    )


def test_out_of_order_records_are_stably_sorted_and_track_is_not_grouping(owner, store) -> None:
    service = EvidenceService(store, config=store.config.evidence, observer=NullObserver())
    later = bind_record(owner, make_input(sequence=1, offset_seconds=10, upstream_subject_ref="track-z"))
    earlier = bind_record(owner, make_input(sequence=0, offset_seconds=0, upstream_subject_ref="track-a"))
    ingest(service, later)
    result = ingest(service, earlier)
    assert result.active_bundle is not None
    assert result.active_bundle.ordered_semantic_record_ids == (
        earlier.semantic_record_id,
        later.semantic_record_id,
    )
    changed_track = bind_record(
        owner,
        make_input(sequence=2, offset_seconds=11, upstream_subject_ref="track-other"),
    )
    assert service.assembler.grouping_key(changed_track) == service.assembler.grouping_key(later)


def test_same_input_different_arrival_and_seal_times_has_same_manifest_identity(tmp_path, owner) -> None:
    identities = []
    for index, order in enumerate(((0, 1), (1, 0))):
        config = BehaviorConfig()
        store = SQLiteBehaviorEvidenceClaimStore(tmp_path / f"behavior-{index}", config=config, initialize=True)
        service = EvidenceService(store, config=config.evidence, observer=NullObserver())
        records = (
            bind_record(owner, make_input(sequence=0, offset_seconds=0)),
            bind_record(owner, make_input(sequence=1, offset_seconds=5)),
        )
        active = None
        for item in order:
            active = ingest(service, records[item]).active_bundle
        assert active is not None
        manifest = service.seal_bundle(active.bundle_id)
        assert manifest is not None
        identities.append((manifest.manifest_id, manifest.manifest_semantic_digest))
    assert identities[0] == identities[1]


def test_late_record_is_explicit_and_never_persisted(owner, store) -> None:
    service = EvidenceService(store, config=store.config.evidence, observer=NullObserver())
    recent = bind_record(owner, make_input(sequence=1, offset_seconds=100))
    ingest(service, recent)
    late = bind_record(owner, make_input(sequence=0, offset_seconds=0))
    result = ingest(service, late)
    assert result.status is SemanticIngestStatus.LATE_REJECTED
    assert result.decision.reason_code == "event_time_before_committed_watermark"
    assert store.read_semantic_record(late.semantic_record_id) is None


@pytest.mark.parametrize(
    ("config", "records", "reason"),
    [
        (
            BehaviorConfig(evidence=replace(BehaviorConfig().evidence, max_gap_seconds=1.0)),
            (make_input(sequence=0), make_input(sequence=1, offset_seconds=2)),
            EvidenceSealReason.MAX_GAP,
        ),
        (
            BehaviorConfig(
                evidence=replace(
                    BehaviorConfig().evidence,
                    max_gap_seconds=1.0,
                    max_bundle_duration_seconds=1.0,
                )
            ),
            (make_input(sequence=0), make_input(sequence=1, offset_seconds=1, duration_seconds=1)),
            EvidenceSealReason.MAX_DURATION,
        ),
        (
            BehaviorConfig(evidence=replace(BehaviorConfig().evidence, max_records_per_bundle=2)),
            (make_input(sequence=0), make_input(sequence=1)),
            EvidenceSealReason.MAX_RECORDS,
        ),
    ],
)
def test_hard_bundle_boundaries_publish_manifest(tmp_path, owner, config, records, reason) -> None:
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / reason.value, config=config, initialize=True)
    service = EvidenceService(store, config=config.evidence, observer=NullObserver())
    result = None
    for value in records:
        result = ingest(service, bind_record(owner, value))
    assert result is not None and result.manifest_ids
    assert store.read_manifest(result.manifest_ids[0]).seal_reason is reason


def test_projection_and_upstream_end_boundaries(tmp_path, owner) -> None:
    size = make_input().projection_chars
    config = BehaviorConfig(
        ingress=replace(BehaviorConfig().ingress, max_payload_chars=size),
        evidence=replace(
            BehaviorConfig().evidence,
            max_projection_chars_per_bundle=size * 2,
        ),
    )
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "projection", config=config, initialize=True)
    service = EvidenceService(store, config=config.evidence, observer=NullObserver())
    ingest(service, bind_record(owner, make_input(sequence=0)))
    result = ingest(service, bind_record(owner, make_input(sequence=1)))
    assert store.read_manifest(result.manifest_ids[0]).seal_reason is EvidenceSealReason.MAX_PROJECTION_SIZE

    config2 = BehaviorConfig()
    store2 = SQLiteBehaviorEvidenceClaimStore(tmp_path / "end", config=config2, initialize=True)
    service2 = EvidenceService(store2, config=config2.evidence, observer=NullObserver())
    result2 = ingest(
        service2,
        bind_record(owner, make_input(boundary_signal=BoundarySignal.END)),
    )
    assert store2.read_manifest(result2.manifest_ids[0]).seal_reason is EvidenceSealReason.UPSTREAM_END


def test_explicit_seal_empty_is_noop_and_active_bundle_recovers(tmp_path, owner) -> None:
    config = BehaviorConfig()
    root = tmp_path / "behavior"
    first = SQLiteBehaviorEvidenceClaimStore(root, config=config, initialize=True)
    assert (
        first.seal_bundle(
            "bundle_" + "0" * 64,
            reason=EvidenceSealReason.EXPLICIT,
            assembler=EvidenceService(first, config=config.evidence, observer=NullObserver()).assembler,
            sealed_at=BASE_TIME,
        )
        is None
    )
    record = bind_record(owner)
    result = ingest(EvidenceService(first, config=config.evidence, observer=NullObserver()), record)
    second = SQLiteBehaviorEvidenceClaimStore(root, config=config, initialize=True)
    assert result.active_bundle is not None
    assert second.read_active_bundle(result.active_bundle.bundle_id) == result.active_bundle


def test_coverage_is_only_from_explicit_coverage_records(tmp_path, owner) -> None:
    config = BehaviorConfig()
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "coverage", config=config, initialize=True)
    service = EvidenceService(store, config=config.evidence, observer=NullObserver())
    normal = bind_record(owner, make_input(sequence=0, duration_seconds=10))
    active = ingest(service, normal).active_bundle
    manifest = service.seal_bundle(active.bundle_id)
    assert manifest.coverage_summary is CoverageSummary.UNKNOWN
    assert manifest.covered_intervals == ()
    assert manifest.unknown_intervals[0].event_time_start == BASE_TIME

    coverage = bind_record(
        owner,
        make_input(
            sequence=1,
            kind=SemanticRecordKind.COVERAGE_INTERVAL,
            payload=CoverageIntervalPayload("VISION", CoverageStatus.COVERED, None),
            modality=SemanticModality.VISION,
            offset_seconds=20,
            duration_seconds=5,
            correlation_id="coverage-correlation",
            boundary_signal=BoundarySignal.END,
        ),
    )
    result = ingest(service, coverage)
    covered = store.read_manifest(result.manifest_ids[0])
    assert covered.coverage_summary is CoverageSummary.COVERED
    assert len(covered.covered_intervals) == 1
    assert covered.blind_intervals == ()


def test_coverage_gaps_remain_unknown(tmp_path, owner) -> None:
    config = BehaviorConfig()
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "coverage-gaps", config=config, initialize=True)
    service = EvidenceService(store, config=config.evidence, observer=NullObserver())
    records = (
        bind_record(
            owner,
            make_input(
                sequence=0,
                kind=SemanticRecordKind.COVERAGE_INTERVAL,
                payload=CoverageIntervalPayload("VISION", CoverageStatus.COVERED, None),
                modality=SemanticModality.VISION,
                duration_seconds=2,
            ),
        ),
        bind_record(
            owner,
            make_input(
                sequence=1,
                kind=SemanticRecordKind.COVERAGE_INTERVAL,
                payload=CoverageIntervalPayload("VISION", CoverageStatus.BLIND, "occluded"),
                modality=SemanticModality.VISION,
                offset_seconds=4,
                duration_seconds=2,
                boundary_signal=BoundarySignal.END,
            ),
        ),
    )
    ingest(service, records[0])
    result = ingest(service, records[1])
    manifest = store.read_manifest(result.manifest_ids[0])
    assert manifest.covered_intervals
    assert manifest.blind_intervals
    assert manifest.unknown_intervals[0].event_time_start == BASE_TIME + timedelta(seconds=2)


def test_manifest_snapshot_keeps_reference_but_not_payload_text(tmp_path, owner) -> None:
    reference = EvidenceReference(
        reference="blob://evidence/frame",
        evidence_kind=EvidenceKind.IMAGE_FRAME,
        digest=digest("frame"),
        event_time_start=BASE_TIME,
        event_time_end=BASE_TIME,
        media_type="image/jpeg",
        size_bytes=10,
        source_system_ref="upstream",
    )
    record = bind_record(owner, make_input(evidence_refs=(reference,), boundary_signal=BoundarySignal.END))
    config = BehaviorConfig()
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "snapshot", config=config, initialize=True)
    result = ingest(EvidenceService(store, config=config.evidence, observer=NullObserver()), record)
    manifest = store.read_manifest(result.manifest_ids[0])
    snapshot = manifest.ordered_record_snapshots[0]
    assert snapshot.evidence_refs == (reference,)
    assert "payload" not in snapshot.to_dict()


def test_manifest_time_query_uses_a_stable_composite_cursor(store, owner) -> None:
    service = EvidenceService(store, config=store.config.evidence, observer=NullObserver())
    first = bind_record(
        owner,
        make_input(
            sequence=30,
            correlation_id="cursor-a",
            boundary_signal=BoundarySignal.END,
        ),
    )
    second = bind_record(
        owner,
        make_input(
            sequence=31,
            offset_seconds=1,
            correlation_id="cursor-b",
            boundary_signal=BoundarySignal.END,
        ),
    )
    ingest(service, first)
    ingest(service, second)
    page_one = store.list_manifests(
        start=BASE_TIME - timedelta(seconds=1),
        end=BASE_TIME + timedelta(seconds=2),
        limit=1,
    )
    page_two = store.list_manifests(
        start=BASE_TIME - timedelta(seconds=1),
        end=BASE_TIME + timedelta(seconds=2),
        limit=1,
        cursor=page_one[0].manifest_id,
    )
    assert len(page_one) == len(page_two) == 1
    assert page_one[0].manifest_id != page_two[0].manifest_id
    with pytest.raises(ValueError, match="cursor"):
        store.list_manifests(
            start=BASE_TIME - timedelta(seconds=1),
            end=BASE_TIME + timedelta(seconds=2),
            limit=1,
            cursor="manifest_missing",
        )


def test_replay_is_idempotent_and_stream_sequence_conflict_rolls_back(store, owner) -> None:
    service = EvidenceService(store, config=store.config.evidence, observer=NullObserver())
    record = bind_record(owner)
    first = ingest(service, record)
    replay = ingest(service, bind_record(owner, ingested_at=BASE_TIME + timedelta(seconds=9)))
    assert replay.status is SemanticIngestStatus.REPLAYED
    assert replay.semantic_record_id == first.semantic_record_id
    conflict = bind_record(owner, make_input(payload=replace(make_input().payload, value="off")))
    with pytest.raises(SemanticRecordConflictError):
        ingest(service, conflict)
    with closing(sqlite3.connect(store.path)) as connection:
        assert connection.execute("SELECT COUNT(*) FROM semantic_records").fetchone()[0] == 1


def test_store_permissions_schema_and_symlink_rejection(tmp_path) -> None:
    config = BehaviorConfig()
    store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "secure", config=config, initialize=True)
    assert os.stat(store.root).st_mode & 0o777 == 0o700
    assert os.stat(store.path).st_mode & 0o777 == 0o600
    store.initialize()
    with closing(sqlite3.connect(store.path)) as connection:
        assert connection.execute("SELECT value FROM behavior_metadata WHERE key='schema_version'").fetchone()[0] == "3"
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ClaimStoreError):
        SQLiteBehaviorEvidenceClaimStore(link, config=config)


def test_schema_validation_rejects_a_same_named_wrong_index(tmp_path) -> None:
    root = tmp_path / "wrong-index"
    config = BehaviorConfig()
    store = SQLiteBehaviorEvidenceClaimStore(root, config=config, initialize=True)
    with closing(sqlite3.connect(store.path)) as connection:
        connection.execute("DROP INDEX idx_claims_record")
        connection.execute("CREATE INDEX idx_claims_record ON claims(manifest_id, claim_id)")
    with pytest.raises(ClaimStoreError, match="index definition"):
        SQLiteBehaviorEvidenceClaimStore(root, config=config, initialize=True)
