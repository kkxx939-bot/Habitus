from __future__ import annotations

import os
import sqlite3
from dataclasses import replace
from datetime import timedelta
from pathlib import Path

import pytest

from behavior.config import BehaviorConfig
from behavior.errors import ClaimStoreError, EvidenceWindowError
from behavior.evidence import (
    EvidenceCoverageState,
    EvidenceSealReason,
    EvidenceService,
    SourceIngestStatus,
)
from behavior.persistence.sqlite import SQLiteBehaviorEvidenceClaimStore
from behavior.source import CaptureState, SourceType
from foundation.observability import NullObserver
from tests.unit.behavior.conftest import BASE_TIME, make_source


def _service(store, evidence_config=None) -> EvidenceService:
    return EvidenceService(
        store,
        config=evidence_config or store.config.evidence,
        observer=NullObserver(),
    )


def test_out_of_order_inputs_have_stable_event_order_and_manifest_digest(tmp_path, owner) -> None:
    records = tuple(
        make_source(
            owner,
            sequence=index,
            offset_seconds=offset,
            ingested_offset_seconds=100 + index,
            semantic_data={},
        )
        for index, offset in enumerate((12, 0, 6))
    )
    manifests = []
    for directory, order in (("first", records), ("second", tuple(reversed(records)))):
        config = BehaviorConfig(
            evidence=replace(BehaviorConfig().evidence, allowed_lateness_seconds=120.0)
        )
        store = SQLiteBehaviorEvidenceClaimStore(tmp_path / directory, config=config)
        store.initialize()
        service = _service(store)
        latest = None
        for record in order:
            latest = service.ingest_source(record)
            assert latest.status is SourceIngestStatus.ACCEPTED
        assert latest is not None and latest.active_window is not None
        manifests.append(service.seal_window(latest.active_window.window_id))
    first, second = manifests
    assert first is not None and second is not None
    assert tuple(item.event_time_start for item in first.ordered_source_records) == tuple(
        sorted(item.event_time_start for item in first.ordered_source_records)
    )
    assert first.to_dict() == second.to_dict()
    assert first.manifest_digest == second.manifest_digest


def test_allowed_lateness_returns_explicit_rejection_without_durable_source(store, owner) -> None:
    service = _service(store, replace(store.config.evidence, allowed_lateness_seconds=5.0))
    service.ingest_source(make_source(owner, sequence=1, offset_seconds=100, semantic_data={}))
    late = make_source(owner, sequence=0, offset_seconds=0, semantic_data={})
    result = service.ingest_source(late)
    assert result.status is SourceIngestStatus.LATE_REJECTED
    assert result.reason_code == "event_time_before_committed_watermark"
    assert store.read_source(late.source_record_id) is None

    active = store.read_active_window(result.active_window.window_id)
    assert active is not None
    manifest = service.seal_window(active.window_id)
    assert manifest is not None
    later_late = make_source(
        owner,
        sequence=2,
        stream_id="late-after-seal",
        offset_seconds=1,
        semantic_data={},
    )
    sealed_result = service.ingest_source(later_late)
    assert sealed_result.status is SourceIngestStatus.LATE_REJECTED
    assert store.read_source(later_late.source_record_id) is None


def test_gap_duration_record_projection_explicit_and_stream_end_seals(tmp_path, owner) -> None:
    cases = (
        (
            replace(BehaviorConfig().evidence, allowed_lateness_seconds=500, max_gap_seconds=5),
            make_source(owner, sequence=1, offset_seconds=20, semantic_data={}),
            EvidenceSealReason.MAX_GAP,
        ),
        (
            replace(
                BehaviorConfig().evidence,
                allowed_lateness_seconds=500,
                max_gap_seconds=5,
                max_window_duration_seconds=5,
            ),
            make_source(owner, sequence=1, offset_seconds=4, duration_seconds=4, semantic_data={}),
            EvidenceSealReason.MAX_DURATION,
        ),
    )
    for index, (evidence_config, second, reason) in enumerate(cases):
        config = BehaviorConfig()
        local_store = SQLiteBehaviorEvidenceClaimStore(tmp_path / f"case-{index}", config=config)
        local_store.initialize()
        service = _service(local_store, evidence_config)
        service.ingest_source(make_source(owner, sequence=0, semantic_data={}))
        result = service.ingest_source(second)
        manifest = local_store.read_manifest(result.manifest_ids[0])
        assert manifest is not None and manifest.seal_reason is reason

    count_store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "count", config=BehaviorConfig())
    count_store.initialize()
    count_service = _service(
        count_store,
        replace(BehaviorConfig().evidence, max_records_per_window=1),
    )
    count_result = count_service.ingest_source(make_source(owner, semantic_data={}))
    count_manifest = count_store.read_manifest(count_result.manifest_ids[0])
    assert count_manifest is not None and count_manifest.seal_reason is EvidenceSealReason.MAX_RECORDS

    projection_store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "projection", config=BehaviorConfig())
    projection_store.initialize()
    projection_service = _service(
        projection_store,
        replace(BehaviorConfig().evidence, max_projection_chars_per_window=2),
    )
    projection_result = projection_service.ingest_source(make_source(owner, semantic_data={}))
    projection_manifest = projection_store.read_manifest(projection_result.manifest_ids[0])
    assert projection_manifest is not None
    assert projection_manifest.seal_reason is EvidenceSealReason.MAX_PROJECTION_SIZE

    explicit_store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "explicit", config=BehaviorConfig())
    explicit_store.initialize()
    explicit_service = _service(explicit_store)
    explicit = explicit_service.ingest_source(make_source(owner, semantic_data={}))
    manifest = explicit_service.seal_window(explicit.active_window.window_id)
    assert manifest is not None and manifest.seal_reason is EvidenceSealReason.EXPLICIT
    assert explicit_service.seal_window(explicit.active_window.window_id) == manifest

    end_store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "stream-end", config=BehaviorConfig())
    end_store.initialize()
    end_result = _service(end_store).ingest_source(
        make_source(owner, semantic_data={}, capture_state=CaptureState.STREAM_END)
    )
    stream_manifest = end_store.read_manifest(end_result.manifest_ids[0])
    assert stream_manifest is not None and stream_manifest.seal_reason is EvidenceSealReason.STREAM_END

    with pytest.raises(EvidenceWindowError, match="duration exceeds"):
        _service(
            end_store,
            replace(
                BehaviorConfig().evidence,
                max_gap_seconds=1,
                max_window_duration_seconds=1,
            ),
        ).ingest_source(
            make_source(
                owner,
                sequence=2,
                stream_id="oversized-duration",
                duration_seconds=2,
                semantic_data={},
            )
        )
    assert _service(end_store).seal_window("window_" + "f" * 64) is None


def test_active_window_recovers_and_source_window_write_rolls_back(tmp_path, owner, monkeypatch) -> None:
    config = BehaviorConfig()
    path = tmp_path / "recovery"
    first_store = SQLiteBehaviorEvidenceClaimStore(path, config=config)
    first_store.initialize()
    accepted = _service(first_store).ingest_source(make_source(owner, semantic_data={}))
    rebuilt = SQLiteBehaviorEvidenceClaimStore(path, config=config)
    rebuilt.initialize()
    assert rebuilt.read_active_window(accepted.active_window.window_id) == accepted.active_window

    failing_store = SQLiteBehaviorEvidenceClaimStore(tmp_path / "rollback", config=config)
    failing_store.initialize()
    record = make_source(owner, sequence=9, semantic_data={})

    def fail_window(*args, **kwargs):
        raise RuntimeError("simulated window failure")

    monkeypatch.setattr(failing_store, "_insert_active_window", fail_window)
    with pytest.raises(RuntimeError, match="simulated"):
        _service(failing_store).ingest_source(record)
    assert failing_store.read_source(record.source_record_id) is None


def test_manifest_coverage_blind_intervals_and_bounded_queries(store, owner) -> None:
    service = _service(store)
    blind = make_source(
        owner,
        source_type=SourceType.COVERAGE_SIGNAL,
        capture_state=CaptureState.BLIND,
        semantic_data={},
    )
    result = service.ingest_source(blind)
    manifest = service.seal_window(result.active_window.window_id)
    assert manifest is not None
    assert manifest.coverage_state is EvidenceCoverageState.BLIND
    assert len(manifest.blind_intervals) == 1
    assert manifest.blind_intervals[0].source_record_id == blind.source_record_id
    assert "raw" not in str(manifest.to_dict()).casefold()
    listed = store.list_manifests(
        start=BASE_TIME - timedelta(seconds=1),
        end=BASE_TIME + timedelta(seconds=1),
        limit=1,
    )
    assert listed == (manifest,)
    assert store.read_manifest_for_window(manifest.window_id) == manifest
    with pytest.raises(ValueError, match="limit"):
        store.list_manifests(start=BASE_TIME, end=BASE_TIME, limit=0)


def test_sqlite_initialization_permissions_symlink_and_schema_validation(tmp_path, owner) -> None:
    config = BehaviorConfig()
    root = tmp_path / "secure"
    store = SQLiteBehaviorEvidenceClaimStore(root, config=config)
    store.initialize()
    store.initialize()
    assert stat_mode(root) == 0o700
    assert stat_mode(store.path) == 0o600
    assert store.readiness()[0]

    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "linked-root"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ClaimStoreError, match="symbolic"):
        SQLiteBehaviorEvidenceClaimStore(link, config=config)

    record = make_source(owner, semantic_data={})
    _service(store).ingest_source(record)
    with sqlite3.connect(store.path) as connection:
        connection.execute(
            "UPDATE source_records SET content_digest=? WHERE source_record_id=?",
            ("0" * 64, record.source_record_id),
        )
    with pytest.raises(ClaimStoreError, match="content digest mismatch"):
        store.read_source(record.source_record_id)

    with sqlite3.connect(store.path) as connection:
        connection.execute("DROP INDEX idx_manifest_time")
    rebuilt = SQLiteBehaviorEvidenceClaimStore(root, config=config)
    with pytest.raises(ClaimStoreError, match="missing required indexes"):
        rebuilt.initialize()


def stat_mode(path: Path) -> int:
    return os.stat(path).st_mode & 0o777
