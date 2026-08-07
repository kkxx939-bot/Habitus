from __future__ import annotations

import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta

import pytest

from behavior._validation import decode_cursor, encode_cursor
from behavior.config import BehaviorConfig
from behavior.errors import (
    BehaviorAdapterCapabilityError,
    BehaviorAdapterError,
    BehaviorEvidenceCapacityError,
    BehaviorEvidenceConflictError,
    BehaviorEvidenceSchemaError,
    BehaviorStoreError,
    LegacyBehaviorStoreError,
)
from behavior.evidence import (
    ActivitySegmentPayload,
    BehaviorModality,
    BehaviorOriginKind,
    BehaviorRecordKind,
    BehaviorRole,
    BehaviorSemanticAdapterRegistry,
    BehaviorSemanticContent,
    BehaviorSemanticInput,
    BehaviorSemanticInputBatch,
    BehaviorSourceTrust,
    BehaviorTimeMode,
    ClockSyncStatus,
    CorrelationRef,
    EvidenceIntegrity,
    FreeTextSemanticPayload,
    IngressReceiptStatus,
    PhaseHint,
)
from behavior.evidence.ingress import BehaviorEvidenceIngressService
from behavior.persistence import BehaviorDatabase, SQLiteBehaviorEvidenceLedger
from tests.unit.behavior.conftest import BASE_TIME, FakeAdapter, FakeClock, digest, source_descriptor


def semantic_input(
    *,
    event: str = "event-1",
    sequence: int = 1,
    item_index: int = 0,
    generation: int = 0,
    event_time=BASE_TIME,
    activity: str = "walking",
    correlation: CorrelationRef | None = None,
) -> BehaviorSemanticInput:
    source = source_descriptor(
        event=event,
        generation=generation,
        sequence=sequence,
        item_index=item_index,
    )
    if correlation is not None:
        source = replace(source, correlation_refs=(correlation,))
    content = BehaviorSemanticContent(
        record_kind=BehaviorRecordKind.ACTIVITY_SEGMENT,
        subject_role=BehaviorRole.USER,
        actor_role=BehaviorRole.USER,
        modality=BehaviorModality.VISION,
        event_time_start=event_time,
        event_time_end=event_time + timedelta(seconds=1),
        event_time_uncertainty_ms=100,
        clock_domain="camera",
        clock_sync_status=ClockSyncStatus.OFFSET_ESTIMATED,
        scene_ref=None,
        location_ref=None,
        object_refs=(),
        entity_refs=(),
        payload=ActivitySegmentPayload(activity, PhaseHint.IN_PROGRESS, {}),
        evidence_refs=(),
        source_confidence=0.8,
        integrity=EvidenceIntegrity.COMPLETE,
    )
    return BehaviorSemanticInput(content, source)


def service(tmp_path, adapter, *, config=None, clock=None):
    resolved = config or BehaviorConfig()
    database = BehaviorDatabase(tmp_path / "behavior", config=resolved, initialize=True)
    ledger = SQLiteBehaviorEvidenceLedger(database)
    registry = BehaviorSemanticAdapterRegistry()
    registry.register(adapter)
    return database, ledger, BehaviorEvidenceIngressService(
        ledger,
        registry,
        config=resolved,
        clock=clock or FakeClock(),
    )


def test_delivery_replay_conflict_request_digest_and_evidence_sequence(tmp_path) -> None:
    adapter = FakeAdapter(semantic_input())
    _, ledger, ingress = service(tmp_path, adapter)
    delivery = digest("delivery")
    first = asyncio.run(ingress.ingest(adapter.name, {"frame": 1}, delivery_id=delivery))
    replay = asyncio.run(ingress.ingest(adapter.name, {"frame": 1}, delivery_id=delivery))
    assert first.receipt.status is IngressReceiptStatus.COMMITTED
    assert not first.reused and replay.reused
    assert replay.receipt == first.receipt
    assert ledger.list_after_sequence(0, 10)[0].sequence == 1
    with pytest.raises(BehaviorEvidenceConflictError):
        asyncio.run(ingress.ingest(adapter.name, {"frame": 2}, delivery_id=delivery))


def test_same_evidence_identity_reuses_original_ingested_time_across_deliveries(tmp_path) -> None:
    adapter = FakeAdapter(semantic_input())
    _, ledger, ingress = service(tmp_path, adapter)
    first = asyncio.run(
        ingress.ingest(adapter.name, {"frame": 1}, delivery_id=digest("delivery-one"))
    )
    later_ingress = BehaviorEvidenceIngressService(
        ledger,
        ingress.adapters,
        config=ingress.config,
        clock=FakeClock(BASE_TIME + timedelta(minutes=1)),
    )
    second = asyncio.run(
        later_ingress.ingest(
            adapter.name,
            {"frame": 1},
            delivery_id=digest("delivery-two"),
        )
    )
    assert second.records == first.records
    assert second.records[0].ingested_at == BASE_TIME
    assert len(ledger.list_after_sequence(0, 10)) == 1


def test_batch_commit_is_atomic_and_one_source_event_can_emit_multiple_items(tmp_path) -> None:
    correlation = CorrelationRef("scene", "tea", "root")
    first = semantic_input(event="same-event", sequence=7, item_index=0, correlation=correlation)
    second = semantic_input(event="same-event", sequence=7, item_index=1, activity="pouring", correlation=correlation)
    adapter = FakeAdapter(BehaviorSemanticInputBatch((first, second)))
    _, ledger, ingress = service(tmp_path, adapter)
    result = asyncio.run(ingress.ingest(adapter.name, [1, 2], delivery_id=digest("batch")))
    assert len(result.records) == 2
    assert [item.sequence for item in ledger.list_after_sequence(0, 10)] == [1, 2]
    by_source, _ = ledger.list_by_source_event(first.source.source_event_ref, 10)
    assert {entry.record.provenance.descriptor.source_item_index for entry in by_source} == {0, 1}
    by_correlation, _ = ledger.list_by_correlation(correlation, 10)
    assert len(by_correlation) == 2


@pytest.mark.parametrize("violation", ["origin", "modality", "role", "batch"])
def test_ingress_mechanically_enforces_adapter_capability(violation, tmp_path) -> None:
    value = semantic_input()
    maximum_batch_size = 1
    if violation == "origin":
        value = replace(
            value,
            source=replace(value.source, origin_kind=BehaviorOriginKind.DIRECT_RUNTIME_EVENT),
        )
    elif violation == "modality":
        value = replace(value, content=replace(value.content, modality=BehaviorModality.AUDIO))
    elif violation == "role":
        value = replace(
            value,
            content=replace(
                value.content,
                subject_role=BehaviorRole.AGENT,
                actor_role=BehaviorRole.AGENT,
            ),
        )
    else:
        value = BehaviorSemanticInputBatch(
            (
                semantic_input(event="one", sequence=1),
                semantic_input(event="two", sequence=2),
            )
        )
    adapter = FakeAdapter(value, maximum_batch_size=maximum_batch_size)
    _, ledger, ingress = service(tmp_path, adapter)
    with pytest.raises(BehaviorAdapterCapabilityError):
        asyncio.run(ingress.ingest(adapter.name, {}, delivery_id=digest(violation)))
    assert ledger.list_after_sequence(0, 10) == ()
    assert ledger.read_ingress_receipt(digest(violation)) is None


def test_free_text_role_pair_requires_explicit_adapter_capability(tmp_path) -> None:
    value = semantic_input()
    value = replace(
        value,
        content=replace(
            value.content,
            record_kind=BehaviorRecordKind.FREE_TEXT_SEMANTIC,
            modality=BehaviorModality.TEXT,
            payload=FreeTextSemanticPayload("bounded", "en", ()),
        ),
    )
    adapter = FakeAdapter(
        value,
        kinds=(BehaviorRecordKind.FREE_TEXT_SEMANTIC,),
        modalities=(BehaviorModality.TEXT,),
        role_pairs=((BehaviorRole.SYSTEM, BehaviorRole.SYSTEM),),
    )
    _, ledger, ingress = service(tmp_path, adapter)
    delivery = digest("free-text-role")
    with pytest.raises(BehaviorAdapterCapabilityError):
        asyncio.run(ingress.ingest(adapter.name, {}, delivery_id=delivery))
    assert ledger.list_after_sequence(0, 10) == ()
    assert ledger.read_ingress_receipt(delivery) is None


@pytest.mark.parametrize("field", ["identifier", "reference"])
def test_ingress_applies_configured_identifier_and_reference_boundaries(field, tmp_path) -> None:
    base = BehaviorConfig().evidence
    if field == "identifier":
        value = semantic_input(event="x" * 17)
        evidence_config = replace(base, max_identifier_chars=16)
    else:
        value = semantic_input()
        value = replace(
            value,
            source=replace(value.source, source_ref="s3://bucket/" + "x" * 40),
        )
        evidence_config = replace(base, max_reference_chars=32)
    config = BehaviorConfig(evidence=evidence_config)
    adapter = FakeAdapter(value)
    _, ledger, ingress = service(tmp_path, adapter, config=config)
    with pytest.raises(BehaviorEvidenceSchemaError):
        asyncio.run(ingress.ingest(adapter.name, {}, delivery_id=digest(field)))
    assert ledger.list_after_sequence(0, 10) == ()


def test_concurrent_same_delivery_publishes_one_receipt_and_one_evidence(tmp_path) -> None:
    barrier = threading.Barrier(2)

    class ConcurrentAdapter(FakeAdapter):
        async def adapt(self, payload: object):
            del payload
            barrier.wait(timeout=5)
            return self.result

    adapter = ConcurrentAdapter(semantic_input())
    _, ledger, ingress = service(tmp_path, adapter)
    delivery = digest("concurrent-delivery")

    def invoke():
        return asyncio.run(ingress.ingest(adapter.name, {"frame": 1}, delivery_id=delivery))

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(future.result() for future in (executor.submit(invoke), executor.submit(invoke)))
    assert sorted(result.reused for result in results) == [False, True]
    assert results[0].receipt == results[1].receipt
    assert len(ledger.list_after_sequence(0, 10)) == 1


def test_adapter_metadata_cannot_change_during_adaptation(tmp_path) -> None:
    class MutatingAdapter(FakeAdapter):
        async def adapt(self, payload: object):
            del payload
            replacement = FakeAdapter(
                self.result,
                name="replacement",
                trust=BehaviorSourceTrust.DIRECT_DEVICE_FACT,
            )
            self.fingerprint = replacement.fingerprint
            self.capabilities = replacement.capabilities
            return self.result

    adapter = MutatingAdapter(semantic_input())
    _, ledger, ingress = service(tmp_path, adapter)
    delivery = digest("mutable-adapter")
    with pytest.raises(BehaviorAdapterError, match="changed during adaptation"):
        asyncio.run(ingress.ingest(adapter.name, {}, delivery_id=delivery))
    assert ledger.list_after_sequence(0, 10) == ()
    assert ledger.read_ingress_receipt(delivery) is None


def test_adapter_receives_digest_bound_canonical_payload_copy(tmp_path) -> None:
    class CapturingAdapter(FakeAdapter):
        received: object | None = None

        async def adapt(self, payload: object):
            self.received = payload
            assert isinstance(payload, dict)
            nested = payload["nested"]
            assert isinstance(nested, list)
            nested.append("adapter-local")
            return self.result

    adapter = CapturingAdapter(semantic_input())
    _, ledger, ingress = service(tmp_path, adapter)
    raw = {"nested": ["source"]}
    result = asyncio.run(ingress.ingest(adapter.name, raw, delivery_id=digest("canonical-copy")))

    assert result.receipt.status is IngressReceiptStatus.COMMITTED
    assert raw == {"nested": ["source"]}
    assert adapter.received == {"nested": ["source", "adapter-local"]}
    assert len(ledger.list_after_sequence(0, 10)) == 1


def test_source_and_stream_identity_conflicts_fail_without_receipt(tmp_path) -> None:
    adapter = FakeAdapter(semantic_input())
    _, ledger, ingress = service(tmp_path, adapter)
    asyncio.run(ingress.ingest(adapter.name, {}, delivery_id=digest("first")))
    adapter.result = semantic_input(activity="changed")
    with pytest.raises(BehaviorEvidenceConflictError):
        asyncio.run(ingress.ingest(adapter.name, {}, delivery_id=digest("second")))
    assert ledger.read_ingress_receipt(digest("second")) is None
    assert len(ledger.list_after_sequence(0, 10)) == 1


def test_different_producer_fingerprint_can_reinterpret_same_source(tmp_path) -> None:
    input_value = semantic_input()
    first = FakeAdapter(input_value, name="first")
    database = BehaviorDatabase(tmp_path / "behavior", config=BehaviorConfig(), initialize=True)
    ledger = SQLiteBehaviorEvidenceLedger(database)
    registry = BehaviorSemanticAdapterRegistry()
    registry.register(first)
    registry.register(FakeAdapter(input_value, name="second"))
    ingress = BehaviorEvidenceIngressService(ledger, registry, config=BehaviorConfig(), clock=FakeClock())
    asyncio.run(ingress.ingest("first", {}, delivery_id=digest("one")))
    asyncio.run(ingress.ingest("second", {}, delivery_id=digest("two")))
    assert len(ledger.list_after_sequence(0, 10)) == 2


def test_live_clock_rejection_backfill_history_and_legal_delay(tmp_path) -> None:
    old = semantic_input(event_time=BASE_TIME - timedelta(days=500))
    live = FakeAdapter(old, time_mode=BehaviorTimeMode.LIVE)
    _, live_ledger, live_ingress = service(tmp_path / "live", live)
    rejected = asyncio.run(live_ingress.ingest(live.name, {}, delivery_id=digest("old-live")))
    assert rejected.receipt.status is IngressReceiptStatus.REJECTED
    assert rejected.receipt.rejected_item_indexes == (0,)
    assert live_ledger.list_after_sequence(0, 10) == ()

    backfill = FakeAdapter(old, time_mode=BehaviorTimeMode.BACKFILL)
    _, _, backfill_ingress = service(tmp_path / "backfill", backfill)
    assert asyncio.run(
        backfill_ingress.ingest(backfill.name, {}, delivery_id=digest("old-backfill"))
    ).receipt.status is IngressReceiptStatus.COMMITTED

    delayed = FakeAdapter(semantic_input(event_time=BASE_TIME - timedelta(minutes=5)), time_mode=BehaviorTimeMode.LIVE)
    config = BehaviorConfig(evidence=replace(BehaviorConfig().evidence, max_live_event_age_seconds=3600.0))
    _, _, delayed_ingress = service(tmp_path / "delayed", delayed, config=config)
    assert asyncio.run(
        delayed_ingress.ingest(delayed.name, {}, delivery_id=digest("delayed"))
    ).receipt.status is IngressReceiptStatus.COMMITTED


def test_capacity_rejection_receipt_publishes_no_partial_records(tmp_path) -> None:
    config = BehaviorConfig(store=replace(BehaviorConfig().store, max_evidence_records=1))
    adapter = FakeAdapter(
        BehaviorSemanticInputBatch(
            (
                semantic_input(event="a", sequence=1),
                semantic_input(event="b", sequence=2),
            )
        )
    )
    _, ledger, ingress = service(tmp_path, adapter, config=config)
    result = asyncio.run(ingress.ingest(adapter.name, {}, delivery_id=digest("capacity")))
    assert result.receipt.status is IngressReceiptStatus.CAPACITY_REJECTED
    assert result.receipt.rejected_item_indexes == (0, 1)
    assert ledger.list_after_sequence(0, 10) == ()


def test_queries_are_bounded_paginated_and_cursors_are_query_bound(tmp_path) -> None:
    items = tuple(semantic_input(event=f"event-{index}", sequence=index) for index in range(1, 4))
    adapter = FakeAdapter(BehaviorSemanticInputBatch(items))
    _, ledger, ingress = service(tmp_path, adapter)
    asyncio.run(ingress.ingest(adapter.name, {}, delivery_id=digest("page")))
    page, cursor = ledger.list_by_record_kind(
        BehaviorRecordKind.ACTIVITY_SEGMENT,
        BASE_TIME - timedelta(seconds=1),
        BASE_TIME + timedelta(seconds=2),
        2,
    )
    assert len(page) == 2 and cursor is not None
    tail, final_cursor = ledger.list_by_record_kind(
        BehaviorRecordKind.ACTIVITY_SEGMENT,
        BASE_TIME - timedelta(seconds=1),
        BASE_TIME + timedelta(seconds=2),
        2,
        cursor,
    )
    assert len(tail) == 1 and final_cursor is None
    by_time, by_time_cursor = ledger.list_by_event_time(
        BASE_TIME,
        BASE_TIME + timedelta(seconds=1),
        2,
    )
    assert len(by_time) == 2 and by_time_cursor is not None
    with pytest.raises(ValueError, match="cursor"):
        ledger.list_by_event_time(BASE_TIME, BASE_TIME, 2, cursor)
    forged = decode_cursor(cursor)
    forged["sequence"] = 999_999
    with pytest.raises(ValueError, match="missing"):
        ledger.list_by_record_kind(
            BehaviorRecordKind.ACTIVITY_SEGMENT,
            BASE_TIME - timedelta(seconds=1),
            BASE_TIME + timedelta(seconds=2),
            2,
            encode_cursor(forged),
        )
    with pytest.raises(ValueError, match="limit"):
        ledger.list_after_sequence(0, 0)


def test_canonical_readback_tamper_schema_permissions_wal_and_legacy_rejection(tmp_path) -> None:
    adapter = FakeAdapter(semantic_input())
    database, ledger, ingress = service(tmp_path, adapter)
    result = asyncio.run(ingress.ingest(adapter.name, {}, delivery_id=digest("tamper")))
    assert os.stat(database.root).st_mode & 0o777 == 0o700
    assert os.stat(database.connection.path).st_mode & 0o777 == 0o600
    with database.connection.read() as connection:
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].casefold() == "wal"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    with database.connection.write() as connection:
        connection.execute(
            "UPDATE behavior_evidence_records SET content_json='{}' WHERE evidence_record_id=?",
            (result.records[0].evidence_record_id,),
        )
    with pytest.raises(BehaviorStoreError):
        ledger.read(result.records[0].evidence_record_id)

    legacy_root = tmp_path / "legacy"
    legacy_root.mkdir()
    (legacy_root / "evidence_claims.sqlite3").write_bytes(b"legacy")
    with pytest.raises(LegacyBehaviorStoreError, match="Memory and Conversation"):
        BehaviorDatabase(legacy_root, config=BehaviorConfig()).initialize()


def test_store_rejects_symlinks_and_detects_schema_or_index_drift(tmp_path) -> None:
    real_root = tmp_path / "real"
    real_root.mkdir()
    linked_root = tmp_path / "linked"
    linked_root.symlink_to(real_root, target_is_directory=True)
    with pytest.raises(BehaviorStoreError, match="symbolic"):
        BehaviorDatabase(linked_root, config=BehaviorConfig()).initialize()

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias-parent"
    alias_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(BehaviorStoreError, match="ancestors"):
        BehaviorDatabase(alias_parent / "nested", config=BehaviorConfig()).initialize()

    database_root = tmp_path / "database-link"
    database_root.mkdir()
    target = tmp_path / "target.sqlite3"
    target.write_bytes(b"")
    (database_root / "behavior.sqlite3").symlink_to(target)
    with pytest.raises(BehaviorStoreError, match="symbolic"):
        BehaviorDatabase(database_root, config=BehaviorConfig()).initialize()

    database = BehaviorDatabase(tmp_path / "drift", config=BehaviorConfig(), initialize=True)
    with database.connection.write() as connection:
        connection.execute("DROP INDEX idx_evidence_event_time")
    ready, detail = database.readiness()
    assert not ready and detail == "BehaviorStoreError"

    extra = BehaviorDatabase(tmp_path / "extra-drift", config=BehaviorConfig(), initialize=True)
    with extra.connection.write() as connection:
        connection.execute("CREATE TABLE unexpected_behavior_table(value TEXT)")
    assert extra.readiness() == (False, "BehaviorStoreError")

    extra_index = BehaviorDatabase(tmp_path / "extra-index", config=BehaviorConfig(), initialize=True)
    with extra_index.connection.write() as connection:
        connection.execute(
            "CREATE INDEX unexpected_behavior_index "
            "ON behavior_evidence_records(source_sequence)"
        )
    assert extra_index.readiness() == (False, "BehaviorStoreError")


def test_concurrent_first_initialization_is_idempotent(tmp_path) -> None:
    root = tmp_path / "concurrent-init"
    barrier = threading.Barrier(2)

    def initialize():
        barrier.wait(timeout=5)
        return BehaviorDatabase(root, config=BehaviorConfig()).initialize()

    with ThreadPoolExecutor(max_workers=2) as executor:
        paths = tuple(
            future.result()
            for future in (executor.submit(initialize), executor.submit(initialize))
        )
    assert paths[0] == paths[1] == root / "behavior.sqlite3"


def test_schema_and_wal_physical_capacity_fail_closed(tmp_path) -> None:
    tiny_store = replace(
        BehaviorConfig().store,
        max_json_bytes=32_768,
        max_database_bytes=32_768,
    )
    with pytest.raises(BehaviorStoreError, match="schema exceeds"):
        BehaviorDatabase(
            tmp_path / "tiny",
            config=BehaviorConfig(store=tiny_store),
            initialize=True,
        )

    bounded_store = replace(
        BehaviorConfig().store,
        max_json_bytes=100_000,
        max_database_bytes=500_000,
    )
    config = BehaviorConfig(store=bounded_store)
    adapter = FakeAdapter(semantic_input())
    database, _, ingress = service(tmp_path / "bounded", adapter, config=config)
    reader = database.connection.connect()
    reader.execute("BEGIN")
    reader.execute("SELECT COUNT(*) FROM behavior_evidence_records").fetchone()
    try:
        capacity_seen = False
        for index in range(1, 30):
            adapter.result = semantic_input(event=f"capacity-{index}", sequence=index)
            try:
                result = asyncio.run(
                    ingress.ingest(
                        adapter.name,
                        {"index": index},
                        delivery_id=digest(f"capacity-{index}"),
                    )
                )
                capacity_seen = result.receipt.status is IngressReceiptStatus.CAPACITY_REJECTED
            except BehaviorEvidenceCapacityError:
                capacity_seen = True
            assert database.connection.database_size() <= config.store.max_database_bytes
            if capacity_seen:
                break
        assert capacity_seen
    finally:
        reader.close()


def test_registry_rejects_duplicates_unknown_and_projection_capability() -> None:
    registry = BehaviorSemanticAdapterRegistry()
    adapter = FakeAdapter(semantic_input())
    registry.register(adapter)
    with pytest.raises(BehaviorAdapterError):
        registry.register(adapter)
    with pytest.raises(BehaviorAdapterError):
        registry.get("missing")
    projection = FakeAdapter(
        semantic_input(),
        origins=(BehaviorOriginKind.CONVERSATION_PROJECTION,),
        trust=BehaviorSourceTrust.MODEL_INFERRED,
    )
    with pytest.raises(BehaviorAdapterError):
        BehaviorSemanticAdapterRegistry().register(projection)
