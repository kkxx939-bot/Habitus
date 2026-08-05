from __future__ import annotations

import pytest

from behavior.errors import (
    BehaviorOwnerConflictError,
    BehaviorOwnerError,
    SourceRecordConflictError,
    SourceRecordError,
)
from behavior.owner import ConfirmedOwnerBinding, OwnerRouteDecision, OwnerRouteStatus
from behavior.source import (
    BehaviorSourceAdapterRegistry,
    Modality,
    SourceRecord,
    SourceRecordBatch,
    SourceRecordService,
    SourceType,
)
from tests.unit.behavior.conftest import BASE_TIME, digest, make_source


@pytest.mark.parametrize("status", [OwnerRouteStatus.OTHER_PERSON, OwnerRouteStatus.UNRESOLVED])
def test_only_confirmed_owner_route_can_enter_behavior(status: OwnerRouteStatus) -> None:
    decision = OwnerRouteDecision(status, None, "router-v1", BASE_TIME, digest(status.value))
    with pytest.raises(BehaviorOwnerError):
        decision.confirm()

    confirmed = OwnerRouteDecision(
        OwnerRouteStatus.OWNER_CONFIRMED,
        "local-binding",
        "router-v1",
        BASE_TIME,
        digest("confirmed"),
    ).confirm()
    assert isinstance(confirmed, ConfirmedOwnerBinding)
    assert confirmed.binding_digest == ConfirmedOwnerBinding.from_dict(confirmed.to_dict()).binding_digest


def test_store_rejects_a_different_owner_binding(store, owner) -> None:
    first = make_source(owner)
    store.ingest_source(first, __import__("behavior").EvidenceWindowAssembler(store.config.evidence))
    other = ConfirmedOwnerBinding("other-local-binding", "router-v1", BASE_TIME, digest("other"))
    with pytest.raises(BehaviorOwnerConflictError):
        store.ingest_source(
            make_source(other, sequence=1),
            __import__("behavior").EvidenceWindowAssembler(store.config.evidence),
        )


@pytest.mark.parametrize("source_type", tuple(SourceType))
def test_every_source_type_is_a_valid_raw_source(source_type: SourceType, owner) -> None:
    record = make_source(
        owner,
        source_type=source_type,
        modality=Modality.TEXT,
        semantic_data={},
    )
    assert record.source_type is source_type


@pytest.mark.parametrize("modality", tuple(Modality))
def test_every_modality_is_supported(modality: Modality, owner) -> None:
    assert make_source(owner, modality=modality).modality is modality


def test_source_record_time_digest_json_and_media_boundaries(owner) -> None:
    record = make_source(owner)
    assert record.event_time_start.tzinfo is not None
    assert SourceRecord.from_dict(record.to_dict()).source_record_id == record.source_record_id
    with pytest.raises(SourceRecordError, match="timezone-aware"):
        SourceRecord.from_dict({**record.to_dict(), "event_time_start": "2026-08-05T01:02:03"})
    with pytest.raises(SourceRecordError, match="earlier"):
        make_source(owner, duration_seconds=-1)
    with pytest.raises(SourceRecordError, match="SHA-256"):
        SourceRecord.from_dict({**record.to_dict(), "payload_digest": "ABC"})
    with pytest.raises(SourceRecordError, match="non-finite"):
        make_source(owner, semantic_data={"score": float("nan")})
    with pytest.raises(SourceRecordError, match="unsupported type"):
        make_source(owner, semantic_data={"object": object()})
    with pytest.raises(SourceRecordError, match="recursive"):
        recursive = {}
        recursive["self"] = recursive
        make_source(owner, semantic_data=recursive)
    with pytest.raises(SourceRecordError, match="base64"):
        data = record.to_dict()
        data["payload_ref"] = "data:image/png;base64,AAAA"
        SourceRecord.from_dict(data)
    with pytest.raises(SourceRecordError, match="URI scheme"):
        data = record.to_dict()
        data["payload_ref"] = "not-an-external-reference"
        SourceRecord.from_dict(data)
    with pytest.raises(SourceRecordError, match="base64 media"):
        make_source(owner, semantic_data={"projection": "data:audio/wav;base64,AAAA"})
    with pytest.raises(SourceRecordError, match="binary"):
        make_source(owner, semantic_data={"raw": b"media"})


def test_source_capacity_identity_replay_and_conflict(store, owner) -> None:
    with pytest.raises(SourceRecordError, match="boundary"):
        make_source(owner, semantic_text="x" * 16_385)

    first = make_source(owner)
    same = SourceRecord.from_dict(first.to_dict())
    assembler = __import__("behavior").EvidenceWindowAssembler(store.config.evidence)
    accepted = store.ingest_source(first, assembler)
    replayed = store.ingest_source(same, assembler)
    assert accepted.source_record_id == replayed.source_record_id == first.source_record_id
    assert replayed.status.value == "REPLAYED"

    conflict_data = first.to_dict()
    conflict_data["modality"] = Modality.TEXT.value
    conflict = SourceRecord.from_dict(conflict_data)
    assert conflict.source_record_id == first.source_record_id
    with pytest.raises(SourceRecordConflictError):
        store.ingest_source(conflict, assembler)


class FakeAdapter:
    name = "Fake-Adapter"

    def __init__(self, record: SourceRecord, result: object | None = None) -> None:
        self.record = record
        self.result = result

    async def adapt(self, payload: object, *, owner_binding: ConfirmedOwnerBinding):
        assert payload == {"external": True}
        assert owner_binding is self.record.owner_binding
        return self.record if self.result is None else self.result


def test_source_adapter_registry_and_return_contract(store, owner) -> None:
    import asyncio

    record = make_source(owner)
    registry = BehaviorSourceAdapterRegistry()
    registry.register(FakeAdapter(record))
    assert registry.names() == ("fake_adapter",)
    with pytest.raises(SourceRecordError, match="already"):
        registry.register(FakeAdapter(record))
    with pytest.raises(SourceRecordError, match="unknown"):
        registry.get("missing")
    service = SourceRecordService(store, registry)
    batch = asyncio.run(service.adapt("fake-adapter", {"external": True}, owner_binding=owner))
    assert batch == SourceRecordBatch((record,))

    bad_registry = BehaviorSourceAdapterRegistry()
    bad_registry.register(FakeAdapter(record, result={"not": "a SourceRecord"}))
    bad_service = SourceRecordService(store, bad_registry)
    with pytest.raises(TypeError, match="unsupported"):
        asyncio.run(bad_service.adapt("fake-adapter", {"external": True}, owner_binding=owner))
