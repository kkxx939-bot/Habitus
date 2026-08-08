from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from behavior._validation import decode_cursor, encode_cursor
from behavior.claim import (
    ClaimKind,
    ClaimNormalizationAttempt,
    ClaimNormalizationReceipt,
    ReceiptStatus,
)
from behavior.claim.proposal import proposal_to_dict
from behavior.claim.publication import ClaimPublication
from behavior.claim.receipt import AttemptStatus
from behavior.config import BehaviorConfig
from behavior.errors import (
    BehaviorClaimCapacityError,
    BehaviorClaimConflictError,
    BehaviorStoreError,
    ClaimNormalizationConflictError,
)
from behavior.persistence import BehaviorAuditService, BehaviorDatabase
from foundation.integrity import canonical_digest
from tests.unit.behavior.conftest import BASE_TIME, FakeModelNormalizer
from tests.unit.behavior.test_claim_normalization import (
    create_claim,
    free_text_input,
    ingest_record,
    normalization_service,
    state_proposal,
)


def test_low_confidence_and_same_semantic_from_independent_evidence_are_all_saved(tmp_path) -> None:
    config = BehaviorConfig()
    database = BehaviorDatabase(tmp_path / "behavior", config=config, initialize=True)
    evidence_ledger, first = asyncio.run(
        ingest_record(database, free_text_input(), config=config, name="first")
    )
    second_input = free_text_input()
    second_input = replace(
        second_input,
        source=replace(
            second_input.source,
            source_event_ref=type(second_input.source.source_event_ref)("test", "second"),
            source_sequence=2,
        ),
    )
    _, second = asyncio.run(ingest_record(database, second_input, config=config, name="second"))
    model = FakeModelNormalizer((state_proposal(confidence=0.0),))
    claim_ledger, service = normalization_service(tmp_path, database, evidence_ledger, config, model)
    first_result = asyncio.run(service.normalize(first.evidence_record_id))
    second_result = asyncio.run(service.normalize(second.evidence_record_id))
    first_claim = claim_ledger.read_claim(first_result.enhancement_receipts[0].claim_ids[0])
    second_claim = claim_ledger.read_claim(second_result.enhancement_receipts[0].claim_ids[0])
    assert first_claim is not None and second_claim is not None
    assert first_claim.effective_confidence == second_claim.effective_confidence == 0.0
    assert first_claim.claim_id != second_claim.claim_id
    assert first_claim.semantic_fingerprint == second_claim.semantic_fingerprint
    matches, _ = claim_ledger.list_by_semantic_fingerprint(first_claim.semantic_fingerprint, 10)
    assert len(matches) == 2


def test_claim_sequence_evidence_pagination_and_exact_replay(tmp_path) -> None:
    config = BehaviorConfig()
    database = BehaviorDatabase(tmp_path / "behavior", config=config, initialize=True)
    evidence_ledger, record = asyncio.run(ingest_record(database, free_text_input(), config=config))
    proposals = (
        state_proposal(),
        replace(
            state_proposal(),
            predicate="awake",
            semantic_payload={"value": True},
            local_alternative_group_id="choice",
        ),
    )
    model = FakeModelNormalizer(proposals)
    claim_ledger, service = normalization_service(tmp_path, database, evidence_ledger, config, model)
    first = asyncio.run(service.normalize(record.evidence_record_id))
    replay = asyncio.run(service.normalize(record.evidence_record_id))
    assert first.enhancement_receipts == replay.enhancement_receipts
    entries = claim_ledger.list_after_sequence(0, 10)
    assert [entry.sequence for entry in entries] == [1, 2]
    page, cursor = claim_ledger.list_for_evidence(record.evidence_record_id, 1)
    assert len(page) == 1 and cursor is not None
    tail, final = claim_ledger.list_for_evidence(record.evidence_record_id, 1, cursor)
    assert len(tail) == 1 and final is None
    by_time, time_cursor = claim_ledger.list_by_event_time(
        record.semantic_content.event_time_start,
        record.semantic_content.event_time_end,
        1,
    )
    assert len(by_time) == 1 and time_cursor is not None
    by_time_tail, final_time_cursor = claim_ledger.list_by_event_time(
        record.semantic_content.event_time_start,
        record.semantic_content.event_time_end,
        1,
        time_cursor,
    )
    assert len(by_time_tail) == 1 and final_time_cursor is None
    with pytest.raises(ValueError, match="cursor"):
        claim_ledger.list_for_evidence(record.evidence_record_id, 1, time_cursor)
    forged = decode_cursor(cursor)
    forged["sequence"] = 999_999
    with pytest.raises(ValueError, match="missing"):
        claim_ledger.list_for_evidence(
            record.evidence_record_id,
            1,
            encode_cursor(forged),
        )


def test_attempt_claim_receipt_publication_rolls_back_as_one_transaction(tmp_path) -> None:
    config = BehaviorConfig(store=replace(BehaviorConfig().store, max_claims=1))
    database = BehaviorDatabase(tmp_path / "behavior", config=config, initialize=True)
    evidence_ledger, record = asyncio.run(ingest_record(database, free_text_input(), config=config))
    model = FakeModelNormalizer(
        (
            state_proposal(),
            replace(state_proposal(), predicate="awake", semantic_payload={"value": True}),
        )
    )
    claim_ledger, service = normalization_service(tmp_path, database, evidence_ledger, config, model)
    with pytest.raises(BehaviorClaimCapacityError):
        asyncio.run(service.normalize(record.evidence_record_id))
    assert claim_ledger.list_after_sequence(0, 10) == ()
    plan = service.planner.plan(record)
    identity = service.route_executor.identity(record, plan, plan.enhancement_routes[0]).value
    assert claim_ledger.read_latest_attempt(identity) is None
    assert claim_ledger.read_receipt(identity) is None


def test_claim_capacity_counts_only_genuinely_new_claim_rows(tmp_path) -> None:
    config = BehaviorConfig(store=replace(BehaviorConfig().store, max_claims=1))
    database = BehaviorDatabase(tmp_path / "behavior", config=config, initialize=True)
    evidence_ledger, record = asyncio.run(
        ingest_record(database, free_text_input(), config=config)
    )
    model = FakeModelNormalizer((state_proposal(),))
    claim_ledger, service = normalization_service(
        tmp_path,
        database,
        evidence_ledger,
        config,
        model,
    )
    first = asyncio.run(service.normalize(record.evidence_record_id))
    claim = claim_ledger.read_claim(first.enhancement_receipts[0].claim_ids[0])
    assert claim is not None
    route = service.planner.plan(record).enhancement_routes[0]
    changed_plan = replace(
        service.planner.plan(record),
        planner_policy_digest=canonical_digest({"planner": "changed"}),
    )
    publication = service.route_executor.publication_factory.success(
        record,
        changed_plan,
        route,
        attempt_number=1,
        started_at=BASE_TIME,
        completed_at=BASE_TIME,
        publication_recorded_at=BASE_TIME,
        proposals=(state_proposal(),),
        claims=(claim,),
    )
    stored, reused = claim_ledger.publish(publication)
    assert not reused
    assert stored.claim_ids == (claim.claim_id,)
    assert len(claim_ledger.list_after_sequence(0, 10)) == 1


def test_normal_read_skips_index_projection_and_deep_audit_detects_tamper(tmp_path) -> None:
    config = BehaviorConfig()
    database = BehaviorDatabase(tmp_path / "behavior", config=config, initialize=True)
    evidence_ledger, record = asyncio.run(ingest_record(database, free_text_input(), config=config))
    model = FakeModelNormalizer((state_proposal(),))
    claim_ledger, service = normalization_service(tmp_path, database, evidence_ledger, config, model)
    result = asyncio.run(service.normalize(record.evidence_record_id))
    claim_id = result.enhancement_receipts[0].claim_ids[0]
    with database.connection.write() as connection:
        connection.execute(
            "UPDATE behavior_claims SET semantic_fingerprint=? WHERE claim_id=?",
            ("0" * 64, claim_id),
        )
    assert claim_ledger.read_claim(claim_id) is not None
    with pytest.raises(BehaviorStoreError):
        BehaviorAuditService(database).deep_check()


def test_deep_audit_detects_tampered_claim_member(tmp_path) -> None:
    config = BehaviorConfig()
    database = BehaviorDatabase(tmp_path / "behavior", config=config, initialize=True)
    evidence_ledger, record = asyncio.run(ingest_record(database, free_text_input(), config=config))
    model = FakeModelNormalizer((state_proposal(),))
    claim_ledger, service = normalization_service(tmp_path, database, evidence_ledger, config, model)
    result = asyncio.run(service.normalize(record.evidence_record_id))
    claim_id = result.enhancement_receipts[0].claim_ids[0]
    with database.connection.write() as connection:
        connection.execute(
            "UPDATE behavior_claims SET content_json='{}' WHERE claim_id=?",
            (claim_id,),
        )
    replay = asyncio.run(service.normalize(record.evidence_record_id))
    assert replay.enhancement_receipts
    with pytest.raises(BehaviorStoreError):
        BehaviorAuditService(database).deep_check()


def test_deep_audit_detects_tampered_attempt_member(tmp_path) -> None:
    config = BehaviorConfig()
    database = BehaviorDatabase(tmp_path / "behavior", config=config, initialize=True)
    evidence_ledger, record = asyncio.run(ingest_record(database, free_text_input(), config=config))
    model = FakeModelNormalizer((state_proposal(),))
    claim_ledger, service = normalization_service(tmp_path, database, evidence_ledger, config, model)
    result = asyncio.run(service.normalize(record.evidence_record_id))
    attempt_id = result.enhancement_receipts[0].attempt_ids[0]
    with database.connection.write() as connection:
        connection.execute(
            "UPDATE claim_normalization_attempts SET content_json='{}' WHERE attempt_id=?",
            (attempt_id,),
        )
    replay = asyncio.run(service.normalize(record.evidence_record_id))
    assert replay.enhancement_receipts
    with pytest.raises(BehaviorStoreError):
        BehaviorAuditService(database).deep_check()


def test_receipt_member_table_is_checked_only_by_deep_audit(tmp_path) -> None:
    config = BehaviorConfig()
    database = BehaviorDatabase(tmp_path / "behavior", config=config, initialize=True)
    evidence_ledger, record = asyncio.run(
        ingest_record(database, free_text_input(), config=config)
    )
    model = FakeModelNormalizer((state_proposal(),))
    claim_ledger, service = normalization_service(
        tmp_path,
        database,
        evidence_ledger,
        config,
        model,
    )
    result = asyncio.run(service.normalize(record.evidence_record_id))
    receipt = result.enhancement_receipts[0]
    with database.connection.write() as connection:
        connection.execute(
            "DELETE FROM claim_receipt_members WHERE processing_identity=? "
            "AND member_kind='CLAIM'",
            (receipt.processing_identity,),
        )
    assert claim_ledger.read_receipt(receipt.processing_identity) == receipt
    with pytest.raises(BehaviorStoreError, match="Claim members"):
        BehaviorAuditService(database).deep_check()


def test_claim_publication_closes_policy_ownership(tmp_path) -> None:
    config = BehaviorConfig()
    database = BehaviorDatabase(tmp_path / "behavior", config=config, initialize=True)
    evidence_ledger, record = asyncio.run(ingest_record(database, free_text_input(), config=config))
    model = FakeModelNormalizer((state_proposal(),))
    claim_ledger, service = normalization_service(tmp_path, database, evidence_ledger, config, model)
    plan = service.planner.plan(record)
    route = plan.enhancement_routes[0]
    identity = service.route_executor.identity(record, plan, route).value
    proposal = state_proposal()
    claim = service.route_executor.claim_factory.create(record, proposal, route, created_at=BASE_TIME)
    tampered = replace(claim, compatibility_policy_digest="0" * 64)
    attempt = ClaimNormalizationAttempt(
        processing_identity=identity,
        evidence_record_id=record.evidence_record_id,
        normalizer_name=route.normalizer_name,
        normalizer_fingerprint=route.normalizer_fingerprint,
        lane=route.lane,
        attempt_number=1,
        status=AttemptStatus.COMPLETED,
        proposal_digest=canonical_digest([proposal_to_dict(proposal)]),
        claim_count=1,
        error_code=None,
        retryable=False,
        started_at=BASE_TIME,
        completed_at=BASE_TIME,
    )
    receipt = ClaimNormalizationReceipt(
        processing_identity=identity,
        evidence_record_id=record.evidence_record_id,
        lane=route.lane,
        normalizer_fingerprint=route.normalizer_fingerprint,
        planner_policy_digest=plan.planner_policy_digest,
        compatibility_policy_digest=service.route_executor.claim_factory.compatibility.digest,
        binding_policy_digest=service.route_executor.claim_factory.binding.digest,
        confidence_policy_digest=service.route_executor.claim_factory.confidence.digest,
        status=ReceiptStatus.COMPLETED,
        attempt_ids=(attempt.attempt_id,),
        claim_ids=(tampered.claim_id,),
        completed_at=BASE_TIME,
        publication_recorded_at=BASE_TIME,
    )
    with pytest.raises(ClaimNormalizationConflictError, match="ownership"):
        ClaimPublication(
            service.route_executor.identity(record, plan, route),
            attempt,
            (tampered,),
            receipt,
        )
    assert claim_ledger.list_after_sequence(0, 10) == ()


def test_processing_receipt_replay_still_validates_full_claim_content(tmp_path) -> None:
    config = BehaviorConfig()
    database = BehaviorDatabase(tmp_path / "behavior", config=config, initialize=True)
    evidence_ledger, record = asyncio.run(ingest_record(database, free_text_input(), config=config))
    model = FakeModelNormalizer((state_proposal(),))
    claim_ledger, service = normalization_service(
        tmp_path,
        database,
        evidence_ledger,
        config,
        model,
    )
    result = asyncio.run(service.normalize(record.evidence_record_id))
    receipt = result.enhancement_receipts[0]
    attempt = claim_ledger.read_attempt(receipt.attempt_ids[0])
    claim = claim_ledger.read_claim(receipt.claim_ids[0])
    assert attempt is not None and claim is not None
    conflicting_claim = create_claim(
        record,
        state_proposal(summary="changed-summary"),
        model,
    )
    assert conflicting_claim.claim_id == claim.claim_id
    assert conflicting_claim.content_digest != claim.content_digest
    plan = service.planner.plan(record)
    route = plan.enhancement_routes[0]
    publication = ClaimPublication(
        service.route_executor.identity(record, plan, route),
        attempt,
        (conflicting_claim,),
        receipt,
    )
    with pytest.raises(BehaviorClaimConflictError):
        claim_ledger.publish(publication)


def test_claim_kind_is_not_filtered_by_a_score_gate() -> None:
    proposal = state_proposal(confidence=0.0, kind=ClaimKind.STATE_ASSERTION)
    assert proposal.normalizer_confidence == 0.0


def test_claim_wal_physical_capacity_fails_closed(tmp_path) -> None:
    store = replace(
        BehaviorConfig().store,
        max_json_bytes=100_000,
        max_database_bytes=2_000_000,
    )
    config = BehaviorConfig(store=store)
    database = BehaviorDatabase(tmp_path / "behavior", config=config, initialize=True)
    records = []
    evidence_ledger = None
    for index in range(64):
        value = free_text_input()
        value = replace(
            value,
            source=replace(
                value.source,
                source_event_ref=type(value.source.source_event_ref)(
                    "test", f"claim-capacity-{index}"
                ),
                source_sequence=index + 1,
            ),
        )
        evidence_ledger, record = asyncio.run(
            ingest_record(database, value, config=config, name=f"capacity-adapter-{index}")
        )
        records.append(record)
    assert evidence_ledger is not None
    model = FakeModelNormalizer((state_proposal(),))
    _, service = normalization_service(
        tmp_path,
        database,
        evidence_ledger,
        config,
        model,
    )
    reader = database.connection.connect()
    reader.execute("BEGIN")
    reader.execute("SELECT COUNT(*) FROM behavior_claims").fetchone()
    try:
        capacity_seen = False
        for record in records:
            try:
                asyncio.run(service.normalize(record.evidence_record_id))
            except BehaviorClaimCapacityError:
                capacity_seen = True
            assert database.connection.database_size() <= config.store.max_database_bytes
            if capacity_seen:
                break
        assert capacity_seen
    finally:
        reader.close()
