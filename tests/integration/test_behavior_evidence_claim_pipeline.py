from __future__ import annotations

import asyncio

from behavior.claim import ClaimKind
from behavior.config import BehaviorConfig
from behavior.evidence import BehaviorSemanticInputBatch
from behavior.persistence import SQLiteBehaviorClaimLedger, SQLiteBehaviorEvidenceLedger
from tests.unit.behavior.conftest import FakeAdapter, digest
from tests.unit.behavior.test_claim_normalization import normalization_service
from tests.unit.behavior.test_evidence_ingress_ledger import semantic_input, service


def test_structured_semantics_flow_through_two_append_only_ledgers_without_cross_evidence_fusion(
    tmp_path,
) -> None:
    config = BehaviorConfig()
    adapter = FakeAdapter(
        BehaviorSemanticInputBatch(
            (
                semantic_input(event="first", sequence=1, activity="walking"),
                semantic_input(event="second", sequence=2, activity="walking"),
            )
        )
    )
    database, evidence_ledger, ingress = service(tmp_path, adapter, config=config)
    ingested = asyncio.run(ingress.ingest(adapter.name, {}, delivery_id=digest("integration")))
    assert [entry.sequence for entry in evidence_ledger.list_after_sequence(0, 10)] == [1, 2]
    claim_ledger, normalization = normalization_service(
        tmp_path,
        database,
        evidence_ledger,
        config,
    )
    results = [
        asyncio.run(normalization.normalize(record.evidence_record_id))
        for record in ingested.records
    ]
    claims = claim_ledger.list_after_sequence(0, 10)
    assert [entry.sequence for entry in claims] == [1, 2]
    assert all(entry.claim.claim_kind is ClaimKind.ACTIVITY for entry in claims)
    assert claims[0].claim.semantic_fingerprint == claims[1].claim.semantic_fingerprint
    assert claims[0].claim.claim_id != claims[1].claim.claim_id
    assert all(len(result.core_receipt.claim_ids) == 1 for result in results)
    assert SQLiteBehaviorEvidenceLedger(database).read(ingested.records[0].evidence_record_id)
    assert SQLiteBehaviorClaimLedger(database).read_claim(claims[0].claim.claim_id)
