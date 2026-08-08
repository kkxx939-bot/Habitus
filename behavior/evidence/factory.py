"""通过已验证输入创建不可变 EvidenceRecord。"""

from __future__ import annotations

from datetime import datetime

from behavior._validation import strict_utc, utc_text
from behavior.errors import BehaviorStoreError
from behavior.evidence.content import BehaviorSemanticContent, content_to_dict
from behavior.evidence.policy import ValidatedEvidenceInput
from behavior.evidence.provenance import (
    BehaviorSourceProvenance,
    ProducerFingerprint,
    provenance_to_dict,
)
from behavior.evidence.record import (
    EVIDENCE_RECORD_SCHEMA_VERSION,
    BehaviorEvidenceRecord,
)
from behavior.evidence.trust import BehaviorSourceTrust
from foundation.integrity import canonical_digest


class EvidenceFactory:
    def create_batch(
        self,
        inputs: tuple[ValidatedEvidenceInput, ...],
        *,
        adapter_name: str,
        producer: ProducerFingerprint,
        capability_digest: str,
        ingested_at: datetime,
    ) -> tuple[BehaviorEvidenceRecord, ...]:
        moment = strict_utc(ingested_at, "ingested_at")
        return tuple(
            self._create(
                item,
                adapter_name=adapter_name,
                producer=producer,
                capability_digest=capability_digest,
                ingested_at=moment,
            )
            for item in inputs
        )

    @staticmethod
    def _create(
        item: ValidatedEvidenceInput,
        *,
        adapter_name: str,
        producer: ProducerFingerprint,
        capability_digest: str,
        ingested_at: datetime,
    ) -> BehaviorEvidenceRecord:
        provenance = BehaviorSourceProvenance(
            descriptor=item.value.source,
            adapter_name=adapter_name,
            producer_fingerprint=producer,
            capability_digest=capability_digest,
        )
        trust = item.output_contract.source_trust
        return EvidenceFactory._build(
            content=item.value.content,
            provenance=provenance,
            trust=trust,
            ingested_at=ingested_at,
        )

    @staticmethod
    def restore(
        *,
        content: BehaviorSemanticContent,
        provenance: BehaviorSourceProvenance,
        trust: BehaviorSourceTrust,
        ingested_at: datetime,
        evidence_record_id: str,
        semantic_digest: str,
        content_digest: str,
        schema_version: str,
    ) -> BehaviorEvidenceRecord:
        if schema_version != EVIDENCE_RECORD_SCHEMA_VERSION:
            raise BehaviorStoreError("Evidence record schema is incompatible")
        restored = EvidenceFactory._build(
            content=content,
            provenance=provenance,
            trust=BehaviorSourceTrust(trust),
            ingested_at=strict_utc(ingested_at, "ingested_at"),
        )
        if (
            restored.evidence_record_id != evidence_record_id
            or restored.semantic_digest != semantic_digest
            or restored.content_digest != content_digest
        ):
            raise BehaviorStoreError("Evidence durable identity or digest has drifted")
        return restored

    @staticmethod
    def _build(
        *,
        content: BehaviorSemanticContent,
        provenance: BehaviorSourceProvenance,
        trust: BehaviorSourceTrust,
        ingested_at: datetime,
    ) -> BehaviorEvidenceRecord:
        semantic_body = {
            "provenance": provenance_to_dict(provenance),
            "schema_version": EVIDENCE_RECORD_SCHEMA_VERSION,
            "semantic_content": content_to_dict(content),
            "source_trust": trust.value,
        }
        semantic_digest = canonical_digest(semantic_body)
        evidence_record_id = "evidence_" + semantic_digest
        content_body = {
            "evidence_record_id": evidence_record_id,
            "ingested_at": utc_text(ingested_at),
            **semantic_body,
            "semantic_digest": semantic_digest,
        }
        return BehaviorEvidenceRecord(
            semantic_content=content,
            provenance=provenance,
            source_trust=trust,
            ingested_at=ingested_at,
            evidence_record_id=evidence_record_id,
            semantic_digest=semantic_digest,
            content_digest=canonical_digest(content_body),
        )
