"""不可变 Evidence Ledger 中的系统绑定记录。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from behavior._validation import positive_int, strict_utc, utc_text
from behavior.evidence.content import BehaviorSemanticContent, content_to_dict
from behavior.evidence.provenance import BehaviorSourceProvenance, provenance_to_dict
from behavior.evidence.trust import BehaviorSourceTrust
from foundation.integrity import canonical_digest

EVIDENCE_RECORD_SCHEMA_VERSION = "behavior_evidence_record_v1"


@dataclass(frozen=True)
class BehaviorEvidenceRecord:
    semantic_content: BehaviorSemanticContent
    provenance: BehaviorSourceProvenance
    source_trust: BehaviorSourceTrust
    ingested_at: datetime
    evidence_record_id: str = field(init=False)
    semantic_digest: str = field(init=False)
    content_digest: str = field(init=False)
    schema_version: str = field(init=False, default=EVIDENCE_RECORD_SCHEMA_VERSION)

    def __post_init__(self) -> None:
        if not isinstance(self.semantic_content, BehaviorSemanticContent):
            raise TypeError("semantic_content must be BehaviorSemanticContent")
        if not isinstance(self.provenance, BehaviorSourceProvenance):
            raise TypeError("provenance must be BehaviorSourceProvenance")
        trust = BehaviorSourceTrust(self.source_trust)
        ingested_at = strict_utc(self.ingested_at, "evidence_record.ingested_at")
        semantic_digest = canonical_digest(
            {
                "provenance": provenance_to_dict(self.provenance),
                "schema_version": EVIDENCE_RECORD_SCHEMA_VERSION,
                "semantic_content": content_to_dict(self.semantic_content),
                "source_trust": trust.value,
            }
        )
        record_id = "evidence_" + semantic_digest
        body = {
            "evidence_record_id": record_id,
            "ingested_at": utc_text(ingested_at),
            "provenance": provenance_to_dict(self.provenance),
            "schema_version": EVIDENCE_RECORD_SCHEMA_VERSION,
            "semantic_content": content_to_dict(self.semantic_content),
            "semantic_digest": semantic_digest,
            "source_trust": trust.value,
        }
        object.__setattr__(self, "source_trust", trust)
        object.__setattr__(self, "ingested_at", ingested_at)
        object.__setattr__(self, "semantic_digest", semantic_digest)
        object.__setattr__(self, "evidence_record_id", record_id)
        object.__setattr__(self, "content_digest", canonical_digest(body))


@dataclass(frozen=True)
class BehaviorEvidenceLedgerEntry:
    sequence: int
    record: BehaviorEvidenceRecord

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequence", positive_int(self.sequence, "evidence_sequence"))
        if not isinstance(self.record, BehaviorEvidenceRecord):
            raise TypeError("record must be BehaviorEvidenceRecord")


def record_to_dict(value: BehaviorEvidenceRecord) -> dict[str, Any]:
    return {
        "evidence_record_id": value.evidence_record_id,
        "semantic_content": content_to_dict(value.semantic_content),
        "provenance": provenance_to_dict(value.provenance),
        "source_trust": value.source_trust.value,
        "ingested_at": utc_text(value.ingested_at),
        "semantic_digest": value.semantic_digest,
        "content_digest": value.content_digest,
        "schema_version": value.schema_version,
    }


__all__ = ["BehaviorEvidenceLedgerEntry", "BehaviorEvidenceRecord", "record_to_dict"]
