"""不可变 Evidence Ledger 中的系统绑定记录。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from behavior._validation import positive_int, utc_text
from behavior.evidence.content import BehaviorSemanticContent, content_to_dict
from behavior.evidence.provenance import BehaviorSourceProvenance, provenance_to_dict
from behavior.evidence.trust import BehaviorSourceTrust

EVIDENCE_RECORD_SCHEMA_VERSION = "behavior_evidence_record_v1"


@dataclass(frozen=True, slots=True)
class BehaviorEvidenceRecord:
    semantic_content: BehaviorSemanticContent
    provenance: BehaviorSourceProvenance
    source_trust: BehaviorSourceTrust
    ingested_at: datetime
    evidence_record_id: str
    semantic_digest: str
    content_digest: str
    schema_version: str = EVIDENCE_RECORD_SCHEMA_VERSION


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
