"""语义 Evidence Bundle 与不可变 Manifest 公共边界。"""

from behavior.evidence.bundle import (
    EvidenceBundleState,
    EvidenceSealReason,
    SemanticEvidenceBundle,
    SemanticEvidenceBundleAssembler,
    SemanticIngestResult,
    SemanticIngestStatus,
)
from behavior.evidence.manifest import (
    CoverageInterval,
    CoverageSummary,
    EvidenceManifest,
    ManifestSemanticRecordSnapshot,
)

__all__ = [
    "CoverageInterval",
    "CoverageSummary",
    "EvidenceBundleState",
    "EvidenceManifest",
    "EvidenceSealReason",
    "ManifestSemanticRecordSnapshot",
    "SemanticEvidenceBundle",
    "SemanticEvidenceBundleAssembler",
    "SemanticIngestResult",
    "SemanticIngestStatus",
]
