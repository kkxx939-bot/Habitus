"""ClaimProposal 生成、验证、准入与发布公共边界。"""

from behavior.claim.admission import ClaimAdmissionDecision, ClaimAdmissionGate, ClaimAdmissionStatus
from behavior.claim.model import (
    Claim,
    ClaimBatch,
    ClaimProcessingReceipt,
    ClaimProducerRun,
    ClaimProducerRunStatus,
)
from behavior.claim.producer import (
    ClaimProducer,
    DirectStructuredClaimProducer,
    ProducerFingerprint,
    StructuredSemanticClaimProducer,
)
from behavior.claim.proposal import (
    ActorRole,
    ClaimKind,
    ClaimProposal,
    ClaimProposalBatch,
    EpistemicClass,
    SubjectRole,
)
from behavior.claim.registry import ClaimProducerRegistry
from behavior.claim.service import ClaimPipelineService, ClaimProcessingResult
from behavior.claim.validator import ClaimValidator

__all__ = [
    "ActorRole",
    "Claim",
    "ClaimAdmissionDecision",
    "ClaimAdmissionGate",
    "ClaimAdmissionStatus",
    "ClaimBatch",
    "ClaimKind",
    "ClaimPipelineService",
    "ClaimProcessingReceipt",
    "ClaimProcessingResult",
    "ClaimProducer",
    "ClaimProducerRegistry",
    "ClaimProducerRun",
    "ClaimProducerRunStatus",
    "ClaimProposal",
    "ClaimProposalBatch",
    "ClaimValidator",
    "DirectStructuredClaimProducer",
    "EpistemicClass",
    "ProducerFingerprint",
    "StructuredSemanticClaimProducer",
    "SubjectRole",
]
