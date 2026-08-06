"""Claim 规范化、系统绑定、准入与发布公共边界。"""

from behavior.claim.admission import (
    ClaimAdmissionDecision,
    ClaimAdmissionPolicy,
    ClaimAdmissionStatus,
    StaticAdmissionResult,
)
from behavior.claim.binder import ClaimBinder
from behavior.claim.confidence import ClaimConfidencePolicy
from behavior.claim.model import (
    Claim,
    ClaimBatch,
    ClaimNormalizerRun,
    ClaimNormalizerRunStatus,
    ClaimProcessingReceipt,
    EpistemicClass,
)
from behavior.claim.normalizer import (
    ClaimNormalizer,
    ClaimNormalizerKind,
    DeterministicClaimNormalizer,
    ModelClaimNormalizer,
    NormalizerFingerprint,
)
from behavior.claim.proposal import ClaimKind, ClaimSemanticProposal, ClaimSemanticProposalBatch
from behavior.claim.registry import ClaimNormalizerRegistry
from behavior.claim.router import ClaimNormalizationRouter
from behavior.claim.service import (
    ClaimPipelineService,
    ClaimProcessingResult,
    SemanticPipelineIngestResult,
)

__all__ = [
    "Claim",
    "ClaimAdmissionDecision",
    "ClaimAdmissionPolicy",
    "ClaimAdmissionStatus",
    "ClaimBatch",
    "ClaimBinder",
    "ClaimConfidencePolicy",
    "ClaimKind",
    "ClaimNormalizationRouter",
    "ClaimNormalizer",
    "ClaimNormalizerKind",
    "ClaimNormalizerRegistry",
    "ClaimNormalizerRun",
    "ClaimNormalizerRunStatus",
    "ClaimPipelineService",
    "ClaimProcessingReceipt",
    "ClaimProcessingResult",
    "ClaimSemanticProposal",
    "ClaimSemanticProposalBatch",
    "DeterministicClaimNormalizer",
    "EpistemicClass",
    "ModelClaimNormalizer",
    "NormalizerFingerprint",
    "SemanticPipelineIngestResult",
    "StaticAdmissionResult",
]
