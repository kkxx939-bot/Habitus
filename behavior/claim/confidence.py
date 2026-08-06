"""Claim 有效置信度的版本化保守策略。"""

from __future__ import annotations

from dataclasses import dataclass

from behavior._validation import finite_score
from behavior.claim.normalizer import ClaimNormalizerKind

CLAIM_CONFIDENCE_POLICY_VERSION = "1"


@dataclass(frozen=True)
class ClaimConfidencePolicy:
    version: str = CLAIM_CONFIDENCE_POLICY_VERSION

    def effective(
        self,
        *,
        source_confidence: float,
        normalizer_confidence: float,
        normalizer_kind: ClaimNormalizerKind,
    ) -> float:
        source = finite_score(source_confidence, "source_confidence")
        normalized = finite_score(normalizer_confidence, "normalizer_confidence")
        kind = ClaimNormalizerKind(normalizer_kind)
        return source if kind is ClaimNormalizerKind.DETERMINISTIC else min(source, normalized)


__all__ = ["CLAIM_CONFIDENCE_POLICY_VERSION", "ClaimConfidencePolicy"]
