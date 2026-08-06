"""Versioned conservative Claim confidence policy."""

from __future__ import annotations

from dataclasses import dataclass

from behavior._validation import finite_score
from behavior.claim.policy import ClaimDerivationClass
from foundation.integrity import canonical_digest

CLAIM_CONFIDENCE_POLICY_VERSION = "3"


@dataclass(frozen=True)
class ClaimConfidencePolicy:
    version: str = CLAIM_CONFIDENCE_POLICY_VERSION

    @property
    def digest(self) -> str:
        return canonical_digest(
            {
                "version": self.version,
                "deterministic": "source_confidence",
                "model": "min(source_confidence,normalizer_confidence)",
            }
        )

    def effective(
        self,
        *,
        source_confidence: float,
        normalizer_confidence: float,
        derivation_class: ClaimDerivationClass,
    ) -> float:
        source = finite_score(source_confidence, "source_confidence")
        normalized = finite_score(normalizer_confidence, "normalizer_confidence")
        derivation = ClaimDerivationClass(derivation_class)
        return source if derivation is ClaimDerivationClass.DETERMINISTIC else min(source, normalized)


__all__ = ["CLAIM_CONFIDENCE_POLICY_VERSION", "ClaimConfidencePolicy"]
