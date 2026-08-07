"""不设置阈值的 Claim 置信度组合策略。"""

from __future__ import annotations

from dataclasses import dataclass, field

from behavior._validation import finite_score
from foundation.integrity import canonical_digest

CLAIM_CONFIDENCE_POLICY_VERSION = "claim_confidence_v1"


@dataclass(frozen=True)
class ClaimConfidencePolicy:
    version: str = CLAIM_CONFIDENCE_POLICY_VERSION
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.version != CLAIM_CONFIDENCE_POLICY_VERSION:
            raise ValueError("unsupported Claim confidence policy version")
        object.__setattr__(
            self,
            "digest",
            canonical_digest(
                {
                    "deterministic": "source_confidence",
                    "model": "minimum",
                    "version": self.version,
                }
            ),
        )

    @staticmethod
    def effective(
        source_confidence: float,
        normalizer_confidence: float,
        *,
        derivation_class: object,
    ) -> float:
        source = finite_score(source_confidence, "source_confidence")
        normalizer = finite_score(normalizer_confidence, "normalizer_confidence")
        value = getattr(derivation_class, "value", derivation_class)
        if value == "DETERMINISTIC":
            return source
        if value == "MODEL":
            return min(source, normalizer)
        raise ValueError("unknown Claim derivation class")


__all__ = ["ClaimConfidencePolicy"]
