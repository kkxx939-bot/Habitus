"""单条 Evidence 的 Core 与 Enhancement 路由计划。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from behavior._validation import identifier, sha256_digest
from behavior.claim.normalizer import ClaimNormalizerKind
from behavior.claim.registry import ClaimNormalizerRegistry
from behavior.config import ClaimNormalizationConfig
from behavior.errors import ClaimNormalizationError
from behavior.evidence.content import BehaviorRecordKind, BehaviorRole
from behavior.evidence.record import BehaviorEvidenceRecord
from foundation.integrity import canonical_digest

CLAIM_PLANNER_POLICY_VERSION = "claim_normalization_planner_v1"


class NormalizationLane(str, Enum):
    CORE = "CORE"
    ENHANCEMENT = "ENHANCEMENT"


@dataclass(frozen=True)
class ClaimNormalizationRoute:
    normalizer_name: str
    normalizer_fingerprint: str
    normalizer_kind: ClaimNormalizerKind
    lane: NormalizationLane

    def __post_init__(self) -> None:
        object.__setattr__(self, "normalizer_name", identifier(self.normalizer_name, "route.normalizer_name"))
        object.__setattr__(
            self,
            "normalizer_fingerprint",
            sha256_digest(self.normalizer_fingerprint, "route.normalizer_fingerprint"),
        )
        kind = ClaimNormalizerKind(self.normalizer_kind)
        lane = NormalizationLane(self.lane)
        if (lane is NormalizationLane.CORE) != (kind is ClaimNormalizerKind.DETERMINISTIC):
            raise ValueError("Core routes must be deterministic and Enhancement routes must be model-backed")
        object.__setattr__(self, "normalizer_kind", kind)
        object.__setattr__(self, "lane", lane)


@dataclass(frozen=True)
class ClaimNormalizationPlan:
    evidence_record_id: str
    evidence_record_digest: str
    core_routes: tuple[ClaimNormalizationRoute, ...]
    enhancement_routes: tuple[ClaimNormalizationRoute, ...]
    planner_policy_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "evidence_record_id", identifier(self.evidence_record_id, "plan.evidence_record_id"))
        object.__setattr__(
            self,
            "evidence_record_digest",
            sha256_digest(self.evidence_record_digest, "plan.evidence_record_digest"),
        )
        if not isinstance(self.core_routes, tuple) or any(
            route.lane is not NormalizationLane.CORE for route in self.core_routes
        ):
            raise TypeError("core_routes must contain Core routes")
        if len(self.core_routes) > 1:
            raise ValueError("one Evidence record can have at most one Core route")
        if not isinstance(self.enhancement_routes, tuple) or any(
            route.lane is not NormalizationLane.ENHANCEMENT for route in self.enhancement_routes
        ):
            raise TypeError("enhancement_routes must contain Enhancement routes")
        object.__setattr__(
            self,
            "planner_policy_digest",
            sha256_digest(self.planner_policy_digest, "plan.planner_policy_digest"),
        )


@dataclass(frozen=True)
class ClaimNormalizationPlanner:
    registry: ClaimNormalizerRegistry
    config: ClaimNormalizationConfig
    policy_version: str = CLAIM_PLANNER_POLICY_VERSION
    policy_digest: str = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.registry, ClaimNormalizerRegistry):
            raise TypeError("registry must be ClaimNormalizerRegistry")
        if not isinstance(self.config, ClaimNormalizationConfig):
            raise TypeError("config must be ClaimNormalizationConfig")
        if self.policy_version != CLAIM_PLANNER_POLICY_VERSION:
            raise ValueError("unsupported Claim planner policy version")
        object.__setattr__(
            self,
            "policy_digest",
            canonical_digest(
                {
                    "normalize_user_utterances": self.config.normalize_user_utterances,
                    "version": self.policy_version,
                }
            ),
        )

    def plan(self, record: BehaviorEvidenceRecord) -> ClaimNormalizationPlan:
        if not isinstance(record, BehaviorEvidenceRecord):
            raise TypeError("record must be BehaviorEvidenceRecord")
        deterministic = self.registry.names(ClaimNormalizerKind.DETERMINISTIC)
        if len(deterministic) != 1:
            raise ClaimNormalizationError("exactly one deterministic Core Normalizer is required")
        model_names = self.registry.names(ClaimNormalizerKind.MODEL)
        content = record.semantic_content
        needs_core = content.record_kind is not BehaviorRecordKind.FREE_TEXT_SEMANTIC
        needs_enhancement = content.record_kind is BehaviorRecordKind.FREE_TEXT_SEMANTIC or (
            content.record_kind is BehaviorRecordKind.UTTERANCE_SEGMENT
            and content.subject_role is BehaviorRole.USER
            and self.config.normalize_user_utterances
        )
        if needs_enhancement and not model_names:
            raise ClaimNormalizationError("planned Model Enhancement has no registered Normalizer")
        if len(model_names) > self.config.max_enhancement_normalizers_per_record:
            raise ClaimNormalizationError("Enhancement Normalizer count exceeds configured boundary")
        core_routes: tuple[ClaimNormalizationRoute, ...] = ()
        if needs_core:
            normalizer = self.registry.get(deterministic[0])
            core_routes = (
                ClaimNormalizationRoute(
                    normalizer_name=normalizer.name,
                    normalizer_fingerprint=normalizer.fingerprint.digest,
                    normalizer_kind=normalizer.kind,
                    lane=NormalizationLane.CORE,
                ),
            )
        enhancement_routes = tuple(
            ClaimNormalizationRoute(
                normalizer_name=self.registry.get(name).name,
                normalizer_fingerprint=self.registry.get(name).fingerprint.digest,
                normalizer_kind=ClaimNormalizerKind.MODEL,
                lane=NormalizationLane.ENHANCEMENT,
            )
            for name in model_names
        ) if needs_enhancement else ()
        return ClaimNormalizationPlan(
            evidence_record_id=record.evidence_record_id,
            evidence_record_digest=record.content_digest,
            core_routes=core_routes,
            enhancement_routes=enhancement_routes,
            planner_policy_digest=self.policy_digest,
        )


__all__ = ["ClaimNormalizationPlan", "ClaimNormalizationPlanner"]
