"""Claim processing lanes and versioned policy identities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from behavior.claim.proposal import ClaimKind
from behavior.config import ClaimConfig
from behavior.ingress.model import SemanticActorRole, SemanticRecordKind, SemanticSubjectRole
from foundation.integrity import canonical_digest


class ClaimProcessingLane(str, Enum):
    CORE = "CORE"
    ENHANCEMENT = "ENHANCEMENT"


class ClaimNormalizerRequirement(str, Enum):
    REQUIRED_CORE = "REQUIRED_CORE"
    OPTIONAL_ENHANCEMENT = "OPTIONAL_ENHANCEMENT"


class ClaimDerivationClass(str, Enum):
    DETERMINISTIC = "DETERMINISTIC"
    MODEL = "MODEL"


@dataclass(frozen=True)
class ClaimCompatibilityResult:
    allowed: bool
    reason_code: str


_DETERMINISTIC_KIND_MAP = {
    SemanticRecordKind.OWNER_ACTIVITY_SEGMENT: ClaimKind.ACTIVITY_PHASE,
    SemanticRecordKind.OWNER_UTTERANCE_SEGMENT: ClaimKind.UTTERANCE,
    SemanticRecordKind.OWNER_STATE_ASSERTION: ClaimKind.STATE_ASSERTION,
    SemanticRecordKind.OWNER_STATE_TRANSITION: ClaimKind.STATE_TRANSITION,
    SemanticRecordKind.OWNER_INTERACTION_SEGMENT: ClaimKind.INTERACTION,
    SemanticRecordKind.ROBOT_ACTION_EVENT: ClaimKind.ROBOT_ACTION,
    SemanticRecordKind.AGENT_ACTION_EVENT: ClaimKind.AGENT_ACTION,
    SemanticRecordKind.TOOL_RESULT_EVENT: ClaimKind.TOOL_RESULT,
    SemanticRecordKind.OWNER_SENSOR_FACT: ClaimKind.STATE_ASSERTION,
    SemanticRecordKind.ENVIRONMENT_SENSOR_FACT: ClaimKind.STATE_ASSERTION,
    SemanticRecordKind.DEVICE_STATE: ClaimKind.STATE_ASSERTION,
    SemanticRecordKind.ENVIRONMENT_CHANGE: ClaimKind.ENVIRONMENT_CHANGE,
    SemanticRecordKind.COVERAGE_INTERVAL: ClaimKind.COVERAGE,
}

_MODEL_FORBIDDEN = frozenset(
    {
        ClaimKind.UTTERANCE,
        ClaimKind.ROBOT_ACTION,
        ClaimKind.AGENT_ACTION,
        ClaimKind.TOOL_RESULT,
        ClaimKind.COVERAGE,
    }
)


@dataclass(frozen=True)
class ClaimCompatibilityPolicy:
    version: str = "3"

    @property
    def digest(self) -> str:
        return canonical_digest(
            {
                "version": self.version,
                "deterministic_kind_map": tuple(
                    sorted((record.value, claim.value) for record, claim in _DETERMINISTIC_KIND_MAP.items())
                ),
                "model_forbidden": tuple(sorted(item.value for item in _MODEL_FORBIDDEN)),
                "role_rules_version": "3",
            }
        )

    def allowed_model_kinds(
        self,
        record_kind: SemanticRecordKind,
        subject_role: SemanticSubjectRole,
        actor_role: SemanticActorRole,
    ) -> frozenset[ClaimKind]:
        record = SemanticRecordKind(record_kind)
        subject = SemanticSubjectRole(subject_role)
        actor = SemanticActorRole(actor_role)
        candidates = {
            ClaimKind.STATE_ASSERTION,
            ClaimKind.STATE_TRANSITION,
        }
        if subject is SemanticSubjectRole.OWNER and actor is SemanticActorRole.OWNER:
            candidates.update({ClaimKind.ACTIVITY_PHASE, ClaimKind.INTERACTION})
        if (
            record is not SemanticRecordKind.OWNER_UTTERANCE_SEGMENT
            and subject is SemanticSubjectRole.ENVIRONMENT
            and actor in {SemanticActorRole.SYSTEM, SemanticActorRole.ENVIRONMENT}
        ):
            candidates.add(ClaimKind.ENVIRONMENT_CHANGE)
        if not (
            subject is SemanticSubjectRole.ENVIRONMENT
            and actor in {SemanticActorRole.SYSTEM, SemanticActorRole.ENVIRONMENT}
        ):
            candidates.discard(ClaimKind.ENVIRONMENT_CHANGE)
        return frozenset(candidates)

    def evaluate(
        self,
        *,
        record_kind: SemanticRecordKind,
        subject_role: SemanticSubjectRole,
        actor_role: SemanticActorRole,
        derivation_class: ClaimDerivationClass,
        claim_kind: ClaimKind,
    ) -> ClaimCompatibilityResult:
        record = SemanticRecordKind(record_kind)
        subject = SemanticSubjectRole(subject_role)
        actor = SemanticActorRole(actor_role)
        derivation = ClaimDerivationClass(derivation_class)
        claim = ClaimKind(claim_kind)
        if derivation is ClaimDerivationClass.DETERMINISTIC:
            if _DETERMINISTIC_KIND_MAP.get(record) is not claim:
                return ClaimCompatibilityResult(False, "deterministic_record_claim_kind_mismatch")
        elif claim in _MODEL_FORBIDDEN or claim not in self.allowed_model_kinds(record, subject, actor):
            return ClaimCompatibilityResult(False, "model_claim_kind_not_allowed")
        role_rules = {
            ClaimKind.ROBOT_ACTION: (SemanticSubjectRole.ROBOT, SemanticActorRole.ROBOT),
            ClaimKind.AGENT_ACTION: (SemanticSubjectRole.AGENT, SemanticActorRole.AGENT),
            ClaimKind.TOOL_RESULT: (SemanticSubjectRole.TOOL, SemanticActorRole.TOOL),
            ClaimKind.COVERAGE: (SemanticSubjectRole.ENVIRONMENT, SemanticActorRole.SYSTEM),
            ClaimKind.UTTERANCE: (SemanticSubjectRole.OWNER, SemanticActorRole.OWNER),
        }
        expected = role_rules.get(claim)
        if expected is not None and (subject, actor) != expected:
            return ClaimCompatibilityResult(False, "claim_role_incompatible")
        if claim is ClaimKind.ENVIRONMENT_CHANGE and not (
            subject is SemanticSubjectRole.ENVIRONMENT
            and actor in {SemanticActorRole.SYSTEM, SemanticActorRole.ENVIRONMENT}
        ):
            return ClaimCompatibilityResult(False, "environment_change_role_incompatible")
        return ClaimCompatibilityResult(True, "claim_compatibility_allowed")


@dataclass(frozen=True)
class ClaimBindingPolicy:
    compatibility: ClaimCompatibilityPolicy = ClaimCompatibilityPolicy()
    version: str = "3"
    alternative_group_policy_version: str = "1"

    @property
    def digest(self) -> str:
        return canonical_digest(
            {
                "version": self.version,
                "compatibility_policy_digest": self.compatibility.digest,
                "alternative_group_policy_version": self.alternative_group_policy_version,
                "binding_algorithm_version": "3",
            }
        )


@dataclass(frozen=True)
class ClaimAdmissionPolicyIdentity:
    version: str
    digest: str

    @classmethod
    def from_config(
        cls,
        config: ClaimConfig,
        *,
        max_accepted_claims: int,
        version: str = "3",
    ) -> ClaimAdmissionPolicyIdentity:
        return cls(
            version=version,
            digest=canonical_digest(
                {
                    "version": version,
                    "min_direct_confidence": config.min_direct_confidence,
                    "min_sensor_confidence": config.min_sensor_confidence,
                    "min_model_confidence": config.min_model_confidence,
                    "repeat_state_suppression_seconds": config.repeat_state_suppression_seconds,
                    "max_accepted_claims": max_accepted_claims,
                    "algorithm_version": "3",
                }
            ),
        )


__all__ = [
    "ClaimAdmissionPolicyIdentity",
    "ClaimBindingPolicy",
    "ClaimCompatibilityPolicy",
    "ClaimCompatibilityResult",
    "ClaimDerivationClass",
    "ClaimNormalizerRequirement",
    "ClaimProcessingLane",
]
