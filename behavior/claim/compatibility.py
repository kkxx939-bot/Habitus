"""RecordKind、角色、Normalizer 与 ClaimKind 的机械兼容策略。"""

from __future__ import annotations

from dataclasses import dataclass, field

from behavior.claim.proposal import ClaimKind
from behavior.evidence.content import BehaviorRecordKind, BehaviorRole
from foundation.integrity import canonical_digest

CLAIM_COMPATIBILITY_POLICY_VERSION = "claim_compatibility_v1"

_DETERMINISTIC_KIND: dict[BehaviorRecordKind, ClaimKind] = {
    BehaviorRecordKind.ACTIVITY_SEGMENT: ClaimKind.ACTIVITY,
    BehaviorRecordKind.UTTERANCE_SEGMENT: ClaimKind.UTTERANCE,
    BehaviorRecordKind.STATE_ASSERTION: ClaimKind.STATE_ASSERTION,
    BehaviorRecordKind.STATE_TRANSITION: ClaimKind.STATE_TRANSITION,
    BehaviorRecordKind.INTERACTION_SEGMENT: ClaimKind.INTERACTION,
    BehaviorRecordKind.ACTION_EVENT: ClaimKind.ACTION,
    BehaviorRecordKind.TOOL_CALL_EVENT: ClaimKind.TOOL_CALL,
    BehaviorRecordKind.TOOL_RESULT_EVENT: ClaimKind.TOOL_RESULT,
    BehaviorRecordKind.ENVIRONMENT_CHANGE: ClaimKind.ENVIRONMENT_CHANGE,
    BehaviorRecordKind.COVERAGE_INTERVAL: ClaimKind.COVERAGE,
    BehaviorRecordKind.FEEDBACK_EVENT: ClaimKind.FEEDBACK,
}

_USER_UTTERANCE_ENHANCEMENTS = frozenset(
    {
        ClaimKind.STATE_ASSERTION,
        ClaimKind.STATE_TRANSITION,
        ClaimKind.ACTIVITY,
        ClaimKind.INTERACTION,
        ClaimKind.FEEDBACK,
    }
)


@dataclass(frozen=True)
class ClaimCompatibilityResult:
    allowed: bool
    reason_code: str


@dataclass(frozen=True)
class ClaimCompatibilityPolicy:
    version: str = CLAIM_COMPATIBILITY_POLICY_VERSION
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        if self.version != CLAIM_COMPATIBILITY_POLICY_VERSION:
            raise ValueError("unsupported Claim compatibility policy version")
        object.__setattr__(
            self,
            "digest",
            canonical_digest(
                {
                    "deterministic": {
                        record.value: claim.value for record, claim in _DETERMINISTIC_KIND.items()
                    },
                    "user_utterance_enhancements": sorted(
                        item.value for item in _USER_UTTERANCE_ENHANCEMENTS
                    ),
                    "version": self.version,
                }
            ),
        )

    def evaluate(
        self,
        *,
        record_kind: BehaviorRecordKind,
        subject_role: BehaviorRole,
        actor_role: BehaviorRole | None,
        normalizer_kind: object,
        claim_kind: ClaimKind,
    ) -> ClaimCompatibilityResult:
        record = BehaviorRecordKind(record_kind)
        subject = BehaviorRole(subject_role)
        if actor_role is not None:
            BehaviorRole(actor_role)
        claim = ClaimKind(claim_kind)
        kind_value = getattr(normalizer_kind, "value", normalizer_kind)
        if kind_value == "DETERMINISTIC":
            expected = _DETERMINISTIC_KIND.get(record)
            if expected is None:
                return ClaimCompatibilityResult(False, "NO_DETERMINISTIC_CLAIM")
            if claim is not expected:
                return ClaimCompatibilityResult(False, "DETERMINISTIC_KIND_MISMATCH")
            return ClaimCompatibilityResult(True, "ALLOWED")
        if kind_value != "MODEL":
            return ClaimCompatibilityResult(False, "UNKNOWN_NORMALIZER_KIND")
        if record is BehaviorRecordKind.UTTERANCE_SEGMENT:
            if subject is not BehaviorRole.USER:
                return ClaimCompatibilityResult(False, "UTTERANCE_ENHANCEMENT_REQUIRES_USER")
            if claim in _USER_UTTERANCE_ENHANCEMENTS:
                return ClaimCompatibilityResult(True, "ALLOWED")
            return ClaimCompatibilityResult(False, "UTTERANCE_ENHANCEMENT_KIND_FORBIDDEN")
        if record is BehaviorRecordKind.FREE_TEXT_SEMANTIC:
            return ClaimCompatibilityResult(True, "ALLOWED")
        return ClaimCompatibilityResult(False, "MODEL_ENHANCEMENT_NOT_PLANNED")


__all__ = ["ClaimCompatibilityPolicy", "ClaimCompatibilityResult"]
