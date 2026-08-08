"""Model Enhancement ClaimKind 的唯一兼容策略。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from behavior.claim.proposal import ClaimKind
from behavior.evidence.content import BehaviorRecordKind, BehaviorRole
from behavior.evidence.specs import RECORD_SPECS
from foundation.integrity import canonical_digest

CLAIM_COMPATIBILITY_POLICY_VERSION = "claim_compatibility_v2"

_USER_UTTERANCE_ENHANCEMENTS = frozenset(
    {
        ClaimKind.STATE_ASSERTION,
        ClaimKind.STATE_TRANSITION,
        ClaimKind.ACTIVITY,
        ClaimKind.INTERACTION,
        ClaimKind.FEEDBACK,
    }
)
_FREE_TEXT_ROLE_MATRIX: Mapping[
    tuple[BehaviorRole, BehaviorRole | None],
    frozenset[ClaimKind],
] = {
    (BehaviorRole.USER, BehaviorRole.USER): frozenset(
        {
            ClaimKind.STATE_ASSERTION,
            ClaimKind.STATE_TRANSITION,
            ClaimKind.ACTIVITY,
            ClaimKind.INTERACTION,
            ClaimKind.FEEDBACK,
            ClaimKind.FREE_TEXT,
        }
    ),
    (BehaviorRole.AGENT, BehaviorRole.AGENT): frozenset(
        {
            ClaimKind.STATE_ASSERTION,
            ClaimKind.STATE_TRANSITION,
            ClaimKind.ACTIVITY,
            ClaimKind.INTERACTION,
            ClaimKind.ACTION,
            ClaimKind.FREE_TEXT,
        }
    ),
    (BehaviorRole.ROBOT, BehaviorRole.ROBOT): frozenset(
        {
            ClaimKind.STATE_ASSERTION,
            ClaimKind.STATE_TRANSITION,
            ClaimKind.ACTIVITY,
            ClaimKind.INTERACTION,
            ClaimKind.ACTION,
            ClaimKind.FREE_TEXT,
        }
    ),
    **{
        (BehaviorRole.ENVIRONMENT, actor): frozenset(
            {
                ClaimKind.STATE_ASSERTION,
                ClaimKind.STATE_TRANSITION,
                ClaimKind.ENVIRONMENT_CHANGE,
                ClaimKind.FREE_TEXT,
            }
        )
        for actor in (
            None,
            BehaviorRole.ENVIRONMENT,
            BehaviorRole.SYSTEM,
            BehaviorRole.TOOL,
        )
    },
    **{
        (BehaviorRole.TOOL, actor): frozenset(
            {
                ClaimKind.STATE_ASSERTION,
                ClaimKind.STATE_TRANSITION,
                ClaimKind.FREE_TEXT,
            }
        )
        for actor in (None, BehaviorRole.TOOL, BehaviorRole.AGENT, BehaviorRole.ROBOT, BehaviorRole.SYSTEM)
    },
    **{
        (BehaviorRole.SYSTEM, actor): frozenset(
            {
                ClaimKind.STATE_ASSERTION,
                ClaimKind.STATE_TRANSITION,
                ClaimKind.FREE_TEXT,
            }
        )
        for actor in (None, BehaviorRole.SYSTEM)
    },
    (BehaviorRole.OTHER_ANONYMOUS, None): frozenset({ClaimKind.FREE_TEXT}),
}


@dataclass(frozen=True, slots=True)
class ClaimCompatibilityResult:
    allowed: bool
    reason_code: str


@dataclass(frozen=True, slots=True)
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
                    "free_text_role_matrix": {
                        f"{subject.value}/{'' if actor is None else actor.value}": sorted(
                            claim.value for claim in claims
                        )
                        for (subject, actor), claims in _FREE_TEXT_ROLE_MATRIX.items()
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
        actor = None if actor_role is None else BehaviorRole(actor_role)
        claim = ClaimKind(claim_kind)
        kind_value = getattr(normalizer_kind, "value", normalizer_kind)
        if kind_value == "DETERMINISTIC":
            mapper = RECORD_SPECS[record].deterministic_mapper
            if mapper is None:
                return ClaimCompatibilityResult(False, "NO_DETERMINISTIC_CLAIM")
            return ClaimCompatibilityResult(
                claim is mapper.claim_kind,
                "ALLOWED" if claim is mapper.claim_kind else "DETERMINISTIC_KIND_MISMATCH",
            )
        if kind_value != "MODEL":
            return ClaimCompatibilityResult(False, "UNKNOWN_NORMALIZER_KIND")
        if record is BehaviorRecordKind.UTTERANCE_SEGMENT:
            allowed = subject is BehaviorRole.USER and claim in _USER_UTTERANCE_ENHANCEMENTS
            return ClaimCompatibilityResult(
                allowed,
                "ALLOWED" if allowed else "UTTERANCE_ENHANCEMENT_KIND_FORBIDDEN",
            )
        if record is BehaviorRecordKind.FREE_TEXT_SEMANTIC:
            allowed = claim in _FREE_TEXT_ROLE_MATRIX.get((subject, actor), frozenset())
            return ClaimCompatibilityResult(
                allowed,
                "ALLOWED" if allowed else "FREE_TEXT_ROLE_KIND_FORBIDDEN",
            )
        return ClaimCompatibilityResult(False, "MODEL_ENHANCEMENT_NOT_PLANNED")
