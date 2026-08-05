"""Behavior 单 Owner 路由与不可变绑定。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from behavior._validation import identifier, sha256_digest, strict_utc, utc_text
from behavior.errors import BehaviorOwnerError
from foundation.integrity import canonical_digest


class OwnerRouteStatus(str, Enum):
    OWNER_CONFIRMED = "OWNER_CONFIRMED"
    OTHER_PERSON = "OTHER_PERSON"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True)
class ConfirmedOwnerBinding:
    """只保存本地引用和解析器规范信息，不保存 Owner PII。"""

    binding_ref: str
    resolver_fingerprint: str
    resolved_at: datetime
    evidence_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "binding_ref", identifier(self.binding_ref, "binding_ref"))
        object.__setattr__(
            self,
            "resolver_fingerprint",
            identifier(self.resolver_fingerprint, "resolver_fingerprint"),
        )
        object.__setattr__(self, "resolved_at", strict_utc(self.resolved_at, "resolved_at"))
        object.__setattr__(self, "evidence_digest", sha256_digest(self.evidence_digest, "evidence_digest"))

    @property
    def binding_digest(self) -> str:
        return canonical_digest(
            {
                "binding_ref": self.binding_ref,
                "resolver_fingerprint": self.resolver_fingerprint,
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "binding_ref": self.binding_ref,
            "resolver_fingerprint": self.resolver_fingerprint,
            "resolved_at": utc_text(self.resolved_at),
            "evidence_digest": self.evidence_digest,
            "binding_digest": self.binding_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> ConfirmedOwnerBinding:
        from behavior._validation import parse_utc, require_fields, strict_fields

        allowed = frozenset(
            {"binding_ref", "resolver_fingerprint", "resolved_at", "evidence_digest", "binding_digest"}
        )
        data = strict_fields(value, "owner_binding", allowed)
        require_fields(data, "owner_binding", allowed)
        result = cls(
            binding_ref=data["binding_ref"],
            resolver_fingerprint=data["resolver_fingerprint"],
            resolved_at=parse_utc(data["resolved_at"], "owner_binding.resolved_at"),
            evidence_digest=data["evidence_digest"],
        )
        if data["binding_digest"] != result.binding_digest:
            raise BehaviorOwnerError("owner binding digest mismatch")
        return result


@dataclass(frozen=True)
class OwnerRouteDecision:
    status: OwnerRouteStatus
    binding_ref: str | None
    resolver_fingerprint: str
    resolved_at: datetime
    evidence_digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", OwnerRouteStatus(self.status))
        if self.binding_ref is not None:
            object.__setattr__(self, "binding_ref", identifier(self.binding_ref, "binding_ref"))
        object.__setattr__(
            self,
            "resolver_fingerprint",
            identifier(self.resolver_fingerprint, "resolver_fingerprint"),
        )
        object.__setattr__(self, "resolved_at", strict_utc(self.resolved_at, "resolved_at"))
        object.__setattr__(self, "evidence_digest", sha256_digest(self.evidence_digest, "evidence_digest"))
        if self.status is OwnerRouteStatus.OWNER_CONFIRMED and self.binding_ref is None:
            raise BehaviorOwnerError("OWNER_CONFIRMED requires binding_ref")

    def confirm(self) -> ConfirmedOwnerBinding:
        if self.status is not OwnerRouteStatus.OWNER_CONFIRMED:
            raise BehaviorOwnerError(f"owner route status {self.status.value} cannot enter Behavior")
        assert self.binding_ref is not None
        return ConfirmedOwnerBinding(
            binding_ref=self.binding_ref,
            resolver_fingerprint=self.resolver_fingerprint,
            resolved_at=self.resolved_at,
            evidence_digest=self.evidence_digest,
        )


__all__ = ["ConfirmedOwnerBinding", "OwnerRouteDecision", "OwnerRouteStatus"]
