"""上游确认后的单 Owner 不可变绑定。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from behavior._validation import identifier, require_fields, sha256_digest, strict_fields, strict_utc, utc_text
from behavior.errors import BehaviorOwnerError
from foundation.integrity import canonical_digest

OWNER_BINDING_SCHEMA_VERSION = "2"


@dataclass(frozen=True, init=False)
class ConfirmedOwnerBinding:
    """只表达上游已经确认的 Owner；解析过程字段只用于来源审计。"""

    owner_ref: str
    resolver_fingerprint: str
    resolved_at: datetime
    resolution_evidence_digest: str
    owner_identity_digest: str

    def __init__(
        self,
        owner_ref: object,
        resolver_fingerprint: object,
        resolved_at: object,
        resolution_evidence_digest: object,
    ) -> None:
        try:
            resolved_ref = identifier(owner_ref, "owner_ref")
            resolver = identifier(resolver_fingerprint, "resolver_fingerprint")
            timestamp = strict_utc(resolved_at, "resolved_at")
            evidence = sha256_digest(
                resolution_evidence_digest,
                "resolution_evidence_digest",
            )
        except (TypeError, ValueError) as exc:
            raise BehaviorOwnerError(str(exc)) from exc
        identity = canonical_digest(
            {
                "owner_binding_schema_version": OWNER_BINDING_SCHEMA_VERSION,
                "owner_ref": resolved_ref,
            }
        )
        object.__setattr__(self, "owner_ref", resolved_ref)
        object.__setattr__(self, "resolver_fingerprint", resolver)
        object.__setattr__(self, "resolved_at", timestamp)
        object.__setattr__(self, "resolution_evidence_digest", evidence)
        object.__setattr__(self, "owner_identity_digest", identity)

    def to_dict(self) -> dict[str, object]:
        return {
            "owner_ref": self.owner_ref,
            "resolver_fingerprint": self.resolver_fingerprint,
            "resolved_at": utc_text(self.resolved_at),
            "resolution_evidence_digest": self.resolution_evidence_digest,
            "owner_identity_digest": self.owner_identity_digest,
        }

    @classmethod
    def from_dict(cls, value: object) -> ConfirmedOwnerBinding:
        from behavior._validation import parse_utc

        fields = frozenset(
            {
                "owner_ref",
                "resolver_fingerprint",
                "resolved_at",
                "resolution_evidence_digest",
                "owner_identity_digest",
            }
        )
        try:
            data = strict_fields(value, "owner_binding", fields)
            require_fields(data, "owner_binding", fields)
            result = cls(
                data["owner_ref"],
                data["resolver_fingerprint"],
                parse_utc(data["resolved_at"], "owner_binding.resolved_at"),
                data["resolution_evidence_digest"],
            )
            if data["owner_identity_digest"] != result.owner_identity_digest:
                raise BehaviorOwnerError("owner identity digest mismatch")
            return result
        except (TypeError, ValueError) as exc:
            if isinstance(exc, BehaviorOwnerError):
                raise
            raise BehaviorOwnerError(str(exc)) from exc


__all__ = ["ConfirmedOwnerBinding", "OWNER_BINDING_SCHEMA_VERSION"]
