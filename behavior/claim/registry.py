"""ClaimNormalizer 的显式、供应商无关注册表。"""

from __future__ import annotations

from behavior.claim.normalizer import ClaimNormalizer, ClaimNormalizerKind
from behavior.claim.policy import ClaimCompatibilityPolicy
from behavior.errors import ClaimProductionError
from behavior.ingress.model import SemanticRecordKind
from ModelClient import StructuredChatClient


class ClaimNormalizerRegistry:
    def __init__(self) -> None:
        self._normalizers: dict[str, ClaimNormalizer] = {}

    @staticmethod
    def normalize_name(value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ClaimProductionError("Normalizer name must be non-empty text")
        normalized = value.strip().casefold().replace("-", "_")
        if not normalized.replace("_", "").isalnum() or len(normalized) > 64:
            raise ClaimProductionError("Normalizer name must be a bounded normalized identifier")
        return normalized

    def register(self, normalizer: ClaimNormalizer) -> None:
        if not isinstance(normalizer, ClaimNormalizer):
            raise TypeError("normalizer must implement ClaimNormalizer")
        name = self.normalize_name(normalizer.name)
        if name in self._normalizers:
            raise ClaimProductionError(f"Claim Normalizer is already registered: {name}")
        if normalizer.fingerprint.normalizer_name.casefold().replace("-", "_") != name:
            raise ClaimProductionError("Normalizer name and fingerprint name must match")
        if normalizer.fingerprint.normalizer_kind is not normalizer.kind:
            raise ClaimProductionError("Normalizer kind and fingerprint kind must match")
        if (
            not isinstance(normalizer.allowed_record_kinds, frozenset)
            or not normalizer.allowed_record_kinds
            or any(not isinstance(item, SemanticRecordKind) for item in normalizer.allowed_record_kinds)
        ):
            raise ClaimProductionError("Normalizer allowed_record_kinds must be a non-empty SemanticRecordKind set")
        if normalizer.kind is ClaimNormalizerKind.DETERMINISTIC and normalizer.model_client is not None:
            raise ClaimProductionError("deterministic Normalizer cannot own a model client")
        if normalizer.kind is ClaimNormalizerKind.DETERMINISTIC and normalizer.compatibility_policy is not None:
            raise ClaimProductionError("deterministic Normalizer cannot own a model compatibility policy")
        if normalizer.kind is ClaimNormalizerKind.MODEL and not isinstance(
            normalizer.model_client, StructuredChatClient
        ):
            raise ClaimProductionError("model Normalizer must expose a StructuredChatClient")
        if normalizer.kind is ClaimNormalizerKind.MODEL and not isinstance(
            normalizer.compatibility_policy,
            ClaimCompatibilityPolicy,
        ):
            raise ClaimProductionError("model Normalizer must expose a ClaimCompatibilityPolicy")
        self._normalizers[name] = normalizer

    def get(self, name: object) -> ClaimNormalizer:
        normalized = self.normalize_name(name)
        try:
            return self._normalizers[normalized]
        except KeyError as exc:
            raise ClaimProductionError(f"unknown Claim Normalizer: {normalized}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._normalizers))


__all__ = ["ClaimNormalizerRegistry"]
