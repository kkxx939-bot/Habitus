"""Claim Normalizer 的显式 Registry。"""

from __future__ import annotations

from behavior._validation import identifier
from behavior.claim.normalizer import ClaimNormalizer, ClaimNormalizerKind
from behavior.errors import ClaimNormalizationError


class ClaimNormalizerRegistry:
    def __init__(self) -> None:
        self._normalizers: dict[str, ClaimNormalizer] = {}

    def register(self, normalizer: ClaimNormalizer) -> None:
        if not isinstance(normalizer, ClaimNormalizer):
            raise TypeError("normalizer must implement ClaimNormalizer")
        name = identifier(normalizer.name, "normalizer.name")
        if name in self._normalizers:
            raise ClaimNormalizationError("normalizer name is already registered")
        if normalizer.fingerprint.normalizer_name != name:
            raise ClaimNormalizationError("normalizer name and fingerprint disagree")
        if normalizer.fingerprint.kind is not normalizer.kind:
            raise ClaimNormalizationError("normalizer kind and fingerprint disagree")
        self._normalizers[name] = normalizer

    def get(self, name: str) -> ClaimNormalizer:
        resolved = identifier(name, "normalizer_name")
        try:
            return self._normalizers[resolved]
        except KeyError as exc:
            raise ClaimNormalizationError("unknown Claim Normalizer") from exc

    def names(self, kind: ClaimNormalizerKind | None = None) -> tuple[str, ...]:
        if kind is None:
            return tuple(sorted(self._normalizers))
        resolved = ClaimNormalizerKind(kind)
        return tuple(sorted(name for name, item in self._normalizers.items() if item.kind is resolved))
