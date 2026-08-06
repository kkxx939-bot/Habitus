"""ClaimNormalizer 的显式、供应商无关注册表。"""

from __future__ import annotations

from behavior.claim.normalizer import ClaimNormalizer
from behavior.errors import ClaimProductionError


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
