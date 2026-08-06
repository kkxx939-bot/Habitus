"""SemanticIngressAdapter 的显式、供应商无关注册表。"""

from __future__ import annotations

from behavior.errors import SemanticIngressError
from behavior.ingress.adapter import SemanticIngressAdapter


class SemanticIngressAdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, SemanticIngressAdapter] = {}

    @staticmethod
    def normalize_name(value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise SemanticIngressError("Adapter name must be non-empty text")
        normalized = value.strip().casefold().replace("-", "_")
        if not normalized.replace("_", "").isalnum() or len(normalized) > 64:
            raise SemanticIngressError("Adapter name must be a bounded normalized identifier")
        return normalized

    def register(self, adapter: SemanticIngressAdapter) -> None:
        if not isinstance(adapter, SemanticIngressAdapter):
            raise TypeError("adapter must implement SemanticIngressAdapter")
        name = self.normalize_name(adapter.name)
        if name in self._adapters:
            raise SemanticIngressError(f"semantic Adapter is already registered: {name}")
        if adapter.fingerprint.producer_name.casefold().replace("-", "_") != name:
            raise SemanticIngressError("Adapter name and ProducerFingerprint name must match")
        self._adapters[name] = adapter

    def get(self, name: object) -> SemanticIngressAdapter:
        normalized = self.normalize_name(name)
        try:
            return self._adapters[normalized]
        except KeyError as exc:
            raise SemanticIngressError(f"unknown semantic Adapter: {normalized}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))


__all__ = ["SemanticIngressAdapterRegistry"]
