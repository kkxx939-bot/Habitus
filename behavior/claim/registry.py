"""ClaimProducer 的显式、无厂商分支注册表。"""

from __future__ import annotations

from behavior.claim.producer import ClaimProducer
from behavior.errors import ClaimProductionError


class ClaimProducerRegistry:
    def __init__(self) -> None:
        self._producers: dict[str, ClaimProducer] = {}

    @staticmethod
    def normalize_name(value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ClaimProductionError("producer name must be non-empty text")
        normalized = value.strip().casefold().replace("-", "_")
        if not normalized.replace("_", "").isalnum() or len(normalized) > 64:
            raise ClaimProductionError("producer name must be a bounded normalized identifier")
        return normalized

    def register(self, producer: ClaimProducer) -> None:
        if not isinstance(producer, ClaimProducer):
            raise TypeError("producer must implement ClaimProducer")
        name = self.normalize_name(producer.name)
        if name in self._producers:
            raise ClaimProductionError(f"ClaimProducer is already registered: {name}")
        self._producers[name] = producer

    def get(self, name: object) -> ClaimProducer:
        normalized = self.normalize_name(name)
        try:
            return self._producers[normalized]
        except KeyError as exc:
            raise ClaimProductionError(f"unknown ClaimProducer: {normalized}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._producers))


__all__ = ["ClaimProducerRegistry"]
