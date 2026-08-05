"""供应商无关的来源 Adapter 边界与注册表。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from behavior.errors import SourceRecordError
from behavior.owner import ConfirmedOwnerBinding
from behavior.source.model import SourceRecord, SourceRecordBatch


@runtime_checkable
class BehaviorSourceAdapter(Protocol):
    name: str

    async def adapt(
        self,
        payload: object,
        *,
        owner_binding: ConfirmedOwnerBinding,
    ) -> SourceRecord | SourceRecordBatch: ...


class BehaviorSourceAdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, BehaviorSourceAdapter] = {}

    @staticmethod
    def normalize_name(value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise SourceRecordError("adapter name must be non-empty text")
        normalized = value.strip().casefold().replace("-", "_")
        if not normalized.replace("_", "").isalnum() or len(normalized) > 64:
            raise SourceRecordError("adapter name must be a bounded normalized identifier")
        return normalized

    def register(self, adapter: BehaviorSourceAdapter) -> None:
        if not isinstance(adapter, BehaviorSourceAdapter):
            raise TypeError("adapter must implement BehaviorSourceAdapter")
        name = self.normalize_name(adapter.name)
        if name in self._adapters:
            raise SourceRecordError(f"source adapter is already registered: {name}")
        self._adapters[name] = adapter

    def get(self, name: object) -> BehaviorSourceAdapter:
        normalized = self.normalize_name(name)
        try:
            return self._adapters[normalized]
        except KeyError as exc:
            raise SourceRecordError(f"unknown source adapter: {normalized}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))


__all__ = ["BehaviorSourceAdapter", "BehaviorSourceAdapterRegistry"]
