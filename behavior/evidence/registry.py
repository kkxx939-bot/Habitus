"""显式注册、无厂商分支的 Behavior Adapter Registry。"""

from __future__ import annotations

from behavior._validation import identifier
from behavior.errors import BehaviorAdapterError
from behavior.evidence.adapter import BehaviorSemanticAdapter
from behavior.evidence.provenance import BehaviorOriginKind, ProducerImplementationKind


class BehaviorSemanticAdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, BehaviorSemanticAdapter] = {}

    def register(self, adapter: BehaviorSemanticAdapter) -> None:
        if not isinstance(adapter, BehaviorSemanticAdapter):
            raise TypeError("adapter must implement BehaviorSemanticAdapter")
        name = identifier(adapter.name, "adapter.name")
        if name in self._adapters:
            raise BehaviorAdapterError("adapter name is already registered")
        if adapter.fingerprint.implementation_kind is ProducerImplementationKind.PROJECTOR:
            raise BehaviorAdapterError("external Behavior Adapter cannot use PROJECTOR fingerprint")
        if BehaviorOriginKind.CONVERSATION_PROJECTION in adapter.capabilities.allowed_origin_kinds:
            raise BehaviorAdapterError("external Behavior Adapter cannot declare conversation projection origin")
        self._adapters[name] = adapter

    def get(self, name: str) -> BehaviorSemanticAdapter:
        resolved = identifier(name, "adapter_name")
        try:
            return self._adapters[resolved]
        except KeyError as exc:
            raise BehaviorAdapterError("unknown Behavior Adapter") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))


__all__ = ["BehaviorSemanticAdapterRegistry"]
