"""显式注册、无厂商分支的 Behavior Adapter Registry。"""

from __future__ import annotations

from dataclasses import dataclass

from behavior._validation import identifier
from behavior.errors import BehaviorAdapterError
from behavior.evidence.adapter import BehaviorSemanticAdapter
from behavior.evidence.provenance import BehaviorOriginKind, ProducerFingerprint, ProducerImplementationKind
from behavior.evidence.trust import BehaviorAdapterCapability


@dataclass(frozen=True, slots=True)
class RegisteredBehaviorAdapter:
    name: str
    adapter: BehaviorSemanticAdapter
    fingerprint: ProducerFingerprint
    capability: BehaviorAdapterCapability


class BehaviorSemanticAdapterRegistry:
    def __init__(self) -> None:
        self._adapters: dict[str, RegisteredBehaviorAdapter] = {}

    def register(self, adapter: BehaviorSemanticAdapter) -> None:
        if not isinstance(adapter, BehaviorSemanticAdapter):
            raise TypeError("adapter must implement BehaviorSemanticAdapter")
        name = identifier(adapter.name, "adapter.name")
        if name in self._adapters:
            raise BehaviorAdapterError("adapter name is already registered")
        fingerprint = adapter.fingerprint
        capability = adapter.capabilities
        if not isinstance(fingerprint, ProducerFingerprint):
            raise TypeError("adapter fingerprint must be ProducerFingerprint")
        if not isinstance(capability, BehaviorAdapterCapability):
            raise TypeError("adapter capability must be BehaviorAdapterCapability")
        if fingerprint.implementation_kind is ProducerImplementationKind.PROJECTOR:
            raise BehaviorAdapterError("external Behavior Adapter cannot use PROJECTOR fingerprint")
        if any(
            item.origin_kind is BehaviorOriginKind.CONVERSATION_PROJECTION
            for item in capability.allowed_outputs
        ):
            raise BehaviorAdapterError("external Behavior Adapter cannot declare conversation projection origin")
        self._adapters[name] = RegisteredBehaviorAdapter(
            name,
            adapter,
            fingerprint,
            capability,
        )

    def get(self, name: str) -> BehaviorSemanticAdapter:
        return self.resolve(name).adapter

    def resolve(self, name: str) -> RegisteredBehaviorAdapter:
        resolved = identifier(name, "adapter_name")
        try:
            return self._adapters[resolved]
        except KeyError as exc:
            raise BehaviorAdapterError("unknown Behavior Adapter") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))
