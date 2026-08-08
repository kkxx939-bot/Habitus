"""外部结构化语义 Adapter 的纯转换契约。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from behavior.evidence.content import BehaviorSemanticContent
from behavior.evidence.provenance import BehaviorSourceDescriptor, ProducerFingerprint
from behavior.evidence.trust import BehaviorAdapterCapability


@dataclass(frozen=True)
class BehaviorSemanticInput:
    content: BehaviorSemanticContent
    source: BehaviorSourceDescriptor

    def __post_init__(self) -> None:
        if not isinstance(self.content, BehaviorSemanticContent):
            raise TypeError("semantic input content must be BehaviorSemanticContent")
        if not isinstance(self.source, BehaviorSourceDescriptor):
            raise TypeError("semantic input source must be BehaviorSourceDescriptor")


@dataclass(frozen=True)
class BehaviorSemanticInputBatch:
    items: tuple[BehaviorSemanticInput, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.items, tuple) or not self.items:
            raise ValueError("semantic input batch must be a non-empty tuple")
        if any(not isinstance(item, BehaviorSemanticInput) for item in self.items):
            raise TypeError("semantic input batch contains an invalid item")


@runtime_checkable
class BehaviorSemanticAdapter(Protocol):
    name: str
    fingerprint: ProducerFingerprint
    capabilities: BehaviorAdapterCapability

    async def adapt(
        self,
        payload: object,
    ) -> BehaviorSemanticInput | BehaviorSemanticInputBatch: ...
