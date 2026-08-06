"""供应商无关的 Owner-scoped 语义入口 Adapter 协议。"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from behavior.ingress.model import SemanticRecordInput, SemanticRecordInputBatch
from behavior.ingress.trust import IngressAdapterCapability, ProducerFingerprint
from behavior.owner import ConfirmedOwnerBinding


@runtime_checkable
class SemanticIngressAdapter(Protocol):
    name: str
    fingerprint: ProducerFingerprint
    capabilities: IngressAdapterCapability

    async def adapt(
        self,
        payload: object,
        *,
        owner_binding: ConfirmedOwnerBinding,
    ) -> SemanticRecordInput | SemanticRecordInputBatch: ...


__all__ = ["SemanticIngressAdapter"]
