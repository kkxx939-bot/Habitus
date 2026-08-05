"""来源 Adapter 调用与 Store 绑定的应用边界。"""

from __future__ import annotations

from behavior.config import BehaviorConfig, SourceConfig
from behavior.owner import ConfirmedOwnerBinding
from behavior.persistence.contracts import BehaviorEvidenceClaimStore
from behavior.source.adapter import BehaviorSourceAdapterRegistry
from behavior.source.model import SourceRecord, SourceRecordBatch


class SourceRecordService:
    def __init__(
        self,
        store: BehaviorEvidenceClaimStore,
        adapters: BehaviorSourceAdapterRegistry | None = None,
        *,
        config: SourceConfig | None = None,
    ) -> None:
        if not isinstance(store, BehaviorEvidenceClaimStore):
            raise TypeError("store must implement BehaviorEvidenceClaimStore")
        if adapters is not None and not isinstance(adapters, BehaviorSourceAdapterRegistry):
            raise TypeError("adapters must be BehaviorSourceAdapterRegistry or None")
        store_config = getattr(store, "config", None)
        inferred = store_config.source if isinstance(store_config, BehaviorConfig) else SourceConfig()
        resolved_config = inferred if config is None else config
        if not isinstance(resolved_config, SourceConfig):
            raise TypeError("config must be SourceConfig")
        self.store = store
        self.adapters = adapters or BehaviorSourceAdapterRegistry()
        self.config = resolved_config

    async def adapt(
        self,
        adapter_name: str,
        payload: object,
        *,
        owner_binding: ConfirmedOwnerBinding,
    ) -> SourceRecordBatch:
        if not isinstance(owner_binding, ConfirmedOwnerBinding):
            raise TypeError("owner_binding must be ConfirmedOwnerBinding")
        result = await self.adapters.get(adapter_name).adapt(payload, owner_binding=owner_binding)
        if isinstance(result, SourceRecord):
            return SourceRecordBatch((result,), config=self.config)
        if isinstance(result, SourceRecordBatch):
            return SourceRecordBatch(result.records, config=self.config)
        raise TypeError("source adapter returned an unsupported value")


__all__ = ["SourceRecordService"]
