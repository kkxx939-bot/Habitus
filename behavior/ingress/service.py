"""Adapter 调用、信任绑定、Clock 校验与语义记录构造。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol, runtime_checkable

from behavior.config import BehaviorConfig, IngressConfig
from behavior.errors import SemanticClockError, SemanticIngressError
from behavior.ingress.model import (
    ClockSyncStatus,
    IngressDecision,
    IngressDecisionStatus,
    OwnerScopedSemanticRecord,
    SemanticRecordInput,
    SemanticRecordInputBatch,
)
from behavior.ingress.registry import SemanticIngressAdapterRegistry
from behavior.ingress.trust import require_record_trust_compatibility
from behavior.owner import ConfirmedOwnerBinding
from behavior.persistence.contracts import BehaviorEvidenceClaimStore


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


@dataclass(frozen=True)
class SemanticIngressPreparation:
    record: OwnerScopedSemanticRecord | None
    decision: IngressDecision

    def __post_init__(self) -> None:
        if self.record is not None and not isinstance(self.record, OwnerScopedSemanticRecord):
            raise TypeError("record must be OwnerScopedSemanticRecord or None")
        if not isinstance(self.decision, IngressDecision):
            raise TypeError("decision must be IngressDecision")
        rejected = self.decision.status not in {
            IngressDecisionStatus.ACCEPTED,
            IngressDecisionStatus.REPLAYED,
        }
        if rejected != (self.record is None):
            raise ValueError("rejected ingress preparations cannot carry a durable record")


class SemanticRecordService:
    """普通输入永远不能越过 Adapter Capability 创建系统绑定字段。"""

    def __init__(
        self,
        store: BehaviorEvidenceClaimStore,
        adapters: SemanticIngressAdapterRegistry | None = None,
        *,
        config: IngressConfig | None = None,
        clock: Clock | None = None,
    ) -> None:
        if not isinstance(store, BehaviorEvidenceClaimStore):
            raise TypeError("store must implement BehaviorEvidenceClaimStore")
        if adapters is not None and not isinstance(adapters, SemanticIngressAdapterRegistry):
            raise TypeError("adapters must be SemanticIngressAdapterRegistry or None")
        store_config = getattr(store, "config", None)
        inferred = store_config.ingress if isinstance(store_config, BehaviorConfig) else IngressConfig()
        resolved_config = inferred if config is None else config
        if not isinstance(resolved_config, IngressConfig):
            raise TypeError("config must be IngressConfig")
        resolved_clock = clock or SystemClock()
        if not isinstance(resolved_clock, Clock):
            raise TypeError("clock must implement Clock")
        self.store = store
        self.adapters = adapters or SemanticIngressAdapterRegistry()
        self.config = resolved_config
        self.clock = resolved_clock

    async def prepare(
        self,
        adapter_name: str,
        payload: object,
        *,
        owner_binding: ConfirmedOwnerBinding,
    ) -> tuple[SemanticIngressPreparation, ...]:
        if not isinstance(owner_binding, ConfirmedOwnerBinding):
            raise TypeError("owner_binding must be ConfirmedOwnerBinding")
        adapter = self.adapters.get(adapter_name)
        adapted = await adapter.adapt(payload, owner_binding=owner_binding)
        raw_records: tuple[SemanticRecordInput, ...]
        if isinstance(adapted, SemanticRecordInput):
            raw_records = (adapted,)
        elif isinstance(adapted, SemanticRecordInputBatch):
            raw_records = adapted.records
        else:
            raise TypeError("semantic Adapter returned an unsupported value")
        maximum = min(self.config.max_batch_size, adapter.capabilities.maximum_batch_size)
        if not 1 <= len(raw_records) <= maximum:
            raise SemanticIngressError("Adapter batch exceeds its declared capability")
        preparations: list[SemanticIngressPreparation] = []
        for raw in raw_records:
            semantic_input = SemanticRecordInput.model_validate(raw.to_dict(), config=self.config)
            if semantic_input.record_kind not in adapter.capabilities.allowed_record_kinds:
                raise SemanticIngressError("Adapter emitted a record kind outside its capability")
            require_record_trust_compatibility(
                semantic_input.record_kind,
                adapter.capabilities.trust_class,
            )
            ingested_at = self._now()
            record = OwnerScopedSemanticRecord(
                semantic_input=semantic_input,
                owner_binding=owner_binding,
                producer_fingerprint=adapter.fingerprint,
                ingress_trust_class=adapter.capabilities.trust_class,
                ingested_at=ingested_at,
            )
            status, reason = self._clock_decision(record, now=ingested_at)
            decision = IngressDecision(
                status=status,
                reason_code=reason,
                record=record,
                decided_at=self._now(),
            )
            if status is IngressDecisionStatus.ACCEPTED:
                preparations.append(SemanticIngressPreparation(record, decision))
                continue
            self.store.record_ingress_decision(decision, record=record)
            preparations.append(SemanticIngressPreparation(None, decision))
        return tuple(preparations)

    def _clock_decision(
        self,
        record: OwnerScopedSemanticRecord,
        *,
        now: datetime,
    ) -> tuple[IngressDecisionStatus, str]:
        value = record.semantic_input
        future = now + timedelta(seconds=self.config.max_future_event_skew_seconds)
        past = now - timedelta(seconds=self.config.max_past_event_age_seconds)
        uncertainty = timedelta(milliseconds=value.event_time_uncertainty_ms)
        if value.clock_sync_status is ClockSyncStatus.SYNCHRONIZED:
            latest_certain = value.event_time_end
            earliest_certain = value.event_time_start
        else:
            latest_certain = value.event_time_end - uncertainty
            earliest_certain = value.event_time_start + uncertainty
        if latest_certain > future:
            return IngressDecisionStatus.CLOCK_SKEW_REJECTED, "event_time_exceeds_future_skew"
        if earliest_certain < past:
            return IngressDecisionStatus.EVENT_TOO_OLD_REJECTED, "event_time_exceeds_past_age"
        return IngressDecisionStatus.ACCEPTED, "semantic_record_clock_accepted"

    def _now(self) -> datetime:
        value = self.clock.now()
        try:
            from behavior._validation import strict_utc

            return strict_utc(value, "clock.now")
        except (TypeError, ValueError) as exc:
            raise SemanticClockError("Clock returned a non-UTC-aware datetime") from exc


__all__ = ["Clock", "SemanticIngressPreparation", "SemanticRecordService", "SystemClock"]
