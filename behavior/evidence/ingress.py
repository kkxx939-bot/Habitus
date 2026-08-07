"""结构化语义输入到不可变 Evidence Ledger 的原子入口。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Protocol

from behavior._validation import (
    identifier,
    json_value_snapshot,
    non_negative_int,
    optional_identifier,
    sha256_digest,
    strict_utc,
    utc_text,
)
from behavior.config import BehaviorConfig
from behavior.errors import (
    BehaviorAdapterCapabilityError,
    BehaviorAdapterError,
    BehaviorEvidenceClockError,
    BehaviorEvidenceConflictError,
    BehaviorEvidenceSchemaError,
)
from behavior.evidence.adapter import (
    BehaviorSemanticAdapter,
    BehaviorSemanticInput,
    BehaviorSemanticInputBatch,
)
from behavior.evidence.content import BehaviorSemanticContent
from behavior.evidence.ledger import BehaviorEvidenceLedger
from behavior.evidence.payloads import payload_to_dict, validate_payload_capacity
from behavior.evidence.provenance import (
    BehaviorOriginKind,
    BehaviorSourceDescriptor,
    BehaviorSourceProvenance,
    ProducerFingerprint,
    ProducerImplementationKind,
    producer_to_dict,
)
from behavior.evidence.record import BehaviorEvidenceRecord
from behavior.evidence.registry import BehaviorSemanticAdapterRegistry
from behavior.evidence.trust import BehaviorAdapterCapability, BehaviorTimeMode
from foundation.integrity import canonical_digest
from foundation.observability import NullObserver, ObservationEvent, ObservationStatus, Observer

INGRESS_CONTRACT_VERSION = "behavior_evidence_ingress_v1"


class IngressReceiptStatus(str, Enum):
    COMMITTED = "COMMITTED"
    REJECTED = "REJECTED"
    CAPACITY_REJECTED = "CAPACITY_REJECTED"


@dataclass(frozen=True)
class BehaviorEvidenceIngressReceipt:
    delivery_id: str
    request_digest: str
    adapter_name: str
    adapter_fingerprint: str
    capability_digest: str
    status: IngressReceiptStatus
    reason_code: str | None
    rejected_item_indexes: tuple[int, ...]
    evidence_record_ids: tuple[str, ...]
    recorded_at: datetime
    content_digest: str = field(init=False)

    def __post_init__(self) -> None:
        delivery_id = sha256_digest(self.delivery_id, "delivery_id")
        request_digest = sha256_digest(self.request_digest, "request_digest")
        adapter_name = identifier(self.adapter_name, "adapter_name")
        adapter_fingerprint = sha256_digest(self.adapter_fingerprint, "adapter_fingerprint")
        capability_digest = sha256_digest(self.capability_digest, "capability_digest")
        status = IngressReceiptStatus(self.status)
        reason = optional_identifier(self.reason_code, "reason_code")
        if not isinstance(self.rejected_item_indexes, tuple):
            raise TypeError("rejected_item_indexes must be a tuple")
        indexes = tuple(non_negative_int(item, "rejected_item_index") for item in self.rejected_item_indexes)
        if indexes != tuple(sorted(set(indexes))):
            raise ValueError("rejected_item_indexes must be sorted and unique")
        if not isinstance(self.evidence_record_ids, tuple):
            raise TypeError("evidence_record_ids must be a tuple")
        record_ids = tuple(identifier(item, "evidence_record_id") for item in self.evidence_record_ids)
        if len(record_ids) != len(set(record_ids)):
            raise ValueError("evidence_record_ids must be unique")
        if status is IngressReceiptStatus.COMMITTED:
            if reason is not None or indexes or not record_ids:
                raise ValueError("COMMITTED receipt requires records and no rejection metadata")
        elif record_ids or reason is None:
            raise ValueError("rejected receipt requires a reason and no records")
        recorded_at = strict_utc(self.recorded_at, "receipt.recorded_at")
        body = {
            "adapter_fingerprint": adapter_fingerprint,
            "adapter_name": adapter_name,
            "capability_digest": capability_digest,
            "delivery_id": delivery_id,
            "evidence_record_ids": record_ids,
            "reason_code": reason,
            "recorded_at": utc_text(recorded_at),
            "rejected_item_indexes": indexes,
            "request_digest": request_digest,
            "status": status.value,
        }
        object.__setattr__(self, "delivery_id", delivery_id)
        object.__setattr__(self, "request_digest", request_digest)
        object.__setattr__(self, "adapter_name", adapter_name)
        object.__setattr__(self, "adapter_fingerprint", adapter_fingerprint)
        object.__setattr__(self, "capability_digest", capability_digest)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "reason_code", reason)
        object.__setattr__(self, "rejected_item_indexes", indexes)
        object.__setattr__(self, "evidence_record_ids", record_ids)
        object.__setattr__(self, "recorded_at", recorded_at)
        object.__setattr__(self, "content_digest", canonical_digest(body))


@dataclass(frozen=True)
class BehaviorEvidenceIngressResult:
    receipt: BehaviorEvidenceIngressReceipt
    records: tuple[BehaviorEvidenceRecord, ...]
    reused: bool

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, BehaviorEvidenceIngressReceipt):
            raise TypeError("receipt must be BehaviorEvidenceIngressReceipt")
        if not isinstance(self.records, tuple) or any(not isinstance(item, BehaviorEvidenceRecord) for item in self.records):
            raise TypeError("records must contain BehaviorEvidenceRecord values")
        if not isinstance(self.reused, bool):
            raise TypeError("reused must be boolean")
        if tuple(record.evidence_record_id for record in self.records) != self.receipt.evidence_record_ids:
            raise ValueError("result records do not match the receipt")


class Clock(Protocol):
    def now(self) -> datetime: ...


class SystemClock:
    def now(self) -> datetime:
        return datetime.now(timezone.utc)


class BehaviorEvidenceIngressService:
    def __init__(
        self,
        ledger: BehaviorEvidenceLedger,
        adapters: BehaviorSemanticAdapterRegistry,
        *,
        config: BehaviorConfig,
        clock: Clock | None = None,
        observer: Observer | None = None,
    ) -> None:
        if not isinstance(adapters, BehaviorSemanticAdapterRegistry):
            raise TypeError("adapters must be BehaviorSemanticAdapterRegistry")
        if not isinstance(config, BehaviorConfig):
            raise TypeError("config must be BehaviorConfig")
        if not callable(getattr(ledger, "append_delivery", None)):
            raise TypeError("ledger must implement BehaviorEvidenceLedger")
        self.ledger = ledger
        self.adapters = adapters
        self.config = config
        self.clock = clock or SystemClock()
        self.observer = observer or NullObserver()

    async def ingest(
        self,
        adapter_name: str,
        payload: object,
        *,
        delivery_id: str,
    ) -> BehaviorEvidenceIngressResult:
        started = time.monotonic()
        delivery = sha256_digest(delivery_id, "delivery_id")
        adapter = self.adapters.get(adapter_name)
        registered_name, producer, capability = self._adapter_snapshot(adapter)
        try:
            raw = json_value_snapshot(
                payload,
                "raw_payload",
                maximum_chars=self.config.store.max_json_bytes,
                maximum_items=max(
                    self.config.evidence.max_payload_items,
                    self.config.evidence.max_batch_size,
                ),
                maximum_depth=self.config.evidence.max_payload_depth,
            )
        except (TypeError, ValueError) as exc:
            raise BehaviorEvidenceSchemaError("raw payload is not canonical bounded JSON") from exc
        request_digest = canonical_digest(
            {
                "adapter_fingerprint": producer_to_dict(producer),
                "adapter_name": registered_name,
                "capability_digest": capability.digest,
                "contract_version": INGRESS_CONTRACT_VERSION,
                "raw_payload": raw,
            }
        )
        prior = self.ledger.read_ingress_receipt(delivery)
        if prior is not None:
            if prior.request_digest != request_digest:
                raise BehaviorEvidenceConflictError("delivery identity already belongs to another request")
            replayed_records = self._read_receipt_records(prior)
            self._observe("behavior_evidence_replayed", started, prior, len(replayed_records))
            return BehaviorEvidenceIngressResult(prior, replayed_records, True)
        try:
            adapted = await adapter.adapt(payload)
        except Exception as exc:
            if isinstance(exc, (BehaviorEvidenceSchemaError, BehaviorAdapterCapabilityError)):
                raise
            raise BehaviorAdapterError(f"Behavior Adapter failed with {type(exc).__name__}") from exc
        if (
            adapter.name != registered_name
            or adapter.fingerprint != producer
            or adapter.capabilities != capability
        ):
            raise BehaviorAdapterError("Behavior Adapter metadata changed during adaptation")
        items = adapted.items if isinstance(adapted, BehaviorSemanticInputBatch) else (adapted,)
        if any(not isinstance(item, BehaviorSemanticInput) for item in items):
            raise BehaviorAdapterError("Behavior Adapter returned an unsupported value")
        if len(items) > capability.maximum_batch_size:
            raise BehaviorAdapterCapabilityError("Adapter exceeded its declared batch capacity")
        now = strict_utc(self.clock.now(), "clock.now")
        if len(items) > self.config.evidence.max_batch_size:
            receipt = self._receipt(
                delivery,
                request_digest,
                registered_name,
                producer.digest,
                capability.digest,
                IngressReceiptStatus.CAPACITY_REJECTED,
                "BATCH_CAPACITY",
                tuple(range(len(items))),
                (),
                now,
            )
            stored, reused = self.ledger.append_delivery((), receipt)
            self._observe("behavior_evidence_capacity_rejected", started, stored, 0)
            return BehaviorEvidenceIngressResult(stored, (), reused)
        self._validate_batch_identity(items, producer.digest)
        pending_records: list[BehaviorEvidenceRecord] = []
        clock_rejections: list[int] = []
        for index, item in enumerate(items):
            self._validate_item(item.content, item.source, capability)
            try:
                self._validate_clock(item.content, capability.time_mode, now)
            except BehaviorEvidenceClockError:
                clock_rejections.append(index)
            provenance = BehaviorSourceProvenance(
                descriptor=item.source,
                adapter_name=registered_name,
                producer_fingerprint=producer,
                capability_digest=capability.digest,
            )
            pending_records.append(
                BehaviorEvidenceRecord(
                    semantic_content=item.content,
                    provenance=provenance,
                    source_trust=capability.source_trust,
                    ingested_at=now,
                )
            )
        if clock_rejections:
            receipt = self._receipt(
                delivery,
                request_digest,
                registered_name,
                producer.digest,
                capability.digest,
                IngressReceiptStatus.REJECTED,
                "CLOCK_POLICY",
                tuple(clock_rejections),
                (),
                now,
            )
            stored, reused = self.ledger.append_delivery((), receipt)
            self._observe("behavior_evidence_rejected", started, stored, 0)
            return BehaviorEvidenceIngressResult(stored, (), reused)
        committed = self._receipt(
            delivery,
            request_digest,
            registered_name,
            producer.digest,
            capability.digest,
            IngressReceiptStatus.COMMITTED,
            None,
            (),
            tuple(record.evidence_record_id for record in pending_records),
            now,
        )
        capacity = self._receipt(
            delivery,
            request_digest,
            registered_name,
            producer.digest,
            capability.digest,
            IngressReceiptStatus.CAPACITY_REJECTED,
            "STORE_CAPACITY",
            tuple(range(len(pending_records))),
            (),
            now,
        )
        stored, reused = self.ledger.append_delivery(
            tuple(pending_records),
            committed,
            capacity_receipt=capacity,
        )
        if stored.status is IngressReceiptStatus.CAPACITY_REJECTED:
            self._observe("behavior_evidence_capacity_rejected", started, stored, 0)
            return BehaviorEvidenceIngressResult(stored, (), reused)
        resolved_records = self._read_receipt_records(stored)
        operation = "behavior_evidence_replayed" if reused else "behavior_evidence_committed"
        self._observe(operation, started, stored, len(resolved_records))
        return BehaviorEvidenceIngressResult(stored, resolved_records, reused)

    def _validate_item(
        self,
        content: BehaviorSemanticContent,
        source: BehaviorSourceDescriptor,
        capability: BehaviorAdapterCapability,
    ) -> None:
        if source.origin_kind is BehaviorOriginKind.CONVERSATION_PROJECTION:
            raise BehaviorAdapterCapabilityError("external Adapter cannot emit conversation projection origin")
        if not capability.permits(
            origin_kind=source.origin_kind,
            record_kind=content.record_kind,
            modality=content.modality,
            subject_role=content.subject_role,
            actor_role=content.actor_role,
        ):
            raise BehaviorAdapterCapabilityError("Adapter output exceeds its declared capability")
        limits = self.config.evidence
        if len(content.evidence_refs) > limits.max_evidence_refs:
            raise BehaviorEvidenceSchemaError("too many Evidence references")
        if len(content.object_refs) > limits.max_object_refs or len(content.entity_refs) > limits.max_entity_refs:
            raise BehaviorEvidenceSchemaError("semantic reference capacity exceeded")
        if content.event_time_uncertainty_ms > limits.max_event_time_uncertainty_ms:
            raise BehaviorEvidenceSchemaError("event time uncertainty exceeds the configured boundary")
        if len(source.parent_source_event_refs) > limits.max_parent_source_refs:
            raise BehaviorEvidenceSchemaError("too many parent source references")
        if len(source.correlation_refs) > limits.max_correlation_refs:
            raise BehaviorEvidenceSchemaError("too many correlation references")
        if len(source.causal_refs) > limits.max_causal_refs:
            raise BehaviorEvidenceSchemaError("too many causal references")
        if source.source_ref is not None and len(source.source_ref) > limits.max_reference_chars:
            raise BehaviorEvidenceSchemaError("source reference exceeds the configured boundary")
        for reference in content.evidence_refs:
            if len(reference.reference) > limits.max_reference_chars:
                raise BehaviorEvidenceSchemaError("Evidence reference exceeds the configured boundary")
            if (
                reference.source_system_ref is not None
                and len(reference.source_system_ref) > limits.max_reference_chars
            ):
                raise BehaviorEvidenceSchemaError(
                    "Evidence source-system reference exceeds the configured boundary"
                )
            if reference.media_type is not None and len(reference.media_type) > limits.max_identifier_chars:
                raise BehaviorEvidenceSchemaError("Evidence media type exceeds the configured boundary")
        for causal_ref in source.causal_refs:
            if len(causal_ref.reference) > limits.max_reference_chars:
                raise BehaviorEvidenceSchemaError("causal reference exceeds the configured boundary")
        identifier_values = [
            content.clock_domain,
            content.scene_ref,
            content.location_ref,
            *content.object_refs,
            *content.entity_refs,
            source.source_event_ref.namespace,
            source.source_event_ref.value,
            source.stream_ref.namespace,
            source.stream_ref.value,
        ]
        for parent in source.parent_source_event_refs:
            identifier_values.extend((parent.namespace, parent.value))
        for correlation in source.correlation_refs:
            identifier_values.extend(
                (correlation.namespace, correlation.value, correlation.root_value)
            )
        if any(
            value is not None and len(value) > limits.max_identifier_chars
            for value in identifier_values
        ):
            raise BehaviorEvidenceSchemaError("semantic identifier exceeds the configured boundary")
        payload = payload_to_dict(content.payload)
        for name in (
            "activity",
            "language",
            "state_name",
            "interaction_type",
            "action_name",
            "phase",
            "capability_ref",
            "target_ref",
            "tool_name",
            "tool_call_id",
            "predicate",
            "coverage_scope_ref",
            "feedback_kind",
        ):
            value = payload.get(name)
            if isinstance(value, str) and len(value) > limits.max_identifier_chars:
                raise BehaviorEvidenceSchemaError(
                    f"payload.{name} exceeds the configured identifier boundary"
                )
        labels = payload.get("labels")
        if isinstance(labels, tuple) and any(
            len(label) > limits.max_identifier_chars for label in labels
        ):
            raise BehaviorEvidenceSchemaError("payload.labels exceed the configured identifier boundary")
        for name in ("result_ref", "explicit_text_ref"):
            value = payload.get(name)
            if isinstance(value, str) and len(value) > limits.max_reference_chars:
                raise BehaviorEvidenceSchemaError(
                    f"payload.{name} exceeds the configured reference boundary"
                )
        try:
            validate_payload_capacity(content.payload, limits)
        except (TypeError, ValueError) as exc:
            raise BehaviorEvidenceSchemaError("semantic payload exceeds the configured boundary") from exc

    def _adapter_snapshot(
        self,
        adapter: BehaviorSemanticAdapter,
    ) -> tuple[str, ProducerFingerprint, BehaviorAdapterCapability]:
        name = identifier(adapter.name, "adapter.name")
        producer = adapter.fingerprint
        capability = adapter.capabilities
        if not isinstance(producer, ProducerFingerprint):
            raise BehaviorAdapterCapabilityError("Adapter fingerprint is invalid")
        if not isinstance(capability, BehaviorAdapterCapability):
            raise BehaviorAdapterCapabilityError("Adapter capability is invalid")
        if producer.implementation_kind is ProducerImplementationKind.PROJECTOR:
            raise BehaviorAdapterCapabilityError("external Adapter cannot use PROJECTOR fingerprint")
        if BehaviorOriginKind.CONVERSATION_PROJECTION in capability.allowed_origin_kinds:
            raise BehaviorAdapterCapabilityError(
                "external Adapter cannot declare conversation projection origin"
            )
        limit = self.config.evidence.max_identifier_chars
        values = (
            name,
            producer.producer_name,
            producer.producer_version,
            producer.pipeline_version,
            producer.output_schema_version,
            producer.model_provider,
            producer.model_name,
            producer.prompt_version,
        )
        if any(value is not None and len(value) > limit for value in values):
            raise BehaviorAdapterCapabilityError(
                "Adapter metadata exceeds the configured identifier boundary"
            )
        return name, producer, capability

    def _validate_clock(
        self,
        content: BehaviorSemanticContent,
        mode: BehaviorTimeMode,
        now: datetime,
    ) -> None:
        uncertainty = timedelta(milliseconds=content.event_time_uncertainty_ms)
        earliest = content.event_time_start - uncertainty
        latest = content.event_time_end + uncertainty
        future_limit = now + timedelta(seconds=self.config.evidence.max_future_event_skew_seconds)
        if earliest > future_limit:
            raise BehaviorEvidenceClockError("event time is entirely beyond the allowed future skew")
        if mode is BehaviorTimeMode.LIVE:
            past_limit = now - timedelta(seconds=self.config.evidence.max_live_event_age_seconds)
            if latest < past_limit:
                raise BehaviorEvidenceClockError("LIVE event time is entirely older than the allowed age")

    @staticmethod
    def _validate_batch_identity(items: tuple[BehaviorSemanticInput, ...], producer_digest: str) -> None:
        first_keys: set[tuple[str, str, str, int]] = set()
        second_keys: set[tuple[str, str, str, int, int, int]] = set()
        for item in items:
            source = item.source
            first = (
                producer_digest,
                source.source_event_ref.namespace,
                source.source_event_ref.value,
                source.source_item_index,
            )
            second = (
                producer_digest,
                source.stream_ref.namespace,
                source.stream_ref.value,
                source.stream_ref.generation,
                source.source_sequence,
                source.source_item_index,
            )
            if first in first_keys or second in second_keys:
                raise BehaviorEvidenceSchemaError("batch contains duplicate Evidence identities")
            first_keys.add(first)
            second_keys.add(second)

    def _read_receipt_records(
        self,
        receipt: BehaviorEvidenceIngressReceipt,
    ) -> tuple[BehaviorEvidenceRecord, ...]:
        records: list[BehaviorEvidenceRecord] = []
        for record_id in receipt.evidence_record_ids:
            record = self.ledger.read(record_id)
            if record is None:
                raise BehaviorEvidenceConflictError("committed receipt references a missing Evidence record")
            records.append(record)
        return tuple(records)

    @staticmethod
    def _receipt(
        delivery_id: str,
        request_digest: str,
        adapter_name: str,
        adapter_fingerprint: str,
        capability_digest: str,
        status: IngressReceiptStatus,
        reason_code: str | None,
        rejected_item_indexes: tuple[int, ...],
        evidence_record_ids: tuple[str, ...],
        recorded_at: datetime,
    ) -> BehaviorEvidenceIngressReceipt:
        return BehaviorEvidenceIngressReceipt(
            delivery_id=delivery_id,
            request_digest=request_digest,
            adapter_name=adapter_name,
            adapter_fingerprint=adapter_fingerprint,
            capability_digest=capability_digest,
            status=status,
            reason_code=reason_code,
            rejected_item_indexes=rejected_item_indexes,
            evidence_record_ids=evidence_record_ids,
            recorded_at=recorded_at,
        )

    def _observe(
        self,
        operation: str,
        started: float,
        receipt: BehaviorEvidenceIngressReceipt,
        count: int,
    ) -> None:
        attributes: dict[str, str | int | float | bool] = {
            "count": count,
            "status": receipt.status.value,
        }
        if receipt.reason_code is not None:
            attributes["error_code"] = receipt.reason_code
        try:
            self.observer.record(
                ObservationEvent(
                    category="behavior",
                    operation=operation,
                    status=(
                        ObservationStatus.SUCCESS
                        if receipt.status is IngressReceiptStatus.COMMITTED
                        else ObservationStatus.DEGRADED
                    ),
                    duration_seconds=max(0.0, time.monotonic() - started),
                    attributes=attributes,
                )
            )
        except Exception:
            return


def ingress_receipt_to_dict(value: BehaviorEvidenceIngressReceipt) -> dict[str, object]:
    return {
        "delivery_id": value.delivery_id,
        "request_digest": value.request_digest,
        "adapter_name": value.adapter_name,
        "adapter_fingerprint": value.adapter_fingerprint,
        "capability_digest": value.capability_digest,
        "status": value.status.value,
        "reason_code": value.reason_code,
        "rejected_item_indexes": value.rejected_item_indexes,
        "evidence_record_ids": value.evidence_record_ids,
        "recorded_at": utc_text(value.recorded_at),
        "content_digest": value.content_digest,
    }


__all__ = [
    "BehaviorEvidenceIngressReceipt",
    "BehaviorEvidenceIngressResult",
    "BehaviorEvidenceIngressService",
    "IngressReceiptStatus",
]
