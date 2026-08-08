"""结构化语义到 Evidence Ledger 的应用编排。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol

from behavior._validation import (
    identifier,
    non_negative_int,
    optional_identifier,
    sha256_digest,
    strict_utc,
    utc_text,
)
from behavior.config import BehaviorConfig
from behavior.errors import (
    BehaviorAdapterError,
    BehaviorEvidenceConflictError,
)
from behavior.evidence.adapter import BehaviorSemanticInput, BehaviorSemanticInputBatch
from behavior.evidence.factory import EvidenceFactory
from behavior.evidence.ledger import BehaviorEvidenceLedger
from behavior.evidence.policy import (
    EvidenceBatchCapacityRejection,
    EvidenceBatchClockRejection,
    EvidencePolicy,
)
from behavior.evidence.provenance import producer_to_dict
from behavior.evidence.raw_payload import RawPayloadCodec
from behavior.evidence.record import BehaviorEvidenceRecord
from behavior.evidence.registry import BehaviorSemanticAdapterRegistry, RegisteredBehaviorAdapter
from foundation.integrity import canonical_digest
from foundation.observability import NullObserver, ObservationEvent, ObservationStatus, Observer

INGRESS_CONTRACT_VERSION = "behavior_evidence_ingress_v1"


class IngressReceiptStatus(str, Enum):
    COMMITTED = "COMMITTED"
    REJECTED = "REJECTED"
    CAPACITY_REJECTED = "CAPACITY_REJECTED"


@dataclass(frozen=True, slots=True)
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
        delivery = sha256_digest(self.delivery_id, "delivery_id")
        request = sha256_digest(self.request_digest, "request_digest")
        name = identifier(self.adapter_name, "adapter_name")
        fingerprint = sha256_digest(self.adapter_fingerprint, "adapter_fingerprint")
        capability = sha256_digest(self.capability_digest, "capability_digest")
        status = IngressReceiptStatus(self.status)
        reason = optional_identifier(self.reason_code, "reason_code")
        indexes = tuple(
            non_negative_int(item, "rejected_item_index")
            for item in self.rejected_item_indexes
        )
        records = tuple(identifier(item, "evidence_record_id") for item in self.evidence_record_ids)
        if indexes != tuple(sorted(set(indexes))) or len(records) != len(set(records)):
            raise ValueError("Receipt members must be ordered and unique")
        if status is IngressReceiptStatus.COMMITTED:
            if reason is not None or indexes or not records:
                raise ValueError("COMMITTED receipt requires records only")
        elif records or reason is None:
            raise ValueError("rejected receipt requires rejection metadata only")
        recorded_at = strict_utc(self.recorded_at, "receipt.recorded_at")
        body = {
            "adapter_fingerprint": fingerprint,
            "adapter_name": name,
            "capability_digest": capability,
            "delivery_id": delivery,
            "evidence_record_ids": records,
            "reason_code": reason,
            "recorded_at": utc_text(recorded_at),
            "rejected_item_indexes": indexes,
            "request_digest": request,
            "status": status.value,
        }
        for field_name, value in (
            ("delivery_id", delivery),
            ("request_digest", request),
            ("adapter_name", name),
            ("adapter_fingerprint", fingerprint),
            ("capability_digest", capability),
            ("status", status),
            ("reason_code", reason),
            ("rejected_item_indexes", indexes),
            ("evidence_record_ids", records),
            ("recorded_at", recorded_at),
            ("content_digest", canonical_digest(body)),
        ):
            object.__setattr__(self, field_name, value)


@dataclass(frozen=True, slots=True)
class BehaviorEvidenceIngressResult:
    receipt: BehaviorEvidenceIngressReceipt
    records: tuple[BehaviorEvidenceRecord, ...]
    reused: bool

    def __post_init__(self) -> None:
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
        raw_payload_codec: RawPayloadCodec | None = None,
        evidence_policy: EvidencePolicy | None = None,
        evidence_factory: EvidenceFactory | None = None,
    ) -> None:
        self.ledger = ledger
        self.adapters = adapters
        self.config = config
        self.clock = clock or SystemClock()
        self.observer = observer or NullObserver()
        self.raw_payload_codec = raw_payload_codec or RawPayloadCodec(config)
        self.evidence_policy = evidence_policy or EvidencePolicy(config.evidence)
        self.evidence_factory = evidence_factory or EvidenceFactory()

    async def ingest(
        self,
        adapter_name: str,
        payload: object,
        *,
        delivery_id: str,
    ) -> BehaviorEvidenceIngressResult:
        started = time.monotonic()
        delivery = sha256_digest(delivery_id, "delivery_id")
        registered = self.adapters.resolve(adapter_name)
        snapshot = self.raw_payload_codec.snapshot(payload)
        request_digest = self._request_digest(snapshot.detached_copy(), registered)
        replay = self.ledger.read_ingress_receipt(delivery)
        if replay is not None:
            return self._resolve_replay(replay, request_digest, started)

        adapted = await self._adapt(registered, snapshot.detached_copy())
        items = adapted.items if isinstance(adapted, BehaviorSemanticInputBatch) else (adapted,)
        now = strict_utc(self.clock.now(), "clock.now")
        try:
            validated = self.evidence_policy.validate_batch(
                items,
                registered.capability,
                producer_digest=registered.fingerprint.digest,
                now=now,
            )
        except EvidenceBatchCapacityRejection as exc:
            return self._reject(
                registered,
                delivery,
                request_digest,
                now,
                IngressReceiptStatus.CAPACITY_REJECTED,
                "BATCH_CAPACITY",
                exc.indexes,
                started,
            )
        except EvidenceBatchClockRejection as exc:
            return self._reject(
                registered,
                delivery,
                request_digest,
                now,
                IngressReceiptStatus.REJECTED,
                "CLOCK_POLICY",
                exc.indexes,
                started,
            )

        records = self.evidence_factory.create_batch(
            validated,
            adapter_name=registered.name,
            producer=registered.fingerprint,
            capability_digest=registered.capability.digest,
            ingested_at=now,
        )
        committed = self._receipt(
            registered,
            delivery,
            request_digest,
            now,
            IngressReceiptStatus.COMMITTED,
            None,
            (),
            tuple(record.evidence_record_id for record in records),
        )
        capacity = self._receipt(
            registered,
            delivery,
            request_digest,
            now,
            IngressReceiptStatus.CAPACITY_REJECTED,
            "STORE_CAPACITY",
            tuple(range(len(records))),
            (),
        )
        stored, reused = self.ledger.append_delivery(
            records,
            committed,
            capacity_receipt=capacity,
        )
        resolved = () if stored.status is IngressReceiptStatus.CAPACITY_REJECTED else self._records(stored)
        operation = (
            "behavior_evidence_capacity_rejected"
            if stored.status is IngressReceiptStatus.CAPACITY_REJECTED
            else "behavior_evidence_replayed" if reused else "behavior_evidence_committed"
        )
        self._observe(operation, started, stored, len(resolved))
        return BehaviorEvidenceIngressResult(stored, resolved, reused)

    async def _adapt(
        self,
        registered: RegisteredBehaviorAdapter,
        payload: object,
    ) -> BehaviorSemanticInput | BehaviorSemanticInputBatch:
        try:
            return await registered.adapter.adapt(payload)
        except Exception as exc:
            from behavior.errors import BehaviorAdapterCapabilityError, BehaviorEvidenceSchemaError

            if isinstance(exc, BehaviorAdapterCapabilityError | BehaviorEvidenceSchemaError):
                raise
            raise BehaviorAdapterError(
                f"Behavior Adapter failed with {type(exc).__name__}"
            ) from exc

    def _resolve_replay(
        self,
        receipt: BehaviorEvidenceIngressReceipt,
        request_digest: str,
        started: float,
    ) -> BehaviorEvidenceIngressResult:
        if receipt.request_digest != request_digest:
            raise BehaviorEvidenceConflictError(
                "delivery identity already belongs to another request"
            )
        records = self._records(receipt)
        self._observe("behavior_evidence_replayed", started, receipt, len(records))
        return BehaviorEvidenceIngressResult(receipt, records, True)

    def _reject(
        self,
        registered: RegisteredBehaviorAdapter,
        delivery: str,
        request_digest: str,
        now: datetime,
        status: IngressReceiptStatus,
        reason: str,
        indexes: tuple[int, ...],
        started: float,
    ) -> BehaviorEvidenceIngressResult:
        receipt = self._receipt(
            registered,
            delivery,
            request_digest,
            now,
            status,
            reason,
            indexes,
            (),
        )
        stored, reused = self.ledger.append_delivery((), receipt)
        operation = (
            "behavior_evidence_capacity_rejected"
            if status is IngressReceiptStatus.CAPACITY_REJECTED
            else "behavior_evidence_rejected"
        )
        self._observe(operation, started, stored, 0)
        return BehaviorEvidenceIngressResult(stored, (), reused)

    def _records(
        self,
        receipt: BehaviorEvidenceIngressReceipt,
    ) -> tuple[BehaviorEvidenceRecord, ...]:
        records: list[BehaviorEvidenceRecord] = []
        for record_id in receipt.evidence_record_ids:
            record = self.ledger.read(record_id)
            if record is None:
                raise BehaviorEvidenceConflictError(
                    "committed receipt references missing Evidence"
                )
            records.append(record)
        return tuple(records)

    @staticmethod
    def _request_digest(raw_payload: object, registered: RegisteredBehaviorAdapter) -> str:
        return canonical_digest(
            {
                "adapter_fingerprint": producer_to_dict(registered.fingerprint),
                "adapter_name": registered.name,
                "capability_digest": registered.capability.digest,
                "contract_version": INGRESS_CONTRACT_VERSION,
                "raw_payload": raw_payload,
            }
        )

    @staticmethod
    def _receipt(
        registered: RegisteredBehaviorAdapter,
        delivery: str,
        request_digest: str,
        now: datetime,
        status: IngressReceiptStatus,
        reason: str | None,
        indexes: tuple[int, ...],
        record_ids: tuple[str, ...],
    ) -> BehaviorEvidenceIngressReceipt:
        return BehaviorEvidenceIngressReceipt(
            delivery,
            request_digest,
            registered.name,
            registered.fingerprint.digest,
            registered.capability.digest,
            status,
            reason,
            indexes,
            record_ids,
            now,
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
