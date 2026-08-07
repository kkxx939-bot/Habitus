"""不可变 Evidence Ledger 的领域协议。"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from behavior.evidence.content import BehaviorRecordKind
from behavior.evidence.record import BehaviorEvidenceLedgerEntry, BehaviorEvidenceRecord
from behavior.evidence.refs import CorrelationRef, SourceEventRef

if TYPE_CHECKING:
    from behavior.evidence.ingress import BehaviorEvidenceIngressReceipt

EvidencePage = tuple[tuple[BehaviorEvidenceLedgerEntry, ...], str | None]


@runtime_checkable
class BehaviorEvidenceLedger(Protocol):
    def append_delivery(
        self,
        records: tuple[BehaviorEvidenceRecord, ...],
        receipt: BehaviorEvidenceIngressReceipt,
        *,
        capacity_receipt: BehaviorEvidenceIngressReceipt | None = None,
    ) -> tuple[BehaviorEvidenceIngressReceipt, bool]: ...

    def read(self, record_id: str) -> BehaviorEvidenceRecord | None: ...

    def read_entry(self, record_id: str) -> BehaviorEvidenceLedgerEntry | None: ...

    def list_after_sequence(self, sequence: int, limit: int) -> tuple[BehaviorEvidenceLedgerEntry, ...]: ...

    def list_by_event_time(
        self,
        start: datetime,
        end: datetime,
        limit: int,
        cursor: str | None = None,
    ) -> EvidencePage: ...

    def list_by_source_event(
        self,
        source_event_ref: SourceEventRef,
        limit: int,
        cursor: str | None = None,
    ) -> EvidencePage: ...

    def list_by_correlation(
        self,
        correlation_ref: CorrelationRef,
        limit: int,
        cursor: str | None = None,
    ) -> EvidencePage: ...

    def list_by_record_kind(
        self,
        record_kind: BehaviorRecordKind,
        start: datetime,
        end: datetime,
        limit: int,
        cursor: str | None = None,
    ) -> EvidencePage: ...

    def read_ingress_receipt(self, delivery_id: str) -> BehaviorEvidenceIngressReceipt | None: ...


__all__ = ["BehaviorEvidenceLedger", "BehaviorEvidenceLedgerEntry"]
