"""SQLite-backed append-only Behavior Evidence Ledger。"""

from __future__ import annotations

import sqlite3
from collections.abc import Mapping
from datetime import datetime

from behavior._validation import (
    decode_sequence_cursor,
    encode_sequence_cursor,
    identifier,
    non_negative_int,
    sha256_digest,
    strict_utc,
    utc_text,
)
from behavior.errors import (
    BehaviorEvidenceCapacityError,
    BehaviorEvidenceConflictError,
)
from behavior.evidence.content import BehaviorRecordKind
from behavior.evidence.ingress import (
    BehaviorEvidenceIngressReceipt,
    IngressReceiptStatus,
    ingress_receipt_to_dict,
)
from behavior.evidence.ledger import EvidencePage
from behavior.evidence.record import BehaviorEvidenceLedgerEntry, BehaviorEvidenceRecord, record_to_dict
from behavior.evidence.refs import CorrelationRef, SourceEventRef
from behavior.persistence.codecs import (
    decode_evidence_record,
    decode_ingress_receipt,
    encode_value,
)
from behavior.persistence.database import BehaviorDatabase


class _CapacitySignal(Exception):
    """使当前写事务回滚后改为发布容量 Receipt。"""


class SQLiteBehaviorEvidenceLedger:
    def __init__(self, database: BehaviorDatabase) -> None:
        if not isinstance(database, BehaviorDatabase):
            raise TypeError("database must be BehaviorDatabase")
        self.database = database
        self.config = database.config

    def append_delivery(
        self,
        records: tuple[BehaviorEvidenceRecord, ...],
        receipt: BehaviorEvidenceIngressReceipt,
        *,
        capacity_receipt: BehaviorEvidenceIngressReceipt | None = None,
    ) -> tuple[BehaviorEvidenceIngressReceipt, bool]:
        if not isinstance(records, tuple) or any(not isinstance(item, BehaviorEvidenceRecord) for item in records):
            raise TypeError("records must contain BehaviorEvidenceRecord values")
        if not isinstance(receipt, BehaviorEvidenceIngressReceipt):
            raise TypeError("receipt must be BehaviorEvidenceIngressReceipt")
        if tuple(record.evidence_record_id for record in records) != receipt.evidence_record_ids:
            raise ValueError("records and receipt Evidence identities disagree")
        if receipt.status is not IngressReceiptStatus.COMMITTED and records:
            raise ValueError("rejected receipt cannot publish Evidence records")
        try:
            with self.database.connection.write() as connection:
                prior = self._read_receipt_row(connection, receipt.delivery_id)
                if prior is not None:
                    return self._resolve_receipt_replay(prior, receipt.request_digest)
                self._require_receipt_capacity(connection)
                encoded_records: list[tuple[str, str]] = []
                new_records: tuple[BehaviorEvidenceRecord, ...] = ()
                if receipt.status is IngressReceiptStatus.COMMITTED:
                    new_records = self._resolve_records(connection, records)
                    count = int(connection.execute("SELECT COUNT(*) FROM behavior_evidence_records").fetchone()[0])
                    if count + len(new_records) > self.config.store.max_evidence_records:
                        raise _CapacitySignal()
                    encoded_records = [encode_value(record, record_to_dict) for record in new_records]
                    self._require_encoded_sizes(encoded_records)
                encoded_receipt = encode_value(receipt, ingress_receipt_to_dict)
                self._require_encoded_sizes([encoded_receipt])
                expected_bytes = len(encoded_receipt[0].encode("utf-8")) + sum(
                    len(text.encode("utf-8")) for text, _ in encoded_records
                )
                btree_writes = 2 + sum(
                    8
                    + 3 * len(record.provenance.descriptor.correlation_refs)
                    + 3 * len(record.provenance.descriptor.causal_refs)
                    + 2 * len(record.provenance.descriptor.parent_source_event_refs)
                    for record in new_records
                )
                if (
                    self.database.connection.projected_write_size(
                        connection,
                        encoded_bytes=expected_bytes,
                        btree_writes=btree_writes,
                    )
                    > self.config.store.max_database_bytes
                ):
                    raise _CapacitySignal()
                for record, encoded in zip(new_records, encoded_records, strict=True):
                    self._insert_record(connection, record, encoded)
                self._insert_receipt(connection, receipt, encoded_receipt)
                if self.database.connection.database_size(connection) > self.config.store.max_database_bytes:
                    raise _CapacitySignal()
                return receipt, False
        except _CapacitySignal:
            if capacity_receipt is None:
                raise BehaviorEvidenceCapacityError("Evidence Ledger capacity is exhausted") from None
            return self._publish_capacity_receipt(capacity_receipt)
        except sqlite3.IntegrityError as exc:
            raise BehaviorEvidenceConflictError("Evidence Ledger uniqueness conflict") from exc

    def read(self, record_id: str) -> BehaviorEvidenceRecord | None:
        entry = self.read_entry(record_id)
        return None if entry is None else entry.record

    def read_entry(self, record_id: str) -> BehaviorEvidenceLedgerEntry | None:
        resolved = identifier(record_id, "evidence_record_id")
        with self.database.connection.read() as connection:
            row = connection.execute(
                "SELECT * FROM behavior_evidence_records WHERE evidence_record_id=?",
                (resolved,),
            ).fetchone()
            return None if row is None else self._entry(row)

    def list_after_sequence(self, sequence: int, limit: int) -> tuple[BehaviorEvidenceLedgerEntry, ...]:
        after = non_negative_int(sequence, "evidence_sequence")
        bounded = self._limit(limit)
        with self.database.connection.read() as connection:
            rows = connection.execute(
                "SELECT * FROM behavior_evidence_records WHERE evidence_sequence>? "
                "ORDER BY evidence_sequence ASC LIMIT ?",
                (after, bounded),
            ).fetchall()
            return tuple(self._entry(row) for row in rows)

    def list_by_event_time(
        self,
        start: datetime,
        end: datetime,
        limit: int,
        cursor: str | None = None,
    ) -> EvidencePage:
        start_value = strict_utc(start, "start")
        end_value = strict_utc(end, "end")
        if end_value < start_value:
            raise ValueError("end cannot precede start")
        query = {"end": utc_text(end_value), "start": utc_text(start_value)}
        return self._page(
            "event_time",
            query,
            "event_time_start<=? AND event_time_end>=?",
            (utc_text(end_value), utc_text(start_value)),
            limit,
            cursor,
        )

    def list_by_source_event(
        self,
        source_event_ref: SourceEventRef,
        limit: int,
        cursor: str | None = None,
    ) -> EvidencePage:
        if not isinstance(source_event_ref, SourceEventRef):
            raise TypeError("source_event_ref must be SourceEventRef")
        query = {
            "namespace": source_event_ref.namespace,
            "value": source_event_ref.value,
        }
        return self._page(
            "source_event",
            query,
            "source_event_namespace=? AND source_event_value=?",
            (source_event_ref.namespace, source_event_ref.value),
            limit,
            cursor,
        )

    def list_by_correlation(
        self,
        correlation_ref: CorrelationRef,
        limit: int,
        cursor: str | None = None,
    ) -> EvidencePage:
        if not isinstance(correlation_ref, CorrelationRef):
            raise TypeError("correlation_ref must be CorrelationRef")
        query = {
            "namespace": correlation_ref.namespace,
            "root_value": correlation_ref.root_value,
            "value": correlation_ref.value,
        }
        root_clause = "c.root_value IS NULL" if correlation_ref.root_value is None else "c.root_value=?"
        parameters: list[object] = [correlation_ref.namespace, correlation_ref.value]
        if correlation_ref.root_value is not None:
            parameters.append(correlation_ref.root_value)
        where = (
            "EXISTS (SELECT 1 FROM behavior_evidence_correlations c "
            "WHERE c.evidence_record_id=behavior_evidence_records.evidence_record_id "
            f"AND c.namespace=? AND c.value=? AND {root_clause})"
        )
        return self._page("correlation", query, where, tuple(parameters), limit, cursor)

    def list_by_record_kind(
        self,
        record_kind: BehaviorRecordKind,
        start: datetime,
        end: datetime,
        limit: int,
        cursor: str | None = None,
    ) -> EvidencePage:
        kind = BehaviorRecordKind(record_kind)
        start_value = strict_utc(start, "start")
        end_value = strict_utc(end, "end")
        if end_value < start_value:
            raise ValueError("end cannot precede start")
        query = {
            "end": utc_text(end_value),
            "record_kind": kind.value,
            "start": utc_text(start_value),
        }
        return self._page(
            "record_kind",
            query,
            "record_kind=? AND event_time_start<=? AND event_time_end>=?",
            (kind.value, utc_text(end_value), utc_text(start_value)),
            limit,
            cursor,
        )

    def read_ingress_receipt(self, delivery_id: str) -> BehaviorEvidenceIngressReceipt | None:
        delivery = sha256_digest(delivery_id, "delivery_id")
        with self.database.connection.read() as connection:
            row = self._read_receipt_row(connection, delivery)
            return None if row is None else self._decode_receipt_row(row)

    def _publish_capacity_receipt(
        self,
        receipt: BehaviorEvidenceIngressReceipt,
    ) -> tuple[BehaviorEvidenceIngressReceipt, bool]:
        if receipt.status is not IngressReceiptStatus.CAPACITY_REJECTED:
            raise ValueError("capacity fallback must be CAPACITY_REJECTED")
        try:
            with self.database.connection.write() as connection:
                prior = self._read_receipt_row(connection, receipt.delivery_id)
                if prior is not None:
                    return self._resolve_receipt_replay(prior, receipt.request_digest)
                self._require_receipt_capacity(connection)
                encoded = encode_value(receipt, ingress_receipt_to_dict)
                self._require_encoded_sizes([encoded])
                if self.database.connection.projected_write_size(
                    connection,
                    encoded_bytes=len(encoded[0].encode("utf-8")),
                    btree_writes=2,
                ) > self.config.store.max_database_bytes:
                    raise BehaviorEvidenceCapacityError("capacity Receipt cannot be durably recorded")
                self._insert_receipt(connection, receipt, encoded)
                if self.database.connection.database_size(connection) > self.config.store.max_database_bytes:
                    raise BehaviorEvidenceCapacityError("capacity Receipt exceeds database boundary")
                return receipt, False
        except sqlite3.IntegrityError as exc:
            raise BehaviorEvidenceConflictError("capacity Receipt uniqueness conflict") from exc

    def _resolve_records(
        self,
        connection: sqlite3.Connection,
        records: tuple[BehaviorEvidenceRecord, ...],
    ) -> tuple[BehaviorEvidenceRecord, ...]:
        new: list[BehaviorEvidenceRecord] = []
        for record in records:
            descriptor = record.provenance.descriptor
            producer = record.provenance.producer_fingerprint.digest
            by_source = connection.execute(
                "SELECT * FROM behavior_evidence_records WHERE producer_fingerprint=? AND "
                "source_event_namespace=? AND source_event_value=? AND source_item_index=?",
                (
                    producer,
                    descriptor.source_event_ref.namespace,
                    descriptor.source_event_ref.value,
                    descriptor.source_item_index,
                ),
            ).fetchone()
            by_stream = connection.execute(
                "SELECT * FROM behavior_evidence_records WHERE producer_fingerprint=? AND "
                "stream_namespace=? AND stream_value=? AND stream_generation=? AND "
                "source_sequence=? AND source_item_index=?",
                (
                    producer,
                    descriptor.stream_ref.namespace,
                    descriptor.stream_ref.value,
                    descriptor.stream_ref.generation,
                    descriptor.source_sequence,
                    descriptor.source_item_index,
                ),
            ).fetchone()
            rows = [row for row in (by_source, by_stream) if row is not None]
            if rows and any(row["evidence_record_id"] != rows[0]["evidence_record_id"] for row in rows):
                raise BehaviorEvidenceConflictError("Evidence unique identities resolve to different records")
            by_id = connection.execute(
                "SELECT * FROM behavior_evidence_records WHERE evidence_record_id=?",
                (record.evidence_record_id,),
            ).fetchone()
            if by_id is not None:
                rows.append(by_id)
            if rows:
                existing = self._entry(rows[0]).record
                if existing.semantic_digest != record.semantic_digest:
                    raise BehaviorEvidenceConflictError("Evidence identity replay changed content")
            else:
                new.append(record)
        return tuple(new)

    def _insert_record(
        self,
        connection: sqlite3.Connection,
        record: BehaviorEvidenceRecord,
        encoded: tuple[str, str],
    ) -> None:
        descriptor = record.provenance.descriptor
        content = record.semantic_content
        connection.execute(
            """INSERT INTO behavior_evidence_records(
                evidence_record_id, producer_fingerprint, capability_digest, source_trust, origin_kind,
                source_event_namespace, source_event_value, source_item_index,
                stream_namespace, stream_value, stream_generation, source_sequence,
                record_kind, subject_role, actor_role, event_time_start, event_time_end, ingested_at,
                semantic_digest, content_digest, encoded_digest, content_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.evidence_record_id,
                record.provenance.producer_fingerprint.digest,
                record.provenance.capability_digest,
                record.source_trust.value,
                descriptor.origin_kind.value,
                descriptor.source_event_ref.namespace,
                descriptor.source_event_ref.value,
                descriptor.source_item_index,
                descriptor.stream_ref.namespace,
                descriptor.stream_ref.value,
                descriptor.stream_ref.generation,
                descriptor.source_sequence,
                content.record_kind.value,
                content.subject_role.value,
                None if content.actor_role is None else content.actor_role.value,
                utc_text(content.event_time_start),
                utc_text(content.event_time_end),
                utc_text(record.ingested_at),
                record.semantic_digest,
                record.content_digest,
                encoded[1],
                encoded[0],
            ),
        )
        for order, correlation in enumerate(descriptor.correlation_refs):
            connection.execute(
                "INSERT INTO behavior_evidence_correlations VALUES (?, ?, ?, ?, ?)",
                (
                    record.evidence_record_id,
                    order,
                    correlation.namespace,
                    correlation.value,
                    correlation.root_value,
                ),
            )
        for order, causal_ref in enumerate(descriptor.causal_refs):
            connection.execute(
                "INSERT INTO behavior_evidence_causal_refs VALUES (?, ?, ?, ?, ?)",
                (
                    record.evidence_record_id,
                    order,
                    causal_ref.kind.value,
                    causal_ref.reference,
                    causal_ref.reference_digest,
                ),
            )
        for order, parent_source in enumerate(descriptor.parent_source_event_refs):
            connection.execute(
                "INSERT INTO behavior_evidence_parent_sources VALUES (?, ?, ?, ?, ?)",
                (
                    record.evidence_record_id,
                    order,
                    parent_source.namespace,
                    parent_source.value,
                    parent_source.identity_digest,
                ),
            )

    @staticmethod
    def _insert_receipt(
        connection: sqlite3.Connection,
        receipt: BehaviorEvidenceIngressReceipt,
        encoded: tuple[str, str],
    ) -> None:
        connection.execute(
            "INSERT INTO behavior_evidence_ingress_receipts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                receipt.delivery_id,
                receipt.request_digest,
                receipt.adapter_name,
                receipt.adapter_fingerprint,
                receipt.capability_digest,
                receipt.status.value,
                utc_text(receipt.recorded_at),
                receipt.content_digest,
                encoded[1],
                encoded[0],
            ),
        )

    @staticmethod
    def _entry(row: sqlite3.Row) -> BehaviorEvidenceLedgerEntry:
        record = decode_evidence_record(row["content_json"], row["encoded_digest"])
        return BehaviorEvidenceLedgerEntry(int(row["evidence_sequence"]), record)

    def _page(
        self,
        kind: str,
        query: Mapping[str, object],
        where: str,
        parameters: tuple[object, ...],
        limit: int,
        cursor: str | None,
    ) -> EvidencePage:
        bounded = self._limit(limit)
        sql = (
            f"SELECT * FROM behavior_evidence_records WHERE {where} AND evidence_sequence>? "
            "ORDER BY evidence_sequence ASC LIMIT ?"
        )
        with self.database.connection.read() as connection:
            after = self._cursor(connection, kind, query, where, parameters, cursor)
            rows = connection.execute(sql, (*parameters, after, bounded + 1)).fetchall()
            entries = tuple(self._entry(row) for row in rows[:bounded])
        next_cursor = None
        if len(rows) > bounded and entries:
            next_cursor = encode_sequence_cursor(kind, query, entries[-1].sequence)
        return entries, next_cursor

    @staticmethod
    def _cursor(
        connection: sqlite3.Connection,
        kind: str,
        query: Mapping[str, object],
        where: str,
        parameters: tuple[object, ...],
        cursor: str | None,
    ) -> int:
        if cursor is None:
            return 0
        sequence = decode_sequence_cursor(
            cursor,
            kind=kind,
            query=query,
            subject="Evidence",
        )
        matches = int(
            connection.execute(
                f"SELECT COUNT(*) FROM behavior_evidence_records WHERE {where} "
                "AND evidence_sequence=?",
                (*parameters, sequence),
            ).fetchone()[0]
        )
        if matches != 1:
            raise ValueError("cursor Evidence entry is missing from this query")
        return sequence

    def _limit(self, limit: int) -> int:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= self.config.evidence.max_query_limit:
            raise ValueError("query limit exceeds the configured Evidence boundary")
        return limit

    def _require_receipt_capacity(self, connection: sqlite3.Connection) -> None:
        count = int(
            connection.execute("SELECT COUNT(*) FROM behavior_evidence_ingress_receipts").fetchone()[0]
        )
        if count >= self.config.store.max_ingress_receipts:
            raise BehaviorEvidenceCapacityError("Evidence ingress Receipt capacity is exhausted")

    def _require_encoded_sizes(self, values: list[tuple[str, str]]) -> None:
        if any(len(text.encode("utf-8")) > self.config.store.max_json_bytes for text, _ in values):
            raise BehaviorEvidenceCapacityError("Behavior durable JSON exceeds configured boundary")

    @staticmethod
    def _read_receipt_row(connection: sqlite3.Connection, delivery_id: str) -> sqlite3.Row | None:
        return connection.execute(
            "SELECT * FROM behavior_evidence_ingress_receipts WHERE delivery_id=?",
            (delivery_id,),
        ).fetchone()

    @staticmethod
    def _decode_receipt_row(row: sqlite3.Row) -> BehaviorEvidenceIngressReceipt:
        return decode_ingress_receipt(row["content_json"], row["encoded_digest"])

    def _resolve_receipt_replay(
        self,
        row: sqlite3.Row,
        request_digest: str,
    ) -> tuple[BehaviorEvidenceIngressReceipt, bool]:
        receipt = self._decode_receipt_row(row)
        if receipt.request_digest != request_digest:
            raise BehaviorEvidenceConflictError("delivery identity already belongs to another request")
        return receipt, True
