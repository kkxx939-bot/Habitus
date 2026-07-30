"""只记录安全与人工运维动作的有界 SQLite 审计旁路。"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

from foundation.observability import ObservationEvent

_AUDITED_OPERATIONS = frozenset(
    {
        ("security", "authentication"),
        ("operations", "retry_job"),
        ("operations", "abandon_job"),
        ("runtime", "initialization"),
    }
)
_SAFE_ATTRIBUTES = frozenset(
    {
        "error_code",
        "error_type",
        "retryable",
        "http_method",
        "http_route",
        "http_status_code",
        "job_status",
    }
)


@dataclass(frozen=True)
class AuditRecord:
    audit_id: str
    occurred_at: datetime
    category: str
    operation: str
    status: str
    request_id: str | None
    memory_sequence: int | None
    transaction_id: str | None
    attributes: dict[str, str | int | float | bool]


class AuditStore:
    """低频同步写入；失败由上层 CompositeObserver 隔离于业务主链。"""

    def __init__(self, path: Path, *, retention_days: int, max_records: int) -> None:
        self.path = Path(path).expanduser().absolute()
        self.retention_days = retention_days
        self.max_records = max_records
        self._guard = threading.RLock()
        self._initialized = False

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> None:
        if self._initialized:
            return
        with self._guard:
            if self._initialized:
                return
            self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            os.chmod(self.path.parent, 0o700)
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_events (
                      audit_id TEXT PRIMARY KEY,
                      occurred_at TEXT NOT NULL,
                      category TEXT NOT NULL,
                      operation TEXT NOT NULL,
                      status TEXT NOT NULL,
                      request_id TEXT,
                      memory_sequence INTEGER,
                      transaction_id TEXT,
                      attributes_json TEXT NOT NULL
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS audit_events_occurred_at ON audit_events(occurred_at)"
                )
            os.chmod(self.path, 0o600)
            self._initialized = True

    def record(self, event: ObservationEvent) -> None:
        if (event.category, event.operation) not in _AUDITED_OPERATIONS:
            return
        self.initialize()
        attributes = {
            key: value
            for key, value in event.attributes.items()
            if key in _SAFE_ATTRIBUTES
        }
        with self._guard, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO audit_events(
                  audit_id, occurred_at, category, operation, status,
                  request_id, memory_sequence, transaction_id, attributes_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid4().hex,
                    event.occurred_at.isoformat(),
                    event.category,
                    event.operation,
                    event.status.value,
                    event.context.request_id,
                    event.context.memory_sequence,
                    event.context.transaction_id,
                    json.dumps(attributes, separators=(",", ":"), sort_keys=True),
                ),
            )
            self._prune(connection, now=event.occurred_at)

    def recent(self, *, limit: int = 100) -> tuple[AuditRecord, ...]:
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
            raise ValueError("audit limit must be between 1 and 500")
        self.initialize()
        with self._guard, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM audit_events ORDER BY occurred_at DESC, audit_id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return tuple(self._record(row) for row in rows)

    def _prune(self, connection: sqlite3.Connection, *, now: datetime) -> None:
        cutoff = (now.astimezone(timezone.utc) - timedelta(days=self.retention_days)).isoformat()
        connection.execute("DELETE FROM audit_events WHERE occurred_at < ?", (cutoff,))
        connection.execute(
            """
            DELETE FROM audit_events
            WHERE audit_id IN (
              SELECT audit_id FROM audit_events
              ORDER BY occurred_at DESC, audit_id DESC
              LIMIT -1 OFFSET ?
            )
            """,
            (self.max_records,),
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _record(row: sqlite3.Row) -> AuditRecord:
        raw_attributes = json.loads(str(row["attributes_json"]))
        attributes = {
            str(key): value
            for key, value in raw_attributes.items()
            if key in _SAFE_ATTRIBUTES and isinstance(value, str | int | float | bool)
        }
        return AuditRecord(
            audit_id=str(row["audit_id"]),
            occurred_at=datetime.fromisoformat(str(row["occurred_at"])).astimezone(timezone.utc),
            category=str(row["category"]),
            operation=str(row["operation"]),
            status=str(row["status"]),
            request_id=None if row["request_id"] is None else str(row["request_id"]),
            memory_sequence=(None if row["memory_sequence"] is None else int(row["memory_sequence"])),
            transaction_id=(None if row["transaction_id"] is None else str(row["transaction_id"])),
            attributes=attributes,
        )


__all__ = ["AuditRecord", "AuditStore"]
