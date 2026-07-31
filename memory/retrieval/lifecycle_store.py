"""以独立 SQLite 状态保存 L2 记忆的成功召回事实。"""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from memory.retrieval.lifecycle import (
    MemoryRecallLifecycleConfig,
    MemoryRecallLifecycleError,
    MemoryRecallState,
    MemoryRecallTarget,
)
from memory.uri import MemoryURI

_TABLE = "memory_recall_lifecycle"
_TABLE_LAYOUT = (
    ("uri", "TEXT", 0, None, 1),
    ("document_revision", "INTEGER", 1, None, 0),
    ("document_created_at", "TEXT", 1, None, 0),
    ("successful_recall_count", "INTEGER", 1, None, 0),
    ("last_successful_recall_at", "TEXT", 1, None, 0),
    ("version", "INTEGER", 1, None, 0),
)


class SQLiteMemoryRecallLifecycleStore:
    def __init__(
        self,
        path: str | Path,
        *,
        config: MemoryRecallLifecycleConfig | None = None,
        initialize: bool = True,
    ) -> None:
        if config is not None and not isinstance(config, MemoryRecallLifecycleConfig):
            raise TypeError("config must be MemoryRecallLifecycleConfig")
        if not isinstance(initialize, bool):
            raise TypeError("initialize must be boolean")
        self.path = Path(path).expanduser().absolute()
        self.config = config or MemoryRecallLifecycleConfig()
        self._initialized = False
        self._initialization_guard = threading.Lock()
        if initialize:
            self.initialize()

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> None:
        if self._initialized:
            return
        with self._initialization_guard:
            if self._initialized:
                return
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                os.chmod(self.path.parent, 0o700)
                with closing(self._connect()) as connection:
                    connection.execute(
                        f"""
                        CREATE TABLE IF NOT EXISTS {_TABLE} (
                            uri TEXT PRIMARY KEY,
                            document_revision INTEGER NOT NULL CHECK(document_revision > 0),
                            document_created_at TEXT NOT NULL,
                            successful_recall_count INTEGER NOT NULL
                                CHECK(successful_recall_count > 0),
                            last_successful_recall_at TEXT NOT NULL,
                            version INTEGER NOT NULL CHECK(version > 0)
                        )
                        """
                    )
                    self._require_schema(connection)
                os.chmod(self.path, 0o600)
            except Exception as exc:
                raise MemoryRecallLifecycleError("failed to initialize memory recall lifecycle store") from exc
            self._initialized = True

    def read_many(self, uris: tuple[MemoryURI, ...]) -> tuple[MemoryRecallState, ...]:
        values = self._uris(uris)
        if not values:
            return ()
        self.initialize()
        placeholders = ",".join("?" for _ in values)
        try:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    f"SELECT * FROM {_TABLE} WHERE uri IN ({placeholders})",
                    tuple(str(uri) for uri in values),
                ).fetchall()
            states = tuple(self._state(row) for row in rows)
        except Exception as exc:
            raise MemoryRecallLifecycleError("failed to read memory recall lifecycle state") from exc
        by_uri = {state.uri: state for state in states}
        if len(by_uri) != len(states):
            raise MemoryRecallLifecycleError("memory recall lifecycle store returned duplicate identities")
        return tuple(by_uri[uri] for uri in values if uri in by_uri)

    def record_success(
        self,
        targets: tuple[MemoryRecallTarget, ...],
        *,
        recalled_at: datetime,
    ) -> tuple[MemoryRecallState, ...]:
        values = self._targets(targets)
        if not values:
            return ()
        timestamp = self._timestamp(recalled_at)
        self.initialize()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            result: list[MemoryRecallState] = []
            for target in values:
                row = connection.execute(
                    f"SELECT * FROM {_TABLE} WHERE uri = ?",
                    (str(target.uri),),
                ).fetchone()
                if row is None:
                    count = 1
                    version = 1
                    last_recalled = timestamp
                else:
                    current = self._state(row)
                    if (
                        current.document_created_at > target.document_created_at
                        or (
                            current.document_created_at == target.document_created_at
                            and current.document_revision > target.document_revision
                        )
                    ):
                        result.append(current)
                        continue
                    version = current.version + 1
                    if (
                        current.document_created_at != target.document_created_at
                        or current.document_revision != target.document_revision
                    ):
                        count = 1
                        last_recalled = timestamp
                    else:
                        count = max(
                            current.successful_recall_count,
                            min(
                                current.successful_recall_count + 1,
                                self.config.max_successful_recall_count,
                            ),
                        )
                        assert current.last_successful_recall_at is not None
                        last_recalled = max(current.last_successful_recall_at, timestamp)
                state = MemoryRecallState(
                    uri=target.uri,
                    document_revision=target.document_revision,
                    document_created_at=target.document_created_at,
                    successful_recall_count=count,
                    last_successful_recall_at=last_recalled,
                    version=version,
                )
                connection.execute(
                    f"""
                    INSERT INTO {_TABLE}(
                        uri,
                        document_revision,
                        document_created_at,
                        successful_recall_count,
                        last_successful_recall_at,
                        version
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(uri) DO UPDATE SET
                        document_revision = excluded.document_revision,
                        document_created_at = excluded.document_created_at,
                        successful_recall_count = excluded.successful_recall_count,
                        last_successful_recall_at = excluded.last_successful_recall_at,
                        version = excluded.version
                    """,
                    (
                        str(state.uri),
                        state.document_revision,
                        self._format_timestamp(state.document_created_at),
                        state.successful_recall_count,
                        self._format_timestamp(last_recalled),
                        state.version,
                    ),
                )
                result.append(state)
            connection.commit()
            return tuple(result)
        except Exception as exc:
            if connection.in_transaction:
                connection.rollback()
            raise MemoryRecallLifecycleError("failed to update memory recall lifecycle state") from exc
        finally:
            connection.close()

    def delete_many(self, uris: tuple[MemoryURI, ...]) -> int:
        values = self._uris(uris)
        if not values:
            return 0
        self.initialize()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            placeholders = ",".join("?" for _ in values)
            cursor = connection.execute(
                f"DELETE FROM {_TABLE} WHERE uri IN ({placeholders})",
                tuple(str(uri) for uri in values),
            )
            deleted = cursor.rowcount
            connection.commit()
            return deleted
        except Exception as exc:
            if connection.in_transaction:
                connection.rollback()
            raise MemoryRecallLifecycleError("failed to delete memory recall lifecycle state") from exc
        finally:
            connection.close()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.path,
            timeout=self.config.sqlite_timeout_seconds,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {int(self.config.sqlite_timeout_seconds * 1_000)}")
        return connection

    @staticmethod
    def _require_schema(connection: sqlite3.Connection) -> None:
        rows = connection.execute(f"PRAGMA table_info({_TABLE})").fetchall()
        layout = tuple(
            (
                str(row["name"]),
                str(row["type"]).upper(),
                int(row["notnull"]),
                row["dflt_value"],
                int(row["pk"]),
            )
            for row in rows
        )
        if layout != _TABLE_LAYOUT:
            raise MemoryRecallLifecycleError("memory recall lifecycle store schema is incompatible")

    def _uris(self, values: tuple[MemoryURI, ...]) -> tuple[MemoryURI, ...]:
        if not isinstance(values, tuple):
            raise TypeError("memory recall lifecycle URIs must be a tuple")
        if len(values) > self.config.max_batch_size:
            raise ValueError("memory recall lifecycle URIs exceed max_batch_size")
        parsed = tuple(MemoryURI.parse(value) for value in values)
        if len(set(parsed)) != len(parsed):
            raise ValueError("memory recall lifecycle URIs must be unique")
        for uri in parsed:
            uri.to_address()
        return parsed

    def _targets(
        self,
        values: tuple[MemoryRecallTarget, ...],
    ) -> tuple[MemoryRecallTarget, ...]:
        if not isinstance(values, tuple) or any(
            not isinstance(value, MemoryRecallTarget) for value in values
        ):
            raise TypeError("memory recall lifecycle targets must be a tuple")
        if len(values) > self.config.max_batch_size:
            raise ValueError("memory recall lifecycle targets exceed max_batch_size")
        by_uri: dict[MemoryURI, MemoryRecallTarget] = {}
        for value in values:
            previous = by_uri.get(value.uri)
            if previous is not None and previous != value:
                raise ValueError("memory recall lifecycle target revisions conflict")
            by_uri[value.uri] = value
        return tuple(sorted(by_uri.values(), key=lambda value: str(value.uri)))

    @classmethod
    def _state(cls, row: sqlite3.Row) -> MemoryRecallState:
        return MemoryRecallState(
            uri=MemoryURI.parse(row["uri"]),
            document_revision=row["document_revision"],
            document_created_at=cls._parse_timestamp(row["document_created_at"]),
            successful_recall_count=row["successful_recall_count"],
            last_successful_recall_at=cls._parse_timestamp(row["last_successful_recall_at"]),
            version=row["version"],
        )

    @staticmethod
    def _timestamp(value: datetime) -> datetime:
        if not isinstance(value, datetime):
            raise TypeError("memory recall lifecycle timestamp must be datetime")
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("memory recall lifecycle timestamp must include a timezone")
        return value.astimezone(timezone.utc)

    @staticmethod
    def _format_timestamp(value: datetime) -> str:
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _parse_timestamp(value: object) -> datetime:
        if not isinstance(value, str) or not value:
            raise ValueError("memory recall lifecycle stored timestamp is invalid")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("memory recall lifecycle stored timestamp lacks timezone")
        return parsed.astimezone(timezone.utc)


__all__ = ["SQLiteMemoryRecallLifecycleStore"]
