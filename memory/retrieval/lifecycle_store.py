"""以独立 SQLite 状态保存 L2 记忆的实际使用和冷记忆生命周期事实。"""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import closing
from datetime import UTC, datetime
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
    ("useful_recall_count", "INTEGER", 1, None, 0),
    ("last_useful_recall_at", "TEXT", 0, None, 0),
    ("lifecycle_activity_at", "TEXT", 0, None, 0),
    ("cold2_probe_count", "INTEGER", 1, None, 0),
    ("last_cold2_probe_at", "TEXT", 0, None, 0),
    ("compacted_at", "TEXT", 0, None, 0),
    ("retire_candidate_at", "TEXT", 0, None, 0),
    ("retired_at", "TEXT", 0, None, 0),
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
                            useful_recall_count INTEGER NOT NULL CHECK(useful_recall_count >= 0),
                            last_useful_recall_at TEXT,
                            lifecycle_activity_at TEXT,
                            cold2_probe_count INTEGER NOT NULL CHECK(cold2_probe_count >= 0),
                            last_cold2_probe_at TEXT,
                            compacted_at TEXT,
                            retire_candidate_at TEXT,
                            retired_at TEXT,
                            version INTEGER NOT NULL CHECK(version > 0),
                            CHECK((useful_recall_count = 0) = (last_useful_recall_at IS NULL)),
                            CHECK((cold2_probe_count = 0) = (last_cold2_probe_at IS NULL)),
                            CHECK(retire_candidate_at IS NULL OR compacted_at IS NOT NULL),
                            CHECK(retired_at IS NULL OR retire_candidate_at IS NOT NULL)
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

    def record_use(
        self,
        targets: tuple[MemoryRecallTarget, ...],
        *,
        used_at: datetime,
    ) -> tuple[MemoryRecallState, ...]:
        values = self._targets(targets)
        if not values:
            return ()
        timestamp = self._timestamp(used_at)

        def update(current: MemoryRecallState | None, target: MemoryRecallTarget) -> MemoryRecallState:
            if current is None or not self._same_generation(current, target):
                return MemoryRecallState(
                    uri=target.uri,
                    document_revision=target.document_revision,
                    document_created_at=target.document_created_at,
                    useful_recall_count=1,
                    last_useful_recall_at=timestamp,
                    lifecycle_activity_at=None,
                    cold2_probe_count=0,
                    last_cold2_probe_at=None,
                    compacted_at=None,
                    retire_candidate_at=None,
                    retired_at=None,
                    version=1 if current is None else current.version + 1,
                )
            revised = current.document_revision != target.document_revision
            if current.retired_at is not None and not revised:
                raise MemoryRecallLifecycleError("cannot record use for a retiring memory")
            return MemoryRecallState(
                uri=target.uri,
                document_revision=target.document_revision,
                document_created_at=target.document_created_at,
                useful_recall_count=max(
                    current.useful_recall_count,
                    min(current.useful_recall_count + 1, self.config.max_useful_recall_count),
                ),
                last_useful_recall_at=max(current.last_useful_recall_at or timestamp, timestamp),
                lifecycle_activity_at=None if revised else current.lifecycle_activity_at,
                cold2_probe_count=0,
                last_cold2_probe_at=None,
                compacted_at=None if revised else current.compacted_at,
                retire_candidate_at=None,
                retired_at=None,
                version=current.version + 1,
            )

        return self._update_targets(values, update, "record memory recall use")

    def record_probe(
        self,
        targets: tuple[MemoryRecallTarget, ...],
        *,
        probed_at: datetime,
    ) -> tuple[MemoryRecallState, ...]:
        values = self._targets(targets)
        if not values:
            return ()
        timestamp = self._timestamp(probed_at)

        def update(current: MemoryRecallState | None, target: MemoryRecallTarget) -> MemoryRecallState:
            if current is None or not self._same_generation(current, target):
                return MemoryRecallState(
                    uri=target.uri,
                    document_revision=target.document_revision,
                    document_created_at=target.document_created_at,
                    useful_recall_count=0,
                    last_useful_recall_at=None,
                    lifecycle_activity_at=None,
                    cold2_probe_count=1,
                    last_cold2_probe_at=timestamp,
                    compacted_at=None,
                    retire_candidate_at=None,
                    retired_at=None,
                    version=1 if current is None else current.version + 1,
                )
            revised = current.document_revision != target.document_revision
            return MemoryRecallState(
                uri=target.uri,
                document_revision=target.document_revision,
                document_created_at=target.document_created_at,
                useful_recall_count=current.useful_recall_count,
                last_useful_recall_at=current.last_useful_recall_at,
                lifecycle_activity_at=None if revised else current.lifecycle_activity_at,
                cold2_probe_count=max(
                    current.cold2_probe_count,
                    min(current.cold2_probe_count + 1, self.config.cold2_probe_limit),
                ),
                last_cold2_probe_at=max(current.last_cold2_probe_at or timestamp, timestamp),
                compacted_at=None if revised else current.compacted_at,
                retire_candidate_at=None if revised else current.retire_candidate_at,
                retired_at=None if revised else current.retired_at,
                version=current.version + 1,
            )

        return self._update_targets(values, update, "record COLD_2 probe")

    def mark_compacted(
        self,
        target: MemoryRecallTarget,
        *,
        lifecycle_activity_at: datetime,
        compacted_at: datetime,
        expected_version: int,
    ) -> MemoryRecallState:
        if not isinstance(target, MemoryRecallTarget):
            raise TypeError("target must be MemoryRecallTarget")
        activity = self._timestamp(lifecycle_activity_at)
        timestamp = self._timestamp(compacted_at)
        if activity > timestamp:
            raise ValueError("memory lifecycle activity cannot follow compaction")

        def update(current: MemoryRecallState | None, value: MemoryRecallTarget) -> MemoryRecallState:
            current_version = 0 if current is None else current.version
            if current_version != expected_version:
                raise MemoryRecallLifecycleError("memory lifecycle version changed before compaction")
            preserve = current is not None and current.document_created_at == value.document_created_at
            useful_recall_count = 0
            last_useful_recall_at = None
            if preserve and current is not None:
                useful_recall_count = current.useful_recall_count
                last_useful_recall_at = current.last_useful_recall_at
            return MemoryRecallState(
                uri=value.uri,
                document_revision=value.document_revision,
                document_created_at=value.document_created_at,
                useful_recall_count=useful_recall_count,
                last_useful_recall_at=last_useful_recall_at,
                lifecycle_activity_at=activity,
                cold2_probe_count=0,
                last_cold2_probe_at=None,
                compacted_at=timestamp,
                retire_candidate_at=None,
                retired_at=None,
                version=1 if current is None else current.version + 1,
            )

        return self._update_targets((target,), update, "mark memory compacted")[0]

    def mark_retired(
        self,
        target: MemoryRecallTarget,
        *,
        retired_at: datetime,
        expected_version: int,
    ) -> MemoryRecallState:
        if not isinstance(target, MemoryRecallTarget):
            raise TypeError("target must be MemoryRecallTarget")
        timestamp = self._timestamp(retired_at)

        def update(current: MemoryRecallState | None, value: MemoryRecallTarget) -> MemoryRecallState:
            if current is None or not self._same_document(current, value):
                raise MemoryRecallLifecycleError("cannot retire an unknown or stale memory lifecycle target")
            if current.version != expected_version:
                raise MemoryRecallLifecycleError("memory lifecycle version changed before retirement")
            if current.compacted_at is None:
                raise MemoryRecallLifecycleError("cannot retire memory before reversible compaction")
            if current.retire_candidate_at is None:
                raise MemoryRecallLifecycleError("cannot retire memory before candidate grace")
            return MemoryRecallState(
                uri=value.uri,
                document_revision=value.document_revision,
                document_created_at=value.document_created_at,
                useful_recall_count=current.useful_recall_count,
                last_useful_recall_at=current.last_useful_recall_at,
                lifecycle_activity_at=current.lifecycle_activity_at,
                cold2_probe_count=current.cold2_probe_count,
                last_cold2_probe_at=current.last_cold2_probe_at,
                compacted_at=current.compacted_at,
                retire_candidate_at=current.retire_candidate_at,
                retired_at=max(current.retired_at or timestamp, timestamp),
                version=current.version + 1,
            )

        return self._update_targets((target,), update, "mark memory retired")[0]

    def mark_retire_candidate(
        self,
        target: MemoryRecallTarget,
        *,
        marked_at: datetime,
        expected_version: int,
    ) -> MemoryRecallState:
        if not isinstance(target, MemoryRecallTarget):
            raise TypeError("target must be MemoryRecallTarget")
        timestamp = self._timestamp(marked_at)

        def update(current: MemoryRecallState | None, value: MemoryRecallTarget) -> MemoryRecallState:
            if current is None or not self._same_document(current, value):
                raise MemoryRecallLifecycleError("cannot mark an unknown or stale retirement candidate")
            if current.version != expected_version:
                raise MemoryRecallLifecycleError("memory lifecycle version changed before candidate marking")
            if current.compacted_at is None or current.retired_at is not None:
                raise MemoryRecallLifecycleError("memory retirement candidate state is invalid")
            return MemoryRecallState(
                uri=value.uri,
                document_revision=value.document_revision,
                document_created_at=value.document_created_at,
                useful_recall_count=current.useful_recall_count,
                last_useful_recall_at=current.last_useful_recall_at,
                lifecycle_activity_at=current.lifecycle_activity_at,
                cold2_probe_count=current.cold2_probe_count,
                last_cold2_probe_at=current.last_cold2_probe_at,
                compacted_at=current.compacted_at,
                retire_candidate_at=current.retire_candidate_at or timestamp,
                retired_at=None,
                version=current.version + 1,
            )

        return self._update_targets((target,), update, "mark memory retirement candidate")[0]

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

    def _update_targets(self, targets, update, operation: str) -> tuple[MemoryRecallState, ...]:
        self.initialize()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            result: list[MemoryRecallState] = []
            for target in targets:
                row = connection.execute(
                    f"SELECT * FROM {_TABLE} WHERE uri = ?",
                    (str(target.uri),),
                ).fetchone()
                current = None if row is None else self._state(row)
                if current is not None and self._is_stale(current, target):
                    result.append(current)
                    continue
                state = update(current, target)
                self._write(connection, state)
                result.append(state)
            connection.commit()
            return tuple(result)
        except MemoryRecallLifecycleError:
            if connection.in_transaction:
                connection.rollback()
            raise
        except Exception as exc:
            if connection.in_transaction:
                connection.rollback()
            raise MemoryRecallLifecycleError(f"failed to {operation}") from exc
        finally:
            connection.close()

    @classmethod
    def _write(cls, connection: sqlite3.Connection, state: MemoryRecallState) -> None:
        connection.execute(
            f"""
            INSERT INTO {_TABLE} VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(uri) DO UPDATE SET
                document_revision = excluded.document_revision,
                document_created_at = excluded.document_created_at,
                useful_recall_count = excluded.useful_recall_count,
                last_useful_recall_at = excluded.last_useful_recall_at,
                lifecycle_activity_at = excluded.lifecycle_activity_at,
                cold2_probe_count = excluded.cold2_probe_count,
                last_cold2_probe_at = excluded.last_cold2_probe_at,
                compacted_at = excluded.compacted_at,
                retire_candidate_at = excluded.retire_candidate_at,
                retired_at = excluded.retired_at,
                version = excluded.version
            """,
            (
                str(state.uri),
                state.document_revision,
                cls._format_timestamp(state.document_created_at),
                state.useful_recall_count,
                cls._optional_timestamp(state.last_useful_recall_at),
                cls._optional_timestamp(state.lifecycle_activity_at),
                state.cold2_probe_count,
                cls._optional_timestamp(state.last_cold2_probe_at),
                cls._optional_timestamp(state.compacted_at),
                cls._optional_timestamp(state.retire_candidate_at),
                cls._optional_timestamp(state.retired_at),
                state.version,
            ),
        )

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

    def _targets(self, values: tuple[MemoryRecallTarget, ...]) -> tuple[MemoryRecallTarget, ...]:
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
            useful_recall_count=row["useful_recall_count"],
            last_useful_recall_at=cls._parse_optional_timestamp(row["last_useful_recall_at"]),
            lifecycle_activity_at=cls._parse_optional_timestamp(row["lifecycle_activity_at"]),
            cold2_probe_count=row["cold2_probe_count"],
            last_cold2_probe_at=cls._parse_optional_timestamp(row["last_cold2_probe_at"]),
            compacted_at=cls._parse_optional_timestamp(row["compacted_at"]),
            retire_candidate_at=cls._parse_optional_timestamp(row["retire_candidate_at"]),
            retired_at=cls._parse_optional_timestamp(row["retired_at"]),
            version=row["version"],
        )

    @staticmethod
    def _same_generation(state: MemoryRecallState, target: MemoryRecallTarget) -> bool:
        return state.document_created_at == target.document_created_at

    @staticmethod
    def _same_document(state: MemoryRecallState, target: MemoryRecallTarget) -> bool:
        return (
            state.document_created_at == target.document_created_at
            and state.document_revision == target.document_revision
        )

    @staticmethod
    def _is_stale(state: MemoryRecallState, target: MemoryRecallTarget) -> bool:
        return state.document_created_at > target.document_created_at or (
            state.document_created_at == target.document_created_at
            and state.document_revision > target.document_revision
        )

    @staticmethod
    def _timestamp(value: datetime) -> datetime:
        if not isinstance(value, datetime):
            raise TypeError("memory recall lifecycle timestamp must be datetime")
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("memory recall lifecycle timestamp must include a timezone")
        return value.astimezone(UTC)

    @classmethod
    def _optional_timestamp(cls, value: datetime | None) -> str | None:
        return None if value is None else cls._format_timestamp(value)

    @staticmethod
    def _format_timestamp(value: datetime) -> str:
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")

    @classmethod
    def _parse_optional_timestamp(cls, value: object) -> datetime | None:
        return None if value is None else cls._parse_timestamp(value)

    @staticmethod
    def _parse_timestamp(value: object) -> datetime:
        if not isinstance(value, str) or not value:
            raise ValueError("memory recall lifecycle stored timestamp is invalid")
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise ValueError("memory recall lifecycle stored timestamp lacks timezone")
        return parsed.astimezone(UTC)


__all__ = ["SQLiteMemoryRecallLifecycleStore"]
