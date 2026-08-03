"""Summary 的实际使用保护与终态退休候选状态，不参与召回排名。"""

from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from memory.conversation.indexing.model import ConversationSummaryReference, summary_reference
from memory.conversation.layout import ConversationAddress, ConversationLayout
from pre.conversation import ConversationRangeSummary, ConversationSegmentSummary


class ConversationSummaryUseError(RuntimeError):
    """Summary 实际使用状态无法安全读取或更新。"""


@dataclass(frozen=True)
class ConversationSummaryUseState:
    reference: ConversationSummaryReference
    useful_recall_count: int
    last_useful_recall_at: datetime | None
    retire_candidate_at: datetime | None
    retiring_at: datetime | None
    version: int

    def __post_init__(self) -> None:
        if not isinstance(self.reference, ConversationSummaryReference):
            raise TypeError("reference must be ConversationSummaryReference")
        for name in ("useful_recall_count", "version"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"summary use state {name} must be non-negative")
        for name in ("last_useful_recall_at", "retire_candidate_at", "retiring_at"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _timestamp(value))
        if (self.useful_recall_count == 0) != (self.last_useful_recall_at is None):
            raise ValueError("summary use count and timestamp must describe the same lifecycle")
        if self.useful_recall_count == 0 and self.retire_candidate_at is None:
            raise ValueError("zero-use Summary state must be a retirement candidate")
        if self.retiring_at is not None and self.retire_candidate_at is None:
            raise ValueError("Summary cannot retire before candidate marking")
        if self.version <= 0:
            raise ValueError("persisted summary use state version must be positive")


class SQLiteConversationSummaryUseStore:
    """独立保存实际使用回执；Summary 文件继续是内容真相源。"""

    def __init__(
        self,
        path: str | Path,
        *,
        sqlite_timeout_seconds: float = 5.0,
        max_batch_size: int = 1_000,
        max_useful_recall_count: int = 1_000_000_000,
        initialize: bool = True,
    ) -> None:
        if (
            isinstance(sqlite_timeout_seconds, bool)
            or not isinstance(sqlite_timeout_seconds, int | float)
            or not 0.001 <= float(sqlite_timeout_seconds) <= 60.0
        ):
            raise ValueError("summary use sqlite timeout is outside its supported range")
        for name, value in {
            "max_batch_size": max_batch_size,
            "max_useful_recall_count": max_useful_recall_count,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"summary use {name} must be positive")
        if not isinstance(initialize, bool):
            raise TypeError("initialize must be boolean")
        self.path = Path(path).expanduser().absolute()
        self.sqlite_timeout_seconds = float(sqlite_timeout_seconds)
        self.max_batch_size = max_batch_size
        self.max_useful_recall_count = max_useful_recall_count
        self._initialized = False
        self._guard = threading.Lock()
        if initialize:
            self.initialize()

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> None:
        if self._initialized:
            return
        with self._guard:
            if self._initialized:
                return
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                os.chmod(self.path.parent, 0o700)
                with closing(self._connect()) as connection:
                    connection.execute(
                        """
                        CREATE TABLE IF NOT EXISTS conversation_summary_use (
                            identity TEXT PRIMARY KEY,
                            started_on TEXT NOT NULL,
                            conversation_id TEXT NOT NULL,
                            stage TEXT NOT NULL,
                            summary_id TEXT NOT NULL,
                            useful_recall_count INTEGER NOT NULL CHECK(useful_recall_count >= 0),
                            last_useful_recall_at TEXT,
                            retire_candidate_at TEXT,
                            retiring_at TEXT,
                            version INTEGER NOT NULL CHECK(version > 0)
                        )
                        """
                    )
                    self._require_schema(connection)
                os.chmod(self.path, 0o600)
            except Exception as exc:
                raise ConversationSummaryUseError("failed to initialize Summary use store") from exc
            self._initialized = True

    def read_many(
        self,
        references: tuple[ConversationSummaryReference, ...],
    ) -> tuple[ConversationSummaryUseState, ...]:
        values = self._references(references)
        if not values:
            return ()
        self.initialize()
        placeholders = ",".join("?" for _ in values)
        try:
            with closing(self._connect()) as connection:
                rows = connection.execute(
                    f"SELECT * FROM conversation_summary_use WHERE identity IN ({placeholders})",
                    tuple(item.identity for item in values),
                ).fetchall()
            by_identity = {row["identity"]: self._state(row) for row in rows}
            return tuple(by_identity[item.identity] for item in values if item.identity in by_identity)
        except Exception as exc:
            raise ConversationSummaryUseError("failed to read Summary use state") from exc

    def record_use(
        self,
        references: tuple[ConversationSummaryReference, ...],
        *,
        used_at: datetime,
    ) -> tuple[ConversationSummaryUseState, ...]:
        values = self._references(references)
        if not values:
            return ()
        timestamp = _timestamp(used_at)
        self.initialize()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            states: list[ConversationSummaryUseState] = []
            for reference in values:
                row = connection.execute(
                    "SELECT * FROM conversation_summary_use WHERE identity = ?",
                    (reference.identity,),
                ).fetchone()
                if row is None:
                    state = ConversationSummaryUseState(reference, 1, timestamp, None, None, 1)
                else:
                    current = self._state(row)
                    if current.retiring_at is not None:
                        raise ConversationSummaryUseError("cannot record use for a retiring Summary")
                    state = ConversationSummaryUseState(
                        reference,
                        max(
                            current.useful_recall_count,
                            min(current.useful_recall_count + 1, self.max_useful_recall_count),
                        ),
                        max(current.last_useful_recall_at or timestamp, timestamp),
                        None,
                        None,
                        current.version + 1,
                    )
                self._write(connection, state)
                states.append(state)
            connection.commit()
            return tuple(states)
        except ConversationSummaryUseError:
            if connection.in_transaction:
                connection.rollback()
            raise
        except Exception as exc:
            if connection.in_transaction:
                connection.rollback()
            raise ConversationSummaryUseError("failed to record Summary actual use") from exc
        finally:
            connection.close()

    def mark_retire_candidate(
        self,
        reference: ConversationSummaryReference,
        *,
        marked_at: datetime,
    ) -> ConversationSummaryUseState:
        if not isinstance(reference, ConversationSummaryReference):
            raise TypeError("reference must be ConversationSummaryReference")
        timestamp = _timestamp(marked_at)
        self.initialize()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM conversation_summary_use WHERE identity = ?",
                (reference.identity,),
            ).fetchone()
            if row is None:
                # 零次使用也要能进入两阶段退休；用零计数状态只在领域对象外持久化。
                connection.execute(
                    """
                    INSERT INTO conversation_summary_use VALUES (?, ?, ?, ?, ?, 0, NULL, ?, NULL, 1)
                    """,
                    (
                        reference.identity,
                        reference.address.started_on.isoformat(),
                        reference.address.conversation_id,
                        reference.stage.value,
                        reference.summary_id,
                        _format_timestamp(timestamp),
                    ),
                )
                state = self._state(
                    connection.execute(
                        "SELECT * FROM conversation_summary_use WHERE identity = ?",
                        (reference.identity,),
                    ).fetchone()
                )
            else:
                current = self._state(row)
                candidate_at = current.retire_candidate_at or timestamp
                connection.execute(
                    "UPDATE conversation_summary_use SET retire_candidate_at = ?, version = ? WHERE identity = ?",
                    (_format_timestamp(candidate_at), current.version + 1, reference.identity),
                )
                state = ConversationSummaryUseState(
                    reference,
                    current.useful_recall_count,
                    current.last_useful_recall_at,
                    candidate_at,
                    current.retiring_at,
                    current.version + 1,
                )
            connection.commit()
            return state
        except Exception as exc:
            if connection.in_transaction:
                connection.rollback()
            raise ConversationSummaryUseError("failed to mark Summary retirement candidate") from exc
        finally:
            connection.close()

    def claim_retirement(
        self,
        reference: ConversationSummaryReference,
        *,
        expected_version: int,
        claimed_at: datetime,
    ) -> ConversationSummaryUseState:
        """以 use version CAS 抢占终态退休，阻止并发成功使用。"""

        if not isinstance(reference, ConversationSummaryReference):
            raise TypeError("reference must be ConversationSummaryReference")
        timestamp = _timestamp(claimed_at)
        self.initialize()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT * FROM conversation_summary_use WHERE identity = ?",
                (reference.identity,),
            ).fetchone()
            if row is None:
                raise ConversationSummaryUseError("cannot claim retirement without candidate state")
            current = self._state(row)
            if current.version != expected_version:
                raise ConversationSummaryUseError("Summary use version changed before retirement claim")
            if current.retire_candidate_at is None:
                raise ConversationSummaryUseError("cannot claim retirement before candidate marking")
            state = ConversationSummaryUseState(
                reference,
                current.useful_recall_count,
                current.last_useful_recall_at,
                current.retire_candidate_at,
                current.retiring_at or timestamp,
                current.version + 1,
            )
            self._write(connection, state)
            connection.commit()
            return state
        except ConversationSummaryUseError:
            if connection.in_transaction:
                connection.rollback()
            raise
        except Exception as exc:
            if connection.in_transaction:
                connection.rollback()
            raise ConversationSummaryUseError("failed to claim Summary retirement") from exc
        finally:
            connection.close()

    def delete_many(self, references: tuple[ConversationSummaryReference, ...]) -> int:
        values = self._references(references)
        if not values:
            return 0
        self.initialize()
        placeholders = ",".join("?" for _ in values)
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                f"DELETE FROM conversation_summary_use WHERE identity IN ({placeholders})",
                tuple(item.identity for item in values),
            )
            return cursor.rowcount

    def delete_coverage(
        self,
        address: ConversationAddress,
        *,
        start_sequence: int,
        end_sequence: int,
    ) -> int:
        """终态清理覆盖范围内各级 Summary 状态，支持内容文件已被部分删除后的续跑。"""

        if not isinstance(address, ConversationAddress):
            raise TypeError("address must be ConversationAddress")
        ConversationLayout.segment_id(start_sequence, end_sequence)
        self.initialize()
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            identities: list[str] = []
            after = ""
            while True:
                rows = connection.execute(
                    """
                    SELECT identity, summary_id FROM conversation_summary_use
                    WHERE started_on = ? AND conversation_id = ? AND identity > ?
                    ORDER BY identity LIMIT ?
                    """,
                    (
                        address.started_on.isoformat(),
                        address.conversation_id,
                        after,
                        self.max_batch_size,
                    ),
                ).fetchall()
                if not rows:
                    break
                selected: list[str] = []
                for row in rows:
                    source_start, source_end = ConversationLayout.segment_range(row["summary_id"])
                    if start_sequence <= source_start and source_end <= end_sequence:
                        selected.append(row["identity"])
                if selected:
                    placeholders = ",".join("?" for _ in selected)
                    connection.execute(
                        f"DELETE FROM conversation_summary_use WHERE identity IN ({placeholders})",
                        tuple(selected),
                    )
                    identities.extend(selected)
                after = rows[-1]["identity"]
                if len(rows) < self.max_batch_size:
                    break
            connection.commit()
            return len(identities)
        except ConversationSummaryUseError:
            if connection.in_transaction:
                connection.rollback()
            raise
        except Exception as exc:
            if connection.in_transaction:
                connection.rollback()
            raise ConversationSummaryUseError("failed to delete Summary use coverage") from exc
        finally:
            connection.close()

    def recently_used(
        self,
        reference: ConversationSummaryReference,
        *,
        now: datetime,
        protection_days: int,
    ) -> bool:
        states = self.read_many((reference,))
        if not states:
            return False
        last = states[0].last_useful_recall_at
        return last is not None and (_timestamp(now) - last).total_seconds() < protection_days * 86_400

    def recently_used_summary(
        self,
        address: ConversationAddress,
        summary: ConversationSegmentSummary | ConversationRangeSummary,
        *,
        now: datetime,
        protection_days: int,
    ) -> bool:
        if not isinstance(address, ConversationAddress):
            raise TypeError("address must be ConversationAddress")
        return self.recently_used(
            summary_reference(address, summary),
            now=now,
            protection_days=protection_days,
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=self.sqlite_timeout_seconds, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {int(self.sqlite_timeout_seconds * 1_000)}")
        return connection

    @staticmethod
    def _require_schema(connection: sqlite3.Connection) -> None:
        names = tuple(row["name"] for row in connection.execute("PRAGMA table_info(conversation_summary_use)"))
        if names != (
            "identity",
            "started_on",
            "conversation_id",
            "stage",
            "summary_id",
            "useful_recall_count",
            "last_useful_recall_at",
            "retire_candidate_at",
            "retiring_at",
            "version",
        ):
            raise ConversationSummaryUseError("Summary use store schema is incompatible")

    @staticmethod
    def _write(connection: sqlite3.Connection, state: ConversationSummaryUseState) -> None:
        reference = state.reference
        connection.execute(
            """
            INSERT INTO conversation_summary_use VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(identity) DO UPDATE SET
                useful_recall_count = excluded.useful_recall_count,
                last_useful_recall_at = excluded.last_useful_recall_at,
                retire_candidate_at = excluded.retire_candidate_at,
                retiring_at = excluded.retiring_at,
                version = excluded.version
            """,
            (
                reference.identity,
                reference.address.started_on.isoformat(),
                reference.address.conversation_id,
                reference.stage.value,
                reference.summary_id,
                state.useful_recall_count,
                (
                    None
                    if state.last_useful_recall_at is None
                    else _format_timestamp(state.last_useful_recall_at)
                ),
                None if state.retire_candidate_at is None else _format_timestamp(state.retire_candidate_at),
                None if state.retiring_at is None else _format_timestamp(state.retiring_at),
                state.version,
            ),
        )

    @classmethod
    def _state(cls, row: sqlite3.Row) -> ConversationSummaryUseState:
        reference = ConversationSummaryReference(
            address=ConversationAddress(
                row["conversation_id"],
                datetime.fromisoformat(row["started_on"]).date(),
            ),
            stage=row["stage"],
            summary_id=row["summary_id"],
        )
        if row["identity"] != reference.identity:
            raise ConversationSummaryUseError("Summary use identity does not match stored fields")
        count = row["useful_recall_count"]
        return ConversationSummaryUseState(
            reference,
            count,
            _parse_optional_timestamp(row["last_useful_recall_at"]),
            _parse_optional_timestamp(row["retire_candidate_at"]),
            _parse_optional_timestamp(row["retiring_at"]),
            row["version"],
        )

    def _references(self, values):
        if not isinstance(values, tuple) or any(not isinstance(item, ConversationSummaryReference) for item in values):
            raise TypeError("Summary use references must be a tuple")
        if len(values) > self.max_batch_size:
            raise ValueError("Summary use references exceed max_batch_size")
        if len({item.identity for item in values}) != len(values):
            raise ValueError("Summary use references must be unique")
        return tuple(sorted(values, key=lambda item: item.identity))


def _timestamp(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Summary use timestamp must be timezone-aware datetime")
    return value.astimezone(timezone.utc)


def _format_timestamp(value: datetime) -> str:
    return _timestamp(value).isoformat().replace("+00:00", "Z")


def _parse_optional_timestamp(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("stored Summary use timestamp is invalid")
    return _timestamp(datetime.fromisoformat(value.replace("Z", "+00:00")))


__all__ = [
    "ConversationSummaryUseError",
    "ConversationSummaryUseState",
    "SQLiteConversationSummaryUseStore",
]
