"""多文档记忆提交在进程崩溃后使用的耐久恢复日志。"""

from __future__ import annotations

import base64
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from foundation.integrity import canonical_json
from infrastructure.store.filesystem import (
    atomic_create_bytes,
    atomic_replace_bytes,
    atomic_temporary_destination,
    read_regular_bytes,
)
from memory.document import MemoryDocument, MemoryDocumentCodec
from memory.uri import MemoryURI

_TRANSACTION_ID = re.compile(r"^[0-9a-f]{32}$")
_SCHEMA = "memory_commit_transaction_v1"


class MemoryTransactionJournalError(RuntimeError):
    """提交日志无法安全创建、读取、推进或清理。"""


class MemoryTransactionJournalState(str, Enum):
    """耐久提交日志的状态。"""

    PREPARED = "prepared"
    COMMITTED = "committed"
    ROLLED_BACK = "rolled_back"


@dataclass(frozen=True)
class MemoryTransactionJournalEntry:
    """一个可能被写入或删除的 URI 在事务前后的完整文档。"""

    uri: MemoryURI
    before: MemoryDocument | None
    after: MemoryDocument | None

    def __post_init__(self) -> None:
        if not isinstance(self.uri, MemoryURI):
            raise TypeError("journal entry uri must be a MemoryURI")
        self.uri.to_address()
        for name, document in {"before": self.before, "after": self.after}.items():
            if document is None:
                continue
            if not isinstance(document, MemoryDocument):
                raise TypeError(f"journal entry {name} must be a MemoryDocument or None")
            if MemoryURI.from_address(document.address) != self.uri:
                raise ValueError(f"journal entry {name} document has the wrong URI")
        if self.before == self.after:
            raise ValueError("journal entry must describe an actual document change")


@dataclass(frozen=True)
class MemoryTransactionJournalRecord:
    """一笔 PREPARED 或已结束事务的完整恢复信息。"""

    transaction_id: str
    state: MemoryTransactionJournalState
    created_at: datetime
    updated_at: datetime
    lock_identities: tuple[str, ...]
    entries: tuple[MemoryTransactionJournalEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.transaction_id, str) or not _TRANSACTION_ID.fullmatch(self.transaction_id):
            raise ValueError("transaction_id must be 32 lowercase hexadecimal characters")
        try:
            state = MemoryTransactionJournalState(self.state)
        except ValueError as exc:
            raise ValueError("transaction journal contains an unsupported state") from exc
        object.__setattr__(self, "state", state)
        for name in ("created_at", "updated_at"):
            value = getattr(self, name)
            if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"journal {name} must be timezone-aware")
            object.__setattr__(self, name, value.astimezone(timezone.utc))
        if self.updated_at < self.created_at:
            raise ValueError("journal updated_at cannot precede created_at")
        if not isinstance(self.lock_identities, tuple) or any(
            not isinstance(identity, str) or not identity for identity in self.lock_identities
        ):
            raise TypeError("lock_identities must contain non-empty strings")
        if self.lock_identities != tuple(sorted(set(self.lock_identities))):
            raise ValueError("lock_identities must be unique and sorted")
        for identity in self.lock_identities:
            MemoryURI.parse(identity).to_address()
        if not isinstance(self.entries, tuple) or any(
            not isinstance(entry, MemoryTransactionJournalEntry) for entry in self.entries
        ):
            raise TypeError("journal entries must contain MemoryTransactionJournalEntry values")
        entry_ids = tuple(str(entry.uri) for entry in self.entries)
        if entry_ids != tuple(sorted(set(entry_ids))):
            raise ValueError("journal entries must be unique and sorted")
        if not set(entry_ids).issubset(self.lock_identities):
            raise ValueError("journal entries must be protected by the recorded lock set")


class MemoryTransactionJournal:
    """在记忆树外保存可恢复的多文档提交清单。"""

    _MAX_RECORD_BYTES = 32 * 1024 * 1024
    _MAX_RECORDS = 1_000

    def __init__(self, root: str | Path, codec: MemoryDocumentCodec) -> None:
        requested = Path(root).expanduser().absolute()
        if requested.is_symlink():
            raise MemoryTransactionJournalError("transaction journal root cannot be a symbolic link")
        if not isinstance(codec, MemoryDocumentCodec):
            raise TypeError("codec must be a MemoryDocumentCodec")
        self.root = requested.resolve(strict=False)
        self.codec = codec

    def prepare(self, record: MemoryTransactionJournalRecord) -> None:
        """仅创建一次 PREPARED 日志，确保先有恢复信息再发布文档。"""

        if not isinstance(record, MemoryTransactionJournalRecord):
            raise TypeError("record must be a MemoryTransactionJournalRecord")
        if record.state is not MemoryTransactionJournalState.PREPARED:
            raise ValueError("new transaction journal must start in PREPARED state")
        encoded = self._encode(record)
        if len(encoded) > self._MAX_RECORD_BYTES:
            raise MemoryTransactionJournalError("transaction journal record is too large")
        try:
            atomic_create_bytes(
                self._path(record.transaction_id),
                encoded,
                artifact_root=self.root,
            )
        except Exception as exc:
            raise MemoryTransactionJournalError("failed to create PREPARED transaction journal") from exc

    def mark(
        self,
        transaction_id: str,
        state: MemoryTransactionJournalState,
        *,
        timestamp: datetime,
    ) -> MemoryTransactionJournalRecord:
        """把 PREPARED 日志推进到一个终态。"""

        normalized_state = MemoryTransactionJournalState(state)
        if normalized_state is MemoryTransactionJournalState.PREPARED:
            raise ValueError("mark requires a terminal transaction state")
        current = self.read(transaction_id)
        if current.state is not MemoryTransactionJournalState.PREPARED:
            if current.state is normalized_state:
                return current
            raise MemoryTransactionJournalError("transaction journal already reached another terminal state")
        updated = MemoryTransactionJournalRecord(
            transaction_id=current.transaction_id,
            state=normalized_state,
            created_at=current.created_at,
            updated_at=timestamp,
            lock_identities=current.lock_identities,
            entries=current.entries,
        )
        try:
            atomic_replace_bytes(
                self._path(transaction_id),
                self._encode(updated),
                artifact_root=self.root,
            )
        except Exception as exc:
            raise MemoryTransactionJournalError("failed to mark transaction journal terminal") from exc
        return updated

    def read(self, transaction_id: str) -> MemoryTransactionJournalRecord:
        """严格读取一条有界日志记录。"""

        path = self._path(transaction_id)
        try:
            encoded = read_regular_bytes(
                path,
                artifact_root=self.root,
                max_bytes=self._MAX_RECORD_BYTES,
            )
            raw = json.loads(encoded)
            return self._parse(raw)
        except Exception as exc:
            if isinstance(exc, MemoryTransactionJournalError):
                raise
            raise MemoryTransactionJournalError("failed to read transaction journal") from exc

    def try_read(
        self,
        transaction_id: str,
    ) -> MemoryTransactionJournalRecord | None:
        """日志不存在时返回 None，存在但损坏时仍明确失败。"""

        try:
            return self.read(transaction_id)
        except MemoryTransactionJournalError as exc:
            if isinstance(exc.__cause__, FileNotFoundError):
                return None
            raise

    def pending(self) -> tuple[MemoryTransactionJournalRecord, ...]:
        """有界枚举仍需恢复的 PREPARED 事务。"""

        if not self.root.exists():
            return ()
        if self.root.is_symlink() or not self.root.is_dir():
            raise MemoryTransactionJournalError("transaction journal root is not a safe directory")
        records: list[MemoryTransactionJournalRecord] = []
        for entry_count, child in enumerate(self.root.iterdir(), start=1):
            if entry_count > self._MAX_RECORDS:
                raise MemoryTransactionJournalError(
                    "transaction journal entry count exceeds its bound"
                )
            if child.is_symlink() or not child.is_file():
                raise MemoryTransactionJournalError("transaction journal contains an unsupported entry")
            temporary_destination = atomic_temporary_destination(child.name)
            if temporary_destination is not None and re.fullmatch(
                r"[0-9a-f]{32}\.json", temporary_destination
            ):
                continue
            if child.suffix != ".json":
                raise MemoryTransactionJournalError("transaction journal contains an unsupported entry")
            records.append(self.read(child.stem))
        return tuple(
            record
            for record in sorted(records, key=lambda item: item.transaction_id)
            if record.state is MemoryTransactionJournalState.PREPARED
        )

    def discard_terminal(self, transaction_id: str) -> None:
        """耐久删除已经标记终态的日志。"""

        record = self.read(transaction_id)
        if record.state is MemoryTransactionJournalState.PREPARED:
            raise MemoryTransactionJournalError("cannot discard a PREPARED transaction journal")
        path = self._path(transaction_id)
        try:
            path.unlink()
            descriptor = os.open(self.root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except Exception as exc:
            raise MemoryTransactionJournalError("failed to discard terminal transaction journal") from exc

    def _path(self, transaction_id: str) -> Path:
        if not isinstance(transaction_id, str) or not _TRANSACTION_ID.fullmatch(transaction_id):
            raise ValueError("transaction_id must be 32 lowercase hexadecimal characters")
        return self.root / f"{transaction_id}.json"

    def _encode(self, record: MemoryTransactionJournalRecord) -> bytes:
        payload = {
            "schema": _SCHEMA,
            "transaction_id": record.transaction_id,
            "state": record.state.value,
            "created_at": self._timestamp(record.created_at),
            "updated_at": self._timestamp(record.updated_at),
            "lock_identities": list(record.lock_identities),
            "entries": [
                {
                    "uri": str(entry.uri),
                    "before": self._document(entry.before),
                    "after": self._document(entry.after),
                }
                for entry in record.entries
            ],
        }
        return canonical_json(payload).encode("utf-8")

    def _parse(self, value: object) -> MemoryTransactionJournalRecord:
        if not isinstance(value, Mapping) or set(value) != {
            "schema",
            "transaction_id",
            "state",
            "created_at",
            "updated_at",
            "lock_identities",
            "entries",
        }:
            raise MemoryTransactionJournalError("transaction journal has an invalid shape")
        if value["schema"] != _SCHEMA:
            raise MemoryTransactionJournalError("transaction journal has an unsupported schema")
        raw_locks = value["lock_identities"]
        raw_entries = value["entries"]
        if not isinstance(raw_locks, list) or not isinstance(raw_entries, list):
            raise MemoryTransactionJournalError("transaction journal arrays are invalid")
        entries: list[MemoryTransactionJournalEntry] = []
        for raw in raw_entries:
            if not isinstance(raw, Mapping) or set(raw) != {"uri", "before", "after"}:
                raise MemoryTransactionJournalError("transaction journal entry has an invalid shape")
            uri = MemoryURI.parse(raw["uri"])
            uri.to_address()
            entries.append(
                MemoryTransactionJournalEntry(
                    uri=uri,
                    before=self._parse_document(raw["before"], uri),
                    after=self._parse_document(raw["after"], uri),
                )
            )
        return MemoryTransactionJournalRecord(
            transaction_id=value["transaction_id"],
            state=MemoryTransactionJournalState(value["state"]),
            created_at=self._parse_timestamp(value["created_at"]),
            updated_at=self._parse_timestamp(value["updated_at"]),
            lock_identities=tuple(raw_locks),
            entries=tuple(entries),
        )

    def _document(self, document: MemoryDocument | None) -> str | None:
        if document is None:
            return None
        encoded = self.codec.encode(document).encode("utf-8")
        return base64.b64encode(encoded).decode("ascii")

    def _parse_document(
        self,
        value: object,
        uri: MemoryURI,
    ) -> MemoryDocument | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise MemoryTransactionJournalError("journal document must be base64 text or null")
        try:
            raw = base64.b64decode(value, validate=True).decode("utf-8")
            return self.codec.decode(raw, expected_address=uri.to_address())
        except Exception as exc:
            raise MemoryTransactionJournalError("journal contains an invalid memory document") from exc

    @staticmethod
    def _timestamp(value: datetime) -> str:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("journal timestamp must be timezone-aware")
        return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")

    @staticmethod
    def _parse_timestamp(value: object) -> datetime:
        if not isinstance(value, str):
            raise MemoryTransactionJournalError("journal timestamp must be a string")
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise MemoryTransactionJournalError("journal timestamp is invalid") from exc
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            raise MemoryTransactionJournalError("journal timestamp lacks a timezone")
        return parsed.astimezone(timezone.utc)


__all__ = [
    "MemoryTransactionJournal",
    "MemoryTransactionJournalEntry",
    "MemoryTransactionJournalError",
    "MemoryTransactionJournalRecord",
    "MemoryTransactionJournalState",
]
