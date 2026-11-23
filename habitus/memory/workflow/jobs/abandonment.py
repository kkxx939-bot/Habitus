"""人工放弃 FAILED 记忆任务后留下的处置记录文件（``job_abandonments/<来源键>.json``）的读写。"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import fields
from datetime import date
from pathlib import Path

from habitus.foundation.decoding import int_field, text_field
from habitus.foundation.integrity import canonical_json
from habitus.foundation.timestamps import format_utc_z, parse_utc_z
from habitus.infrastructure.store.filesystem import durable_unlink, read_regular_bytes
from habitus.memory.conversation import ConversationAddress
from habitus.memory.workflow.jobs.model import MemoryJob, MemoryJobAbandonment, MemoryJobConfig, MemoryJobError

_NORMALIZED_MESSAGE = "{} must be non-empty normalized text"


def memory_job_source_key(address: ConversationAddress, segment_id: str, source_segment_digest: str) -> str:
    """一段对话片段的来源键：任务文件与放弃记录都以它命名。"""

    return hashlib.sha256(
        (f"{address.identity}\0{address.started_on.isoformat()}\0{segment_id}\0{source_segment_digest}").encode()
    ).hexdigest()


class MemoryJobAbandonmentFiles:
    """一个来源一份、只增不改的处置记录；写入与队列栅栏由 ``MemoryJobStore`` 负责。"""

    def __init__(self, artifact_root: Path, directory: Path, config: MemoryJobConfig) -> None:
        self.artifact_root = artifact_root
        self.directory = directory
        self.config = config

    def path(self, source_key: str) -> Path:
        if not MemoryJob._hex(source_key, 64):
            raise ValueError("memory job source key must be lowercase SHA-256 text")
        return self.directory / f"{source_key}.json"

    def try_read(self, source_key: str) -> MemoryJobAbandonment | None:
        try:
            return self.read(self.path(source_key))
        except MemoryJobError as exc:
            if isinstance(exc.__cause__, FileNotFoundError):
                return None
            raise

    def discard(self, record: MemoryJobAbandonment) -> bool:
        """这一段的原文已经不在、放弃满原文保留期后删掉这份放弃记录（裁定 111）。"""

        address = ConversationAddress(record.conversation_id, record.started_on)
        key = memory_job_source_key(address, record.segment_id, record.source_segment_digest)
        return durable_unlink(self.path(key), artifact_root=self.artifact_root)

    def list(self) -> tuple[MemoryJobAbandonment, ...]:
        if not self.directory.exists():
            return ()
        if self.directory.is_symlink() or not self.directory.is_dir():
            raise MemoryJobError("memory job abandonments path is not a safe directory")
        records: list[MemoryJobAbandonment] = []
        for entry_count, child in enumerate(self.directory.iterdir(), start=1):
            if entry_count > self.config.max_files:
                raise MemoryJobError("memory job abandonment count exceeds its configured bound")
            if child.suffix != ".json" or child.is_symlink() or not child.is_file():
                raise MemoryJobError("memory job abandonments directory contains an unsupported entry")
            records.append(self.read(child))
        return tuple(sorted(records, key=lambda item: item.memory_sequence))

    def encode(self, record: MemoryJobAbandonment) -> bytes:
        encoded = canonical_json(
            {
                "schema": record.SCHEMA_VERSION,
                "memory_sequence": record.memory_sequence,
                "conversation_id": record.conversation_id,
                "started_on": record.started_on.isoformat(),
                "segment_id": record.segment_id,
                "source_segment_digest": record.source_segment_digest,
                "attempts": record.attempts,
                "last_error": record.last_error,
                "reason": record.reason,
                "abandoned_at": format_utc_z(record.abandoned_at, microseconds=True),
            }
        ).encode("utf-8")
        if len(encoded) > self.config.max_file_bytes:
            raise MemoryJobError("memory job abandonment exceeds its configured file bound")
        return encoded

    def read(self, path: Path) -> MemoryJobAbandonment:
        try:
            encoded = read_regular_bytes(path, artifact_root=self.artifact_root, max_bytes=self.config.max_file_bytes)
            value = json.loads(encoded)
            if isinstance(value, Mapping) and value.get("schema") != MemoryJobAbandonment.SCHEMA_VERSION:
                raise MemoryJobError(
                    f"memory job abandonment uses schema {value.get('schema')!r}, this Habitus only reads "
                    f"{MemoryJobAbandonment.SCHEMA_VERSION!r}; old workflow state is not migrated"
                )
            expected = {"schema", *(field.name for field in fields(MemoryJobAbandonment))}
            if not isinstance(value, Mapping) or set(value) != expected:
                raise MemoryJobError("memory job abandonment has an invalid shape")
            record = MemoryJobAbandonment(
                memory_sequence=int_field(value["memory_sequence"], "memory_sequence must be a positive integer"),
                conversation_id=text_field(value["conversation_id"], _NORMALIZED_MESSAGE.format("conversation_id")),
                started_on=date.fromisoformat(value["started_on"]),
                segment_id=text_field(value["segment_id"], _NORMALIZED_MESSAGE.format("segment_id")),
                source_segment_digest=text_field(
                    value["source_segment_digest"], "source_segment_digest must be lowercase SHA-256 text"
                ),
                attempts=int_field(value["attempts"], "attempts must be a positive integer"),
                last_error=text_field(value["last_error"], "last_error must be non-empty bounded text"),
                reason=text_field(value["reason"], "reason must be non-empty bounded text"),
                abandoned_at=parse_utc_z(
                    value["abandoned_at"],
                    invalid="memory job timestamp must be a string",
                    naive="memory job timestamp must include a timezone",
                    empty_is_invalid=False,
                ),
            )
            if encoded != self.encode(record):
                raise MemoryJobError("memory job abandonment is not canonically encoded")
            return record
        except Exception as exc:
            if isinstance(exc, MemoryJobError):
                raise
            raise MemoryJobError("failed to read memory job abandonment") from exc


__all__ = ["MemoryJobAbandonmentFiles", "memory_job_source_key"]
