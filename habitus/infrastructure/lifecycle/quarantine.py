"""隔离区：确定性失败的对象存一份副本搬到这里，不再钉住热存储、不再每轮重试。

副本是重放的唯一来源（原件随后会被正常释放），所以先落副本、再由调用方删原件。重放由调用方编排：
``payload`` 取副本、放回原处成功之后再 ``discard``——中途崩溃副本还在，可以重来。
放弃的项按天数清（``purge``），没放弃的不自动清。
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

from habitus.foundation.ids import require_safe_path_segment
from habitus.foundation.integrity import canonical_json, text_digest
from habitus.infrastructure.store.filesystem import (
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    atomic_replace_bytes,
    durable_rmdir,
    durable_unlink,
    list_real_directory,
    read_regular_bytes,
)

_SCHEMA = "lifecycle_quarantine_v1"
_ITEM = "item.json"
_PAYLOAD = "payload.bin"
_MAX_ITEM_BYTES = 64 * 1024


@dataclass(frozen=True)
class QuarantinedItem:
    item_id: str
    reason: str
    quarantined_at: datetime
    meta: Mapping[str, str] = field(default_factory=dict)
    abandoned_at: datetime | None = None


class Quarantine:
    def __init__(
        self,
        root: str | Path,
        *,
        owner: str,
        clock: Callable[[], datetime],
        max_items: int = 100_000,
        max_payload_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        for label, value in (("max_items", max_items), ("max_payload_bytes", max_payload_bytes)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        self.root = Path(root).expanduser().resolve(strict=False)
        self.owner = require_safe_path_segment(owner, "owner")
        self.directory = self.root / "lifecycle" / "quarantine" / self.owner
        self.max_items = max_items
        self.max_payload_bytes = max_payload_bytes
        self._clock = clock

    def put(self, item_id: str, payload: bytes, *, reason: str, meta: Mapping[str, str] | None = None) -> bool:
        """存副本；同一个 ``item_id`` 已在隔离区时不覆盖，返回 False。"""

        if not isinstance(payload, bytes):
            raise TypeError("payload must be bytes")
        if len(payload) > self.max_payload_bytes:
            raise ValueError("quarantined payload exceeds its size bound")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("quarantine reason must be non-empty text")
        folder = self._folder(item_id)
        try:
            created = atomic_create_bytes(folder / _PAYLOAD, payload, artifact_root=self.root)
        except ImmutableArtifactConflictError:
            created = False  # 同一项已在，内容不同：留先到的那份
        if created or self._read_item(folder) is None:
            # 副本落了、记录还没写就崩溃过：这次把记录补上，否则这一项永远看不见
            record = QuarantinedItem(
                item_id=item_id, reason=reason.strip(), quarantined_at=self._now(), meta=dict(meta or {})
            )
            self._write_item(folder, record)
        return created

    def items(self) -> tuple[QuarantinedItem, ...]:
        found = []
        for entry in list_real_directory(self.directory, artifact_root=self.root, max_entries=self.max_items):
            if entry.is_dir():
                item = self._read_item(self.directory / entry.name)
                if item is not None:
                    found.append(item)
        return tuple(sorted(found, key=lambda item: (item.quarantined_at, item.item_id)))

    def count(self) -> int:
        """隔离区里有几项（容量读数用）：只数目录项，超过 ``max_items`` 也照样数得出来。"""

        try:
            with os.scandir(self.directory) as entries:
                return sum(1 for entry in entries if entry.is_dir())
        except FileNotFoundError:
            return 0

    def contains(self, item_id: str) -> bool:
        """这一项在隔离区里（副本与记录都在）。"""

        return self._read_item(self._folder(item_id)) is not None

    def payload(self, item_id: str) -> bytes:
        return read_regular_bytes(
            self._folder(item_id) / _PAYLOAD, artifact_root=self.root, max_bytes=self.max_payload_bytes
        )

    def abandon(self, item_id: str) -> bool:
        folder = self._folder(item_id)
        item = self._read_item(folder)
        if item is None or item.abandoned_at is not None:
            return False
        self._write_item(folder, replace(item, abandoned_at=self._now()))
        return True

    def discard(self, item_id: str) -> bool:
        """删掉这一项（重放成功后，或清理放弃的项）。"""

        folder = self._folder(item_id)
        removed = durable_unlink(folder / _PAYLOAD, artifact_root=self.root)
        removed = durable_unlink(folder / _ITEM, artifact_root=self.root) or removed
        durable_rmdir(folder, artifact_root=self.root)
        return removed

    def purge(self, before: datetime) -> int:
        """清掉放弃时间早于 ``before`` 的项；没放弃的不动。"""

        doomed = [item.item_id for item in self.items() if item.abandoned_at is not None and item.abandoned_at < before]
        return sum(1 for item_id in doomed if self.discard(item_id))

    # ── 内部 ──────────────────────────────────────────────────────────────────────────

    def _now(self) -> datetime:
        now = self._clock()
        if now.utcoffset() is None:
            raise ValueError("clock must return timezone-aware datetimes")
        return now

    def _folder(self, item_id: str) -> Path:
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("item_id must be non-empty text")
        return self.directory / text_digest(item_id)[:40]

    def _write_item(self, folder: Path, item: QuarantinedItem) -> None:
        payload = {
            "schema": _SCHEMA,
            "item_id": item.item_id,
            "reason": item.reason,
            "quarantined_at": item.quarantined_at.isoformat(),
            "meta": dict(item.meta),
            "abandoned_at": None if item.abandoned_at is None else item.abandoned_at.isoformat(),
        }
        atomic_replace_bytes(folder / _ITEM, canonical_json(payload).encode("utf-8"), artifact_root=self.root)

    def _read_item(self, folder: Path) -> QuarantinedItem | None:
        try:
            encoded = read_regular_bytes(folder / _ITEM, artifact_root=self.root, max_bytes=_MAX_ITEM_BYTES)
        except FileNotFoundError:
            return None  # 副本落了、记录还没写就崩溃：下次 put 同一个 id 时补写
        payload = json.loads(encoded.decode("utf-8"))
        if payload.get("schema") != _SCHEMA:
            raise ValueError(f"quarantine item {folder} has an unexpected schema")
        abandoned = payload.get("abandoned_at")
        return QuarantinedItem(
            item_id=str(payload["item_id"]),
            reason=str(payload["reason"]),
            quarantined_at=datetime.fromisoformat(str(payload["quarantined_at"])),
            meta={str(key): str(value) for key, value in dict(payload.get("meta") or {}).items()},
            abandoned_at=None if abandoned is None else datetime.fromisoformat(str(abandoned)),
        )


__all__ = ["QuarantinedItem", "Quarantine"]
