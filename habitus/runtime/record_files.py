"""习惯卡片与提醒记录共用的文件读写（裁定 113）：树上一个可改的 Markdown 文件，正文由字段渲染、末尾一个 ``HABITUS_FIELDS``
注释（类型键 ``type``、版本、建立与更新时间、字段），与卡片、事件同一种外壳（``infrastructure.tree``）。

只做记录、由系统写：读回时外壳必须严格合法、类型对、字段键集对，并且按字段重新渲染、重新编码与盘上逐字节相同（手改过的、
旧写法的都按坏文件报错，不静默修）。"没变不写"比的是字段：字段相同就不写，版本与时间不动。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from habitus.infrastructure.tree.envelope import (
    TYPE_KEY,
    TreeEnvelopeError,
    decode,
    encode,
    format_timestamp,
    parse_revision,
    parse_timestamp,
    require_plain_body,
)
from habitus.infrastructure.tree.files import TreeFiles
from habitus.infrastructure.tree.metadata import DocumentMetadata

_KEYS = frozenset({TYPE_KEY, "revision", "created_at", "updated_at", "fields"})


class RecordFileError(ValueError):
    """习惯卡片或提醒记录的文件不合格：外壳、类型、字段键集不对，或者不是由字段渲染出来的那一份。"""


@dataclass(frozen=True)
class StoredRecord:
    metadata: DocumentMetadata
    fields: dict[str, Any]


class RecordFiles:
    """树根下某一类记录的文件。``kind`` 是类型键的值，``keys`` 是字段的键集，``render`` 把字段渲染成正文。"""

    def __init__(
        self,
        tree_root: str | Path,
        *,
        kind: str,
        keys: frozenset[str],
        render: Callable[[Mapping[str, Any]], str],
        max_children: int,
        max_bytes: int,
    ) -> None:
        self.files = TreeFiles(tree_root, max_children=max_children, error=RecordFileError, conflict=RecordFileError)
        self.root = self.files.root
        self.kind, self.keys, self.render, self.max_bytes = kind, keys, render, max_bytes

    def path(self, *parts: str) -> Path:
        """树根下的这个相对路径（逐级按规范身份解析，别名报错）。"""

        return self.files.resolve(Path(*parts), leaf_is_file=True)

    def exists(self, path: Path) -> bool:
        return self.files.file_exists(path)

    def read(self, path: Path) -> StoredRecord | None:
        """盘上那一份；没有是 None。"""

        if not self.files.file_exists(path):
            return None
        raw = self.files.read_utf8(path, label=f"{self.kind} record", max_bytes=self.max_bytes)
        try:
            metadata = decode(raw, keys=_KEYS)[1]
            stored = StoredRecord(
                DocumentMetadata(
                    parse_revision(metadata["revision"]),
                    parse_timestamp(metadata["created_at"], "created_at"),
                    parse_timestamp(metadata["updated_at"], "updated_at"),
                ),
                metadata["fields"],
            )
            shaped = metadata[TYPE_KEY] == self.kind and set(stored.fields) == self.keys
            rendered = shaped and self._encoded(stored) == raw
        except (TreeEnvelopeError, TypeError, KeyError, ValueError) as exc:
            raise RecordFileError(f"{path.name} is not a valid {self.kind} record: {exc}") from exc
        if not rendered:
            raise RecordFileError(f"{path.name} is not a {self.kind} record rendered from its own fields")
        return stored

    def write(self, path: Path, fields: Mapping[str, Any], *, old: StoredRecord | None, now: datetime) -> bool:
        """字段和 ``old`` 相同就不写（返回 False）；否则推进一个版本（新建是版本 1）原子替换。"""

        fields = dict(fields)
        if set(fields) != self.keys:
            raise RecordFileError(f"{self.kind} record fields have an invalid shape")
        if old is not None and old.fields == fields:
            return False
        # 更新时间不往回走：时钟回拨时沿用上一次的
        metadata = (
            DocumentMetadata.initial(now)
            if old is None
            else old.metadata.next_revision(max(now, old.metadata.updated_at))
        )
        payload = self._encoded(StoredRecord(metadata, fields)).encode("utf-8")
        if len(payload) > self.max_bytes:
            raise RecordFileError(f"{self.kind} record exceeds its byte bound")
        self.files.ensure_directory(path.parent, check_capacity=True)
        self.files.require_child_capacity(path.parent, (path.name,))
        self.files.replace(path, payload)
        return True

    def _encoded(self, stored: StoredRecord) -> str:
        body = self.render(stored.fields)
        require_plain_body(body)
        return encode(
            body,
            {
                TYPE_KEY: self.kind,
                "revision": stored.metadata.revision,
                "created_at": format_timestamp(stored.metadata.created_at),
                "updated_at": format_timestamp(stored.metadata.updated_at),
                "fields": stored.fields,
            },
        )


__all__ = ["RecordFileError", "RecordFiles", "StoredRecord"]
