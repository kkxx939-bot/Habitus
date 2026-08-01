"""结构化字段与可读正文一致的 L2 记忆文档。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any

from memory.document.link import MemoryStoredLink, normalize_stored_links
from memory.model import MemoryAddress, MemoryKind
from memory.uri import MemoryURI


def _utc_timestamp(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"memory document {field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"memory document {field_name} must include a timezone")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class MemoryDocumentMetadata:
    """不由 LLM 或内容 Schema 控制的文档系统字段。"""

    revision: int
    created_at: datetime
    updated_at: datetime
    last_confirmed_at: datetime | None

    def __post_init__(self) -> None:
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision <= 0:
            raise ValueError("memory document revision must be a positive integer")
        created_at = _utc_timestamp(self.created_at, "created_at")
        updated_at = _utc_timestamp(self.updated_at, "updated_at")
        last_confirmed_at = (
            None
            if self.last_confirmed_at is None
            else _utc_timestamp(self.last_confirmed_at, "last_confirmed_at")
        )
        if updated_at < created_at:
            raise ValueError("memory document updated_at cannot precede created_at")
        if last_confirmed_at is not None and last_confirmed_at > updated_at:
            raise ValueError("memory document last_confirmed_at cannot follow updated_at")
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)
        object.__setattr__(self, "last_confirmed_at", last_confirmed_at)

    @classmethod
    def initial(
        cls,
        timestamp: datetime,
        *,
        confirmed: bool = False,
    ) -> MemoryDocumentMetadata:
        normalized = _utc_timestamp(timestamp, "timestamp")
        if not isinstance(confirmed, bool):
            raise TypeError("confirmed must be boolean")
        return cls(
            revision=1,
            created_at=normalized,
            updated_at=normalized,
            last_confirmed_at=normalized if confirmed else None,
        )

    def next_revision(
        self,
        timestamp: datetime,
        *,
        refresh_confirmation: bool = False,
    ) -> MemoryDocumentMetadata:
        """构造下一版本元数据；实际旧版本校验仍由 Memory Editor 负责。"""

        normalized = _utc_timestamp(timestamp, "timestamp")
        if not isinstance(refresh_confirmation, bool):
            raise TypeError("refresh_confirmation must be boolean")
        if normalized < self.updated_at:
            raise ValueError("next memory revision timestamp cannot move backwards")
        return MemoryDocumentMetadata(
            revision=self.revision + 1,
            created_at=self.created_at,
            updated_at=normalized,
            last_confirmed_at=(normalized if refresh_confirmation else self.last_confirmed_at),
        )


@dataclass(frozen=True)
class MemoryDocument:
    """一个已经通过 Schema 校验且可以原子持久化的 L2 文档。"""

    kind: MemoryKind
    address: MemoryAddress
    metadata: MemoryDocumentMetadata
    fields: Mapping[str, Any]
    markdown_body: str
    links: tuple[MemoryStoredLink, ...] = ()
    backlinks: tuple[MemoryStoredLink, ...] = ()

    def __post_init__(self) -> None:
        kind = MemoryKind(self.kind)
        object.__setattr__(self, "kind", kind)
        if not isinstance(self.address, MemoryAddress) or self.address.kind is not kind:
            raise ValueError("memory document address does not match its kind")
        if not isinstance(self.metadata, MemoryDocumentMetadata):
            raise TypeError("memory document metadata must be MemoryDocumentMetadata")
        if kind is MemoryKind.INTENTION:
            if self.metadata.last_confirmed_at is None:
                raise ValueError("Intention memory requires last_confirmed_at")
        elif self.metadata.last_confirmed_at is not None:
            raise ValueError("only Intention memory accepts last_confirmed_at")
        if not isinstance(self.fields, Mapping) or any(not isinstance(name, str) for name in self.fields):
            raise TypeError("memory document fields must be a mapping with string keys")
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))
        if not isinstance(self.markdown_body, str) or not self.markdown_body.strip():
            raise ValueError("memory document Markdown body must be non-empty")
        if not self.markdown_body.endswith("\n"):
            raise ValueError("memory document Markdown body must end with a newline")
        links = normalize_stored_links(self.links, label="memory document links")
        backlinks = normalize_stored_links(
            self.backlinks,
            label="memory document backlinks",
        )
        uri = MemoryURI.from_address(self.address)
        if any(link.from_uri != uri for link in links):
            raise ValueError("memory document forward link has the wrong source URI")
        if any(link.to_uri != uri for link in backlinks):
            raise ValueError("memory document backlink has the wrong target URI")
        object.__setattr__(self, "links", links)
        object.__setattr__(self, "backlinks", backlinks)


__all__ = ["MemoryDocument", "MemoryDocumentMetadata"]
