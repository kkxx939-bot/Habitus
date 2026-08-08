"""可读正文与结构字段一致的行为语义 L2 文档。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from behavior.document.link import BehaviorStoredLink, normalize_stored_links
from behavior.model import BehaviorAddress, BehaviorKind
from behavior.uri import BehaviorURI
from foundation.integrity import immutable_snapshot


def utc_timestamp(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"behavior document {field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"behavior document {field_name} must include a timezone")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class BehaviorDocumentMetadata:
    """完全由系统控制、不能由融合模型填写的版本字段。"""

    revision: int
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        if isinstance(self.revision, bool) or not isinstance(self.revision, int) or self.revision <= 0:
            raise ValueError("behavior document revision must be a positive integer")
        created_at = utc_timestamp(self.created_at, "created_at")
        updated_at = utc_timestamp(self.updated_at, "updated_at")
        if updated_at < created_at:
            raise ValueError("behavior document updated_at cannot precede created_at")
        object.__setattr__(self, "created_at", created_at)
        object.__setattr__(self, "updated_at", updated_at)

    @classmethod
    def initial(cls, timestamp: datetime) -> BehaviorDocumentMetadata:
        normalized = utc_timestamp(timestamp, "timestamp")
        return cls(revision=1, created_at=normalized, updated_at=normalized)

    def next_revision(self, timestamp: datetime) -> BehaviorDocumentMetadata:
        normalized = utc_timestamp(timestamp, "timestamp")
        if normalized < self.updated_at:
            raise ValueError("next behavior revision timestamp cannot move backwards")
        return BehaviorDocumentMetadata(
            revision=self.revision + 1,
            created_at=self.created_at,
            updated_at=normalized,
        )


@dataclass(frozen=True)
class BehaviorDocument:
    """已经通过 Schema 校验、字段为 JSON-safe 快照的规范 L2 文档。"""

    kind: BehaviorKind
    address: BehaviorAddress
    metadata: BehaviorDocumentMetadata
    fields: Mapping[str, Any]
    markdown_body: str
    links: tuple[BehaviorStoredLink, ...] = ()
    backlinks: tuple[BehaviorStoredLink, ...] = ()

    def __post_init__(self) -> None:
        kind = BehaviorKind(self.kind)
        object.__setattr__(self, "kind", kind)
        if not isinstance(self.address, BehaviorAddress) or self.address.kind is not kind:
            raise ValueError("behavior document address does not match its kind")
        if not isinstance(self.metadata, BehaviorDocumentMetadata):
            raise TypeError("behavior document metadata must be BehaviorDocumentMetadata")
        if not isinstance(self.fields, Mapping) or any(not isinstance(name, str) for name in self.fields):
            raise TypeError("behavior document fields must be a mapping with string keys")
        object.__setattr__(self, "fields", immutable_snapshot(self.fields))
        if not isinstance(self.markdown_body, str) or not self.markdown_body.strip():
            raise ValueError("behavior document Markdown body must be non-empty")
        if not self.markdown_body.endswith("\n"):
            raise ValueError("behavior document Markdown body must end with a newline")
        links = normalize_stored_links(self.links, label="behavior document links")
        backlinks = normalize_stored_links(self.backlinks, label="behavior document backlinks")
        uri = BehaviorURI.from_address(self.address)
        if any(link.from_uri != uri for link in links):
            raise ValueError("behavior document forward link has the wrong source URI")
        if any(link.to_uri != uri for link in backlinks):
            raise ValueError("behavior document backlink has the wrong target URI")
        object.__setattr__(self, "links", links)
        object.__setattr__(self, "backlinks", backlinks)


__all__ = ["BehaviorDocument", "BehaviorDocumentMetadata", "utc_timestamp"]
