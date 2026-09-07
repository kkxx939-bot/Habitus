"""可读正文与结构字段一致的情景 L2 文档。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from habitus.foundation.integrity import immutable_snapshot
from habitus.scene.document.link import SceneStoredLink, normalize_stored_links
from habitus.scene.model import SceneAddress
from habitus.scene.uri import SceneURI


def utc_timestamp(value: datetime, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"scene document {field_name} must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"scene document {field_name} must include a timezone")
    return value.astimezone(UTC)


@dataclass(frozen=True)
class SceneDocumentMetadata:
    """由系统控制的版本字段。天是重建单位，文档没有修订号：重建一天就是一整代新文档。"""

    created_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(self, "created_at", utc_timestamp(self.created_at, "created_at"))


@dataclass(frozen=True)
class SceneDocument:
    """已经通过 Schema 校验、字段为 JSON-safe 快照的规范情景文档。"""

    address: SceneAddress
    metadata: SceneDocumentMetadata
    fields: Mapping[str, Any]
    markdown_body: str
    links: tuple[SceneStoredLink, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.address, SceneAddress):
            raise TypeError("scene document address must be a SceneAddress")
        if not isinstance(self.metadata, SceneDocumentMetadata):
            raise TypeError("scene document metadata must be SceneDocumentMetadata")
        if not isinstance(self.fields, Mapping) or any(not isinstance(name, str) for name in self.fields):
            raise TypeError("scene document fields must be a mapping with string keys")
        object.__setattr__(self, "fields", immutable_snapshot(self.fields))
        if not isinstance(self.markdown_body, str) or not self.markdown_body.strip():
            raise ValueError("scene document Markdown body must be non-empty")
        if not self.markdown_body.endswith("\n"):
            raise ValueError("scene document Markdown body must end with a newline")
        links = normalize_stored_links(self.links, label="scene document links")
        uri = SceneURI.from_address(self.address)
        if any(link.from_uri != uri for link in links):
            raise ValueError("scene document forward link has the wrong source URI")
        object.__setattr__(self, "links", links)

    @property
    def uri(self) -> SceneURI:
        return SceneURI.from_address(self.address)


__all__ = ["SceneDocument", "SceneDocumentMetadata", "utc_timestamp"]
