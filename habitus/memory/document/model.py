"""结构化字段与可读正文一致的 L2 记忆文档。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from habitus.infrastructure.tree.metadata import DocumentMetadata
from habitus.memory.document.link import MemoryStoredLink, normalize_stored_links
from habitus.memory.model import MemoryAddress, MemoryKind
from habitus.memory.uri import MemoryURI


@dataclass(frozen=True)
class MemoryDocument:
    """一个已经通过 Schema 校验且可以原子持久化的 L2 文档。"""

    kind: MemoryKind
    address: MemoryAddress
    metadata: DocumentMetadata
    fields: Mapping[str, Any]
    markdown_body: str
    links: tuple[MemoryStoredLink, ...] = ()
    backlinks: tuple[MemoryStoredLink, ...] = ()

    def __post_init__(self) -> None:
        kind = MemoryKind(self.kind)
        object.__setattr__(self, "kind", kind)
        if self.address.kind is not kind:
            raise ValueError("memory document address does not match its kind")
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))
        if not self.markdown_body.strip():
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


__all__ = ["MemoryDocument"]
