"""可读正文与结构字段一致的行为语义 L2 文档。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from habitus.behavior.model import BehaviorAddress, BehaviorKind
from habitus.foundation.integrity import immutable_snapshot
from habitus.infrastructure.tree.metadata import DocumentMetadata


@dataclass(frozen=True)
class BehaviorDocument:
    """已经通过 Schema 校验、字段为 JSON-safe 快照的规范 L2 文档。"""

    kind: BehaviorKind
    address: BehaviorAddress
    metadata: DocumentMetadata
    fields: Mapping[str, Any]
    markdown_body: str

    def __post_init__(self) -> None:
        kind = BehaviorKind(self.kind)
        object.__setattr__(self, "kind", kind)
        if self.address.kind is not kind:
            raise ValueError("behavior document address does not match its kind")
        object.__setattr__(self, "fields", immutable_snapshot(self.fields))
        if not self.markdown_body.strip():
            raise ValueError("behavior document Markdown body must be non-empty")
        if not self.markdown_body.endswith("\n"):
            raise ValueError("behavior document Markdown body must end with a newline")


__all__ = ["BehaviorDocument"]
