"""Behavior L2 到预测投影器之间的只读、防漂移来源视图。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from behavior import (
    BehaviorDocument,
    BehaviorKind,
    BehaviorSchemaRegistry,
    BehaviorTree,
    BehaviorURI,
)
from foundation.integrity import text_digest


@dataclass(frozen=True)
class BehaviorSourceRecord:
    """已经回读并重新通过强类型 Schema 的一个 Behavior 来源。"""

    uri: str
    document: BehaviorDocument
    fields: dict[str, Any]
    digest: str

    def binding(self, *, member_type: str, member_id: str | None = None) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "revision": self.document.metadata.revision,
            "digest": self.digest,
            "member_type": member_type,
            "member_id": member_id,
        }


class BehaviorProjectionSource:
    """只允许规范 Behavior 文档进入 Prediction 投影。"""

    def __init__(
        self,
        tree: BehaviorTree,
        *,
        registry: BehaviorSchemaRegistry | None = None,
    ) -> None:
        if not isinstance(tree, BehaviorTree):
            raise TypeError("tree must be a BehaviorTree")
        if registry is not None and not isinstance(registry, BehaviorSchemaRegistry):
            raise TypeError("registry must be a BehaviorSchemaRegistry")
        self.tree = tree
        self.registry = registry or BehaviorSchemaRegistry.load_default()

    def read(self, uri: BehaviorURI | str, *, expected_kind: BehaviorKind) -> BehaviorSourceRecord:
        parsed = BehaviorURI.parse(uri)
        address = parsed.to_address()
        normalized_kind = BehaviorKind(expected_kind)
        if address.kind is not normalized_kind:
            raise ValueError(f"Behavior source must identify {normalized_kind.value}")
        document = self.tree.read(address)
        if document.kind is not normalized_kind:
            raise ValueError("Behavior source kind does not match its address")
        return BehaviorSourceRecord(
            uri=str(parsed),
            document=document,
            fields=self.registry.validate(normalized_kind, document.fields),
            digest=text_digest(self.tree.document_codec.encode(document)),
        )


__all__ = ["BehaviorProjectionSource", "BehaviorSourceRecord"]
