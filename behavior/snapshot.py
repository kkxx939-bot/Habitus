"""使用公共版本快照机制读取完整行为语义 L2。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeAlias

from behavior.document import BehaviorDocument
from behavior.tree import BehaviorTree
from behavior.uri import BehaviorURI
from infrastructure.editor.snapshot import (
    SnapshotBatch,
    SnapshotReadConfig,
    SnapshotReader,
    VersionedSnapshot,
)

BehaviorSnapshot: TypeAlias = VersionedSnapshot[BehaviorDocument]
BehaviorSnapshotBatch: TypeAlias = SnapshotBatch[BehaviorDocument]


class BehaviorSnapshotReader:
    """把 Behavior URI 和文档适配到领域无关的快照读取器。"""

    def __init__(
        self,
        tree: BehaviorTree,
        *,
        config: SnapshotReadConfig | None = None,
    ) -> None:
        if not isinstance(tree, BehaviorTree):
            raise TypeError("tree must be a BehaviorTree")
        self.tree = tree
        if config is None:
            default = SnapshotReadConfig()
            max_item_bytes = max(default.max_item_bytes, tree.document_config.max_encoded_bytes)
            config = SnapshotReadConfig(
                max_items=default.max_items,
                max_item_bytes=max_item_bytes,
                max_total_bytes=max(default.max_total_bytes, max_item_bytes),
            )
        self._reader = SnapshotReader[BehaviorDocument](
            load=self._load,
            revision_of=lambda document: document.metadata.revision,
            serialize=self._serialize,
            config=config,
        )

    @property
    def config(self) -> SnapshotReadConfig:
        return self._reader.config

    def read(self, uri: BehaviorURI | str) -> BehaviorSnapshot:
        parsed = self._document_uri(uri)
        return self._reader.read(str(parsed))

    def read_many(self, uris: Iterable[BehaviorURI | str]) -> BehaviorSnapshotBatch:
        if isinstance(uris, str) or not isinstance(uris, Iterable):
            raise TypeError("uris must be an iterable of BehaviorURI or string values")
        return self._reader.read_many(str(self._document_uri(uri)) for uri in uris)

    def _load(self, identity: str) -> BehaviorDocument:
        uri = self._document_uri(identity)
        return self.tree.read(uri.to_address())

    @staticmethod
    def _document_uri(value: BehaviorURI | str) -> BehaviorURI:
        parsed = BehaviorURI.parse(value)
        parsed.to_address()
        return parsed

    def _serialize(self, document: BehaviorDocument) -> bytes:
        return self.tree.document_codec.encode(document).encode("utf-8")


__all__ = [
    "BehaviorSnapshot",
    "BehaviorSnapshotBatch",
    "BehaviorSnapshotReader",
]
