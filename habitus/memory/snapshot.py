"""使用严格 ``habitus://`` L2 URI 读取完整记忆快照。"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeAlias

from habitus.foundation.integrity.canonical_json import canonical_json
from habitus.infrastructure.editor.snapshot import (
    SnapshotBatch,
    SnapshotReadConfig,
    SnapshotReader,
    SnapshotReadLimitError,
    VersionedSnapshot,
)
from habitus.memory.document import MemoryDocument
from habitus.memory.tree import MemoryTree
from habitus.memory.uri import MemoryURI

MemorySnapshot: TypeAlias = VersionedSnapshot[MemoryDocument]
MemorySnapshotBatch: TypeAlias = SnapshotBatch[MemoryDocument]


class MemorySnapshotReader:
    """把记忆领域的 URI 和文档模型适配到公共版本快照机制。"""

    def __init__(
        self,
        tree: MemoryTree,
        *,
        config: SnapshotReadConfig | None = None,
    ) -> None:
        self.tree = tree
        self._reader = SnapshotReader[MemoryDocument](
            load=self._load,
            revision_of=lambda document: document.metadata.revision,
            serialize=self._serialize,
            config=config,
        )

    @property
    def config(self) -> SnapshotReadConfig:
        """返回当前批量读取使用的显式资源边界。"""

        return self._reader.config

    def read(self, uri: MemoryURI | str) -> MemorySnapshot:
        """读取一个 L2 URI；目录和 L0/L1 URI 会被严格拒绝。"""

        return self._reader.read(str(self._document_uri(uri)))

    def read_many(self, uris: Iterable[MemoryURI | str]) -> MemorySnapshotBatch:
        """先验证全部 L2 URI，再执行去重且有界的批量读取。"""

        identities_list: list[str] = []
        for uri in uris:
            identities_list.append(str(self._document_uri(uri)))
            if len(identities_list) > self.config.max_items:
                raise SnapshotReadLimitError("snapshot batch exceeds its configured item limit")
        return self._reader.read_many(tuple(identities_list))

    def _load(self, identity: str) -> MemoryDocument:
        uri = self._document_uri(identity)
        return self.tree.read(uri.to_address())

    @staticmethod
    def _document_uri(value: MemoryURI | str) -> MemoryURI:
        parsed = MemoryURI.parse(value)
        parsed.to_address()
        return parsed

    @staticmethod
    def _serialize(document: MemoryDocument) -> bytes:
        """序列化完整规范内容，生成与读取版本绑定的稳定摘要输入。"""

        payload = {
            "memory_type": document.kind.value,
            "address": str(MemoryURI.from_address(document.address)),
            "revision": document.metadata.revision,
            "created_at": document.metadata.created_at,
            "updated_at": document.metadata.updated_at,
            "fields": document.fields,
            "markdown_body": document.markdown_body,
            "links": [link.to_dict() for link in document.links],
            "backlinks": [link.to_dict() for link in document.backlinks],
        }
        return canonical_json(payload).encode("utf-8")


__all__ = [
    "MemorySnapshot",
    "MemorySnapshotBatch",
    "MemorySnapshotReader",
]
