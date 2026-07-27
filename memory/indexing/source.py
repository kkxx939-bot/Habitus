"""从当前 MemoryTree 生成可重建的 L0/L1/L2 向量源记录。"""

from __future__ import annotations

from collections.abc import Mapping

from memory.indexing.config import MemoryVectorIndexConfig
from memory.indexing.model import MemoryIndexSource
from memory.intention import memory_index_kind
from memory.model import MemoryAddress, MemoryDirectory, MemoryKind, MemoryLevel
from memory.tree import MemoryTree
from memory.uri import MemoryURI, MemoryURINodeType


class MemoryIndexSourceReader:
    """只读取记忆真相源；不生成向量，也不访问具体向量数据库。"""

    def __init__(self, tree: MemoryTree, *, config: MemoryVectorIndexConfig) -> None:
        if not isinstance(tree, MemoryTree):
            raise TypeError("tree must be MemoryTree")
        if not isinstance(config, MemoryVectorIndexConfig):
            raise TypeError("config must be MemoryVectorIndexConfig")
        self.tree = tree
        self.config = config

    def read_uri(self, uri: MemoryURI | str) -> MemoryIndexSource | None:
        parsed = MemoryURI.parse(uri)
        if parsed.node_type is MemoryURINodeType.DOCUMENT:
            address = parsed.to_address()
            if not self.tree.exists(address):
                return None
            return self._document(address)
        if parsed.node_type is MemoryURINodeType.LAYER:
            directory, level = parsed.to_layer()
            return self._layer(directory, level)
        raise ValueError("memory index source URI must identify a document or semantic layer")

    def read_directory_layers(self, directory: MemoryDirectory) -> tuple[MemoryIndexSource, ...]:
        if not isinstance(directory, MemoryDirectory):
            raise TypeError("directory must be MemoryDirectory")
        sources = (
            self._layer(directory, MemoryLevel.ABSTRACT),
            self._layer(directory, MemoryLevel.OVERVIEW),
        )
        return tuple(source for source in sources if source is not None)

    def walk(self) -> tuple[MemoryIndexSource, ...]:
        """按 URI 排序有界遍历整棵树，供启动恢复和管理重建使用。"""

        self.tree.initialize()
        pending = [MemoryDirectory.root()]
        visited: set[MemoryDirectory] = set()
        sources: dict[str, MemoryIndexSource] = {}
        while pending:
            directory = pending.pop()
            if directory in visited:
                continue
            visited.add(directory)
            if len(visited) > self.config.max_directories:
                raise ValueError("memory vector rebuild exceeded its directory bound")
            for address in self.tree.direct_addresses(
                directory,
                limit=self.config.max_direct_entries,
            ):
                source = self._document(address)
                if source is not None:
                    sources[source.identity] = source
                    self._require_record_bound(sources)
            for source in self.read_directory_layers(directory):
                sources[source.identity] = source
                self._require_record_bound(sources)
            children = self.tree.child_directories(
                directory,
                limit=self.config.max_direct_entries,
            )
            pending.extend(reversed(children))
        return tuple(sources[identity] for identity in sorted(sources))

    def _document(self, address: MemoryAddress) -> MemoryIndexSource | None:
        document = self.tree.read(address)
        uri = MemoryURI.from_address(address)
        content = self._bounded(f"[{address.kind.value}] {uri.decoded_path}\n{document.markdown_body.strip()}")
        intention_status = (
            document.fields.get("status") if address.kind is MemoryKind.INTENTION else None
        )
        return MemoryIndexSource(
            uri=uri,
            level=MemoryLevel.DETAIL,
            directory=MemoryDirectory.for_address(address),
            content=content,
            index_kind=memory_index_kind(
                address.kind,
                intention_status=intention_status,
            ),
            revision=document.metadata.revision,
        )

    def _layer(
        self,
        directory: MemoryDirectory,
        level: MemoryLevel,
    ) -> MemoryIndexSource | None:
        if not self.tree.directory_exists(directory) or not self.tree.layer_exists(directory, level):
            return None
        content = self.tree.read_layer_bounded(
            directory,
            level,
            max_bytes=self.config.max_record_chars * 4 + 4,
        ).strip()
        if not content:
            raise ValueError("memory semantic layer is empty")
        uri = MemoryURI.from_layer(directory, level)
        return MemoryIndexSource(
            uri=uri,
            level=level,
            directory=directory,
            content=self._bounded(f"[directory] {uri.decoded_path}\n{content}"),
            index_kind="directory",
            revision=0,
        )

    def _bounded(self, value: str) -> str:
        maximum = self.config.max_record_chars
        if len(value) <= maximum:
            return value
        if maximum <= 3:
            return value[:maximum]
        return value[: maximum - 3].rstrip() + "..."

    def _require_record_bound(self, values: Mapping[str, MemoryIndexSource]) -> None:
        if len(values) > self.config.max_records:
            raise ValueError("memory vector rebuild exceeded its record bound")


__all__ = ["MemoryIndexSourceReader"]
