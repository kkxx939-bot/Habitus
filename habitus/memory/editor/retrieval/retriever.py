"""提取记忆时读取固定旧卡片，并给出偏好主题与实体名字清单。

不靠向量搜索：档案和按本段真实工具名对应的工具卡片固定直读；偏好与实体靠名字清单定位，清单放进第一轮输入，
清单里出现过的地址模型可以要求读。名字取自记忆树目录本身（完整），说明取自模板概览（读不到就只给名字）。
"""

from __future__ import annotations

import asyncio

from habitus.memory.editor.retrieval.model import (
    MemoryNameEntry,
    MemoryRelatedContext,
    MemoryRetrievalConfig,
    MemoryRetrievalError,
)
from habitus.memory.editor.retrieval.search import MemorySearchHit, memory_content_search
from habitus.memory.model import MemoryAddress, MemoryDirectory, MemoryLevel
from habitus.memory.recall import MemoryRecallLifecycle
from habitus.memory.schema import MemorySchemaRegistry
from habitus.memory.semantic import MemoryOverview, MemoryOverviewFormatError
from habitus.memory.snapshot import MemorySnapshotReader
from habitus.memory.tree import MemoryTree
from habitus.memory.uri import MemoryURI
from habitus.pre.conversation import ConversationMessageRole, ConversationSegment

_LISTED_DIRECTORIES = (MemoryDirectory(("preferences",)), MemoryDirectory(("entities",)))
_OVERVIEW_MAX_BYTES = 4 * 1024 * 1024


class MemoryRelatedRetriever:
    """固定节点直读、名字清单有界列出。"""

    def __init__(
        self,
        *,
        schema_registry: MemorySchemaRegistry,
        snapshot_reader: MemorySnapshotReader,
        config: MemoryRetrievalConfig | None = None,
        recall_lifecycle: MemoryRecallLifecycle | None = None,
    ) -> None:
        self.schema_registry = schema_registry
        self.recall_lifecycle = recall_lifecycle  # 搜内容时剔掉退休的卡片
        self.snapshot_reader = snapshot_reader
        self.config = config or MemoryRetrievalConfig()
        if 1 + self.config.max_tool_uris > self.snapshot_reader.config.max_items:
            raise ValueError("fixed old-memory reads can exceed the snapshot reader item limit")

    @property
    def tree(self) -> MemoryTree:
        return self.snapshot_reader.tree

    async def retrieve(self, segment: ConversationSegment) -> MemoryRelatedContext:
        """读档案与本段工具卡片的完整快照，并列出偏好主题与实体名字；读盘放到线程里，不堵事件循环。"""

        return await asyncio.to_thread(self._retrieve, segment)

    def _retrieve(self, segment: ConversationSegment) -> MemoryRelatedContext:
        fixed = (MemoryURI.from_address(MemoryAddress.profile()), *self._tool_uris(segment))
        snapshots = self.snapshot_reader.read_many(tuple(sorted({str(uri) for uri in fixed})))
        entries: list[MemoryNameEntry] = []
        for directory in _LISTED_DIRECTORIES:
            if not self.tree.directory_exists(directory):
                continue
            children = self.tree.child_directories(directory)
            for child in (directory, *children):
                entries.extend(memory_directory_listing(self.tree, child))
        kept = tuple(entries[: self.config.max_listing_entries])
        return MemoryRelatedContext(
            conversation_id=segment.conversation_id,
            segment_id=segment.segment_id,
            source_segment_digest=segment.digest,
            listing=kept,
            snapshots=snapshots,
            omitted_listing_entries=len(entries) - len(kept),
        )

    def search(
        self, keywords: tuple[str, ...], directory: MemoryDirectory | None, *, limit: int
    ) -> tuple[MemorySearchHit, ...]:
        """按关键词在卡片字段文字里字面查找（同步读盘，调用方放进线程）。"""

        return memory_content_search(
            self.tree, self.recall_lifecycle, keywords=keywords, directory=directory, limit=limit
        )

    def _tool_uris(self, segment: ConversationSegment) -> tuple[MemoryURI, ...]:
        names: set[str] = set()
        for message in segment.messages:
            if message.role is not ConversationMessageRole.TOOL_CALL:
                continue
            assert message.tool_name is not None
            names.add(message.tool_name)
            if len(names) > self.config.max_tool_uris:
                raise MemoryRetrievalError("conversation segment contains too many distinct tool names")
        try:
            return tuple(sorted((MemoryURI.from_address(MemoryAddress.tool(name)) for name in names), key=str))
        except (TypeError, ValueError) as exc:
            raise MemoryRetrievalError(
                "conversation tool name cannot map exactly to the confirmed memory tree"
            ) from exc


def memory_directory_listing(tree: MemoryTree, directory: MemoryDirectory) -> tuple[MemoryNameEntry, ...]:
    """一个目录里直接的卡片：名字取目录本身（完整），说明取模板概览；最近更新的在前，没有概览的按名字。"""

    if not tree.directory_exists(directory):
        return ()
    descriptions: dict[str, tuple[str, float]] = {}
    if tree.layer_exists(directory, MemoryLevel.OVERVIEW):
        try:
            overview = MemoryOverview.parse(
                directory, tree.read_layer_bounded(directory, MemoryLevel.OVERVIEW, max_bytes=_OVERVIEW_MAX_BYTES)
            )
        except (MemoryOverviewFormatError, ValueError):
            overview = None
        if overview is not None:
            descriptions = {card.filename: (card.description, card.updated_at.timestamp()) for card in overview.cards}
    entries = []
    for address in tree.direct_addresses(directory):
        filename = tree.path_for(address).name
        description, updated = descriptions.get(filename, ("", 0.0))
        entries.append((-updated, filename, MemoryNameEntry(MemoryURI.from_address(address), description)))
    entries.sort(key=lambda item: (item[0], item[1]))
    return tuple(entry for _updated, _filename, entry in entries)


__all__ = ["MemoryRelatedRetriever", "memory_directory_listing"]
