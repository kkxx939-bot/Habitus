"""受控 ReAct 使用的完整旧记忆上下文和只读动作执行器。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from habitus.foundation.integrity import canonical_json
from habitus.infrastructure.editor.snapshot import SnapshotBatch, VersionedSnapshot
from habitus.memory.document import MemoryDocument, MemoryStoredLink
from habitus.memory.editor.extraction.config import MemoryExtractionConfig
from habitus.memory.editor.extraction.model import (
    MemoryExtractionCapacityError,
    MemoryExtractionError,
    MemoryRetrievalAction,
    MemoryRetrievalDecision,
    MemoryRetrievalObservation,
)
from habitus.memory.editor.page_id import EXISTING_PAGE_ID_MAX, MemoryPageIdMap
from habitus.memory.editor.retrieval import (
    MemoryNameEntry,
    MemoryRelatedContext,
    MemorySearchHit,
    memory_content_search,
    memory_directory_listing,
)
from habitus.memory.model import MemoryDirectory
from habitus.memory.snapshot import MemorySnapshotBatch, MemorySnapshotReader
from habitus.memory.uri import MemoryURI, MemoryURINodeType
from habitus.model_client import estimate_text_tokens, estimate_utf8_bytes_tokens

_LISTABLE_ROOTS = ("preferences", "entities", "tools")
MemorySearcher = Callable[[tuple[str, ...], MemoryDirectory | None, int], tuple[MemorySearchHit, ...]]


class MemoryExtractionContext:
    """累计完整快照；每轮只允许列一个目录、按词搜一次卡片内容，或读若干个已经暴露过的地址。"""

    def __init__(
        self,
        initial: MemoryRelatedContext,
        *,
        snapshot_reader: MemorySnapshotReader,
        config: MemoryExtractionConfig,
        searcher: MemorySearcher | None = None,
    ) -> None:
        if config.max_old_memory_items > snapshot_reader.config.max_items:
            raise ValueError("extraction item limit cannot exceed the snapshot reader item limit")
        if config.max_old_memory_bytes > snapshot_reader.config.max_total_bytes:
            raise ValueError("extraction byte limit cannot exceed the snapshot reader total-byte limit")

        self.initial = initial
        self.snapshot_reader = snapshot_reader
        self.config = config
        self._snapshots: dict[str, VersionedSnapshot[MemoryDocument]] = {}
        self._total_bytes = 0
        self._total_tokens = 0
        self._page_ids = MemoryPageIdMap()
        self._allowed_read_uris: set[str] = set()
        self._listed: dict[str, tuple[MemoryNameEntry, ...]] = {}
        self._searched: dict[str, tuple[MemorySearchHit, ...]] = {}
        self._searcher = searcher or _tree_searcher(snapshot_reader)
        self._relation_neighbor_uris: set[str] = set()
        self._expanded_relation_seeds: set[str] = set()
        self._relationship_expectations: dict[str, list[tuple[str, MemoryStoredLink, bool]]] = {}

        self.ingest(initial.snapshots)
        self._allowed_read_uris.update(self._snapshots)
        self.shown_listing, self.omitted_listing = _fit_listing(initial, config.max_listing_tokens)
        self._allowed_read_uris.update(str(entry.uri) for entry in self.shown_listing)
        self._expand_relation_seeds(
            tuple(snapshot.identity for snapshot in initial.snapshots.snapshots if snapshot.exists)
        )

    @property
    def snapshots(self) -> MemorySnapshotBatch:
        """返回当前解析周期已经完整读取的稳定快照批次。"""

        values = tuple(self._snapshots[identity] for identity in sorted(self._snapshots))
        return SnapshotBatch(snapshots=values, total_bytes=sum(item.size_bytes for item in values))

    @property
    def page_ids(self) -> MemoryPageIdMap:
        """返回当前稳定的旧节点临时编号副本。"""

        return self._page_ids.copy()

    @property
    def allowed_read_uris(self) -> tuple[str, ...]:
        """返回模型可以请求读取的已发现 L2 URI。"""

        return tuple(sorted(self._allowed_read_uris))

    async def execute(
        self,
        decision: MemoryRetrievalDecision,
        *,
        iteration: int,
    ) -> MemoryRetrievalObservation:
        """执行 Grader 选择的一个只读动作。"""

        if decision.action is MemoryRetrievalAction.FINISH:
            raise ValueError("finish does not execute a retrieval action")
        if decision.action is MemoryRetrievalAction.LIST:
            assert decision.directory is not None
            observation = await asyncio.to_thread(self._list, decision.directory, iteration=iteration)
        elif decision.action is MemoryRetrievalAction.SEARCH:
            observation = await asyncio.to_thread(self._search, decision, iteration=iteration)
        else:
            observation = self._read(decision.uris[: self.config.max_reads_per_action], iteration=iteration)
        if len(canonical_json(observation.to_dict())) > self.config.max_observation_chars:
            raise MemoryExtractionError("retrieval observation exceeds its configured limit")
        return observation

    @property
    def listed_directories(self) -> dict[str, tuple[MemoryNameEntry, ...]]:
        return dict(self._listed)  # 列过的目录内容（地址与说明），给下一轮的模型看

    @property
    def searched(self) -> dict[str, tuple[MemorySearchHit, ...]]:
        return dict(self._searched)  # 搜过的词与命中的卡片（地址、名字、片段），给下一轮的模型看

    def _search(self, decision: MemoryRetrievalDecision, *, iteration: int) -> MemoryRetrievalObservation:
        directory = None if decision.directory is None else _directory(decision.directory, "memory_search")
        inputs = (*(() if directory is None else (str(MemoryURI.from_directory(directory)),)), *decision.keywords)
        key = " ".join(inputs)
        hits = self._searched.get(key)
        cached = hits is not None
        if hits is None:
            hits = self._searcher(decision.keywords, directory, self.config.max_search_results)
            self._searched[key] = hits
            self._allowed_read_uris.update(str(hit.uri) for hit in hits)
        return _seen(iteration, MemoryRetrievalAction.SEARCH, inputs, tuple(str(hit.uri) for hit in hits), cached)

    def _list(self, directory: str, *, iteration: int) -> MemoryRetrievalObservation:
        target = _directory(directory, "memory_list")
        parsed = MemoryURI.from_directory(target)
        identity = str(parsed)
        if identity in self._listed:
            entries = self._listed[identity]
            return _seen(iteration, MemoryRetrievalAction.LIST, (identity,), tuple(str(e.uri) for e in entries), True)
        tree = self.snapshot_reader.tree
        entries = memory_directory_listing(tree, target)
        children = tree.child_directories(target) if tree.directory_exists(target) else ()
        self._listed[identity] = entries
        self._allowed_read_uris.update(str(entry.uri) for entry in entries)
        results = (*(str(entry.uri) for entry in entries), *(str(MemoryURI.from_directory(c)) for c in children))
        return _seen(iteration, MemoryRetrievalAction.LIST, (identity,), results)

    def _read(self, uris: tuple[str, ...], *, iteration: int) -> MemoryRetrievalObservation:
        identities: list[str] = []
        for uri in uris:
            try:
                parsed = MemoryURI.parse(uri)
                parsed.to_address()
            except (TypeError, ValueError) as exc:
                raise MemoryExtractionError("memory_read requires valid L2 memory URIs") from exc
            identity = str(parsed)
            if identity not in self._allowed_read_uris:
                raise MemoryExtractionError(
                    "memory_read URI was not exposed by the name listing, a directory listing, a content search, "
                    "or one-hop relations"
                )
            identities.append(identity)
        unread = tuple(identity for identity in dict.fromkeys(identities) if identity not in self._snapshots)
        if not unread:
            return _seen(iteration, MemoryRetrievalAction.READ, tuple(identities), tuple(identities), True)
        batch = self.snapshot_reader.read_many(unread)
        added = self.ingest(batch)
        for identity in unread:
            self._validate_relationship_neighbor(identity)
        relation_uris = self._expand_relation_seeds(
            tuple(identity for identity in unread if identity in self._snapshots)
        )
        return MemoryRetrievalObservation(
            iteration=iteration,
            action=MemoryRetrievalAction.READ,
            inputs=tuple(identities),
            result_uris=tuple(identities),
            added_uris=added,
            relation_expanded_uris=relation_uris,
        )

    def _expand_relation_seeds(self, seed_uris: tuple[str, ...]) -> tuple[str, ...]:
        selected_neighbors: set[str] = set()
        unread_selected: set[str] = set()
        remaining_item_slots = self.config.max_old_memory_items - len(self._snapshots)
        for seed_uri in sorted(set(seed_uris)):
            if seed_uri in self._expanded_relation_seeds:
                continue
            self._expanded_relation_seeds.add(seed_uri)
            snapshot = self._snapshots.get(seed_uri)
            if snapshot is None or not snapshot.exists:
                continue
            assert isinstance(snapshot.value, MemoryDocument)
            grouped: dict[str, list[tuple[MemoryStoredLink, bool]]] = {}
            for link in snapshot.value.links:
                grouped.setdefault(str(link.to_uri), []).append((link, True))
            for backlink in snapshot.value.backlinks:
                grouped.setdefault(str(backlink.from_uri), []).append((backlink, False))

            selected_for_seed = 0
            for neighbor_uri in sorted(grouped):
                if selected_for_seed >= self.config.max_relation_neighbors_per_seed:
                    break
                if (
                    neighbor_uri not in self._relation_neighbor_uris
                    and len(self._relation_neighbor_uris) >= self.config.max_relation_neighbors_total
                ):
                    continue
                if (
                    neighbor_uri not in self._snapshots
                    and neighbor_uri not in unread_selected
                    and len(unread_selected) >= remaining_item_slots
                ):
                    continue
                selected_for_seed += 1
                selected_neighbors.add(neighbor_uri)
                if neighbor_uri not in self._snapshots:
                    unread_selected.add(neighbor_uri)
                self._relation_neighbor_uris.add(neighbor_uri)
                self._allowed_read_uris.add(neighbor_uri)
                expectations = self._relationship_expectations.setdefault(neighbor_uri, [])
                for link, expect_backlink in grouped[neighbor_uri]:
                    expectation = (seed_uri, link, expect_backlink)
                    if expectation not in expectations:
                        expectations.append(expectation)

        unread = tuple(sorted(uri for uri in selected_neighbors if uri not in self._snapshots))
        if unread:
            batch = self.snapshot_reader.read_many(unread)
            self.ingest(batch)
        for neighbor_uri in sorted(selected_neighbors):
            self._validate_relationship_neighbor(neighbor_uri)
        return tuple(sorted(selected_neighbors))

    def _validate_relationship_neighbor(self, identity: str) -> None:
        expectations = self._relationship_expectations.get(identity, [])
        if not expectations:
            return
        snapshot = self._snapshots.get(identity)
        if snapshot is None or not snapshot.exists or snapshot.value is None:
            raise MemoryExtractionError(f"one-hop relation target does not exist as a complete L2 memory: {identity}")
        document = snapshot.value
        for seed_uri, link, expect_backlink in expectations:
            counterpart = document.backlinks if expect_backlink else document.links
            if link not in counterpart:
                raise MemoryExtractionError(
                    f"stored memory relation is not bidirectionally consistent between {seed_uri} and {identity}"
                )

    def ingest(self, batch: MemorySnapshotBatch) -> tuple[str, ...]:
        pending: list[VersionedSnapshot[MemoryDocument]] = []
        for snapshot in batch.snapshots:
            current = self._snapshots.get(snapshot.identity)
            if current is not None:
                if current != snapshot:
                    raise MemoryExtractionError(f"old memory changed during extraction: {snapshot.identity}")
                continue
            if snapshot.exists:
                if snapshot.value is None:
                    raise MemoryExtractionError("old-memory snapshot contains an invalid document")
                expected = str(MemoryURI.from_address(snapshot.value.address))
                if expected != snapshot.identity:
                    raise MemoryExtractionError("old-memory snapshot identity does not match its document")
            pending.append(snapshot)

        item_count = len(self._snapshots) + len(pending)
        byte_count = self._total_bytes + sum(snapshot.size_bytes for snapshot in pending)
        found_count = sum(snapshot.exists for snapshot in self._snapshots.values()) + sum(
            snapshot.exists for snapshot in pending
        )
        if item_count > self.config.max_old_memory_items:
            raise MemoryExtractionCapacityError("old-memory context exceeds its item limit")
        if byte_count > self.config.max_old_memory_bytes:
            raise MemoryExtractionCapacityError("old-memory context exceeds its byte limit")
        token_count = self._total_tokens + sum(estimate_utf8_bytes_tokens(snapshot.size_bytes) for snapshot in pending)
        if token_count > self.config.max_old_memory_tokens:
            raise MemoryExtractionCapacityError("old-memory context exceeds its token limit")
        if found_count > EXISTING_PAGE_ID_MAX:
            raise MemoryExtractionCapacityError("old-memory context exhausts the existing page_id range")

        for snapshot in pending:
            self._snapshots[snapshot.identity] = snapshot
            self._total_bytes += snapshot.size_bytes
            self._total_tokens += estimate_utf8_bytes_tokens(snapshot.size_bytes)
            self._allowed_read_uris.add(snapshot.identity)
            if snapshot.exists:
                self._page_ids.register_existing(snapshot.identity)
        return tuple(snapshot.identity for snapshot in pending)


def _seen(
    iteration: int,
    action: MemoryRetrievalAction,
    inputs: tuple[str, ...],
    results: tuple[str, ...],
    cached: bool = False,
) -> MemoryRetrievalObservation:
    """没有新读进快照的一次动作结果（列目录、搜内容、读已读过的地址）。"""

    return MemoryRetrievalObservation(
        iteration=iteration,
        action=action,
        inputs=inputs,
        result_uris=results,
        added_uris=(),
        relation_expanded_uris=(),
        cached=cached,
    )


def _tree_searcher(reader: MemorySnapshotReader) -> MemorySearcher:
    """没有注入搜索时直接在记忆树上搜（不剔退休卡片，只给单测与不接冷热的场合用）。"""

    def search(keywords: tuple[str, ...], directory: MemoryDirectory | None, limit: int) -> tuple[MemorySearchHit, ...]:
        return memory_content_search(reader.tree, None, keywords=keywords, directory=directory, limit=limit)

    return search


def _directory(value: str, action: str) -> MemoryDirectory:
    """列目录、搜内容时给的目录：必须是 preferences、entities、tools 或它们下面的目录。"""

    try:
        parsed = MemoryURI.parse(value)
    except (TypeError, ValueError) as exc:
        raise MemoryExtractionError(f"{action} requires a valid memory:// directory URI") from exc
    if parsed.node_type is not MemoryURINodeType.DIRECTORY:
        raise MemoryExtractionError(f"{action} requires a directory URI")
    target = parsed.to_directory()
    if not target.parts or target.parts[0] not in _LISTABLE_ROOTS:
        raise MemoryExtractionError(f"{action} only lists or searches preferences, entities or tools")
    return target


def _fit_listing(initial: MemoryRelatedContext, max_tokens: int) -> tuple[tuple[MemoryNameEntry, ...], int]:
    """名字清单在 ``max_listing_tokens`` 之内放得下的前缀，以及没给模型看到的张数。

    只有给模型看到的名字算暴露过；按输入预算截掉的那部分要先列目录才能读。
    """

    entries, used = initial.listing, 0
    for index, entry in enumerate(entries):
        used += estimate_text_tokens(entry.line()) + 1
        if used > max_tokens:
            return entries[:index], len(entries) - index + initial.omitted_listing_entries
    return entries, initial.omitted_listing_entries


__all__ = ["MemoryExtractionContext"]
