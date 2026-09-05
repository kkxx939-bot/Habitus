"""受控 ReAct 使用的完整旧记忆上下文和只读动作执行器。"""

from __future__ import annotations

from collections.abc import Sequence

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
    MemoryRelatedContext,
)
from habitus.memory.intention import MemoryIntentionRecallScope
from habitus.memory.retrieval import MemorySearchHit, MemorySemanticSearch
from habitus.memory.snapshot import MemorySnapshotBatch, MemorySnapshotReader
from habitus.memory.uri import MemoryURI, MemoryURINodeType
from habitus.model_client import estimate_utf8_bytes_tokens


class MemoryExtractionContext:
    """累计完整快照，并只允许一次搜索或一次受控读取。"""

    def __init__(
        self,
        initial: MemoryRelatedContext,
        *,
        snapshot_reader: MemorySnapshotReader,
        semantic_search: MemorySemanticSearch,
        config: MemoryExtractionConfig,
    ) -> None:
        if not isinstance(initial, MemoryRelatedContext):
            raise TypeError("initial must be a MemoryRelatedContext")
        if not isinstance(snapshot_reader, MemorySnapshotReader):
            raise TypeError("snapshot_reader must be a MemorySnapshotReader")
        if not callable(getattr(semantic_search, "search", None)):
            raise TypeError("semantic_search must implement async search")
        if not isinstance(config, MemoryExtractionConfig):
            raise TypeError("config must be a MemoryExtractionConfig")
        if config.max_old_memory_items > snapshot_reader.config.max_items:
            raise ValueError("extraction item limit cannot exceed the snapshot reader item limit")
        if config.max_old_memory_bytes > snapshot_reader.config.max_total_bytes:
            raise ValueError("extraction byte limit cannot exceed the snapshot reader total-byte limit")

        self.initial = initial
        self.snapshot_reader = snapshot_reader
        self.semantic_search = semantic_search
        self.config = config
        self._snapshots: dict[str, VersionedSnapshot[MemoryDocument]] = {}
        self._total_bytes = 0
        self._total_tokens = 0
        self._page_ids = MemoryPageIdMap()
        self._allowed_read_uris: set[str] = set()
        self._search_hits: dict[str, MemorySearchHit] = {}
        self._query_hits: dict[str, tuple[MemorySearchHit, ...]] = {}
        self._relation_neighbor_uris: set[str] = set()
        self._expanded_relation_seeds: set[str] = set()
        self._relationship_expectations: dict[
            str,
            list[tuple[str, MemoryStoredLink, bool]],
        ] = {}

        self._ingest(initial.snapshots)
        for hit in initial.search_hits:
            snapshot = self._snapshots.get(str(hit.uri))
            if snapshot is None or not snapshot.exists:
                raise MemoryExtractionError("initial semantic search hit has no complete existing snapshot")
            self._remember_search_hit(hit)
        initial_query = self._query(initial.query)
        self._query_hits[initial_query] = initial.search_hits
        self._allowed_read_uris.update(self._snapshots)
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

    @property
    def search_hits(self) -> tuple[MemorySearchHit, ...]:
        """返回所有查询中每个 URI 的最佳命中。"""

        return tuple(sorted(self._search_hits.values(), key=lambda hit: (-hit.score, str(hit.uri))))

    async def execute(
        self,
        decision: MemoryRetrievalDecision,
        *,
        iteration: int,
    ) -> MemoryRetrievalObservation:
        """执行 Grader 选择的一个只读动作。"""

        if not isinstance(decision, MemoryRetrievalDecision):
            raise TypeError("decision must be a MemoryRetrievalDecision")
        if decision.action is MemoryRetrievalAction.FINISH:
            raise ValueError("finish does not execute a retrieval action")
        if decision.action is MemoryRetrievalAction.SEARCH:
            assert decision.query is not None
            observation = await self._search(decision.query, iteration=iteration)
        else:
            assert decision.uri is not None
            observation = self._read(decision.uri, iteration=iteration)
        if len(canonical_json(observation.to_dict())) > self.config.max_observation_chars:
            raise MemoryExtractionError("retrieval observation exceeds its configured limit")
        return observation

    async def _search(self, query: str, *, iteration: int) -> MemoryRetrievalObservation:
        normalized_query = self._query(query)
        cached_hits = self._query_hits.get(normalized_query)
        if cached_hits is not None:
            return MemoryRetrievalObservation(
                iteration=iteration,
                action=MemoryRetrievalAction.SEARCH,
                input_value=normalized_query,
                result_uris=tuple(str(hit.uri) for hit in cached_hits),
                added_uris=(),
                relation_expanded_uris=(),
                cached=True,
            )
        if not self.initial.search_roots:
            raise MemoryExtractionError("memory_search has no allowed memory roots")
        try:
            raw_hits = await self.semantic_search.search(
                normalized_query,
                roots=self.initial.search_roots,
                kinds=(),
                intention_scope=MemoryIntentionRecallScope.ALL,
                limit=self.config.additional_search_limit,
            )
        except Exception as exc:
            raise MemoryExtractionError("additional memory semantic search failed") from exc
        hits = self._fit_hits(self._validated_hits(raw_hits))
        self._query_hits[normalized_query] = hits
        for hit in hits:
            self._remember_search_hit(hit)
            self._allowed_read_uris.add(str(hit.uri))

        before = set(self._snapshots)
        hit_uris = tuple(str(hit.uri) for hit in hits)
        if hit_uris:
            batch = self.snapshot_reader.read_many(hit_uris)
            missing = tuple(snapshot.identity for snapshot in batch.snapshots if not snapshot.exists)
            if missing:
                raise MemoryExtractionError(f"semantic search returned memory that disappeared before read: {missing}")
            self._ingest(batch)
        relation_uris = self._expand_relation_seeds(hit_uris)
        added = tuple(sorted(set(self._snapshots) - before))
        return MemoryRetrievalObservation(
            iteration=iteration,
            action=MemoryRetrievalAction.SEARCH,
            input_value=normalized_query,
            result_uris=hit_uris,
            added_uris=added,
            relation_expanded_uris=relation_uris,
        )

    def _read(self, uri: str, *, iteration: int) -> MemoryRetrievalObservation:
        try:
            parsed = MemoryURI.parse(uri)
            parsed.to_address()
        except (TypeError, ValueError) as exc:
            raise MemoryExtractionError("memory_read requires a valid L2 memory URI") from exc
        identity = str(parsed)
        if identity not in self._allowed_read_uris:
            raise MemoryExtractionError("memory_read URI was not exposed by prefetch, search, or one-hop relations")
        if identity in self._snapshots:
            return MemoryRetrievalObservation(
                iteration=iteration,
                action=MemoryRetrievalAction.READ,
                input_value=identity,
                result_uris=(identity,),
                added_uris=(),
                relation_expanded_uris=(),
                cached=True,
            )
        snapshot = self.snapshot_reader.read(parsed)
        self._ingest(SnapshotBatch(snapshots=(snapshot,), total_bytes=snapshot.size_bytes))
        self._validate_relationship_neighbor(identity)
        return MemoryRetrievalObservation(
            iteration=iteration,
            action=MemoryRetrievalAction.READ,
            input_value=identity,
            result_uris=(identity,),
            added_uris=(identity,),
            relation_expanded_uris=(),
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
            self._ingest(batch)
        for neighbor_uri in sorted(selected_neighbors):
            self._validate_relationship_neighbor(neighbor_uri)
        return tuple(sorted(selected_neighbors))

    def _validate_relationship_neighbor(self, identity: str) -> None:
        expectations = self._relationship_expectations.get(identity, [])
        if not expectations:
            return
        snapshot = self._snapshots.get(identity)
        if snapshot is None or not snapshot.exists or not isinstance(snapshot.value, MemoryDocument):
            raise MemoryExtractionError(f"one-hop relation target does not exist as a complete L2 memory: {identity}")
        document = snapshot.value
        for seed_uri, link, expect_backlink in expectations:
            counterpart = document.backlinks if expect_backlink else document.links
            if link not in counterpart:
                raise MemoryExtractionError(
                    f"stored memory relation is not bidirectionally consistent between {seed_uri} and {identity}"
                )

    def _ingest(self, batch: MemorySnapshotBatch) -> tuple[str, ...]:
        if not isinstance(batch, SnapshotBatch):
            raise TypeError("batch must be a MemorySnapshotBatch")
        pending: list[VersionedSnapshot[MemoryDocument]] = []
        for snapshot in batch.snapshots:
            current = self._snapshots.get(snapshot.identity)
            if current is not None:
                if current != snapshot:
                    raise MemoryExtractionError(f"old memory changed during extraction: {snapshot.identity}")
                continue
            if snapshot.exists:
                if not isinstance(snapshot.value, MemoryDocument):
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
        token_count = self._total_tokens + sum(
            estimate_utf8_bytes_tokens(snapshot.size_bytes) for snapshot in pending
        )
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

    def _validated_hits(self, raw_hits: object) -> tuple[MemorySearchHit, ...]:
        if isinstance(raw_hits, str) or not isinstance(raw_hits, Sequence):
            raise MemoryExtractionError("memory semantic search must return a sequence of hits")
        if len(raw_hits) > self.config.additional_search_limit:
            raise MemoryExtractionError("memory semantic search exceeded its requested limit")
        by_uri: dict[MemoryURI, MemorySearchHit] = {}
        for hit in raw_hits:
            if not isinstance(hit, MemorySearchHit):
                raise MemoryExtractionError("memory semantic search returned an invalid hit")
            if hit.uri.node_type is not MemoryURINodeType.DOCUMENT:
                raise MemoryExtractionError("memory semantic search returned a non-L2 URI")
            if not any(hit.uri.matches_prefix(root) for root in self.initial.search_roots):
                raise MemoryExtractionError("memory semantic search returned an out-of-scope URI")
            current = by_uri.get(hit.uri)
            if current is None or hit.score > current.score:
                by_uri[hit.uri] = hit
        return tuple(sorted(by_uri.values(), key=lambda hit: (-hit.score, str(hit.uri))))

    def _remember_search_hit(self, hit: MemorySearchHit) -> None:
        current = self._search_hits.get(str(hit.uri))
        if current is None or hit.score > current.score:
            self._search_hits[str(hit.uri)] = hit

    def _fit_hits(self, hits: tuple[MemorySearchHit, ...]) -> tuple[MemorySearchHit, ...]:
        """在累计快照上限内保留相关性最高的已有或新 URI。"""

        remaining = self.config.max_old_memory_items - len(self._snapshots)
        selected: list[MemorySearchHit] = []
        selected_new: set[str] = set()
        for hit in hits:
            identity = str(hit.uri)
            if identity not in self._snapshots and identity not in selected_new:
                if len(selected_new) >= remaining:
                    continue
                selected_new.add(identity)
            selected.append(hit)
        return tuple(selected)

    def _query(self, value: object) -> str:
        if not isinstance(value, str):
            raise MemoryExtractionError("memory search query must be text")
        normalized = " ".join(value.split())
        if not normalized:
            raise MemoryExtractionError("memory search query must be non-empty")
        if len(normalized) > self.config.max_query_chars:
            raise MemoryExtractionError("memory search query exceeds its character limit")
        return normalized


__all__ = ["MemoryExtractionContext"]
