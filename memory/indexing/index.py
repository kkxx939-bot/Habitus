"""MemoryTree 真相源与远程 VectorStore 之间的可恢复索引。"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from foundation.integrity import canonical_digest
from infrastructure.store.contracts import PathLock
from infrastructure.vector import (
    VectorStore,
    VectorStoreConflictError,
    VectorStoreFilter,
    VectorStoreMatch,
    VectorStoreRecord,
    VectorStoreState,
)
from memory.indexing.config import MemoryVectorIndexConfig
from memory.indexing.contracts import (
    MemoryVectorIndexError,
    MemoryVectorMatch,
)
from memory.indexing.model import MemoryIndexSource, MemoryVectorConsistencyReport
from memory.indexing.source import MemoryIndexSourceReader
from memory.intention import MemoryIntentionRecallScope, allowed_memory_index_kinds
from memory.model import MemoryKind, MemoryLevel
from memory.semantic import MemorySemanticRefreshResult
from memory.tree import MemoryTree
from memory.uri import MemoryURI, MemoryURINodeType
from ModelClient import Embedder, EmbeddingVector

_SCHEMA_VERSION = "habitus_memory_vector_v2"


class PersistentMemoryVectorIndex:
    """远程索引是可重建派生数据；MemoryTree 始终是唯一内容真相源。"""

    def __init__(
        self,
        tree: MemoryTree,
        embedder: Embedder,
        store: VectorStore,
        *,
        dimension: int,
        embedding_fingerprint: str,
        config: MemoryVectorIndexConfig | None = None,
        path_lock: PathLock | None = None,
    ) -> None:
        if not isinstance(tree, MemoryTree):
            raise TypeError("tree must be MemoryTree")
        if not callable(getattr(embedder, "embed_documents", None)):
            raise TypeError("embedder must implement embed_documents")
        for name in ("initialize", "state", "read", "replace_all", "apply", "search", "scan"):
            if not callable(getattr(store, name, None)):
                raise TypeError(f"store must implement VectorStore.{name}")
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("memory vector dimension must be positive")
        if (
            not isinstance(embedding_fingerprint, str)
            or not embedding_fingerprint
            or embedding_fingerprint != embedding_fingerprint.strip()
        ):
            raise ValueError("embedding_fingerprint must be normalized non-empty text")
        if path_lock is not None and not isinstance(path_lock, PathLock):
            raise TypeError("path_lock must be PathLock or None")
        self.tree = tree
        self.embedder = embedder
        self.store = store
        self.dimension = dimension
        self.embedding_fingerprint = embedding_fingerprint
        self.config = config or MemoryVectorIndexConfig()
        self.sources = MemoryIndexSourceReader(
            tree,
            config=self.config,
            path_lock=path_lock,
        )
        self._lock = asyncio.Lock()

    async def ensure_ready(self) -> VectorStoreState:
        """验证远端集合；缺失或模型语义变化时从 MemoryTree 完整重建。"""

        await self.store.initialize()
        async with self._lock:
            state = await self.store.state()
            if self._state_matches(state):
                assert state is not None
                return state
            checkpoint = 0 if state is None else state.checkpoint
            return await self._rebuild_locked(checkpoint=checkpoint)

    async def synchronize(
        self,
        *,
        changed_uris: tuple[MemoryURI, ...],
        semantic_results: tuple[MemorySemanticRefreshResult, ...],
        checkpoint: int,
    ) -> VectorStoreState:
        """在 L2 和 L0/L1 均发布后，以 Job 全局序号推进远程索引。"""

        if not isinstance(changed_uris, tuple) or any(not isinstance(uri, MemoryURI) for uri in changed_uris):
            raise TypeError("changed_uris must contain MemoryURI values")
        if not isinstance(semantic_results, tuple) or any(
            not isinstance(result, MemorySemanticRefreshResult) for result in semantic_results
        ):
            raise TypeError("semantic_results must contain MemorySemanticRefreshResult values")
        if isinstance(checkpoint, bool) or not isinstance(checkpoint, int) or checkpoint <= 0:
            raise ValueError("memory vector checkpoint must be positive")
        await self.ensure_ready()
        async with self._lock:
            for _ in range(self.config.stale_retries + 1):
                state = await self.store.state()
                if not self._state_matches(state):
                    state = await self._rebuild_locked(checkpoint=0 if state is None else state.checkpoint)
                assert state is not None
                if checkpoint == state.checkpoint:
                    return state
                if checkpoint < state.checkpoint:
                    raise MemoryVectorIndexError("memory vector checkpoint attempted to move backwards")
                sources, identities = self._incremental_sources(changed_uris, semantic_results)
                existing = {record.identity: record for record in await self.store.read(identities)}
                upserts = await self._materialize(sources, existing=existing)
                present = {source.identity for source in sources}
                deletes = tuple(identity for identity in identities if identity not in present)
                try:
                    return await self.store.apply(
                        upserts,
                        deletes,
                        checkpoint=checkpoint,
                        expected_generation=state.generation,
                        expected_checkpoint=state.checkpoint,
                    )
                except VectorStoreConflictError:
                    continue
            raise MemoryVectorIndexError("memory vector state changed repeatedly during synchronization")

    async def rebuild(self, *, checkpoint: int | None = None) -> VectorStoreState:
        """管理入口：完整扫描 MemoryTree 并发布一代新的远程索引。"""

        await self.store.initialize()
        async with self._lock:
            current = await self.store.state()
            selected = (0 if current is None else current.checkpoint) if checkpoint is None else checkpoint
            if isinstance(selected, bool) or not isinstance(selected, int) or selected < 0:
                raise ValueError("memory vector rebuild checkpoint must be non-negative")
            return await self._rebuild_locked(checkpoint=selected)

    async def check_consistency(self) -> MemoryVectorConsistencyReport:
        """显式管理检查会先确保索引存在，再执行只读差异审计。"""

        await self.ensure_ready()
        return await self.audit_consistency()

    async def audit_consistency(self) -> MemoryVectorConsistencyReport:
        """无副作用比较全部 URI 和内容摘要，不触发重建。"""

        state = await self.store.state()
        if not self._state_matches(state):
            raise MemoryVectorIndexError("memory vector index is not ready for consistency audit")
        assert state is not None
        expected = {source.identity: source for source in self.sources.walk()}
        indexed = {
            record.identity: record
            for record in await self.store.scan(
                filters=_empty_filter(),
                limit=self.config.max_records,
            )
        }
        if len(indexed) != state.record_count:
            raise MemoryVectorIndexError("remote vector record count does not match published state")
        expected_ids = set(expected)
        indexed_ids = set(indexed)
        stale = tuple(
            sorted(
                identity
                for identity in expected_ids & indexed_ids
                if expected[identity].content_digest != indexed[identity].content_digest
            )
        )
        return MemoryVectorConsistencyReport(
            expected_count=len(expected),
            indexed_count=len(indexed),
            missing_identities=tuple(sorted(expected_ids - indexed_ids)),
            stale_identities=stale,
            orphan_identities=tuple(sorted(indexed_ids - expected_ids)),
        )

    async def search(
        self,
        query_vector: EmbeddingVector,
        *,
        roots: tuple[MemoryURI, ...],
        levels: tuple[MemoryLevel, ...],
        kinds: tuple[MemoryKind, ...],
        intention_scope: MemoryIntentionRecallScope,
        limit: int,
    ) -> tuple[MemoryVectorMatch, ...]:
        await self.ensure_ready()
        vector = self._query_vector(query_vector)
        normalized_roots = self._roots(roots)
        normalized_levels = self._levels(levels)
        maximum = self._limit(limit)
        one_of: dict[str, tuple[str | int, ...]] = {
            "level": tuple(level.value for level in normalized_levels),
            "scope_roots": tuple(str(root) for root in normalized_roots),
        }
        if MemoryLevel.DETAIL in normalized_levels:
            if normalized_levels != (MemoryLevel.DETAIL,):
                raise ValueError("L2 kind filtering cannot be mixed with L0/L1 in one vector query")
            one_of["kind"] = allowed_memory_index_kinds(kinds, intention_scope)
        matches = await self.store.search(
            vector,
            filters=VectorStoreFilter(
                equals={},
                one_of=one_of,
            ),
            limit=maximum,
        )
        return await self._matches(matches)

    async def search_children(
        self,
        query_vector: EmbeddingVector,
        *,
        parent: MemoryURI,
        kinds: tuple[MemoryKind, ...],
        intention_scope: MemoryIntentionRecallScope,
        limit: int,
    ) -> tuple[MemoryVectorMatch, ...]:
        await self.ensure_ready()
        vector = self._query_vector(query_vector)
        parsed = MemoryURI.parse(parent)
        if parsed.node_type is not MemoryURINodeType.DIRECTORY:
            raise ValueError("memory vector child search parent must be a directory URI")
        maximum = self._limit(limit)
        direct, directories = await asyncio.gather(
            self.store.search(
                vector,
                filters=VectorStoreFilter(
                    equals={"directory_key": str(parsed)},
                    one_of={
                        "level": (MemoryLevel.DETAIL.value,),
                        "kind": allowed_memory_index_kinds(kinds, intention_scope),
                    },
                ),
                limit=maximum,
            ),
            self.store.search(
                vector,
                filters=VectorStoreFilter(
                    equals={"parent_key": str(parsed)},
                    one_of={
                        "level": (
                            MemoryLevel.ABSTRACT.value,
                            MemoryLevel.OVERVIEW.value,
                        )
                    },
                ),
                limit=maximum,
            ),
        )
        matches = list(await self._matches(tuple(direct) + tuple(directories)))
        matches.sort(key=lambda item: (-item.score, str(item.uri)))
        return tuple(matches[:maximum])

    async def close(self) -> None:
        await self.store.close()

    async def _rebuild_locked(self, *, checkpoint: int) -> VectorStoreState:
        for _ in range(self.config.stale_retries + 1):
            state = await self.store.state()
            sources = self.sources.walk()
            reusable: dict[str, VectorStoreRecord] = {}
            if self._state_matches(state):
                assert state is not None
                if state.record_count > self.config.max_records:
                    raise MemoryVectorIndexError("published vector state exceeds rebuild capacity")
                reusable = {
                    record.identity: record
                    for record in await self.store.scan(
                        filters=_empty_filter(),
                        limit=self.config.max_records,
                    )
                }
            records = await self._materialize(sources, existing=reusable)
            try:
                return await self.store.replace_all(
                    records,
                    schema_version=_SCHEMA_VERSION,
                    embedding_fingerprint=self.embedding_fingerprint,
                    dimension=self.dimension,
                    checkpoint=checkpoint,
                    expected_generation=None if state is None else state.generation,
                )
            except VectorStoreConflictError:
                continue
        raise MemoryVectorIndexError("memory vector state changed repeatedly during rebuild")

    async def _materialize(
        self,
        sources: Sequence[MemoryIndexSource],
        *,
        existing: dict[str, VectorStoreRecord],
    ) -> tuple[VectorStoreRecord, ...]:
        resolved: dict[str, VectorStoreRecord] = {}
        missing: list[MemoryIndexSource] = []
        for source in sources:
            current = existing.get(source.identity)
            if (
                current is not None
                and current.content_digest == source.content_digest
                and current.vector.dimension == self.dimension
            ):
                resolved[source.identity] = self._record(source, current.vector)
            else:
                missing.append(source)
        for offset in range(0, len(missing), self.config.embedding_batch_size):
            batch = missing[offset : offset + self.config.embedding_batch_size]
            vectors = await self.embedder.embed_documents(tuple(source.content for source in batch))
            if not isinstance(vectors, tuple) or len(vectors) != len(batch):
                raise MemoryVectorIndexError("embedder returned an unexpected vector count")
            for source, vector in zip(batch, vectors, strict=True):
                if not isinstance(vector, EmbeddingVector) or vector.dimension != self.dimension:
                    raise MemoryVectorIndexError("embedder returned an invalid vector dimension")
                resolved[source.identity] = self._record(source, vector)
        if len(resolved) != len(sources):
            raise AssertionError("memory vector materialization lost an index source")
        return tuple(resolved[identity] for identity in sorted(resolved))

    @staticmethod
    def _record(source: MemoryIndexSource, vector: EmbeddingVector) -> VectorStoreRecord:
        return VectorStoreRecord(
            identity=source.identity,
            vector=vector,
            content=source.content,
            content_digest=source.content_digest,
            attributes={
                "uri": source.identity,
                "level": source.level.value,
                "directory_key": source.directory_key,
                "parent_key": source.parent_key,
                "scope_roots": source.scope_roots,
                "kind": source.index_kind,
                "revision": source.revision,
            },
        )

    def _incremental_sources(
        self,
        changed_uris: tuple[MemoryURI, ...],
        semantic_results: tuple[MemorySemanticRefreshResult, ...],
    ) -> tuple[tuple[MemoryIndexSource, ...], tuple[str, ...]]:
        desired: dict[str, MemoryIndexSource] = {}
        identities: set[str] = set()
        for uri in changed_uris:
            if uri.node_type is not MemoryURINodeType.DOCUMENT:
                raise ValueError("changed memory URI must identify an L2 document")
            identities.add(str(uri))
            source = self.sources.read_uri(uri)
            if source is not None:
                desired[source.identity] = source
        for result in semantic_results:
            for level in (MemoryLevel.ABSTRACT, MemoryLevel.OVERVIEW):
                uri = MemoryURI.from_layer(result.directory, level)
                identities.add(str(uri))
                source = self.sources.read_uri(uri)
                if source is not None:
                    desired[source.identity] = source
        return (
            tuple(desired[identity] for identity in sorted(desired)),
            tuple(sorted(identities)),
        )

    async def _matches(
        self,
        values: Sequence[VectorStoreMatch],
    ) -> tuple[MemoryVectorMatch, ...]:
        matches: list[MemoryVectorMatch] = []
        for value in values:
            if not isinstance(value, VectorStoreMatch):
                raise MemoryVectorIndexError("vector store returned an invalid match")
            record = value.record
            uri_value = record.attributes.get("uri")
            level_value = record.attributes.get("level")
            directory_value = record.attributes.get("directory_key")
            index_kind = record.attributes.get("kind")
            if (
                not isinstance(uri_value, str)
                or not isinstance(level_value, int)
                or not isinstance(directory_value, str)
                or not isinstance(index_kind, str)
            ):
                raise MemoryVectorIndexError("vector record is missing memory identity attributes")
            uri = MemoryURI.parse(uri_value)
            level = MemoryLevel(level_value)
            directory_uri = MemoryURI.parse(directory_value)
            if directory_uri.node_type is not MemoryURINodeType.DIRECTORY:
                raise MemoryVectorIndexError("vector record directory attribute is invalid")
            current = self.sources.read_uri(uri)
            if current is None or current.content_digest != record.content_digest:
                raise MemoryVectorIndexError("remote vector match is stale relative to MemoryTree")
            if current.index_kind != index_kind:
                raise MemoryVectorIndexError("remote vector match has a stale retrieval partition")
            matches.append(
                MemoryVectorMatch(
                    uri=uri,
                    level=level,
                    directory=directory_uri.to_directory(),
                    content=record.content,
                    score=value.score,
                )
            )
        return tuple(matches)

    def _state_matches(self, state: VectorStoreState | None) -> bool:
        return bool(
            state is not None
            and state.ready
            and state.schema_version == _SCHEMA_VERSION
            and state.embedding_fingerprint == self.embedding_fingerprint
            and state.dimension == self.dimension
        )

    def _limit(self, value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= self.config.max_search_hits:
            raise ValueError("memory vector search limit is outside its configured bound")
        return value

    @staticmethod
    def _query_vector(value: EmbeddingVector) -> EmbeddingVector:
        if not isinstance(value, EmbeddingVector):
            raise TypeError("query_vector must be EmbeddingVector")
        return value

    @staticmethod
    def _roots(values: tuple[MemoryURI, ...]) -> tuple[MemoryURI, ...]:
        if not isinstance(values, tuple) or not values:
            raise ValueError("memory vector search requires at least one root")
        roots = tuple(MemoryURI.parse(value) for value in values)
        if any(root.node_type is not MemoryURINodeType.DIRECTORY for root in roots):
            raise ValueError("memory vector roots must identify directories")
        if len(roots) != len(set(roots)):
            raise ValueError("memory vector roots must be unique")
        return roots

    @staticmethod
    def _levels(values: tuple[MemoryLevel, ...]) -> tuple[MemoryLevel, ...]:
        if not isinstance(values, tuple) or not values:
            raise ValueError("memory vector search requires at least one level")
        levels = tuple(MemoryLevel(value) for value in values)
        if len(levels) != len(set(levels)):
            raise ValueError("memory vector levels must be unique")
        return levels


def memory_embedding_fingerprint(
    *,
    provider: str,
    adapter: str,
    model: str,
    base_url: str,
    dimension: int,
    input_mode: str,
    extra_body: object,
    document_parameters: object,
) -> str:
    """模型或文档编码参数变化时确定性触发全量重建。"""

    return canonical_digest(
        {
            "schema": _SCHEMA_VERSION,
            "provider": provider,
            "adapter": adapter,
            "model": model,
            "base_url": base_url,
            "dimension": dimension,
            "input_mode": input_mode,
            "extra_body": extra_body,
            "document_parameters": document_parameters,
        }
    )


def _empty_filter() -> VectorStoreFilter:
    return VectorStoreFilter(equals={}, one_of={})


__all__ = ["PersistentMemoryVectorIndex", "memory_embedding_fingerprint"]
