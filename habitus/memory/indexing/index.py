"""MemoryTree 真相源与远程 VectorStore 之间的可恢复索引。"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from habitus.foundation.integrity import canonical_digest
from habitus.infrastructure.store.contracts import PathLock
from habitus.infrastructure.vector import (
    VectorStore,
    VectorStoreConflictError,
    VectorStoreFilter,
    VectorStoreMatch,
    VectorStoreRecord,
    VectorStoreState,
)
from habitus.memory.indexing.config import MemoryVectorIndexConfig
from habitus.memory.indexing.contracts import (
    MemoryVectorIndexError,
    MemoryVectorMatch,
)
from habitus.memory.indexing.model import MemoryIndexSource, MemoryVectorConsistencyReport
from habitus.memory.indexing.source import MemoryIndexSourceReader
from habitus.memory.intention import MemoryIntentionRecallScope, allowed_memory_index_kinds
from habitus.memory.model import MemoryKind, MemoryLevel
from habitus.memory.semantic import MemorySemanticRefreshResult
from habitus.memory.tree import MemoryTree
from habitus.memory.uri import MemoryURI, MemoryURINodeType
from habitus.memory.vector_index import PersistentVectorIndex, VectorIndexMessages
from habitus.model_client import Embedder, EmbeddingVector

_SCHEMA_VERSION = "habitus_memory_vector_v2"


class PersistentMemoryVectorIndex(
    PersistentVectorIndex[MemoryIndexSource, MemoryVectorConsistencyReport, MemoryVectorIndexConfig]
):
    """远程索引是可重建派生数据；MemoryTree 始终是唯一内容真相源。"""

    _schema_version = _SCHEMA_VERSION
    _messages = VectorIndexMessages(
        rebuild_checkpoint="memory vector rebuild checkpoint must be non-negative",
        rebuild_conflict="memory vector state changed repeatedly during rebuild",
        audit_not_ready="memory vector index is not ready for consistency audit",
        audit_record_count="remote vector record count does not match published state",
        embedding_count="embedder returned an unexpected vector count",
        embedding_dimension="embedder returned an invalid vector dimension",
        materialization_lost="memory vector materialization lost an index source",
        search_limit="memory vector search limit is outside its configured bound",
    )
    _error_type = MemoryVectorIndexError

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
        if dimension <= 0:
            raise ValueError("memory vector dimension must be positive")
        if not embedding_fingerprint or embedding_fingerprint != embedding_fingerprint.strip():
            raise ValueError("embedding_fingerprint must be normalized non-empty text")
        super().__init__(
            embedder,
            store,
            dimension=dimension,
            embedding_fingerprint=embedding_fingerprint,
            config=config or MemoryVectorIndexConfig(),
        )
        self.tree = tree
        self.sources = MemoryIndexSourceReader(
            tree,
            config=self.config,
            path_lock=path_lock,
        )

    async def synchronize(
        self,
        *,
        changed_uris: tuple[MemoryURI, ...],
        semantic_results: tuple[MemorySemanticRefreshResult, ...],
        checkpoint: int,
    ) -> VectorStoreState:
        """在 L2 和 L0/L1 均发布后，以 Job 全局序号推进远程索引。"""

        if checkpoint <= 0:
            raise ValueError("memory vector checkpoint must be positive")
        await self.ensure_ready()
        async with self._lock:
            for _ in range(self.config.stale_retries + 1):
                state = await self._ready_state_locked()
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
                    # generation/checkpoint CAS 冲突：重读状态后在有界次数内重试，用尽后由循环外明确报错。
                    continue
            raise MemoryVectorIndexError("memory vector state changed repeatedly during synchronization")

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

    def _walk_sources(self) -> tuple[MemoryIndexSource, ...]:
        return self.sources.walk()

    def _scan_filter(self) -> VectorStoreFilter:
        return _empty_filter()

    def _require_rebuild_capacity(self, state: VectorStoreState) -> None:
        if state.record_count > self.config.max_records:
            raise MemoryVectorIndexError("published vector state exceeds rebuild capacity")

    def _consistency_report(
        self,
        *,
        expected_count: int,
        indexed_count: int,
        missing_identities: tuple[str, ...],
        stale_identities: tuple[str, ...],
        orphan_identities: tuple[str, ...],
    ) -> MemoryVectorConsistencyReport:
        return MemoryVectorConsistencyReport(
            expected_count=expected_count,
            indexed_count=indexed_count,
            missing_identities=missing_identities,
            stale_identities=stale_identities,
            orphan_identities=orphan_identities,
        )

    def _record(self, source: MemoryIndexSource, vector: EmbeddingVector) -> VectorStoreRecord:
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

    @staticmethod
    def _query_vector(value: EmbeddingVector) -> EmbeddingVector:
        return value

    @staticmethod
    def _roots(values: tuple[MemoryURI, ...]) -> tuple[MemoryURI, ...]:
        if not values:
            raise ValueError("memory vector search requires at least one root")
        roots = tuple(MemoryURI.parse(value) for value in values)
        if any(root.node_type is not MemoryURINodeType.DIRECTORY for root in roots):
            raise ValueError("memory vector roots must identify directories")
        if len(roots) != len(set(roots)):
            raise ValueError("memory vector roots must be unique")
        return roots

    @staticmethod
    def _levels(values: tuple[MemoryLevel, ...]) -> tuple[MemoryLevel, ...]:
        if not values:
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
    document_prefix: str,
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
            "document_prefix": document_prefix,
            "extra_body": extra_body,
            "document_parameters": document_parameters,
        }
    )


def _empty_filter() -> VectorStoreFilter:
    return VectorStoreFilter(equals={}, one_of={})


__all__ = ["PersistentMemoryVectorIndex", "memory_embedding_fingerprint"]
