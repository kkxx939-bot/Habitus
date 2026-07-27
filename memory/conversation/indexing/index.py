"""Conversation Summary 真相源与独立远程 VectorStore 之间的可恢复索引。"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from datetime import date

from foundation.integrity import canonical_digest
from infrastructure.vector import (
    VectorStore,
    VectorStoreConflictError,
    VectorStoreFilter,
    VectorStoreMatch,
    VectorStoreRecord,
    VectorStoreState,
)
from memory.conversation.compaction import ConversationSummaryCompactor
from memory.conversation.indexing.config import ConversationSummaryVectorIndexConfig
from memory.conversation.indexing.model import (
    ConversationSummaryIndexError,
    ConversationSummaryIndexSource,
    ConversationSummaryMatch,
    ConversationSummaryReference,
    ConversationSummaryStage,
)
from memory.conversation.indexing.source import ConversationSummaryIndexSourceReader
from memory.conversation.layout import ConversationAddress
from memory.conversation.messages import ConversationMessageJournal
from ModelClient import Embedder, EmbeddingVector, Reranker

_SCHEMA_VERSION = "m2bos_conversation_summary_vector_v1"
_SUMMARY_KIND = "conversation_summary"
_SUMMARY_LEVEL = 2


class PersistentConversationSummaryVectorIndex:
    """仅为低频后备召回服务；Summary 文件始终是唯一内容真相源。"""

    def __init__(
        self,
        journal: ConversationMessageJournal,
        compactor: ConversationSummaryCompactor,
        embedder: Embedder,
        store: VectorStore,
        *,
        dimension: int,
        embedding_fingerprint: str,
        reranker: Reranker | None = None,
        config: ConversationSummaryVectorIndexConfig | None = None,
    ) -> None:
        if not isinstance(journal, ConversationMessageJournal):
            raise TypeError("journal must be ConversationMessageJournal")
        if not isinstance(compactor, ConversationSummaryCompactor):
            raise TypeError("compactor must be ConversationSummaryCompactor")
        if compactor.journal is not journal:
            raise ValueError("summary vector index must share the Conversation journal")
        if not callable(getattr(embedder, "embed_query", None)) or not callable(
            getattr(embedder, "embed_documents", None)
        ):
            raise TypeError("embedder must implement query and document embedding")
        for name in ("initialize", "state", "read", "replace_all", "apply", "search", "scan", "close"):
            if not callable(getattr(store, name, None)):
                raise TypeError(f"store must implement VectorStore.{name}")
        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("summary vector dimension must be positive")
        if not isinstance(embedding_fingerprint, str) or not embedding_fingerprint.strip():
            raise ValueError("summary embedding_fingerprint must be non-empty text")
        if reranker is not None and not callable(getattr(reranker, "rerank", None)):
            raise TypeError("reranker must implement rerank")
        resolved_config = config or ConversationSummaryVectorIndexConfig()
        self.journal = journal
        self.compactor = compactor
        self.embedder = embedder
        self.store = store
        self.dimension = dimension
        self.embedding_fingerprint = embedding_fingerprint.strip()
        self.reranker = reranker
        self.config = resolved_config
        self.sources = ConversationSummaryIndexSourceReader(
            journal,
            compactor,
            config=resolved_config,
        )
        self._lock = asyncio.Lock()

    async def ensure_ready(self) -> VectorStoreState:
        """验证独立集合；缺失或模型语义变化时从活跃 Summary 前沿重建。"""

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
        address: ConversationAddress,
        *,
        checkpoint: int | None = None,
    ) -> VectorStoreState:
        """让一个 Conversation 的远程记录精确等于其当前活跃摘要前沿。"""

        if not isinstance(address, ConversationAddress):
            raise TypeError("address must be ConversationAddress")
        if checkpoint is not None and (
            isinstance(checkpoint, bool) or not isinstance(checkpoint, int) or checkpoint < 0
        ):
            raise ValueError("summary vector checkpoint must be non-negative or None")
        await self.ensure_ready()
        async with self._lock:
            for _ in range(self.config.stale_retries + 1):
                state = await self.store.state()
                if not self._state_matches(state):
                    state = await self._rebuild_locked(checkpoint=0 if state is None else state.checkpoint)
                assert state is not None
                selected_checkpoint = state.checkpoint if checkpoint is None else checkpoint
                if selected_checkpoint < state.checkpoint:
                    raise ConversationSummaryIndexError("summary vector checkpoint attempted to move backwards")
                active = self.sources.active(address)
                active_by_id = {source.identity: source for source in active}
                known_ids = {reference.identity for reference in self.sources.all_references(address)}
                known_ids.update(active_by_id)
                existing = {
                    record.identity: record
                    for record in await self.store.read(tuple(sorted(known_ids)))
                }
                upserts = await self._materialize(active, existing=existing)
                deletes = tuple(sorted(set(existing) - set(active_by_id)))
                try:
                    return await self.store.apply(
                        upserts,
                        deletes,
                        checkpoint=selected_checkpoint,
                        expected_generation=state.generation,
                        expected_checkpoint=state.checkpoint,
                    )
                except VectorStoreConflictError:
                    continue
            raise ConversationSummaryIndexError(
                "summary vector state changed repeatedly during synchronization"
            )

    async def rebuild(self, *, checkpoint: int | None = None) -> VectorStoreState:
        """管理入口：从全部 Conversation 的活跃摘要前沿完整重建。"""

        await self.store.initialize()
        async with self._lock:
            current = await self.store.state()
            selected = (0 if current is None else current.checkpoint) if checkpoint is None else checkpoint
            if isinstance(selected, bool) or not isinstance(selected, int) or selected < 0:
                raise ValueError("summary vector rebuild checkpoint must be non-negative")
            return await self._rebuild_locked(checkpoint=selected)

    async def search(self, query: str, *, limit: int) -> tuple[ConversationSummaryMatch, ...]:
        """只在调用方确认 Memory 不充分后执行一次独立 Summary 语义召回。"""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("summary fallback query must be non-empty text")
        normalized = query.strip()
        maximum = self._limit(limit)
        await self.ensure_ready()
        vector = await self.embedder.embed_query(normalized)
        if not isinstance(vector, EmbeddingVector) or vector.dimension != self.dimension:
            raise ConversationSummaryIndexError("embedder returned an invalid Summary query vector")
        candidate_limit = min(
            self.config.max_search_hits,
            max(
                self.config.min_vector_candidates,
                maximum * self.config.candidate_multiplier,
            ),
        )
        raw = await self.store.search(
            vector,
            filters=VectorStoreFilter(equals={"kind": _SUMMARY_KIND}, one_of={}),
            limit=candidate_limit,
        )
        candidates = self._resolve_matches(raw)
        candidates = tuple(
            item for item in candidates if item.vector_score >= self.config.vector_score_threshold
        )
        if self.reranker is None or not candidates:
            return candidates[:maximum]
        selected = candidates[: self.config.max_rerank_candidates]
        documents = tuple(
            item.content[: self.config.max_rerank_document_chars] for item in selected
        )
        scores = await self.reranker.rerank(normalized, documents)
        if not isinstance(scores, tuple) or len(scores) != len(selected):
            raise ConversationSummaryIndexError("reranker returned an unexpected Summary score count")
        reranked: list[ConversationSummaryMatch] = []
        for item, score in zip(selected, scores, strict=True):
            if isinstance(score, bool) or not isinstance(score, int | float):
                raise ConversationSummaryIndexError("reranker returned a non-numeric Summary score")
            rerank_score = float(score)
            if not math.isfinite(rerank_score):
                raise ConversationSummaryIndexError("reranker returned a non-finite Summary score")
            if rerank_score < self.config.rerank_score_threshold:
                continue
            reranked.append(
                ConversationSummaryMatch(
                    reference=item.reference,
                    summary=item.summary,
                    content=item.content,
                    score=rerank_score,
                    vector_score=item.vector_score,
                    rerank_score=rerank_score,
                )
            )
        reranked.sort(key=lambda item: (-item.score, item.reference.identity))
        return tuple(reranked[:maximum])

    async def close(self) -> None:
        await self.store.close()

    async def _rebuild_locked(self, *, checkpoint: int) -> VectorStoreState:
        for _ in range(self.config.stale_retries + 1):
            state = await self.store.state()
            sources = self.sources.walk()
            reusable: dict[str, VectorStoreRecord] = {}
            if self._state_matches(state):
                assert state is not None
                reusable = {
                    record.identity: record
                    for record in await self.store.scan(
                        filters=VectorStoreFilter(equals={"kind": _SUMMARY_KIND}, one_of={}),
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
        raise ConversationSummaryIndexError("summary vector state changed repeatedly during rebuild")

    async def _materialize(
        self,
        sources: Sequence[ConversationSummaryIndexSource],
        *,
        existing: dict[str, VectorStoreRecord],
    ) -> tuple[VectorStoreRecord, ...]:
        resolved: dict[str, VectorStoreRecord] = {}
        missing: list[ConversationSummaryIndexSource] = []
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
                raise ConversationSummaryIndexError("embedder returned an unexpected Summary vector count")
            for source, vector in zip(batch, vectors, strict=True):
                if not isinstance(vector, EmbeddingVector) or vector.dimension != self.dimension:
                    raise ConversationSummaryIndexError("embedder returned an invalid Summary vector")
                resolved[source.identity] = self._record(source, vector)
        if len(resolved) != len(sources):
            raise AssertionError("Summary vector materialization lost an index source")
        return tuple(resolved[identity] for identity in sorted(resolved))

    @staticmethod
    def _record(
        source: ConversationSummaryIndexSource,
        vector: EmbeddingVector,
    ) -> VectorStoreRecord:
        reference = source.reference
        summary = source.summary
        return VectorStoreRecord(
            identity=source.identity,
            vector=vector,
            content=source.content,
            content_digest=source.content_digest,
            attributes={
                "uri": source.identity,
                "level": _SUMMARY_LEVEL,
                "directory_key": reference.conversation_scope,
                "parent_key": "conversation-summary-root",
                "scope_roots": ("conversation-summary-root", reference.conversation_scope),
                "kind": _SUMMARY_KIND,
                "revision": 0,
                "started_on": reference.address.started_on.isoformat(),
                "conversation_id": reference.address.conversation_id,
                "stage": reference.stage.value,
                "summary_id": reference.summary_id,
                "start_sequence": summary.start_sequence,
                "end_sequence": summary.end_sequence,
            },
        )

    def _resolve_matches(
        self,
        values: Sequence[VectorStoreMatch],
    ) -> tuple[ConversationSummaryMatch, ...]:
        matches: list[ConversationSummaryMatch] = []
        active_cache: dict[ConversationAddress, dict[str, ConversationSummaryIndexSource]] = {}
        seen: set[str] = set()
        for value in values:
            if not isinstance(value, VectorStoreMatch):
                raise ConversationSummaryIndexError("vector store returned an invalid Summary match")
            reference = self._reference_from_record(value.record)
            if reference.identity in seen:
                raise ConversationSummaryIndexError("vector store returned duplicate Summary matches")
            seen.add(reference.identity)
            active = active_cache.get(reference.address)
            if active is None:
                active = {
                    source.identity: source
                    for source in self.sources.active(reference.address)
                }
                active_cache[reference.address] = active
            source = active.get(reference.identity)
            if source is None:
                raise ConversationSummaryIndexError(
                    "remote Summary match is no longer part of the active frontier"
                )
            if source.content_digest != value.record.content_digest:
                raise ConversationSummaryIndexError(
                    "remote Summary match is stale relative to its truth source"
                )
            matches.append(
                ConversationSummaryMatch(
                    reference=reference,
                    summary=source.summary,
                    content=source.content,
                    score=value.score,
                    vector_score=value.score,
                )
            )
        matches.sort(key=lambda item: (-item.score, item.reference.identity))
        return tuple(matches)

    @staticmethod
    def _reference_from_record(record: VectorStoreRecord) -> ConversationSummaryReference:
        values = record.attributes
        started_on = values.get("started_on")
        conversation_id = values.get("conversation_id")
        stage = values.get("stage")
        summary_id = values.get("summary_id")
        if (
            not isinstance(started_on, str)
            or not started_on
            or not isinstance(conversation_id, str)
            or not conversation_id
            or not isinstance(stage, str)
            or not stage
            or not isinstance(summary_id, str)
            or not summary_id
        ):
            raise ConversationSummaryIndexError("Summary vector record is missing source identity attributes")
        try:
            reference = ConversationSummaryReference(
                address=ConversationAddress(
                    conversation_id=conversation_id,
                    started_on=date.fromisoformat(started_on),
                ),
                stage=ConversationSummaryStage(stage),
                summary_id=summary_id,
            )
        except (TypeError, ValueError) as exc:
            raise ConversationSummaryIndexError("Summary vector record source identity is invalid") from exc
        if record.identity != reference.identity or values.get("uri") != reference.identity:
            raise ConversationSummaryIndexError("Summary vector record identity does not match its attributes")
        return reference

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
            raise ValueError("Summary vector search limit is outside its configured bound")
        return value


def conversation_summary_embedding_fingerprint(
    *,
    provider: str,
    model: str,
    dimension: int,
    input_mode: str,
    document_parameters: object,
) -> str:
    """模型、编码参数或 Summary 索引语义变化时触发独立重建。"""

    return canonical_digest(
        {
            "schema": _SCHEMA_VERSION,
            "provider": provider,
            "model": model,
            "dimension": dimension,
            "input_mode": input_mode,
            "document_parameters": document_parameters,
        }
    )


__all__ = [
    "PersistentConversationSummaryVectorIndex",
    "conversation_summary_embedding_fingerprint",
]
