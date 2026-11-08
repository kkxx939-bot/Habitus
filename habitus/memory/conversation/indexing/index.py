"""Conversation Summary 真相源与独立远程 VectorStore 之间的可恢复索引。"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from datetime import date

from habitus.foundation.integrity import canonical_digest
from habitus.foundation.observability import NullObserver, ObservationEvent, ObservationStatus, Observer
from habitus.infrastructure.vector import (
    VectorStore,
    VectorStoreConflictError,
    VectorStoreFilter,
    VectorStoreMatch,
    VectorStoreRecord,
    VectorStoreState,
)
from habitus.memory.conversation.compaction import ConversationSummaryCompactor
from habitus.memory.conversation.indexing.config import ConversationSummaryVectorIndexConfig
from habitus.memory.conversation.indexing.model import (
    ConversationSummaryIndexError,
    ConversationSummaryIndexSource,
    ConversationSummaryMatch,
    ConversationSummaryReference,
    ConversationSummaryStage,
    ConversationSummaryVectorConsistencyReport,
)
from habitus.memory.conversation.indexing.source import (
    ConversationSummaryIndexSourceReader,
    ConversationSummaryRetirementFilter,
)
from habitus.memory.conversation.layout import ConversationAddress
from habitus.memory.conversation.messages import ConversationMessageJournal
from habitus.memory.vector_index import PersistentVectorIndex, VectorIndexMessages
from habitus.model_client import Embedder, EmbeddingVector, Reranker

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "habitus_conversation_summary_vector_v1"
_SUMMARY_KIND = "conversation_summary"
_SUMMARY_LEVEL = 2


class PersistentConversationSummaryVectorIndex(
    PersistentVectorIndex[
        ConversationSummaryIndexSource,
        ConversationSummaryVectorConsistencyReport,
        ConversationSummaryVectorIndexConfig,
    ]
):
    """仅为低频后备召回服务；Summary 文件始终是唯一内容真相源。"""

    _schema_version = _SCHEMA_VERSION
    _messages = VectorIndexMessages(
        rebuild_checkpoint="summary vector rebuild checkpoint must be non-negative",
        rebuild_conflict="summary vector state changed repeatedly during rebuild",
        audit_not_ready="summary vector index is not ready for consistency audit",
        audit_record_count="remote Summary vector record count does not match published state",
        embedding_count="embedder returned an unexpected Summary vector count",
        embedding_dimension="embedder returned an invalid Summary vector",
        materialization_lost="Summary vector materialization lost an index source",
        search_limit="Summary vector search limit is outside its configured bound",
    )
    _error_type = ConversationSummaryIndexError

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
        observer: Observer | None = None,
        retirement_store: ConversationSummaryRetirementFilter | None = None,
    ) -> None:
        if compactor.journal is not journal:
            raise ValueError("summary vector index must share the Conversation journal")
        if dimension <= 0:
            raise ValueError("summary vector dimension must be positive")
        if not embedding_fingerprint.strip():
            raise ValueError("summary embedding_fingerprint must be non-empty text")
        resolved_config = config or ConversationSummaryVectorIndexConfig()
        super().__init__(
            embedder,
            store,
            dimension=dimension,
            embedding_fingerprint=embedding_fingerprint.strip(),
            config=resolved_config,
        )
        self.journal = journal
        self.compactor = compactor
        self.reranker = reranker
        self.observer = observer or NullObserver()
        self.sources = ConversationSummaryIndexSourceReader(
            journal,
            compactor,
            config=resolved_config,
            retirement_store=retirement_store,
        )

    async def synchronize(
        self,
        address: ConversationAddress,
        *,
        checkpoint: int | None = None,
        removed_references: tuple[ConversationSummaryReference, ...] = (),
    ) -> VectorStoreState:
        """让一个 Conversation 的远程记录精确等于其当前活跃摘要前沿。"""

        if checkpoint is not None and checkpoint < 0:
            raise ValueError("summary vector checkpoint must be non-negative or None")
        if any(item.address != address for item in removed_references):
            raise ValueError("removed Summary references belong to another Conversation")
        await self.ensure_ready()
        async with self._lock:
            for _ in range(self.config.stale_retries + 1):
                state = await self._ready_state_locked()
                selected_checkpoint = state.checkpoint if checkpoint is None else checkpoint
                if selected_checkpoint < state.checkpoint:
                    raise ConversationSummaryIndexError("summary vector checkpoint attempted to move backwards")
                active = self.sources.active(address)
                active_by_id = {source.identity: source for source in active}
                known_ids = {reference.identity for reference in self.sources.all_references(address)}
                known_ids.update(active_by_id)
                known_ids.update(reference.identity for reference in removed_references)
                existing = {record.identity: record for record in await self.store.read(tuple(sorted(known_ids)))}
                upserts = await self._materialize(active, existing=existing)
                deletes = tuple(sorted(set(existing) - set(active_by_id)))
                try:
                    published = await self.store.apply(
                        upserts,
                        deletes,
                        checkpoint=selected_checkpoint,
                        expected_generation=state.generation,
                        expected_checkpoint=state.checkpoint,
                    )
                    if removed_references:
                        remaining = await self.store.read(tuple(reference.identity for reference in removed_references))
                        if remaining:
                            raise ConversationSummaryIndexError(
                                "retiring Summary vector records failed exact deletion verification"
                            )
                    return published
                except VectorStoreConflictError:
                    # generation/checkpoint CAS 冲突：重读状态后在有界次数内重试，用尽后由循环外明确报错。
                    continue
            raise ConversationSummaryIndexError("summary vector state changed repeatedly during synchronization")

    async def search(self, query: str, *, limit: int) -> tuple[ConversationSummaryMatch, ...]:
        """只在调用方确认 Memory 不充分后执行一次独立 Summary 语义召回。"""

        if not query.strip():
            raise ValueError("summary fallback query must be non-empty text")
        normalized = query.strip()
        maximum = self._limit(limit)
        await self.ensure_ready()
        vector = await self.embedder.embed_query(normalized)
        if vector.dimension != self.dimension:
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
        candidates = tuple(item for item in candidates if item.vector_score >= self.config.vector_score_threshold)
        if self.reranker is None or not candidates:
            return candidates[:maximum]
        selected = candidates[: self.config.max_rerank_candidates]
        documents = tuple(item.content[: self.config.max_rerank_document_chars] for item in selected)
        try:
            scores = await self.reranker.rerank(normalized, documents)
            if len(scores) != len(selected):
                raise ConversationSummaryIndexError("reranker returned an unexpected Summary score count")
        except Exception as exc:
            logger.warning("Summary reranker failed; using vector scores", exc_info=exc)
            self._observe_rerank_fallback(exc)
            return candidates[:maximum]
        reranked: list[ConversationSummaryMatch] = []
        try:
            for item, score in zip(selected, scores, strict=True):
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
        except Exception as exc:
            logger.warning("Summary reranker scores were invalid; using vector scores", exc_info=exc)
            self._observe_rerank_fallback(exc)
            return candidates[:maximum]
        reranked.sort(key=lambda item: (-item.score, item.reference.identity))
        return tuple(reranked[:maximum])

    def _observe_rerank_fallback(self, error: BaseException) -> None:
        self.observer.record(
            ObservationEvent(
                category="retrieval",
                operation="rerank_fallback",
                status=ObservationStatus.DEGRADED,
                duration_seconds=0.0,
                attributes={"stage": "summary", "error_type": type(error).__name__},
            )
        )

    def _walk_sources(self) -> tuple[ConversationSummaryIndexSource, ...]:
        return self.sources.walk()

    def _scan_filter(self) -> VectorStoreFilter:
        return VectorStoreFilter(equals={"kind": _SUMMARY_KIND}, one_of={})

    def _consistency_report(
        self,
        *,
        expected_count: int,
        indexed_count: int,
        missing_identities: tuple[str, ...],
        stale_identities: tuple[str, ...],
        orphan_identities: tuple[str, ...],
    ) -> ConversationSummaryVectorConsistencyReport:
        return ConversationSummaryVectorConsistencyReport(
            expected_count=expected_count,
            indexed_count=indexed_count,
            missing_identities=missing_identities,
            stale_identities=stale_identities,
            orphan_identities=orphan_identities,
        )

    def _record(
        self,
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
            reference = self._reference_from_record(value.record)
            if reference.identity in seen:
                raise ConversationSummaryIndexError("vector store returned duplicate Summary matches")
            seen.add(reference.identity)
            active = active_cache.get(reference.address)
            if active is None:
                active = {source.identity: source for source in self.sources.active(reference.address)}
                active_cache[reference.address] = active
            source = active.get(reference.identity)
            if source is None:
                raise ConversationSummaryIndexError("remote Summary match is no longer part of the active frontier")
            if source.content_digest != value.record.content_digest:
                raise ConversationSummaryIndexError("remote Summary match is stale relative to its truth source")
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


def conversation_summary_embedding_fingerprint(
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
    """模型、编码参数或 Summary 索引语义变化时触发独立重建。"""

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


__all__ = [
    "PersistentConversationSummaryVectorIndex",
    "conversation_summary_embedding_fingerprint",
]
