"""面向 Agent 的完整只读记忆 SearchService。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import Protocol

from habitus.foundation.observability import NullObserver, ObservationEvent, ObservationStatus, Observer
from habitus.memory.conversation import ConversationAddress
from habitus.memory.conversation.indexing import ConversationSummaryMatch
from habitus.memory.document import MemoryDocument
from habitus.memory.intention import (
    MemoryIntentionRecallScope,
    MemoryIntentionReview,
    MemoryIntentionReviewer,
    intention_matches_scope,
)
from habitus.memory.model import MemoryDirectory, MemoryKind, MemoryLevel
from habitus.memory.recall import (
    MemoryRecallCandidate,
    MemoryRecallLifecycle,
    MemoryRecallLifecycleError,
    MemoryRecallTarget,
)
from habitus.memory.retrieval.acknowledgement import (
    MemoryColdProbeExpander,
    append_degradation,
    final_context_acknowledger,
)
from habitus.memory.retrieval.assembler import MemoryContextAssembler, MemoryContextAssembly
from habitus.memory.retrieval.context import ConversationSearchContextReader
from habitus.memory.retrieval.grader import MemoryRetrievalGrader
from habitus.memory.retrieval.model import (
    MemoryMatchedMemory,
    MemoryQueryPlan,
    MemoryQueryResult,
    MemoryRetrievalAssessment,
    MemoryRetrievalSufficiency,
    MemorySearchDegradation,
    MemorySearchDegradationStage,
    MemorySearchError,
    MemorySearchHit,
    MemorySearchResult,
    MemorySearchServiceConfig,
    MemoryTypedQuery,
)
from habitus.memory.retrieval.planner import MemorySearchQueryPlanner
from habitus.memory.retrieval.relations import RelationExpander, review_intention
from habitus.memory.retrieval.request import MemorySearchRequestNormalizer
from habitus.memory.snapshot import MemorySnapshotReader
from habitus.memory.uri import MemoryURI


class MemorySemanticSearch(Protocol):
    """公共 SearchService 和 Memory Editor 共同依赖的语义搜索契约。"""

    async def search(
        self,
        query: str,
        *,
        roots: tuple[MemoryURI, ...],
        kinds: tuple[MemoryKind, ...],
        intention_scope: MemoryIntentionRecallScope,
        limit: int,
    ) -> Sequence[MemorySearchHit]: ...


class ConversationSummarySemanticSearch(Protocol):
    """仅供 Memory 召回不足后使用的历史 Summary 搜索契约。"""

    async def search(
        self,
        query: str,
        *,
        limit: int,
    ) -> Sequence[ConversationSummaryMatch]: ...


class ConversationSummaryFallbackExpander(Protocol):
    def expand(
        self,
        match: ConversationSummaryMatch,
        *,
        max_chars: int,
    ) -> ConversationSummaryMatch: ...


class ConversationSummaryUseRecorder(Protocol):
    def record_use(self, references: tuple, *, used_at: datetime) -> tuple: ...


class MemoryDirectoryLayerReader(Protocol):
    """SearchService 读取目录 L0 目标上下文所需的最小记忆树契约；``MemoryTree`` 结构化满足它。"""

    def directory_exists(self, directory: MemoryDirectory) -> bool: ...

    def layer_exists(self, directory: MemoryDirectory, level: MemoryLevel) -> bool: ...

    def read_layer_bounded(self, directory: MemoryDirectory, level: MemoryLevel, *, max_bytes: int) -> str: ...


def _temperature_count(memories: tuple[MemoryMatchedMemory, ...], temperature: str) -> int:
    return sum(
        memory.hit.lifecycle_temperature is not None and memory.hit.lifecycle_temperature.value == temperature
        for memory in memories
    )


def _recall_target(memory: MemoryMatchedMemory) -> MemoryRecallTarget:
    return MemoryRecallTarget(
        memory.uri,
        memory.document.metadata.revision,
        memory.document.metadata.created_at,
    )


def _search_observation(plan: MemoryQueryPlan, result: MemorySearchResult, *, started: float) -> ObservationEvent:
    """一次 find/search 完成后的检索观察事件；有任何降级即记为 DEGRADED。"""

    return ObservationEvent(
        category="retrieval",
        operation="search",
        status=(ObservationStatus.DEGRADED if result.degradations else ObservationStatus.SUCCESS),
        duration_seconds=max(0.0, time.monotonic() - started),
        attributes={
            "queries": len(plan.queries),
            "memories": len(result.memories),
            "summaries": len(result.summary_fallbacks),
            "degradations": len(result.degradations),
            "result_count": len(result.memories) + len(result.summary_fallbacks),
            "degradation_count": len(result.degradations),
            "hot_memories": _temperature_count(result.memories, "hot"),
            "warm_memories": _temperature_count(result.memories, "warm"),
            "cold1_memories": _temperature_count(result.memories, "cold_1"),
            "cold2_memories": _temperature_count(result.memories, "cold_2"),
        },
    )


class SearchService:
    """按 find/search 分层返回完整且可溯源的 L2 记忆上下文。"""

    def __init__(
        self,
        *,
        tree: MemoryDirectoryLayerReader,
        snapshot_reader: MemorySnapshotReader,
        semantic_search: MemorySemanticSearch,
        summary_search: ConversationSummarySemanticSearch,
        query_planner: MemorySearchQueryPlanner,
        retrieval_grader: MemoryRetrievalGrader,
        recall_lifecycle: MemoryRecallLifecycle,
        conversation_context: ConversationSearchContextReader,
        assembler: MemoryContextAssembler | None = None,
        config: MemorySearchServiceConfig | None = None,
        intention_reviewer: MemoryIntentionReviewer | None = None,
        cold_probe_expander: MemoryColdProbeExpander | None = None,
        summary_fallback_expander: ConversationSummaryFallbackExpander | None = None,
        summary_use_recorder: ConversationSummaryUseRecorder | None = None,
        clock: Callable[[], datetime] | None = None,
        observer: Observer | None = None,
    ) -> None:
        if snapshot_reader.tree is not tree:
            raise ValueError("SearchService and snapshot reader must share one memory tree")
        resolved_config = config or MemorySearchServiceConfig()
        resolved_assembler = assembler or MemoryContextAssembler(config=resolved_config)
        if query_planner.config != resolved_config:
            raise ValueError("SearchService and query planner must share one search config")
        if retrieval_grader.config != resolved_config:
            raise ValueError("SearchService and retrieval grader must share one search config")
        if conversation_context.config != resolved_config:
            raise ValueError("SearchService and Conversation context reader must share one search config")
        if resolved_assembler.config != resolved_config:
            raise ValueError("SearchService and context assembler must share one search config")
        self.tree = tree
        self.snapshot_reader = snapshot_reader
        self.semantic_search = semantic_search
        self.summary_search = summary_search
        self.query_planner = query_planner
        self.retrieval_grader = retrieval_grader
        self.recall_lifecycle = recall_lifecycle
        self.conversation_context = conversation_context
        self.assembler = resolved_assembler
        self.config = resolved_config
        self.intention_reviewer = intention_reviewer or MemoryIntentionReviewer()
        self.cold_probe_expander = cold_probe_expander
        self.summary_fallback_expander = summary_fallback_expander
        self.summary_use_recorder = summary_use_recorder
        self.clock = clock or (lambda: datetime.now(UTC))
        self.observer = observer or NullObserver()
        self.request_normalizer = MemorySearchRequestNormalizer(resolved_config)
        self.relation_expander = RelationExpander(
            snapshot_reader=snapshot_reader,
            config=resolved_config,
            intention_reviewer=self.intention_reviewer,
        )

    async def find(
        self,
        query: str,
        *,
        target_uris: MemoryURI | str | tuple[MemoryURI | str, ...] | None = None,
        limit: int | None = None,
        score_threshold: float | None = None,
        kinds: tuple[MemoryKind, ...] = (),
        intention_scope: MemoryIntentionRecallScope = MemoryIntentionRecallScope.ACTIVE,
    ) -> MemorySearchResult:
        """不调用查询规划模型，直接执行一次完整语义检索。"""

        request = self.request_normalizer.normalize(
            query,
            target_uris=target_uris,
            limit=limit,
            score_threshold=score_threshold,
            kinds=kinds,
            intention_scope=intention_scope,
        )
        plan = self.query_planner.direct(request.query)
        return await self._execute(
            plan,
            request.roots,
            request.limit,
            request.score_threshold,
            request.kinds,
            request.intention_scope,
            assess_for_summary=False,
            degradations=(),
        )

    async def search(
        self,
        query: str,
        *,
        conversation: ConversationAddress | None = None,
        target_uris: MemoryURI | str | tuple[MemoryURI | str, ...] | None = None,
        limit: int | None = None,
        score_threshold: float | None = None,
        kinds: tuple[MemoryKind, ...] = (),
        intention_scope: MemoryIntentionRecallScope = MemoryIntentionRecallScope.ACTIVE,
    ) -> MemorySearchResult:
        """完成 Memory 主召回，并仅在 Grader 判定不足时补查历史 Summary。"""

        request = self.request_normalizer.normalize(
            query,
            target_uris=target_uris,
            limit=limit,
            score_threshold=score_threshold,
            kinds=kinds,
            intention_scope=intention_scope,
        )
        degradations: tuple[MemorySearchDegradation, ...] = ()
        if conversation is None:
            plan = self.query_planner.direct(request.query)
        else:
            try:
                context = self.conversation_context.read(conversation)
                plan = await self.query_planner.plan(
                    request.query,
                    context,
                    target_context=self._target_context(request.roots),
                )
            except Exception as exc:
                plan = self.query_planner.direct(request.query)
                degradations = (
                    MemorySearchDegradation(
                        MemorySearchDegradationStage.QUERY_PLANNER,
                        type(exc).__name__,
                    ),
                )
        return await self._execute(
            plan,
            request.roots,
            request.limit,
            request.score_threshold,
            request.kinds,
            request.intention_scope,
            assess_for_summary=True,
            degradations=degradations,
        )

    async def _execute(
        self,
        plan: MemoryQueryPlan,
        roots: tuple[MemoryURI, ...],
        limit: int,
        score_threshold: float | None,
        kinds: tuple[MemoryKind, ...],
        intention_scope: MemoryIntentionRecallScope,
        *,
        assess_for_summary: bool,
        degradations: tuple[MemorySearchDegradation, ...] = (),
    ) -> MemorySearchResult:
        started = time.monotonic()
        applied_degradations = list(degradations)
        query_results = await self._execute_queries(plan, roots, limit, score_threshold, kinds, intention_scope)
        hits, matched_queries = self._aggregate(query_results)
        review_now = self._timestamp()
        direct = self._select_direct(
            hits,
            matched_queries,
            limit=limit,
            review_now=review_now,
            kinds=kinds,
            intention_scope=intention_scope,
            degradations=applied_degradations,
        )
        direct_targets = {memory.uri: _recall_target(memory) for memory in direct}
        expanded = self.relation_expander.expand(
            self._expand_cold_for_budget(direct),
            roots=roots,
            intention_scope=intention_scope,
            review_now=review_now,
        )
        assembly, assessment, summary_fallback_attempted = await self._assemble_final_context(
            plan,
            expanded,
            targets=direct_targets,
            used_at=review_now,
            assess_for_summary=assess_for_summary,
            degradations=applied_degradations,
        )
        result = MemorySearchResult(
            query=plan.original_query,
            target_roots=roots,
            kinds=kinds,
            intention_scope=intention_scope,
            plan=plan,
            query_results=query_results,
            memories=assembly.memories,
            retrieval_assessment=assessment,
            summary_fallback_attempted=summary_fallback_attempted,
            summary_fallbacks=assembly.summary_fallbacks,
            context=assembly.context,
            budget_exhausted=assembly.budget_exhausted,
            degradations=tuple(applied_degradations),
        )
        self.observer.record(_search_observation(plan, result, started=started))
        return result

    async def _execute_queries(
        self,
        plan: MemoryQueryPlan,
        roots: tuple[MemoryURI, ...],
        limit: int,
        score_threshold: float | None,
        kinds: tuple[MemoryKind, ...],
        intention_scope: MemoryIntentionRecallScope,
    ) -> tuple[MemoryQueryResult, ...]:
        """并发执行查询计划里的每条查询，每条按候选倍数放宽到有界上限。"""

        candidate_limit = min(
            1_000,
            max(limit, limit * self.config.candidate_multiplier),
        )
        return tuple(
            await asyncio.gather(
                *(
                    self._execute_query(
                        query,
                        roots=roots,
                        limit=candidate_limit,
                        score_threshold=score_threshold,
                        kinds=kinds,
                        intention_scope=intention_scope,
                    )
                    for query in plan.queries
                )
            )
        )

    def _select_direct(
        self,
        hits: tuple[MemorySearchHit, ...],
        matched_queries: dict[MemoryURI, tuple[str, ...]],
        *,
        limit: int,
        review_now: datetime,
        kinds: tuple[MemoryKind, ...],
        intention_scope: MemoryIntentionRecallScope,
        degradations: list[MemorySearchDegradation],
    ) -> tuple[MemoryMatchedMemory, ...]:
        """读取有界数量的完整 L2，再按召回生命周期排序截到 ``limit``；生命周期失败时退回语义顺序。"""

        read_limit = min(
            self.snapshot_reader.config.max_items,
            max(limit, limit * self.config.candidate_multiplier),
        )
        direct_candidates = self._read_direct(
            hits[:read_limit],
            matched_queries,
            review_now=review_now,
            kinds=kinds,
            intention_scope=intention_scope,
        )
        try:
            return self._rank_recall_lifecycle(
                direct_candidates,
                limit=limit,
                now=review_now,
            )
        except MemoryRecallLifecycleError as exc:
            append_degradation(
                degradations,
                MemorySearchDegradationStage.RECALL_LIFECYCLE,
                exc,
            )
            return direct_candidates[:limit]

    async def _assemble_final_context(
        self,
        plan: MemoryQueryPlan,
        expanded: tuple[MemoryMatchedMemory, ...],
        *,
        targets: dict[MemoryURI, MemoryRecallTarget],
        used_at: datetime,
        assess_for_summary: bool,
        degradations: list[MemorySearchDegradation],
    ) -> tuple[MemoryContextAssembly, MemoryRetrievalAssessment | None, bool]:
        """组装、评估与按需补查 Summary，使用确认剔除了记忆且尚未补查时再评估一次。"""

        assembly = self.assembler.assemble(expanded)
        assessment: MemoryRetrievalAssessment | None = None
        summary_fallback_attempted = False
        if assess_for_summary:
            assembly, assessment, summary_fallback_attempted = await self._assess_with_summary_fallback(
                plan,
                assembly,
                memories=expanded,
                degradations=degradations,
            )
        pre_ack_uris = {memory.uri for memory in assembly.memories}
        assembly = await final_context_acknowledger(
            recall_lifecycle=self.recall_lifecycle,
            cold_probe_expander=self.cold_probe_expander,
            assembler=self.assembler,
        ).acknowledge(assembly, targets=targets, used_at=used_at, degradations=degradations)
        post_ack_uris = {memory.uri for memory in assembly.memories}
        if assess_for_summary and not summary_fallback_attempted and post_ack_uris != pre_ack_uris:
            assembly, assessment, summary_fallback_attempted = await self._assess_with_summary_fallback(
                plan,
                assembly,
                memories=assembly.memories,
                degradations=degradations,
            )
        assembly = await self._acknowledge_final_summaries(
            assembly,
            used_at=used_at,
            degradations=degradations,
        )
        return assembly, assessment, summary_fallback_attempted

    async def _assess_with_summary_fallback(
        self,
        plan: MemoryQueryPlan,
        assembly: MemoryContextAssembly,
        *,
        memories: tuple[MemoryMatchedMemory, ...],
        degradations: list[MemorySearchDegradation],
    ) -> tuple[MemoryContextAssembly, MemoryRetrievalAssessment | None, bool]:
        """Grader 判定不足且预算未耗尽时补查一次 Summary，并以 ``memories`` 重新组装；返回是否尝试过补查。"""

        try:
            assessment: MemoryRetrievalAssessment | None = await self.retrieval_grader.assess(
                plan,
                assembly.memories,
                assembly.context,
            )
        except Exception as exc:
            assessment = None
            append_degradation(degradations, MemorySearchDegradationStage.RETRIEVAL_GRADER, exc)
        if not (
            self.config.summary_fallback_enabled
            and assessment is not None
            and assessment.decision is MemoryRetrievalSufficiency.INSUFFICIENT
            and not assembly.budget_exhausted
        ):
            return assembly, assessment, False
        assert assessment.summary_query is not None
        try:
            summaries = await self._search_summaries(assessment.summary_query)
        except Exception as exc:
            append_degradation(degradations, MemorySearchDegradationStage.SUMMARY_FALLBACK, exc)
            return assembly, assessment, True
        return self.assembler.assemble(memories, summary_fallbacks=summaries), assessment, True

    async def _search_summaries(
        self,
        query: str,
    ) -> tuple[ConversationSummaryMatch, ...]:
        """只在 Grader 明确判定不足后执行一次有界 Summary 后备检索。"""

        try:
            raw = await self.summary_search.search(
                query,
                limit=self.config.summary_fallback_limit,
            )
        except Exception as exc:
            raise MemorySearchError("Conversation Summary fallback search failed") from exc
        if len(raw) > self.config.summary_fallback_limit:
            raise MemorySearchError("Summary fallback search exceeded its requested limit")
        values = tuple(raw)
        expected = tuple(sorted(values, key=lambda item: (-item.score, item.reference.identity)))
        if values != expected or len({item.reference.identity for item in values}) != len(values):
            raise MemorySearchError("Summary fallback matches must be unique and relevance-sorted")
        if self.summary_fallback_expander is None:
            return values
        expanded: list[ConversationSummaryMatch] = []
        remaining = self.config.summary_fallback_max_context_chars
        for item in values:
            value = self.summary_fallback_expander.expand(item, max_chars=max(1, remaining))
            expanded.append(value)
            remaining = max(0, remaining - len(value.content))
        return tuple(expanded)

    async def _execute_query(
        self,
        query: MemoryTypedQuery,
        *,
        roots: tuple[MemoryURI, ...],
        limit: int,
        score_threshold: float | None,
        kinds: tuple[MemoryKind, ...],
        intention_scope: MemoryIntentionRecallScope,
    ) -> MemoryQueryResult:
        try:
            raw_hits = await self.semantic_search.search(
                query.query,
                roots=roots,
                kinds=kinds,
                intention_scope=intention_scope,
                limit=limit,
            )
        except Exception as exc:
            self.observer.record(
                ObservationEvent(
                    category="retrieval",
                    operation="semantic_search",
                    status=ObservationStatus.FAILURE,
                    duration_seconds=0.0,
                    attributes={"error_type": type(exc).__name__},
                )
            )
            raise MemorySearchError("memory semantic search failed") from exc
        if len(raw_hits) > limit:
            raise MemorySearchError("memory semantic search exceeded its requested result limit")
        by_uri: dict[MemoryURI, MemorySearchHit] = {}
        for hit in raw_hits:
            if not any(hit.uri.matches_prefix(root) for root in roots):
                raise MemorySearchError("memory semantic search returned an out-of-scope URI")
            if kinds and hit.uri.to_address().kind not in kinds:
                raise MemorySearchError("memory semantic search ignored its direct kind filter")
            if score_threshold is not None and hit.score < score_threshold:
                continue
            previous = by_uri.get(hit.uri)
            if previous is None or hit.score > previous.score:
                by_uri[hit.uri] = hit
        hits = tuple(sorted(by_uri.values(), key=lambda hit: (-hit.score, str(hit.uri))))
        return MemoryQueryResult(query=query, hits=hits)

    @staticmethod
    def _aggregate(
        results: tuple[MemoryQueryResult, ...],
    ) -> tuple[tuple[MemorySearchHit, ...], dict[MemoryURI, tuple[str, ...]]]:
        best: dict[MemoryURI, MemorySearchHit] = {}
        queries: dict[MemoryURI, list[str]] = {}
        for result in results:
            for hit in result.hits:
                previous = best.get(hit.uri)
                if previous is None or hit.score > previous.score:
                    best[hit.uri] = hit
                matched = queries.setdefault(hit.uri, [])
                if result.query.query not in matched:
                    matched.append(result.query.query)
        hits = tuple(sorted(best.values(), key=lambda hit: (-hit.score, str(hit.uri))))
        return hits, {uri: tuple(values) for uri, values in queries.items()}

    def _read_direct(
        self,
        hits: tuple[MemorySearchHit, ...],
        matched_queries: dict[MemoryURI, tuple[str, ...]],
        *,
        review_now: datetime,
        kinds: tuple[MemoryKind, ...],
        intention_scope: MemoryIntentionRecallScope,
    ) -> tuple[MemoryMatchedMemory, ...]:
        if not hits:
            return ()
        try:
            snapshots = self.snapshot_reader.read_many(tuple(hit.uri for hit in hits))
        except Exception as exc:
            raise MemorySearchError("failed to read complete matched memory documents") from exc
        selected: list[MemoryMatchedMemory] = []
        for hit in hits:
            snapshot = snapshots.get(str(hit.uri))
            if snapshot is None or not snapshot.exists or snapshot.value is None:
                raise MemorySearchError("semantic index returned a missing or invalid L2 memory document")
            document = snapshot.value
            if kinds and document.kind not in kinds:
                raise MemorySearchError("semantic index kind does not match the complete L2 document")
            if document.kind is MemoryKind.INTENTION and not intention_matches_scope(
                document.fields.get("status"),
                intention_scope,
            ):
                raise MemorySearchError("semantic index returned an Intention outside the requested scope")
            selected.append(
                MemoryMatchedMemory(
                    hit=hit,
                    document=document,
                    matched_queries=matched_queries[hit.uri],
                    intention_review=self._intention_review(document, now=review_now),
                )
            )
        return tuple(selected)

    def _rank_recall_lifecycle(
        self,
        memories: tuple[MemoryMatchedMemory, ...],
        *,
        limit: int,
        now: datetime,
    ) -> tuple[MemoryMatchedMemory, ...]:
        if not memories:
            return ()
        rankings = self.recall_lifecycle.select(
            tuple(
                MemoryRecallCandidate(
                    target=MemoryRecallTarget(
                        memory.uri,
                        memory.document.metadata.revision,
                        memory.document.metadata.created_at,
                    ),
                    kind=memory.document.kind,
                    updated_at=memory.document.metadata.updated_at,
                    semantic_score=memory.hit.score,
                )
                for memory in memories
            ),
            limit=limit,
            now=now,
        )
        by_uri = {memory.uri: memory for memory in memories}
        ranked: list[MemoryMatchedMemory] = []
        for ranking in rankings:
            memory = by_uri[ranking.candidate.uri]
            hit = memory.hit
            if ranking.hotness is not None and ranking.temperature is not None:
                hit = replace(
                    hit,
                    score=ranking.final_score,
                    lifecycle_semantic_score=ranking.candidate.semantic_score,
                    lifecycle_hotness=ranking.hotness,
                    lifecycle_temperature=ranking.temperature,
                )
            ranked.append(replace(memory, hit=hit))
        return tuple(ranked)

    def _expand_cold_for_budget(
        self,
        memories: tuple[MemoryMatchedMemory, ...],
    ) -> tuple[MemoryMatchedMemory, ...]:
        if self.cold_probe_expander is None:
            return memories
        expanded: list[MemoryMatchedMemory] = []
        for memory in memories:
            document = memory.document
            if memory.hit.lifecycle_temperature is not None and memory.hit.lifecycle_temperature.value == "cold_2":
                document = self.cold_probe_expander.expand_for_probe(document)
            expanded.append(replace(memory, document=document))
        return tuple(expanded)

    async def _acknowledge_final_summaries(
        self,
        assembly: MemoryContextAssembly,
        *,
        used_at: datetime,
        degradations: list[MemorySearchDegradation],
    ) -> MemoryContextAssembly:
        if self.summary_use_recorder is None or not assembly.summary_fallbacks:
            return assembly
        retained: list[ConversationSummaryMatch] = []
        for summary in assembly.summary_fallbacks:
            try:
                states = await asyncio.to_thread(
                    self.summary_use_recorder.record_use,
                    (summary.reference,),
                    used_at=used_at,
                )
            except Exception as exc:
                append_degradation(
                    degradations,
                    MemorySearchDegradationStage.SUMMARY_FALLBACK,
                    exc,
                )
                continue
            if states:
                retained.append(summary)
        return self.assembler.assemble(
            assembly.memories,
            summary_fallbacks=tuple(retained),
        )

    def _intention_review(
        self,
        document: MemoryDocument,
        *,
        now: datetime,
    ) -> MemoryIntentionReview | None:
        return review_intention(self.intention_reviewer, document, now=now)

    def _timestamp(self) -> datetime:
        timestamp = self.clock()
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("memory search clock must return a timezone-aware datetime")
        return timestamp.astimezone(UTC)

    def _target_context(self, roots: tuple[MemoryURI, ...]) -> str:
        rendered: list[str] = []
        used = 0
        for root in roots:
            directory = root.to_directory()
            if not self.tree.directory_exists(directory) or not self.tree.layer_exists(
                directory,
                MemoryLevel.ABSTRACT,
            ):
                continue
            abstract = self.tree.read_layer_bounded(
                directory,
                MemoryLevel.ABSTRACT,
                max_bytes=self.config.max_target_context_chars * 4 + 4,
            ).strip()
            block = f"[{root}] {abstract}"
            separator = 1 if rendered else 0
            if used + separator + len(block) > self.config.max_target_context_chars:
                break
            rendered.append(block)
            used += separator + len(block)
        return "\n".join(rendered)


__all__ = ["ConversationSummarySemanticSearch", "MemoryDirectoryLayerReader", "MemorySemanticSearch", "SearchService"]
