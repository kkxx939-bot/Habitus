"""面向 Agent 的完整只读记忆 SearchService。"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Protocol

from foundation.observability import NullObserver, ObservationEvent, ObservationStatus, Observer
from memory.conversation import ConversationAddress
from memory.conversation.indexing import ConversationSummaryMatch
from memory.document import MemoryDocument, MemoryStoredLink
from memory.intention import (
    MemoryIntentionRecallScope,
    MemoryIntentionReview,
    MemoryIntentionReviewer,
    intention_matches_scope,
)
from memory.model import MemoryKind, MemoryLevel
from memory.retrieval.assembler import MemoryContextAssembler
from memory.retrieval.context import ConversationSearchContextReader
from memory.retrieval.grader import MemoryRetrievalGrader
from memory.retrieval.model import (
    MemoryMatchedMemory,
    MemoryQueryPlan,
    MemoryQueryResult,
    MemoryRelatedMemory,
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
from memory.retrieval.planner import MemorySearchQueryPlanner
from memory.snapshot import MemorySnapshotReader
from memory.tree import MemoryTree
from memory.uri import MemoryURI, MemoryURINodeType


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


class SearchService:
    """参考 OpenViking 的 find/search 分层，返回完整且可溯源的 L2 记忆上下文。"""

    def __init__(
        self,
        *,
        tree: MemoryTree,
        snapshot_reader: MemorySnapshotReader,
        semantic_search: MemorySemanticSearch,
        summary_search: ConversationSummarySemanticSearch,
        query_planner: MemorySearchQueryPlanner,
        retrieval_grader: MemoryRetrievalGrader,
        conversation_context: ConversationSearchContextReader,
        assembler: MemoryContextAssembler | None = None,
        config: MemorySearchServiceConfig | None = None,
        intention_reviewer: MemoryIntentionReviewer | None = None,
        clock: Callable[[], datetime] | None = None,
        observer: Observer | None = None,
    ) -> None:
        if not isinstance(tree, MemoryTree):
            raise TypeError("tree must be MemoryTree")
        if not isinstance(snapshot_reader, MemorySnapshotReader):
            raise TypeError("snapshot_reader must be MemorySnapshotReader")
        if snapshot_reader.tree is not tree:
            raise ValueError("SearchService and snapshot reader must share one memory tree")
        if not callable(getattr(semantic_search, "search", None)):
            raise TypeError("semantic_search must implement async search")
        if not callable(getattr(summary_search, "search", None)):
            raise TypeError("summary_search must implement async search")
        if not isinstance(query_planner, MemorySearchQueryPlanner):
            raise TypeError("query_planner must be MemorySearchQueryPlanner")
        if not isinstance(retrieval_grader, MemoryRetrievalGrader):
            raise TypeError("retrieval_grader must be MemoryRetrievalGrader")
        if not isinstance(conversation_context, ConversationSearchContextReader):
            raise TypeError("conversation_context must be ConversationSearchContextReader")
        if config is not None and not isinstance(config, MemorySearchServiceConfig):
            raise TypeError("config must be MemorySearchServiceConfig")
        if intention_reviewer is not None and not isinstance(
            intention_reviewer,
            MemoryIntentionReviewer,
        ):
            raise TypeError("intention_reviewer must be MemoryIntentionReviewer")
        if clock is not None and not callable(clock):
            raise TypeError("clock must be callable")
        resolved_config = config or MemorySearchServiceConfig()
        resolved_assembler = assembler or MemoryContextAssembler(config=resolved_config)
        if not isinstance(resolved_assembler, MemoryContextAssembler):
            raise TypeError("assembler must be MemoryContextAssembler")
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
        self.conversation_context = conversation_context
        self.assembler = resolved_assembler
        self.config = resolved_config
        self.intention_reviewer = intention_reviewer or MemoryIntentionReviewer()
        self.clock = clock or (lambda: datetime.now(timezone.utc))
        self.observer = observer or NullObserver()

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

        normalized = self._query(query)
        roots = self._roots(target_uris)
        maximum = self._limit(limit)
        threshold = self._threshold(score_threshold)
        allowed_kinds, allowed_intention_scope = self._recall_filters(kinds, intention_scope)
        plan = self.query_planner.direct(normalized)
        return await self._execute(
            plan,
            roots,
            maximum,
            threshold,
            allowed_kinds,
            allowed_intention_scope,
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

        normalized = self._query(query)
        roots = self._roots(target_uris)
        maximum = self._limit(limit)
        threshold = self._threshold(score_threshold)
        allowed_kinds, allowed_intention_scope = self._recall_filters(kinds, intention_scope)
        degradations: tuple[MemorySearchDegradation, ...] = ()
        if conversation is None:
            plan = self.query_planner.direct(normalized)
        else:
            if not isinstance(conversation, ConversationAddress):
                raise TypeError("conversation must be ConversationAddress or None")
            try:
                context = self.conversation_context.read(conversation)
                plan = await self.query_planner.plan(
                    normalized,
                    context,
                    target_context=self._target_context(roots),
                )
            except Exception as exc:
                plan = self.query_planner.direct(normalized)
                degradations = (
                    MemorySearchDegradation(
                        MemorySearchDegradationStage.QUERY_PLANNER,
                        type(exc).__name__,
                    ),
                )
        return await self._execute(
            plan,
            roots,
            maximum,
            threshold,
            allowed_kinds,
            allowed_intention_scope,
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
        if not isinstance(assess_for_summary, bool):
            raise TypeError("assess_for_summary must be boolean")
        started = time.monotonic()
        applied_degradations = list(degradations)
        candidate_limit = min(
            1_000,
            max(limit, limit * self.config.candidate_multiplier),
        )
        query_results = tuple(
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
        hits, matched_queries = self._aggregate(query_results)
        review_now = self._timestamp()
        read_limit = min(
            self.snapshot_reader.config.max_items,
            max(limit, limit * self.config.candidate_multiplier),
        )
        direct = self._read_direct(
            hits[:read_limit],
            matched_queries,
            limit,
            review_now=review_now,
            kinds=kinds,
            intention_scope=intention_scope,
        )
        expanded = self._expand_relations(
            direct,
            roots=roots,
            intention_scope=intention_scope,
            review_now=review_now,
        )
        assembly = self.assembler.assemble(expanded)
        assessment: MemoryRetrievalAssessment | None = None
        summary_fallback_attempted = False
        if assess_for_summary:
            try:
                assessment = await self.retrieval_grader.assess(
                    plan,
                    assembly.memories,
                    assembly.context,
                )
            except Exception as exc:
                applied_degradations.append(
                    MemorySearchDegradation(
                        MemorySearchDegradationStage.RETRIEVAL_GRADER,
                        type(exc).__name__,
                    )
                )
            if (
                self.config.summary_fallback_enabled
                and assessment is not None
                and assessment.decision is MemoryRetrievalSufficiency.INSUFFICIENT
                and not assembly.budget_exhausted
            ):
                assert assessment.summary_query is not None
                summary_fallback_attempted = True
                try:
                    summaries = await self._search_summaries(assessment.summary_query)
                except Exception as exc:
                    applied_degradations.append(
                        MemorySearchDegradation(
                            MemorySearchDegradationStage.SUMMARY_FALLBACK,
                            type(exc).__name__,
                        )
                    )
                else:
                    assembly = self.assembler.assemble(
                        expanded,
                        summary_fallbacks=summaries,
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
        self.observer.record(
            ObservationEvent(
                category="retrieval",
                operation="search",
                status=(ObservationStatus.DEGRADED if applied_degradations else ObservationStatus.SUCCESS),
                duration_seconds=max(0.0, time.monotonic() - started),
                attributes={
                    "queries": len(plan.queries),
                    "memories": len(result.memories),
                    "summaries": len(result.summary_fallbacks),
                    "degradations": len(result.degradations),
                },
            )
        )
        return result

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
        if isinstance(raw, str) or not isinstance(raw, Sequence):
            raise MemorySearchError("Summary fallback search must return a sequence")
        if len(raw) > self.config.summary_fallback_limit:
            raise MemorySearchError("Summary fallback search exceeded its requested limit")
        values = tuple(raw)
        if any(not isinstance(item, ConversationSummaryMatch) for item in values):
            raise MemorySearchError("Summary fallback search returned an invalid match")
        expected = tuple(
            sorted(values, key=lambda item: (-item.score, item.reference.identity))
        )
        if values != expected or len({item.reference.identity for item in values}) != len(values):
            raise MemorySearchError("Summary fallback matches must be unique and relevance-sorted")
        return values

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
        if isinstance(raw_hits, str) or not isinstance(raw_hits, Sequence):
            raise MemorySearchError("memory semantic search must return a sequence of hits")
        if len(raw_hits) > limit:
            raise MemorySearchError("memory semantic search exceeded its requested result limit")
        by_uri: dict[MemoryURI, MemorySearchHit] = {}
        for hit in raw_hits:
            if not isinstance(hit, MemorySearchHit):
                raise MemorySearchError("memory semantic search returned an invalid hit")
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
        limit: int,
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
            if snapshot is None or not snapshot.exists or not isinstance(snapshot.value, MemoryDocument):
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
            if len(selected) == limit:
                break
        return tuple(selected)

    def _expand_relations(
        self,
        memories: tuple[MemoryMatchedMemory, ...],
        *,
        roots: tuple[MemoryURI, ...],
        intention_scope: MemoryIntentionRecallScope,
        review_now: datetime,
    ) -> tuple[MemoryMatchedMemory, ...]:
        if not memories or self.config.max_relation_neighbors_total == 0:
            return memories
        direct_uris = {memory.uri for memory in memories}
        direct_documents = {memory.uri: memory.document for memory in memories}
        selected_relations: dict[MemoryURI, list[MemoryStoredLink]] = {memory.uri: [] for memory in memories}
        neighbor_uris: list[MemoryURI] = []
        assigned_neighbors: set[MemoryURI] = set()
        assigned_relations: set[tuple[str, str, str]] = set()
        for memory in memories:
            relations = tuple(
                sorted((*memory.document.links, *memory.document.backlinks), key=lambda link: link.identity)
            )
            for relation in relations:
                neighbor = relation.to_uri if relation.from_uri == memory.uri else relation.from_uri
                if relation.identity in assigned_relations:
                    continue
                if not any(neighbor.matches_prefix(root) for root in roots):
                    continue
                if len(selected_relations[memory.uri]) >= self.config.max_relation_neighbors_per_match:
                    break
                if len(assigned_relations) >= self.config.max_relation_neighbors_total:
                    break
                selected_relations[memory.uri].append(relation)
                assigned_relations.add(relation.identity)
                if neighbor not in direct_uris and neighbor not in assigned_neighbors:
                    neighbor_uris.append(neighbor)
                    assigned_neighbors.add(neighbor)
            if len(assigned_relations) >= self.config.max_relation_neighbors_total:
                break
        try:
            neighbors = self.snapshot_reader.read_many(tuple(neighbor_uris))
        except Exception as exc:
            raise MemorySearchError("failed to read one-hop related memory documents") from exc

        expanded: list[MemoryMatchedMemory] = []
        for memory in memories:
            related: list[MemoryRelatedMemory] = []
            for relation in selected_relations[memory.uri]:
                neighbor_uri = relation.to_uri if relation.from_uri == memory.uri else relation.from_uri
                document = direct_documents.get(neighbor_uri)
                if document is None:
                    snapshot = neighbors.get(str(neighbor_uri))
                    if snapshot is None or not snapshot.exists or not isinstance(snapshot.value, MemoryDocument):
                        raise MemorySearchError("memory relation points to a missing L2 document")
                    document = snapshot.value
                self._require_reciprocal(memory.document, document, relation)
                if (
                    intention_scope is MemoryIntentionRecallScope.ACTIVE
                    and document.kind is MemoryKind.INTENTION
                    and document.fields.get("status") == "completed"
                ):
                    continue
                related.append(
                    MemoryRelatedMemory(
                        relation=relation,
                        document=document,
                        intention_review=self._intention_review(document, now=review_now),
                    )
                )
            expanded.append(
                MemoryMatchedMemory(
                    hit=memory.hit,
                    document=memory.document,
                    matched_queries=memory.matched_queries,
                    intention_review=memory.intention_review,
                    related=tuple(related),
                )
            )
        return tuple(expanded)

    def _intention_review(
        self,
        document: MemoryDocument,
        *,
        now: datetime,
    ) -> MemoryIntentionReview | None:
        if document.kind is not MemoryKind.INTENTION:
            return None
        return self.intention_reviewer.review(document, now=now)

    def _timestamp(self) -> datetime:
        timestamp = self.clock()
        if not isinstance(timestamp, datetime):
            raise TypeError("memory search clock must return datetime")
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise ValueError("memory search clock must return a timezone-aware datetime")
        return timestamp.astimezone(timezone.utc)

    @staticmethod
    def _require_reciprocal(
        seed: MemoryDocument,
        neighbor: MemoryDocument,
        relation: MemoryStoredLink,
    ) -> None:
        seed_uri = MemoryURI.from_address(seed.address)
        neighbor_uri = MemoryURI.from_address(neighbor.address)
        if {seed_uri, neighbor_uri} != {relation.from_uri, relation.to_uri}:
            raise MemorySearchError("memory relation endpoint does not match the loaded documents")
        if relation.from_uri == seed_uri:
            valid = relation in seed.links and relation in neighbor.backlinks
        else:
            valid = relation in seed.backlinks and relation in neighbor.links
        if not valid:
            raise MemorySearchError("memory relation Links/Backlinks are inconsistent")

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

    def _query(self, value: object) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("memory search query must be non-empty text")
        normalized = value.strip()
        if len(normalized) > self.config.max_query_chars:
            raise ValueError("memory search query exceeds its configured character bound")
        return normalized

    def _roots(
        self,
        value: MemoryURI | str | tuple[MemoryURI | str, ...] | None,
    ) -> tuple[MemoryURI, ...]:
        if value is None:
            raw: tuple[MemoryURI | str, ...] = (MemoryURI.root(),)
        elif isinstance(value, MemoryURI | str):
            raw = (value,)
        elif isinstance(value, tuple):
            raw = value
        else:
            raise TypeError("target_uris must be a URI, string, tuple, or None")
        if not raw or len(raw) > self.config.max_target_roots:
            raise ValueError("memory search target root count is outside its configured bound")
        parsed = tuple(
            sorted({MemoryURI.parse(root) for root in raw}, key=lambda root: (len(root.segments), str(root)))
        )
        if any(root.node_type is not MemoryURINodeType.DIRECTORY for root in parsed):
            raise ValueError("memory search target URIs must identify directories")
        roots: list[MemoryURI] = []
        for root in parsed:
            if any(root.matches_prefix(parent) for parent in roots):
                continue
            roots.append(root)
        return tuple(roots)

    def _limit(self, value: int | None) -> int:
        limit = self.config.default_limit if value is None else value
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= self.config.max_limit:
            raise ValueError("memory search limit is outside its configured bound")
        return limit

    @staticmethod
    def _threshold(value: float | None) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise ValueError("memory search score_threshold must be finite numeric data")
        return float(value)

    @staticmethod
    def _kinds(value: tuple[MemoryKind, ...]) -> tuple[MemoryKind, ...]:
        if not isinstance(value, tuple):
            raise TypeError("memory search kinds must be a tuple")
        normalized = tuple(MemoryKind(kind) for kind in value)
        if len(normalized) != len(set(normalized)):
            raise ValueError("memory search kinds must be unique")
        return normalized

    @classmethod
    def _recall_filters(
        cls,
        kinds: tuple[MemoryKind, ...],
        intention_scope: MemoryIntentionRecallScope,
    ) -> tuple[tuple[MemoryKind, ...], MemoryIntentionRecallScope]:
        """规范直接候选过滤，并给历史 completed Intention 提供显式入口。"""

        normalized_kinds = cls._kinds(kinds)
        scope = MemoryIntentionRecallScope(intention_scope)
        if scope is MemoryIntentionRecallScope.ALL:
            raise ValueError("ALL Intention recall is reserved for Memory Editor old-memory reads")
        if scope is not MemoryIntentionRecallScope.COMPLETED:
            return normalized_kinds, scope
        if normalized_kinds and normalized_kinds != (MemoryKind.INTENTION,):
            raise ValueError("completed Intention search cannot include other memory kinds")
        return (MemoryKind.INTENTION,), scope


__all__ = ["ConversationSummarySemanticSearch", "MemorySemanticSearch", "SearchService"]
