"""Memory 主召回、关系补充、Intention 过滤与 Summary 条件后备测试。"""

import asyncio
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from infrastructure.store.contracts import PathLock
from infrastructure.store.locks import ProcessLocalLockStore
from memory.compaction import MemoryContextUseResult
from memory.conversation import (
    ConversationAddress,
    ConversationMessageJournal,
    ConversationRangeSummaryGenerator,
    ConversationRangeSummaryStore,
    ConversationSummaryCompactor,
    ConversationSummaryStore,
)
from memory.conversation.indexing import ConversationSummaryMatch
from memory.conversation.indexing.model import summary_reference
from memory.intention import MemoryIntentionRecallScope
from memory.model import MemoryDirectory, MemoryKind, MemoryLevel
from memory.retrieval import (
    ConversationSearchContextReader,
    MemoryContextAssembler,
    MemoryQueryResult,
    MemoryRecallLifecycle,
    MemoryRecallLifecycleConfig,
    MemoryRecallLifecycleError,
    MemoryRecallTarget,
    MemoryRetrievalGrader,
    MemorySearchHit,
    MemorySearchMode,
    MemorySearchQueryPlanner,
    MemorySearchServiceConfig,
    MemorySemanticSearchConfig,
    MemorySemanticSearchEngine,
    MemoryTypedQuery,
    MemoryVectorMatch,
    SearchService,
    SQLiteMemoryRecallLifecycleStore,
)
from memory.snapshot import MemorySnapshotReader
from memory.tree import MemoryTree
from memory.uri import MemoryURI
from ModelClient import (
    ChatClient,
    ChatModelConfig,
    EmbeddingVector,
    ModelResponse,
    ProviderCapabilities,
    ProviderConfig,
    StructuredChatClient,
)
from tests.helpers import document, segment, segment_summary
from tests.model_helpers import prepare_chat_request


@dataclass
class QueueChatProvider:
    responses: list[dict[str, object]]
    provider_name: str = "test"
    model: str = "test"
    is_remote: bool = False
    capabilities: ProviderCapabilities = ProviderCapabilities()

    prepare = staticmethod(prepare_chat_request)

    def complete(self, request):
        return ModelResponse(json.dumps(self.responses.pop(0), ensure_ascii=False), self.model, self.provider_name)

    async def complete_async(self, request):
        return self.complete(request)

    def stream(self, request):
        return iter(())

    async def stream_async(self, request):
        if False:
            yield None

    def health_check(self):
        return {"ok": True}


class SemanticSearch:
    def __init__(self, hits=()):
        self.hits = hits
        self.calls = []

    async def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        return self.hits


class StaticAdmissionEmbedder:
    async def embed_query(self, query: str) -> EmbeddingVector:
        return EmbeddingVector((1.0, 0.0))


class StaticAdmissionIndex:
    def __init__(self, matches: tuple[MemoryVectorMatch, ...]) -> None:
        self.matches = matches

    async def search(self, query_vector, **kwargs):
        return self.matches

    async def search_children(self, query_vector, **kwargs):
        return ()


class StaticAdmissionReranker:
    def __init__(self, scores: tuple[float, ...]) -> None:
        self.scores = scores

    async def rerank(self, query: str, documents: Sequence[str]) -> tuple[float, ...]:
        return self.scores


class SummarySearch:
    def __init__(self, matches=()):
        self.matches = matches
        self.calls = []

    async def search(self, query, *, limit):
        self.calls.append((query, limit))
        return self.matches


def structured(responses: list[dict[str, object]]) -> StructuredChatClient:
    route = ProviderConfig(
        provider="test",
        adapter="test",
        model="test",
        base_url="https://example.com",
        max_retries=0,
    )
    return StructuredChatClient(ChatClient(ChatModelConfig(route), QueueChatProvider(responses)))


def conversation_context_reader(
    tmp_path: Path,
    config: MemorySearchServiceConfig,
) -> ConversationSearchContextReader:
    journal = ConversationMessageJournal(
        tmp_path / "conversation",
        PathLock(ProcessLocalLockStore()),
    )
    segment_store = ConversationSummaryStore(journal.layout)
    range_store = ConversationRangeSummaryStore(journal.layout)
    compactor = ConversationSummaryCompactor(
        journal,
        segment_store,
        range_store,
        ConversationRangeSummaryGenerator(structured([])),
    )
    return ConversationSearchContextReader(journal, compactor, config=config)


def service(
    tmp_path: Path,
    *,
    semantic: SemanticSearch,
    summaries: SummarySearch,
    responses: list[dict[str, object]] | None = None,
    config: MemorySearchServiceConfig | None = None,
    lifecycle_config: MemoryRecallLifecycleConfig | None = None,
    recall_lifecycle: MemoryRecallLifecycle | None = None,
    clock: Callable[[], datetime] | None = None,
) -> SearchService:
    config = config or MemorySearchServiceConfig()
    lifecycle_config = lifecycle_config or MemoryRecallLifecycleConfig()
    tree = MemoryTree(tmp_path / "memory")
    reader = MemorySnapshotReader(tree)
    client = structured([] if responses is None else responses)
    planner = MemorySearchQueryPlanner(client, config=config)
    grader = MemoryRetrievalGrader(client, config=config)
    context_reader = conversation_context_reader(tmp_path, config)
    lifecycle = recall_lifecycle or MemoryRecallLifecycle(
        SQLiteMemoryRecallLifecycleStore(
            tmp_path / "workflow" / "memory_recall_lifecycle.sqlite3",
            config=lifecycle_config,
        ),
        config=lifecycle_config,
    )
    return SearchService(
        tree=tree,
        snapshot_reader=reader,
        semantic_search=semantic,
        summary_search=summaries,
        query_planner=planner,
        retrieval_grader=grader,
        recall_lifecycle=lifecycle,
        conversation_context=context_reader,
        assembler=MemoryContextAssembler(config=config),
        config=config,
        clock=clock,
    )


def summary_match() -> ConversationSummaryMatch:
    source = segment(segment_id="000000000000-000000000001")
    summary = segment_summary(source)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    return ConversationSummaryMatch(
        summary_reference(address, summary),
        summary,
        "历史过程：用户先提出简洁回答，助手随后确认。",
        0.8,
        0.8,
    )


def test_find_never_uses_summary_fallback_even_when_memory_has_no_hits(tmp_path: Path) -> None:
    semantic = SemanticSearch()
    summaries = SummarySearch((summary_match(),))
    result = asyncio.run(service(tmp_path, semantic=semantic, summaries=summaries).find("历史偏好"))
    assert result.memories == ()
    assert result.retrieval_assessment is None
    assert not result.summary_fallback_attempted
    assert summaries.calls == []


def test_search_uses_summary_only_after_memory_is_assessed_insufficient(tmp_path: Path) -> None:
    semantic = SemanticSearch()
    summaries = SummarySearch((summary_match(),))
    result = asyncio.run(service(tmp_path, semantic=semantic, summaries=summaries).search("之前怎么决定的"))
    assert result.memories == ()
    assert result.retrieval_assessment.decision.value == "insufficient"
    assert result.summary_fallback_attempted
    assert result.summary_fallbacks == (summary_match(),)
    assert "Conversation Summary" in result.context
    assert len(summaries.calls) == 1


def test_sufficient_memory_is_primary_and_prevents_summary_search(tmp_path: Path) -> None:
    preference = document(MemoryKind.PREFERENCE)
    uri = MemoryURI.from_address(preference.address)
    semantic = SemanticSearch((MemorySearchHit(uri, 0.9),))
    summaries = SummarySearch((summary_match(),))
    instance = service(
        tmp_path,
        semantic=semantic,
        summaries=summaries,
        responses=[
            {
                "decision": "sufficient",
                "reason": "长期偏好足够回答。",
                "missing_information": [],
                "summary_query": None,
            }
        ],
    )
    instance.tree.write(preference)
    result = asyncio.run(instance.search("用户喜欢什么回答风格"))
    assert tuple(item.document for item in result.memories) == (preference,)
    assert result.summary_fallbacks == ()
    assert summaries.calls == []
    assert result.context.index("长期记忆") < result.context.index("偏好")


def test_find_records_only_the_final_model_visible_memory_as_success(tmp_path: Path) -> None:
    now = datetime(2026, 7, 1, 8, 0, tzinfo=timezone.utc)
    preference = document(MemoryKind.PREFERENCE, timestamp=now)
    uri = MemoryURI.from_address(preference.address)
    instance = service(
        tmp_path,
        semantic=SemanticSearch((MemorySearchHit(uri, 0.9),)),
        summaries=SummarySearch(),
        clock=lambda: now,
    )
    instance.tree.write(preference)

    result = asyncio.run(instance.find("回答偏好"))
    assert result.memories[0].hit.lifecycle_semantic_score == 0.9
    assert result.memories[0].hit.lifecycle_hotness == pytest.approx(0.5)
    assert result.memories[0].hit.lifecycle_temperature.value == "warm"
    assert "temperature" not in result.context
    assert instance.recall_lifecycle.store.read_many((uri,))[0].useful_recall_count == 1


def test_sufficient_grader_result_records_only_retained_direct_memory_and_keeps_l2(tmp_path: Path) -> None:
    now = datetime(2026, 7, 1, 8, 0, tzinfo=timezone.utc)
    first = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "回答风格-a", "content": "- 偏好简洁回答"},
        timestamp=now,
    )
    second = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "回答风格-b", "content": "- 偏好分点回答"},
        timestamp=now,
    )
    first_uri = MemoryURI.from_address(first.address)
    second_uri = MemoryURI.from_address(second.address)
    instance = service(
        tmp_path,
        semantic=SemanticSearch((MemorySearchHit(first_uri, 0.9), MemorySearchHit(second_uri, 0.8))),
        summaries=SummarySearch(),
        responses=[
            {
                "decision": "sufficient",
                "reason": "第一条长期偏好足够支持回答。",
                "missing_information": [],
                "summary_query": None,
            }
        ],
        clock=lambda: now,
    )
    instance.tree.write(first)
    instance.tree.write(second)

    result = asyncio.run(instance.search("回答偏好", limit=1))
    assert tuple(memory.uri for memory in result.memories) == (first_uri,)
    assert result.memories[0].hit.lifecycle_temperature.value == "warm"
    states = instance.recall_lifecycle.store.read_many((first_uri, second_uri))
    assert tuple((state.uri, state.useful_recall_count) for state in states) == (
        (first_uri, 1),
    )
    assert instance.tree.read(first.address) == first

    recalled = asyncio.run(instance.find("回答偏好", limit=1))
    assert recalled.memories[0].hit.lifecycle_temperature.value == "hot"
    assert instance.recall_lifecycle.store.read_many((first_uri,))[0].useful_recall_count == 2


def test_final_context_exposure_heats_each_retained_direct_memory(tmp_path: Path) -> None:
    now = datetime(2026, 7, 1, 8, 0, tzinfo=timezone.utc)
    first = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "回答长度", "content": "- 偏好简洁回答"},
        timestamp=now,
    )
    second = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "回答结构", "content": "- 偏好分点回答"},
        timestamp=now,
    )
    first_uri = MemoryURI.from_address(first.address)
    second_uri = MemoryURI.from_address(second.address)
    instance = service(
        tmp_path,
        semantic=SemanticSearch(
            (MemorySearchHit(first_uri, 0.9), MemorySearchHit(second_uri, 0.8))
        ),
        summaries=SummarySearch(),
        responses=[
            {
                "decision": "sufficient",
                "reason": "最终召回的长期偏好集合足够支持回答。",
                "missing_information": [],
                "summary_query": None,
            }
        ],
        clock=lambda: now,
    )
    instance.tree.write(first)
    instance.tree.write(second)

    result = asyncio.run(instance.search("回答应该多长并如何组织", limit=2))
    assert len(result.memories) == 2
    states = instance.recall_lifecycle.store.read_many((first_uri, second_uri))
    assert tuple((state.uri, state.useful_recall_count) for state in states) == (
        (first_uri, 1),
        (second_uri, 1),
    )


def test_hot_low_relevance_memory_cannot_bypass_rerank_admission_or_enter_agent_context(
    tmp_path: Path,
) -> None:
    now = datetime(2026, 7, 1, 8, 0, tzinfo=timezone.utc)
    relevant = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "回答长度", "content": "- 偏好简洁回答"},
        timestamp=now,
    )
    hitchhiker = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "界面主题", "content": "- 偏好深色主题"},
        timestamp=now,
    )
    relevant_uri = MemoryURI.from_address(relevant.address)
    hitchhiker_uri = MemoryURI.from_address(hitchhiker.address)
    semantic = MemorySemanticSearchEngine(
        embedder=StaticAdmissionEmbedder(),
        index=StaticAdmissionIndex(
            (
                MemoryVectorMatch(
                    hitchhiker_uri,
                    MemoryLevel.DETAIL,
                    MemoryDirectory.for_address(hitchhiker.address),
                    hitchhiker.markdown_body,
                    0.95,
                ),
                MemoryVectorMatch(
                    relevant_uri,
                    MemoryLevel.DETAIL,
                    MemoryDirectory.for_address(relevant.address),
                    relevant.markdown_body,
                    0.8,
                ),
            )
        ),
        reranker=StaticAdmissionReranker((0.19, 0.91)),
        config=MemorySemanticSearchConfig(mode=MemorySearchMode.VECTOR),
    )
    instance = service(
        tmp_path,
        semantic=semantic,  # type: ignore[arg-type]
        summaries=SummarySearch(),
        responses=[
            {
                "decision": "sufficient",
                "reason": "回答长度偏好足够支持回答。",
                "missing_information": [],
                "summary_query": None,
            }
        ],
        clock=lambda: now,
    )
    instance.tree.write(relevant)
    instance.tree.write(hitchhiker)
    hitchhiker_target = MemoryRecallTarget(
        hitchhiker_uri,
        hitchhiker.metadata.revision,
        hitchhiker.metadata.created_at,
    )
    for index in range(3):
        instance.recall_lifecycle.record_use(
            (hitchhiker_target,),
            used_at=now + timedelta(seconds=index),
        )

    result = asyncio.run(instance.search("回答应该多长", limit=2))

    assert tuple(memory.uri for memory in result.memories) == (relevant_uri,)
    assert "偏好深色主题" not in result.context
    states = instance.recall_lifecycle.store.read_many((relevant_uri, hitchhiker_uri))
    assert tuple((state.uri, state.useful_recall_count) for state in states) == (
        (relevant_uri, 1),
        (hitchhiker_uri, 3),
    )


def test_insufficient_memory_with_summary_still_records_final_direct_context_use(tmp_path: Path) -> None:
    now = datetime(2026, 7, 1, 8, 0, tzinfo=timezone.utc)
    preference = document(MemoryKind.PREFERENCE, timestamp=now)
    uri = MemoryURI.from_address(preference.address)
    instance = service(
        tmp_path,
        semantic=SemanticSearch((MemorySearchHit(uri, 0.9),)),
        summaries=SummarySearch((summary_match(),)),
        responses=[
            {
                "decision": "insufficient",
                "reason": "长期记忆缺少决定过程。",
                "missing_information": ["决定过程"],
                "summary_query": "决定过程",
            }
        ],
        clock=lambda: now,
    )
    instance.tree.write(preference)

    result = asyncio.run(instance.search("之前怎么决定的"))
    assert result.summary_fallbacks == (summary_match(),)
    assert instance.recall_lifecycle.store.read_many((uri,))[0].useful_recall_count == 1


def test_rejected_final_context_is_regraded_and_can_trigger_summary_fallback(tmp_path: Path) -> None:
    now = datetime(2026, 7, 1, 8, 0, tzinfo=timezone.utc)
    preference = document(MemoryKind.PREFERENCE, timestamp=now)
    uri = MemoryURI.from_address(preference.address)
    summaries = SummarySearch((summary_match(),))
    instance = service(
        tmp_path,
        semantic=SemanticSearch((MemorySearchHit(uri, 0.9),)),
        summaries=summaries,
        responses=[
            {
                "decision": "sufficient",
                "reason": "候选记忆看起来足够。",
                "missing_information": [],
                "summary_query": None,
            },
            {
                "decision": "insufficient",
                "reason": "最终记忆因 revision fence 被拒绝。",
                "missing_information": ["决定过程"],
                "summary_query": "决定过程",
            },
        ],
        clock=lambda: now,
    )
    instance.tree.write(preference)

    class RejectingContextUse:
        @staticmethod
        def expand_for_probe(value):
            return value

        @staticmethod
        async def record_context_use(targets, *, used_at):
            return MemoryContextUseResult((), tuple(target.uri for target in targets))

    instance.cold_probe_expander = RejectingContextUse()
    result = asyncio.run(instance.search("之前怎么决定的"))

    assert result.memories == ()
    assert result.retrieval_assessment is not None
    assert result.retrieval_assessment.decision.value == "insufficient"
    assert result.summary_fallback_attempted
    assert result.summary_fallbacks == (summary_match(),)
    assert summaries.calls == [("之前怎么决定的", instance.config.summary_fallback_limit)]
    assert tuple(item.stage.value for item in result.degradations) == ("recall_lifecycle",)


class FailingRecallStateStore:
    def __init__(self, *, fail_read: bool = False, fail_write: bool = False) -> None:
        self.fail_read = fail_read
        self.fail_write = fail_write

    def initialize(self) -> None:
        return None

    def read_many(self, uris):
        if self.fail_read:
            raise MemoryRecallLifecycleError("read failed")
        return ()

    def record_use(self, targets, *, used_at):
        if self.fail_write:
            raise MemoryRecallLifecycleError("write failed")
        return ()

    def record_probe(self, targets, *, probed_at):
        if self.fail_write:
            raise MemoryRecallLifecycleError("write failed")
        return ()

    def mark_compacted(
        self,
        target,
        *,
        lifecycle_activity_at,
        compacted_at,
        expected_version,
    ):
        if self.fail_write:
            raise MemoryRecallLifecycleError("write failed")
        raise AssertionError("not used by SearchService")

    def mark_retire_candidate(self, target, *, marked_at, expected_version):
        if self.fail_write:
            raise MemoryRecallLifecycleError("write failed")
        raise AssertionError("not used by SearchService")

    def mark_retired(self, target, *, retired_at, expected_version):
        if self.fail_write:
            raise MemoryRecallLifecycleError("write failed")
        raise AssertionError("not used by SearchService")

    def delete_many(self, uris):
        return 0


def test_lifecycle_read_failure_keeps_semantic_order_and_marks_degradation(tmp_path: Path) -> None:
    now = datetime(2026, 7, 1, 8, 0, tzinfo=timezone.utc)
    preference = document(MemoryKind.PREFERENCE, timestamp=now)
    uri = MemoryURI.from_address(preference.address)
    lifecycle = MemoryRecallLifecycle(FailingRecallStateStore(fail_read=True))
    instance = service(
        tmp_path,
        semantic=SemanticSearch((MemorySearchHit(uri, 0.9),)),
        summaries=SummarySearch(),
        recall_lifecycle=lifecycle,
        clock=lambda: now,
    )
    instance.tree.write(preference)

    result = asyncio.run(instance.find("回答偏好"))
    assert result.memories[0].hit.score == 0.9
    assert result.memories[0].hit.lifecycle_temperature is None
    assert tuple(item.stage.value for item in result.degradations) == ("recall_lifecycle",)


def test_actual_use_write_failure_keeps_detailed_memory_and_marks_degradation(tmp_path: Path) -> None:
    now = datetime(2026, 7, 1, 8, 0, tzinfo=timezone.utc)
    preference = document(MemoryKind.PREFERENCE, timestamp=now)
    uri = MemoryURI.from_address(preference.address)
    lifecycle = MemoryRecallLifecycle(FailingRecallStateStore(fail_write=True))
    instance = service(
        tmp_path,
        semantic=SemanticSearch((MemorySearchHit(uri, 0.9),)),
        summaries=SummarySearch(),
        responses=[
            {
                "decision": "sufficient",
                "reason": "长期偏好足够支持回答。",
                "missing_information": [],
                "summary_query": None,
            }
        ],
        recall_lifecycle=lifecycle,
        clock=lambda: now,
    )
    instance.tree.write(preference)

    result = asyncio.run(instance.search("回答偏好"))
    assert result.retrieval_assessment.decision.value == "sufficient"
    assert result.memories[0].hit.lifecycle_temperature.value == "warm"
    assert tuple(item.stage.value for item in result.degradations) == ("recall_lifecycle",)


def test_completed_intention_has_explicit_entry_and_cannot_mix_other_kinds() -> None:
    assert SearchService._recall_filters((), MemoryIntentionRecallScope.COMPLETED) == (
        (MemoryKind.INTENTION,),
        MemoryIntentionRecallScope.COMPLETED,
    )
    with pytest.raises(ValueError, match="cannot include"):
        SearchService._recall_filters(
            (MemoryKind.PROFILE, MemoryKind.INTENTION),
            MemoryIntentionRecallScope.COMPLETED,
        )
    with pytest.raises(ValueError, match="reserved"):
        SearchService._recall_filters((), MemoryIntentionRecallScope.ALL)


def test_aggregate_deduplicates_same_uri_and_records_each_matching_query_once() -> None:
    uri = MemoryURI.from_address(document(MemoryKind.PROFILE).address)
    first_query = MemoryTypedQuery("用户是谁", "身份", 1)
    second_query = MemoryTypedQuery("用户背景", "背景", 2)
    hit = MemorySearchHit(uri, 0.8)
    hits, queries = SearchService._aggregate(
        (
            MemoryQueryResult(first_query, (hit,)),
            MemoryQueryResult(second_query, (MemorySearchHit(uri, 0.9),)),
        )
    )
    assert hits[0].score == 0.9
    assert queries[uri] == ("用户是谁", "用户背景")
