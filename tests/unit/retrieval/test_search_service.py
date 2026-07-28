"""Memory 主召回、关系补充、Intention 过滤与 Summary 条件后备测试。"""

import asyncio
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pytest

from memory.conversation import ConversationAddress
from memory.conversation.indexing import ConversationSummaryMatch
from memory.conversation.indexing.model import summary_reference
from memory.intention import MemoryIntentionRecallScope
from memory.model import MemoryKind
from memory.retrieval import (
    ConversationSearchContextReader,
    MemoryContextAssembler,
    MemoryQueryResult,
    MemoryRetrievalGrader,
    MemorySearchHit,
    MemorySearchQueryPlanner,
    MemorySearchServiceConfig,
    MemoryTypedQuery,
    SearchService,
)
from memory.snapshot import MemorySnapshotReader
from memory.tree import MemoryTree
from memory.uri import MemoryURI
from ModelClient import (
    ChatClient,
    ChatModelConfig,
    ModelResponse,
    ProviderCapabilities,
    ProviderConfig,
    StructuredChatClient,
)
from tests.helpers import document, segment, segment_summary


@dataclass
class QueueChatProvider:
    responses: list[dict[str, object]]
    provider_name: str = "test"
    model: str = "test"
    is_remote: bool = False
    capabilities: ProviderCapabilities = ProviderCapabilities()

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


def service(
    tmp_path: Path,
    *,
    semantic: SemanticSearch,
    summaries: SummarySearch,
    responses: list[dict[str, object]] | None = None,
) -> SearchService:
    config = MemorySearchServiceConfig()
    tree = MemoryTree(tmp_path / "memory")
    reader = MemorySnapshotReader(tree)
    client = structured([] if responses is None else responses)
    planner = MemorySearchQueryPlanner(client, config=config)
    grader = MemoryRetrievalGrader(client, config=config)
    context_reader = object.__new__(ConversationSearchContextReader)
    context_reader.config = config
    return SearchService(
        tree=tree,
        snapshot_reader=reader,
        semantic_search=semantic,
        summary_search=summaries,
        query_planner=planner,
        retrieval_grader=grader,
        conversation_context=context_reader,
        assembler=MemoryContextAssembler(config=config),
        config=config,
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
