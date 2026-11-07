"""完整 Conversation 经受控检索和一次模型候选生成的解析流程测试。"""

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC
from pathlib import Path

import pytest

from habitus.memory.editor import (
    MemoryExtractionConfig,
    MemoryExtractionError,
    MemoryExtractionLoop,
    MemoryRelatedRetriever,
    MemoryRetrievalFinalDecision,
)
from habitus.memory.model import MemoryKind
from habitus.memory.retrieval import MemorySearchHit
from habitus.memory.schema import MemorySchemaRegistry
from habitus.memory.snapshot import MemorySnapshotReader
from habitus.memory.tree import MemoryTree
from habitus.memory.uri import MemoryURI
from habitus.model_client import (
    ChatClient,
    ChatModelConfig,
    ModelResponse,
    ProviderCapabilities,
    ProviderConfig,
    StructuredChatClient,
)
from tests.helpers import memory_fields, segment, tool_turn
from tests.model_helpers import prepare_chat_request


@dataclass
class QueueProvider:
    responses: list[dict[str, object]]
    requests: list[object] = field(default_factory=list)
    provider_name: str = "test"
    model: str = "test"
    is_remote: bool = False
    capabilities: ProviderCapabilities = ProviderCapabilities()

    prepare = staticmethod(prepare_chat_request)

    def complete(self, request):
        self.requests.append(request.request)
        return ModelResponse(
            json.dumps(self.responses.pop(0), ensure_ascii=False),
            self.model,
            self.provider_name,
        )

    async def complete_async(self, request):
        return self.complete(request)

    def stream(self, _request):
        return iter(())

    async def stream_async(self, _request):
        if False:
            yield None

    def health_check(self):
        return {"ok": True}


class EmptySemanticSearch:
    def __init__(self) -> None:
        self.calls = 0

    async def search(self, _query, **_kwargs):
        self.calls += 1
        return ()


class FailingAdditionalSearch(EmptySemanticSearch):
    """首次检索正常返回空，受控 ReAct 的追加搜索遇到向量服务不可用。"""

    async def search(self, query, **kwargs):
        if self.calls >= 1:
            self.calls += 1
            raise ConnectionError("vector service unavailable")
        return await super().search(query, **kwargs)


class OutOfScopeAdditionalSearch(EmptySemanticSearch):
    """追加搜索返回搜索范围之外的 URI。"""

    async def search(self, query, **kwargs):
        if self.calls >= 1:
            self.calls += 1
            return (MemorySearchHit(MemoryURI.parse("memory://profile.md"), 0.9),)
        return await super().search(query, **kwargs)


def finish() -> dict[str, object]:
    return {
        "status": "sufficient",
        "action": "finish",
        "query": None,
        "uri": None,
        "reason": "当前旧记忆已经足够。",
    }


def empty_candidates() -> dict[str, object]:
    return {
        "profile": [],
        "preferences": [],
        "entities": [],
        "tools": [],
        "events": [],
        "intentions": [],
        "identity_proposals": [],
        "relations": [],
    }


def candidate_item(kind: MemoryKind, page_id: int, **controls: object) -> dict[str, object]:
    """构造一个符合对应记忆 YAML 的结构化候选。"""

    item = {"page_id": page_id, **memory_fields(kind), **controls}
    if kind is MemoryKind.EVENT:
        item["event_date"] = "2026-07-01"
    return item


def all_kind_candidates() -> dict[str, object]:
    """构造覆盖六类记忆且 page_id 唯一的新节点批次。"""

    result = empty_candidates()
    result["profile"] = [candidate_item(MemoryKind.PROFILE, 100)]
    result["preferences"] = [candidate_item(MemoryKind.PREFERENCE, 101)]
    result["entities"] = [candidate_item(MemoryKind.ENTITY, 102)]
    result["tools"] = [candidate_item(MemoryKind.TOOL, 103)]
    result["events"] = [candidate_item(MemoryKind.EVENT, 104)]
    result["intentions"] = [candidate_item(MemoryKind.INTENTION, 105, confirmed=True)]
    return result


def search_more(query: str = "继续搜索偏好") -> dict[str, object]:
    return {
        "status": "insufficient",
        "action": "memory_search",
        "query": query,
        "uri": None,
        "reason": "旧记忆不足。",
    }


def ended_insufficient() -> dict[str, object]:
    """主链端到端实测 005 的终轮答复形状：树空、两次搜索无结果、如实说不足并结束。"""

    return {
        "status": "insufficient",
        "action": "finish",
        "query": None,
        "uri": None,
        "reason": "完整旧 L2 快照均缺失，语义检索两次均无结果。",
    }


def extraction_loop(
    tmp_path: Path,
    responses: list[dict[str, object]],
    *,
    config: MemoryExtractionConfig | None = None,
    semantic_search: EmptySemanticSearch | None = None,
    provider: QueueProvider | None = None,
) -> MemoryExtractionLoop:
    route = ProviderConfig(
        provider="test",
        adapter="test",
        model="test",
        base_url="https://example.com",
        max_retries=0,
    )
    client = StructuredChatClient(
        ChatClient(ChatModelConfig(route), provider or QueueProvider(responses)),
        validation_retries=0,
    )
    tree = MemoryTree(tmp_path / "memory")
    retriever = MemoryRelatedRetriever(
        schema_registry=MemorySchemaRegistry.load_default(),
        snapshot_reader=MemorySnapshotReader(tree),
        semantic_search=semantic_search or EmptySemanticSearch(),
    )
    return MemoryExtractionLoop(client=client, retriever=retriever, zone=UTC, config=config)


def test_loop_returns_model_candidates_without_second_semantic_review(tmp_path: Path) -> None:
    responses = [finish(), empty_candidates()]
    result = asyncio.run(extraction_loop(tmp_path, responses).extract(segment()))

    assert responses == []
    assert result.candidates.iter_candidates() == ()
    assert result.mutations.mutations == ()
    assert result.retrieval_decisions[0].action.value == "finish"
    assert result.source_segment_digest == segment().digest


def test_final_round_insufficient_on_empty_tree_ends_retrieval_and_still_extracts(tmp_path: Path) -> None:
    provider = QueueProvider([search_more(), search_more("再换一个查询"), ended_insufficient(), all_kind_candidates()])
    search = EmptySemanticSearch()
    loop = extraction_loop(
        tmp_path,
        [],
        config=MemoryExtractionConfig(max_retrieval_iterations=3),
        semantic_search=search,
        provider=provider,
    )

    result = asyncio.run(loop.extract(segment(messages=tool_turn())))

    assert provider.responses == []
    assert result.retrieval_ended_at_limit is True
    assert tuple(item.status.value for item in result.retrieval_decisions) == (
        "insufficient",
        "insufficient",
        "insufficient",
    )
    assert isinstance(result.retrieval_decisions[-1], MemoryRetrievalFinalDecision)
    assert len(result.retrieval_observations) == 2
    assert search.calls == 3
    assert all(mutation.action.value == "create" for mutation in result.mutations.mutations)
    assert len(result.mutations.mutations) == len(MemoryKind)

    final_request = provider.requests[2]
    final_schema = final_request.response_format.schema
    assert final_schema["properties"]["action"]["enum"] == ["finish"]
    final_system = final_request.messages[1].content
    assert "预算已经用完" in final_system
    assert "明确失败" not in final_system
    action_schema = provider.requests[0].response_format.schema
    assert action_schema["properties"]["action"]["enum"] == ["finish", "memory_search", "memory_read"]


def test_final_round_that_still_requests_a_tool_ends_at_limit_without_running_it(tmp_path: Path) -> None:
    search = EmptySemanticSearch()
    responses = [search_more(), empty_candidates()]
    loop = extraction_loop(
        tmp_path,
        responses,
        config=MemoryExtractionConfig(max_retrieval_iterations=1),
        semantic_search=search,
    )

    result = asyncio.run(loop.extract(segment()))

    assert responses == []
    assert result.retrieval_ended_at_limit is True
    assert result.retrieval_decisions == ()
    assert result.retrieval_observations == ()
    assert search.calls == 1
    assert result.candidates.iter_candidates() == ()


def test_final_round_sufficient_finish_is_not_recorded_as_limit(tmp_path: Path) -> None:
    result = asyncio.run(
        extraction_loop(
            tmp_path,
            [search_more(), finish(), empty_candidates()],
            config=MemoryExtractionConfig(max_retrieval_iterations=2),
        ).extract(segment())
    )

    assert result.retrieval_ended_at_limit is False
    assert isinstance(result.retrieval_decisions[-1], MemoryRetrievalFinalDecision)


def test_finish_before_the_final_round_is_not_recorded_as_limit(tmp_path: Path) -> None:
    result = asyncio.run(extraction_loop(tmp_path, [finish(), empty_candidates()]).extract(segment()))

    assert result.retrieval_ended_at_limit is False
    assert not isinstance(result.retrieval_decisions[0], MemoryRetrievalFinalDecision)


@pytest.mark.parametrize(
    ("semantic_search", "message"),
    [
        (FailingAdditionalSearch(), "additional memory semantic search failed"),
        (OutOfScopeAdditionalSearch(), "out-of-scope URI"),
    ],
)
def test_search_failure_still_fails_instead_of_ending_at_limit(
    tmp_path: Path,
    semantic_search: EmptySemanticSearch,
    message: str,
) -> None:
    responses = [search_more(), ended_insufficient(), empty_candidates()]
    loop = extraction_loop(
        tmp_path,
        responses,
        config=MemoryExtractionConfig(max_retrieval_iterations=2),
        semantic_search=semantic_search,
    )

    with pytest.raises(MemoryExtractionError, match=message):
        asyncio.run(loop.extract(segment()))
    assert responses == [ended_insufficient(), empty_candidates()]


def test_loop_executes_one_search_then_finishes_with_auditable_observation(tmp_path: Path) -> None:
    search = {
        "status": "insufficient",
        "action": "memory_search",
        "query": "回答风格",
        "uri": None,
        "reason": "需要确认相关旧偏好。",
    }
    result = asyncio.run(
        extraction_loop(
            tmp_path,
            [search, finish(), empty_candidates()],
            config=MemoryExtractionConfig(max_retrieval_iterations=2),
        ).extract(segment())
    )

    assert tuple(item.action.value for item in result.retrieval_decisions) == (
        "memory_search",
        "finish",
    )
    assert len(result.retrieval_observations) == 1
    assert result.retrieval_observations[0].input_value == "回答风格"


def _provider(responses: list[dict[str, object]]) -> QueueProvider:
    return QueueProvider(responses)


def test_source_context_failure_is_sent_back_once_and_the_repaired_batch_is_used(tmp_path: Path) -> None:
    """裁定 82 第 1 条：来源校验不过，把哪一条、违反什么发回模型改一次；改对了就照改后的写。"""

    invalid = empty_candidates()
    invalid["preferences"] = [candidate_item(MemoryKind.PREFERENCE, 1)]
    repaired = empty_candidates()
    repaired["preferences"] = [candidate_item(MemoryKind.PREFERENCE, 100)]
    provider = _provider([finish(), invalid, repaired])

    result = asyncio.run(extraction_loop(tmp_path, [], provider=provider).extract(segment()))

    assert provider.responses == []
    assert [candidate.page_id for candidate in result.candidates.iter_candidates()] == [100]
    assert result.source_rejections == ()
    repair = provider.requests[-1]
    assert [message.role for message in repair.messages[-3:]] == ["user", "assistant", "user"]
    assert json.loads(str(repair.messages[-2].content)) == invalid
    assert "preferences page_id=1: existing memory candidate must reuse the page_id assigned to its exact URI" in str(
        repair.messages[-1].content
    )


def test_source_context_failure_after_repair_drops_only_the_failing_item(tmp_path: Path) -> None:
    """裁定 82 第 2 条：改一次仍不过只丢那一条（及引用它的关系），同批其余照常规划。"""

    batch = empty_candidates()
    batch["preferences"] = [candidate_item(MemoryKind.PREFERENCE, 100)]
    batch["tools"] = [candidate_item(MemoryKind.TOOL, 101)]
    batch["relations"] = [{"action": "add", "from_page_id": 100, "to_page_id": 101, "link_type": "related_to"}]
    provider = _provider([finish(), batch, batch])

    result = asyncio.run(extraction_loop(tmp_path, [], provider=provider).extract(segment()))

    assert provider.responses == []
    assert [candidate.page_id for candidate in result.candidates.iter_candidates()] == [100]
    assert result.candidates.relations == ()
    assert [mutation.match.candidate.page_id for mutation in result.mutations.mutations] == [100]
    assert [item.describe() for item in result.source_rejections] == [
        "tools page_id=101: tool candidate requires a matching successful tool_call/tool_result pair",
        "relations add 100->101 related_to: relation references a dropped memory candidate",
    ]


def test_every_candidate_failing_after_repair_leaves_an_empty_batch(tmp_path: Path) -> None:
    """修补后全部不合格 = 这一批没有可写的候选，按空批正常返回，不判失败。"""

    batch = empty_candidates()
    batch["tools"] = [candidate_item(MemoryKind.TOOL, 100)]
    provider = _provider([finish(), batch, batch])

    result = asyncio.run(extraction_loop(tmp_path, [], provider=provider).extract(segment()))

    assert result.candidates.iter_candidates() == ()
    assert result.mutations.mutations == ()
    assert len(result.source_rejections) == 1


def test_unusable_repair_answer_falls_back_to_the_first_batch_minus_failing_items(tmp_path: Path) -> None:
    """修补轮的答复用完结构纠错仍不合格：退回第一次的答复，只剔除不合格的条目。"""

    batch = empty_candidates()
    batch["preferences"] = [candidate_item(MemoryKind.PREFERENCE, 100)]
    batch["tools"] = [candidate_item(MemoryKind.TOOL, 101)]
    broken = {**empty_candidates(), "unexpected": []}
    provider = _provider([finish(), batch, broken])

    result = asyncio.run(extraction_loop(tmp_path, [], provider=provider).extract(segment()))

    assert provider.responses == []
    assert [candidate.page_id for candidate in result.candidates.iter_candidates()] == [100]
    assert len(result.source_rejections) == 1


def test_preliminary_mutation_failure_does_not_trigger_semantic_regeneration(
    tmp_path: Path,
) -> None:
    unconfirmed_intention = empty_candidates()
    unconfirmed_intention["intentions"] = [candidate_item(MemoryKind.INTENTION, 100, confirmed=False)]
    unused_retry = empty_candidates()
    responses = [finish(), unconfirmed_intention, unused_retry]
    loop = extraction_loop(tmp_path, responses)

    with pytest.raises(MemoryExtractionError, match="preliminary mutation planning"):
        asyncio.run(loop.extract(segment()))
    assert responses == [unused_retry]


def test_successful_loop_plans_all_six_memory_kinds_without_writing_tree(tmp_path: Path) -> None:
    source = segment(messages=tool_turn())
    tree_root = tmp_path / "memory"
    result = asyncio.run(
        extraction_loop(
            tmp_path,
            [finish(), all_kind_candidates()],
        ).extract(source)
    )

    planned_kinds = tuple(mutation.match.candidate.kind for mutation in result.mutations.mutations)
    assert len(planned_kinds) == len(MemoryKind)
    assert set(planned_kinds) == set(MemoryKind)
    assert all(mutation.action.value == "create" for mutation in result.mutations.mutations)
    assert result.page_ids.page_ids() == frozenset()
    assert tuple(tree_root.rglob("*.md")) == ()


class SizedQueueProvider(QueueProvider):
    """按模型可见文字的字节数估 token 的替身（与真实 Provider 同一估法：字节数 / 3）。"""

    @staticmethod
    def prepare(request, *, stream):
        from habitus.model_client import PreparedChatRequest

        visible = json.dumps([message.content for message in request.messages], ensure_ascii=False).encode()
        return PreparedChatRequest(
            request=request,
            body=b"{}",
            model_visible_body=visible,
            reserved_output_tokens=request.max_output_tokens or 0,
            stream=stream,
        )


def _summaries(source, count: int):
    from dataclasses import replace

    from tests.helpers import segment_summary

    base = segment_summary(source)
    return tuple(
        replace(base, segment_id=f"{index:012d}-{index:012d}", overview="前面一段在读源码。" * 40)
        for index in range(count)
    )


def test_the_turn_budget_reserves_the_old_memory_budget_and_counts_the_context_char_bound(tmp_path: Path) -> None:
    """整轮放不放得下：不带旧记忆的两种请求都要放进 max_input_tokens − max_old_memory_tokens；超过 max_context_chars
    的整轮直接算放不下（按段抽取），不再永久失败。"""

    from habitus.memory.editor import MemoryCandidateBatch

    material = segment(messages=tool_turn())
    probe = extraction_loop(tmp_path / "probe", [], provider=SizedQueueProvider([]))
    candidate = probe.client.estimate_model_input_tokens(
        probe.prompt_builder.candidate_request(material, None),
        model_class=MemoryCandidateBatch,
    )
    fits = extraction_loop(
        tmp_path / "fits",
        [],
        provider=SizedQueueProvider([]),
        config=MemoryExtractionConfig(max_input_tokens=candidate + 5_000, max_old_memory_tokens=5_000),
    )
    assert fits.turn_fits_whole(material)
    tight = extraction_loop(
        tmp_path / "tight",
        [],
        provider=SizedQueueProvider([]),
        config=MemoryExtractionConfig(max_input_tokens=candidate + 5_000, max_old_memory_tokens=5_001),
    )
    assert not tight.turn_fits_whole(material)
    huge = extraction_loop(tmp_path / "huge", [], config=MemoryExtractionConfig(max_context_chars=100))
    assert not huge.turn_fits_whole(material)


def test_a_turn_part_keeps_the_nearest_earlier_summaries_that_fit(tmp_path: Path) -> None:
    from habitus.memory.editor import MemoryCandidateBatch, MemoryTurnContext
    from habitus.memory.editor.extraction import MemoryRetrievalDecision
    from habitus.pre.conversation import ConversationMessageRole

    part = segment(messages=tool_turn(start_sequence=10)[1:])
    opening = tool_turn()[:1]
    assert opening[0].role is ConversationMessageRole.PROMPT
    summaries = _summaries(part, 6)
    probe = extraction_loop(tmp_path / "probe", [], provider=SizedQueueProvider([]))

    def tokens(context: MemoryTurnContext) -> int:
        builder = probe.prompt_builder
        return max(
            probe.client.estimate_model_input_tokens(
                builder.retrieval_request(
                    part, None, decisions=(), observations=(), allow_action=True, turn_context=context
                ),
                model_class=MemoryRetrievalDecision,
            ),
            probe.client.estimate_model_input_tokens(
                builder.candidate_request(part, None, turn_context=context), model_class=MemoryCandidateBatch
            ),
        )

    full = MemoryTurnContext(opening, summaries, earlier_segment_count=7)
    two = MemoryTurnContext(opening, summaries[-2:], earlier_segment_count=7)
    provider = SizedQueueProvider([finish(), empty_candidates()])
    loop = extraction_loop(
        tmp_path / "loop",
        [],
        provider=provider,
        config=MemoryExtractionConfig(max_input_tokens=tokens(two) + 2_000, max_old_memory_tokens=2_000),
    )
    assert tokens(full) > tokens(two)
    asyncio.run(loop.extract(part, turn_context=full))

    sent = json.loads(str(provider.requests[-1].messages[-1].content).split("\n", 1)[1])
    context = sent["turn_context"]
    # 舍远留近：离本段最近的两份留下，前面四份与没读到的一段如实计数
    assert [item["segment_id"] for item in context["earlier_segment_summaries"]] == [
        s.segment_id for s in summaries[-2:]
    ]
    assert context["omitted_earlier_segment_count"] == 5
    assert [item["content"] for item in context["opening_prompts"]] == [opening[0].content]
    assert "按段分开抽取" in str(provider.requests[-1].messages[1].content)


def test_extraction_material_carries_local_times_with_offset(tmp_path: Path) -> None:
    """消息时刻按主体所在地的本地时间带偏移给模型（M6）：UTC 08:00 在上海是 16:00+08:00。"""

    from zoneinfo import ZoneInfo

    provider = QueueProvider([finish(), empty_candidates()])
    loop = extraction_loop(tmp_path, [], provider=provider)
    loop = MemoryExtractionLoop(client=loop.client, retriever=loop.retriever, zone=ZoneInfo("Asia/Shanghai"))
    asyncio.run(loop.extract(segment()))

    sent = json.loads(str(provider.requests[-1].messages[-1].content).split("\n", 1)[1])
    times = [item["occurred_at"] for item in sent["conversation_segment"]["messages"]]
    assert times == ["2026-07-01T16:00:00.000000+08:00", "2026-07-01T16:00:01.000000+08:00"]
