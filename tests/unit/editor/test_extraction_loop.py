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
from tests.helpers import document, memory_fields, segment, tool_turn
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


def finish() -> dict[str, object]:
    return {
        "status": "sufficient",
        "action": "finish",
        "directory": None,
        "uris": [],
        "keywords": [],
        "reason": "当前旧记忆已经足够。",
    }


def empty_candidates() -> dict[str, object]:
    return {
        "profile": [],
        "preferences": [],
        "entities": [],
        "tools": [],
        "identity_proposals": [],
        "relations": [],
    }


def candidate_item(kind: MemoryKind, page_id: int, **controls: object) -> dict[str, object]:
    """构造一个符合对应记忆 YAML 的结构化候选。"""

    return {"page_id": page_id, **memory_fields(kind), **controls}


def all_kind_candidates() -> dict[str, object]:
    """构造覆盖四类记忆且 page_id 唯一的新节点批次。"""

    result = empty_candidates()
    result["profile"] = [candidate_item(MemoryKind.PROFILE, 100)]
    result["preferences"] = [candidate_item(MemoryKind.PREFERENCE, 101)]
    result["entities"] = [candidate_item(MemoryKind.ENTITY, 102)]
    result["tools"] = [candidate_item(MemoryKind.TOOL, 103)]
    return result


def list_more(directory: str = "memory://preferences") -> dict[str, object]:
    return {
        "status": "insufficient",
        "action": "memory_list",
        "directory": directory,
        "uris": [],
        "keywords": [],
        "reason": "旧记忆不足。",
    }


def read_more(*uris: str) -> dict[str, object]:
    return {
        "status": "insufficient",
        "action": "memory_read",
        "directory": None,
        "uris": list(uris),
        "keywords": [],
        "reason": "需要读取完整旧记忆。",
    }


def ended_insufficient() -> dict[str, object]:
    """终轮答复形状：树空、两次列目录都是空的、如实说不足并结束。"""

    return {
        "status": "insufficient",
        "action": "finish",
        "directory": None,
        "uris": [],
        "keywords": [],
        "reason": "完整旧 L2 快照均缺失，列目录两次均为空。",
    }


def extraction_loop(
    tmp_path: Path,
    responses: list[dict[str, object]],
    *,
    config: MemoryExtractionConfig | None = None,
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
    provider = QueueProvider([list_more(), list_more("memory://entities"), ended_insufficient(), all_kind_candidates()])
    loop = extraction_loop(
        tmp_path,
        [],
        config=MemoryExtractionConfig(max_retrieval_iterations=3),
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
    assert all(mutation.action.value == "create" for mutation in result.mutations.mutations)
    assert len(result.mutations.mutations) == len(MemoryKind)

    final_request = provider.requests[2]
    final_schema = final_request.response_format.schema
    assert final_schema["properties"]["action"]["enum"] == ["finish"]
    final_system = final_request.messages[1].content
    assert "预算已经用完" in final_system
    assert "明确失败" not in final_system
    action_schema = provider.requests[0].response_format.schema
    assert action_schema["properties"]["action"]["enum"] == ["finish", "memory_list", "memory_search", "memory_read"]


def test_final_round_that_still_requests_a_tool_ends_at_limit_without_running_it(tmp_path: Path) -> None:
    responses = [list_more(), empty_candidates()]
    loop = extraction_loop(
        tmp_path,
        responses,
        config=MemoryExtractionConfig(max_retrieval_iterations=1),
    )

    result = asyncio.run(loop.extract(segment()))

    assert responses == []
    assert result.retrieval_ended_at_limit is True
    assert result.retrieval_decisions == ()
    assert result.retrieval_observations == ()
    assert result.candidates.iter_candidates() == ()


def test_final_round_sufficient_finish_is_not_recorded_as_limit(tmp_path: Path) -> None:
    result = asyncio.run(
        extraction_loop(
            tmp_path,
            [list_more(), finish(), empty_candidates()],
            config=MemoryExtractionConfig(max_retrieval_iterations=2),
        ).extract(segment())
    )

    assert result.retrieval_ended_at_limit is False
    assert isinstance(result.retrieval_decisions[-1], MemoryRetrievalFinalDecision)


def test_finish_before_the_final_round_is_not_recorded_as_limit(tmp_path: Path) -> None:
    result = asyncio.run(extraction_loop(tmp_path, [finish(), empty_candidates()]).extract(segment()))

    assert result.retrieval_ended_at_limit is False
    assert not isinstance(result.retrieval_decisions[0], MemoryRetrievalFinalDecision)


def test_read_outside_the_listing_still_fails_instead_of_ending_at_limit(tmp_path: Path) -> None:
    """读名字清单、列目录结果和一跳关系之外的地址明确失败，不当作在上限处结束。"""

    MemoryTree(tmp_path / "memory").write(document(MemoryKind.PROFILE))
    responses = [read_more("memory://preferences/未列出.md"), ended_insufficient(), empty_candidates()]
    loop = extraction_loop(
        tmp_path,
        responses,
        config=MemoryExtractionConfig(max_retrieval_iterations=2),
    )

    with pytest.raises(MemoryExtractionError, match="not exposed"):
        asyncio.run(loop.extract(segment()))
    assert responses == [ended_insufficient(), empty_candidates()]


def test_loop_executes_one_list_then_finishes_with_auditable_observation(tmp_path: Path) -> None:
    result = asyncio.run(
        extraction_loop(
            tmp_path,
            [list_more(), finish(), empty_candidates()],
            config=MemoryExtractionConfig(max_retrieval_iterations=2),
        ).extract(segment())
    )

    assert tuple(item.action.value for item in result.retrieval_decisions) == (
        "memory_list",
        "finish",
    )
    assert len(result.retrieval_observations) == 1
    assert result.retrieval_observations[0].inputs == ("memory://preferences",)


def test_name_listing_goes_into_the_first_round_and_listed_cards_can_be_read(tmp_path: Path) -> None:
    """偏好主题与实体名字清单放进第一轮输入；清单里的地址可以一次读进来，读到的成为旧记忆快照。"""

    preference = document(MemoryKind.PREFERENCE)
    entity = document(MemoryKind.ENTITY)
    tree = MemoryTree(tmp_path / "memory")
    tree.write(preference)
    tree.write(entity)
    uris = sorted(str(MemoryURI.from_address(item.address)) for item in (preference, entity))
    provider = QueueProvider([read_more(*uris), finish(), empty_candidates()])

    result = asyncio.run(extraction_loop(tmp_path, [], provider=provider).extract(segment()))

    first = json.loads(str(provider.requests[0].messages[-1].content).split("\n", 1)[1])
    for uri in uris:
        assert uri in first["name_listing"]
    assert result.retrieval_observations[0].inputs == tuple(uris)
    assert {snapshot.identity for snapshot in result.old_memories.snapshots if snapshot.exists} == set(uris)
    assert result.page_ids.page_ids() == frozenset({1, 2})


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


def test_a_candidate_written_to_an_unread_old_card_is_sent_back_with_that_card_and_updates_it(tmp_path: Path) -> None:
    """用户 2026-11-23 定：候选写到已有旧卡片的地址、这次又没读过，系统把旧卡片读出来，连同说明交回修补轮改一次；
    改成复用旧 page_id 就照常更新这张卡片，任务不失败。"""

    old = document(MemoryKind.PREFERENCE)
    MemoryTree(tmp_path / "memory").write(old)
    uri = str(MemoryURI.from_address(old.address))
    unread_target = empty_candidates()
    unread_target["preferences"] = [candidate_item(MemoryKind.PREFERENCE, 100)]
    rewritten = empty_candidates()
    rewritten["preferences"] = [candidate_item(MemoryKind.PREFERENCE, 1, content="- 偏好简洁直接的回答\n- 列表用短句")]
    provider = _provider([finish(), unread_target, rewritten])

    result = asyncio.run(extraction_loop(tmp_path, [], provider=provider).extract(segment()))

    assert provider.responses == []
    assert result.source_rejections == ()
    assert [mutation.match.status.value for mutation in result.mutations.mutations] == ["existing"]
    repair = provider.requests[-1]
    payload = json.loads(str(repair.messages[-3].content).split("\n", 1)[1])  # 修补请求带上了补读进来的旧卡片
    assert uri in {item["uri"] for item in payload["old_memory_snapshots"]}
    assert f"page_id=100 写到 {uri}，这个地址已有旧卡片" in str(repair.messages[-1].content)


@pytest.mark.parametrize("repair_answer", ["still-new", "unusable"])
def test_a_candidate_still_written_to_an_old_card_after_repair_is_dropped_alone(
    tmp_path: Path, repair_answer: str
) -> None:
    """修补后仍把新 page_id 写到旧卡片的地址（或修补答复不能用）：只剔掉这一条和引用它的关系，同批其余照常。"""

    MemoryTree(tmp_path / "memory").write(document(MemoryKind.PREFERENCE))
    first = empty_candidates()
    first["preferences"] = [candidate_item(MemoryKind.PREFERENCE, 100)]
    first["entities"] = [candidate_item(MemoryKind.ENTITY, 101)]
    first["relations"] = [{"action": "add", "from_page_id": 100, "to_page_id": 101, "link_type": "related_to"}]
    second: object = first if repair_answer == "still-new" else {"not": "a batch"}
    provider = _provider([finish(), first, *([second] * 3)])

    result = asyncio.run(extraction_loop(tmp_path, [], provider=provider).extract(segment()))

    assert [candidate.page_id for candidate in result.candidates.iter_candidates()] == [101]
    assert result.candidates.relations == ()
    reasons = [violation.reason for violation in result.source_rejections]
    # 旧卡片补读进来后，来源校验本身就认出"写到已有记忆却没复用它的 page_id"
    assert "candidate for an existing memory URI must reuse its existing page_id" in reasons
    assert "relation references a dropped memory candidate" in reasons


def test_successful_loop_plans_all_four_memory_kinds_without_writing_tree(tmp_path: Path) -> None:
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
    """整轮放不放得下：不带旧记忆与名字清单的两种请求都要放进 max_input_tokens − max_old_memory_tokens − max_listing_tokens；超过 max_context_chars
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
        config=MemoryExtractionConfig(
            max_input_tokens=candidate + 6_000, max_old_memory_tokens=5_000, max_listing_tokens=1_000
        ),
    )
    assert fits.turn_fits_whole(material)
    tight = extraction_loop(
        tmp_path / "tight",
        [],
        provider=SizedQueueProvider([]),
        config=MemoryExtractionConfig(
            max_input_tokens=candidate + 6_000, max_old_memory_tokens=5_001, max_listing_tokens=1_000
        ),
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
        config=MemoryExtractionConfig(
            max_input_tokens=tokens(two) + 3_000, max_old_memory_tokens=2_000, max_listing_tokens=1_000
        ),
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


def search_more(*keywords: str, directory: str | None = None) -> dict[str, object]:
    return {
        "status": "insufficient",
        "action": "memory_search",
        "directory": directory,
        "uris": [],
        "keywords": list(keywords),
        "reason": "名字看不出，按内容找。",
    }


def test_a_content_search_returns_matching_cards_with_snippets_and_makes_them_readable(tmp_path: Path) -> None:
    """用户 2026-11-23 定：提取时可以按词搜卡片内容（字面匹配）。工具卡片不在名字清单里，搜到才能读；
    返回地址、名字和命中片段，不超过条数上限；没搜到的地址照旧拒读。"""

    tree = MemoryTree(tmp_path / "memory")
    for name in ("disk.clean", "disk.scan", "disk.mount"):
        tree.write(
            document(
                MemoryKind.TOOL, fields={"tool_name": name, "verification": f"{name} 跑完后看剩余空间，再看日志。"}
            )
        )
    tree.write(document(MemoryKind.TOOL, fields={"tool_name": "net.ping", "verification": "看丢包率。"}))
    provider = QueueProvider([search_more("剩余空间", directory="memory://tools"), finish(), empty_candidates()])
    config = MemoryExtractionConfig(max_search_results=2)

    result = asyncio.run(extraction_loop(tmp_path, [], provider=provider, config=config).extract(segment()))

    observation = result.retrieval_observations[0]
    assert observation.inputs == ("memory://tools", "剩余空间")
    assert len(observation.result_uris) == 2
    assert all(uri.startswith("memory://tools/disk.") for uri in observation.result_uris)
    second = json.loads(str(provider.requests[1].messages[-1].content).split("\n", 1)[1])
    lines = second["searched"]["memory://tools 剩余空间"]
    assert len(lines) == 2 and all("剩余空间" in line and "（disk." in line for line in lines)

    found = observation.result_uris[0]
    responses = [search_more("剩余空间"), read_more(found), finish(), empty_candidates()]
    read = asyncio.run(
        extraction_loop(tmp_path, responses, config=MemoryExtractionConfig(max_retrieval_iterations=4)).extract(
            segment()
        )
    )
    assert found in {snapshot.identity for snapshot in read.old_memories.snapshots if snapshot.exists}

    refused = [
        search_more("剩余空间"),
        read_more("memory://tools/net.ping.md"),
        ended_insufficient(),
        empty_candidates(),
    ]
    with pytest.raises(MemoryExtractionError, match="not exposed"):
        asyncio.run(
            extraction_loop(tmp_path, refused, config=MemoryExtractionConfig(max_retrieval_iterations=3)).extract(
                segment()
            )
        )


def test_a_content_search_skips_retired_cards(tmp_path: Path) -> None:
    from datetime import datetime, timedelta

    from habitus.memory.editor.retrieval import memory_content_search
    from habitus.memory.recall import (
        MemoryRecallLifecycle,
        MemoryRecallLifecycleConfig,
        MemoryRecallTarget,
        SQLiteMemoryRecallLifecycleStore,
    )

    tree = MemoryTree(tmp_path / "memory")
    kept = document(MemoryKind.TOOL, fields={"tool_name": "disk.clean", "verification": "看剩余空间。"})
    gone = document(MemoryKind.TOOL, fields={"tool_name": "disk.old", "verification": "看剩余空间。"})
    tree.write(kept)
    tree.write(gone)
    config = MemoryRecallLifecycleConfig()
    lifecycle = MemoryRecallLifecycle(
        SQLiteMemoryRecallLifecycleStore(tmp_path / "recall.sqlite3", config=config), config=config
    )
    target = MemoryRecallTarget(MemoryURI.from_address(gone.address), 1, gone.metadata.created_at)
    now = datetime.now(UTC)
    lifecycle.store.mark_retire_candidate(target, marked_at=now - timedelta(days=40), expected_version=0)
    lifecycle.store.mark_retired(target, retired_at=now - timedelta(days=1), expected_version=1)

    hits = memory_content_search(tree, lifecycle, keywords=("剩余空间",), directory=None, limit=8)

    assert [str(hit.uri) for hit in hits] == [str(MemoryURI.from_address(kept.address))]
