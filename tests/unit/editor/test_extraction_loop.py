"""完整 Conversation 经受控检索、候选生成和二次审查的解析循环测试。"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from memory.editor import (
    MemoryCandidateBatch,
    MemoryCandidateRejectedError,
    MemoryExtractionConfig,
    MemoryExtractionError,
    MemoryExtractionLoop,
    MemoryIdentityPlanner,
    MemoryIdentityPlanningError,
    MemoryRelatedRetriever,
    MemoryRetrievalIncompleteError,
)
from memory.model import MemoryKind
from memory.schema import MemorySchemaRegistry
from memory.snapshot import MemorySnapshotReader
from memory.tree import MemoryTree
from ModelClient import (
    ChatClient,
    ChatModelConfig,
    ModelResponse,
    ProviderCapabilities,
    ProviderConfig,
    StructuredChatClient,
)
from tests.helpers import memory_fields, segment, tool_turn


@dataclass
class QueueProvider:
    responses: list[dict[str, object]]
    provider_name: str = "test"
    model: str = "test"
    is_remote: bool = False
    capabilities: ProviderCapabilities = ProviderCapabilities()

    def complete(self, _request):
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
    async def search(self, _query, **_kwargs):
        return ()


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


def accepted() -> dict[str, object]:
    return {"decision": "accept", "issues": []}


def rejected() -> dict[str, object]:
    return {
        "decision": "reject",
        "issues": [
            {
                "code": "unsupported_long_term",
                "detail": "当前内容不足以形成长期记忆。",
            }
        ],
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


class RejectingIdentityPlanner(MemoryIdentityPlanner):
    """模拟第二遍审查通过后确定性身份规则仍拒绝候选。"""

    def plan(self, _extraction):
        raise MemoryIdentityPlanningError("same_memory target does not preserve source")


def extraction_loop(
    tmp_path: Path,
    responses: list[dict[str, object]],
    *,
    config: MemoryExtractionConfig | None = None,
) -> MemoryExtractionLoop:
    route = ProviderConfig(
        provider="test",
        adapter="test",
        model="test",
        base_url="https://example.com",
        max_retries=0,
    )
    client = StructuredChatClient(
        ChatClient(ChatModelConfig(route), QueueProvider(responses)),
        validation_retries=0,
    )
    tree = MemoryTree(tmp_path / "memory")
    retriever = MemoryRelatedRetriever(
        schema_registry=MemorySchemaRegistry.load_default(),
        snapshot_reader=MemorySnapshotReader(tree),
        semantic_search=EmptySemanticSearch(),
    )
    return MemoryExtractionLoop(client=client, retriever=retriever, config=config)


def test_loop_accepts_empty_memory_batch_after_complete_retrieval_review(tmp_path: Path) -> None:
    result = asyncio.run(
        extraction_loop(tmp_path, [finish(), empty_candidates(), accepted()]).extract(segment())
    )

    assert result.candidates.iter_candidates() == ()
    assert result.mutations.mutations == ()
    assert result.retrieval_decisions[0].action.value == "finish"
    assert result.candidate_attempts == 1
    assert result.source_segment_digest == segment().digest


def test_rejected_candidate_is_regenerated_as_a_complete_batch_within_bound(tmp_path: Path) -> None:
    result = asyncio.run(
        extraction_loop(
            tmp_path,
            [finish(), empty_candidates(), rejected(), empty_candidates(), accepted()],
            config=MemoryExtractionConfig(max_candidate_regenerations=1),
        ).extract(segment())
    )

    assert result.candidate_attempts == 2
    assert result.review.decision.value == "accept"


def test_rejection_after_last_attempt_fails_without_returning_partial_candidates(tmp_path: Path) -> None:
    loop = extraction_loop(
        tmp_path,
        [finish(), empty_candidates(), rejected()],
        config=MemoryExtractionConfig(max_candidate_regenerations=0),
    )
    with pytest.raises(MemoryCandidateRejectedError, match="remained rejected"):
        asyncio.run(loop.extract(segment()))


def test_last_retrieval_round_cannot_execute_another_search(tmp_path: Path) -> None:
    search = {
        "status": "insufficient",
        "action": "memory_search",
        "query": "继续搜索偏好",
        "uri": None,
        "reason": "旧记忆不足。",
    }
    loop = extraction_loop(
        tmp_path,
        [search],
        config=MemoryExtractionConfig(max_retrieval_iterations=1),
    )
    with pytest.raises(MemoryRetrievalIncompleteError, match="final round"):
        asyncio.run(loop.extract(segment()))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("client", object(), "client"),
        ("retriever", object(), "retriever"),
        ("config", object(), "config"),
        ("mutation_reader", object(), "mutation_reader"),
        ("mutation_planner", object(), "mutation_planner"),
        ("identity_planner", object(), "identity_planner"),
    ],
)
def test_loop_rejects_each_invalid_collaborator(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    current = extraction_loop(tmp_path, [])
    values = {
        "client": current.client,
        "retriever": current.retriever,
        "config": current.config,
        "mutation_reader": current.mutation_reader,
        "mutation_planner": current.mutation_planner,
        "identity_planner": current.identity_planner,
        field: value,
    }
    with pytest.raises(TypeError, match=message):
        MemoryExtractionLoop(**values)


@pytest.mark.parametrize("invalid", [None, object(), "segment", 1, True])
def test_loop_requires_complete_conversation_segment(tmp_path: Path, invalid: object) -> None:
    loop = extraction_loop(tmp_path, [])
    with pytest.raises(TypeError, match="ConversationSegment"):
        asyncio.run(loop.extract(invalid))  # type: ignore[arg-type]


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
            [search, finish(), empty_candidates(), accepted()],
            config=MemoryExtractionConfig(max_retrieval_iterations=2),
        ).extract(segment())
    )

    assert tuple(item.action.value for item in result.retrieval_decisions) == (
        "memory_search",
        "finish",
    )
    assert len(result.retrieval_observations) == 1
    assert result.retrieval_observations[0].input_value == "回答风格"


def test_source_context_failure_regenerates_complete_candidate_batch(tmp_path: Path) -> None:
    invalid = empty_candidates()
    invalid["preferences"] = [candidate_item(MemoryKind.PREFERENCE, 1)]
    result = asyncio.run(
        extraction_loop(
            tmp_path,
            [finish(), invalid, empty_candidates(), accepted()],
            config=MemoryExtractionConfig(max_candidate_regenerations=1),
        ).extract(segment())
    )

    assert result.candidate_attempts == 2
    assert result.candidates == MemoryCandidateBatch.model_validate(empty_candidates())


def test_source_context_failure_exhaustion_returns_no_partial_result(tmp_path: Path) -> None:
    invalid = empty_candidates()
    invalid["preferences"] = [candidate_item(MemoryKind.PREFERENCE, 1)]
    loop = extraction_loop(
        tmp_path,
        [finish(), invalid],
        config=MemoryExtractionConfig(max_candidate_regenerations=0),
    )

    with pytest.raises(MemoryExtractionError, match="source-context validation"):
        asyncio.run(loop.extract(segment()))


def test_preliminary_mutation_failure_regenerates_candidate_batch(tmp_path: Path) -> None:
    unconfirmed_intention = empty_candidates()
    unconfirmed_intention["intentions"] = [
        candidate_item(MemoryKind.INTENTION, 100, confirmed=False)
    ]
    result = asyncio.run(
        extraction_loop(
            tmp_path,
            [finish(), unconfirmed_intention, empty_candidates(), accepted()],
            config=MemoryExtractionConfig(max_candidate_regenerations=1),
        ).extract(segment())
    )

    assert result.candidate_attempts == 2
    assert result.mutations.mutations == ()


def test_preliminary_mutation_failure_exhaustion_returns_no_partial_plan(tmp_path: Path) -> None:
    unconfirmed_intention = empty_candidates()
    unconfirmed_intention["intentions"] = [
        candidate_item(MemoryKind.INTENTION, 100, confirmed=False)
    ]
    loop = extraction_loop(
        tmp_path,
        [finish(), unconfirmed_intention],
        config=MemoryExtractionConfig(max_candidate_regenerations=0),
    )

    with pytest.raises(MemoryExtractionError, match="preliminary mutation planning"):
        asyncio.run(loop.extract(segment()))


def test_identity_planning_failure_regenerates_after_review_acceptance(tmp_path: Path) -> None:
    loop = extraction_loop(
        tmp_path,
        [finish(), empty_candidates(), accepted(), empty_candidates(), accepted()],
        config=MemoryExtractionConfig(max_candidate_regenerations=1),
    )
    loop.identity_planner = RejectingIdentityPlanner()

    with pytest.raises(MemoryExtractionError, match="deterministic planning"):
        asyncio.run(loop.extract(segment()))


def test_successful_loop_plans_all_six_memory_kinds_without_writing_tree(tmp_path: Path) -> None:
    source = segment(messages=tool_turn())
    tree_root = tmp_path / "memory"
    result = asyncio.run(
        extraction_loop(
            tmp_path,
            [finish(), all_kind_candidates(), accepted()],
        ).extract(source)
    )

    planned_kinds = tuple(mutation.match.candidate.kind for mutation in result.mutations.mutations)
    assert len(planned_kinds) == len(MemoryKind)
    assert set(planned_kinds) == set(MemoryKind)
    assert all(mutation.action.value == "create" for mutation in result.mutations.mutations)
    assert result.page_ids.page_ids() == frozenset()
    assert tuple(tree_root.rglob("*.md")) == ()


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("same_memory merge rejected", "unjustified_identity_merge"),
        ("remove_memory delete rejected", "unjustified_memory_delete"),
        ("tool evidence missing", "invalid_tool_generalization"),
        ("intention transition invalid", "event_intention_confusion"),
        ("relation remove missing", "invalid_relation_remove"),
        ("relation unsupported", "unjustified_relation"),
        ("unknown page", "invalid_page_identity"),
    ],
)
def test_context_failure_is_mapped_to_controlled_review_issue(message: str, expected: str) -> None:
    issue = MemoryExtractionLoop._context_issue(ValueError(message))
    assert issue.code.value == expected
    assert issue.detail == message
