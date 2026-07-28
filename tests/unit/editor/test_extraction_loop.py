"""完整 Conversation 经受控检索、候选生成和二次审查的解析循环测试。"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from memory.editor import (
    MemoryCandidateRejectedError,
    MemoryExtractionConfig,
    MemoryExtractionLoop,
    MemoryRelatedRetriever,
    MemoryRetrievalIncompleteError,
)
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
from tests.helpers import segment


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

