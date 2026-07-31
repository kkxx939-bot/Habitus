"""查询计划、搜索命中、下游上下文预算和角色化 Conversation 补充测试。"""

import pytest

from memory.model import MemoryKind
from memory.retrieval import (
    MemoryContextAssembler,
    MemoryMatchedMemory,
    MemoryQueryPlanContent,
    MemorySearchError,
    MemorySearchHit,
    MemorySearchServiceConfig,
    MemoryTemperature,
    MemoryTypedQuery,
)
from memory.retrieval.context import render_recent_messages
from memory.uri import MemoryURI
from tests.helpers import document, tool_turn


def test_query_plan_rejects_duplicate_semantics_and_invalid_priority() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        MemoryQueryPlanContent(
            (
                MemoryTypedQuery("用户偏好", "偏好", 1),
                MemoryTypedQuery("用户偏好", "偏好", 2),
            )
        )
    with pytest.raises(ValueError, match="between one and five"):
        MemoryTypedQuery("用户偏好", "偏好", 0)


def test_search_hit_requires_l2_uri_and_final_score_equals_reranker() -> None:
    uri = MemoryURI.from_address(document(MemoryKind.PROFILE).address)
    assert MemorySearchHit(uri, 0.7, vector_score=0.5, rerank_score=0.7).score == 0.7
    with pytest.raises(ValueError, match="equal"):
        MemorySearchHit(uri, 0.7, vector_score=0.5, rerank_score=0.6)
    with pytest.raises(ValueError, match="L2"):
        MemorySearchHit(MemoryURI.root(), 0.5)


def test_search_hit_preserves_semantic_score_and_validates_complete_lifecycle_fields() -> None:
    uri = MemoryURI.from_address(document(MemoryKind.PROFILE).address)
    hit = MemorySearchHit(
        uri,
        0.63,
        vector_score=0.8,
        rerank_score=0.7,
        lifecycle_semantic_score=0.7,
        lifecycle_hotness=0.5,
        lifecycle_temperature=MemoryTemperature.WARM,
    )
    assert hit.score == 0.63
    assert hit.rerank_score == 0.7
    assert hit.lifecycle_temperature is MemoryTemperature.WARM
    with pytest.raises(ValueError, match="complete"):
        MemorySearchHit(uri, 0.63, lifecycle_semantic_score=0.7)
    with pytest.raises(ValueError, match="cannot improve"):
        MemorySearchHit(
            uri,
            0.71,
            lifecycle_semantic_score=0.7,
            lifecycle_hotness=0.5,
            lifecycle_temperature=MemoryTemperature.WARM,
        )
    with pytest.raises(ValueError, match="rerank"):
        MemorySearchHit(
            uri,
            0.63,
            rerank_score=0.8,
            lifecycle_semantic_score=0.7,
            lifecycle_hotness=0.5,
            lifecycle_temperature=MemoryTemperature.WARM,
        )


def test_context_assembler_emits_memory_as_primary_read_only_context() -> None:
    preference = document(MemoryKind.PREFERENCE)
    memory = MemoryMatchedMemory(
        MemorySearchHit(MemoryURI.from_address(preference.address), 0.9),
        preference,
        ("用户回答偏好",),
    )
    result = MemoryContextAssembler().assemble((memory,))
    assert result.memories == (memory,)
    assert "只能作为回答事实背景" in result.context
    assert '<memory uri="memory://preferences/' in result.context
    assert "matched_queries: 用户回答偏好" in result.context


def test_context_assembler_exposes_auditable_lifecycle_score_without_changing_l2_body() -> None:
    preference = document(MemoryKind.PREFERENCE)
    memory = MemoryMatchedMemory(
        MemorySearchHit(
            MemoryURI.from_address(preference.address),
            0.81,
            lifecycle_semantic_score=0.9,
            lifecycle_hotness=0.5,
            lifecycle_temperature=MemoryTemperature.WARM,
        ),
        preference,
        ("回答偏好",),
    )
    result = MemoryContextAssembler().assemble((memory,))
    assert 'score="0.900000"' in result.context
    assert "semantic_score" not in result.context
    assert "hotness" not in result.context
    assert "temperature" not in result.context
    assert preference.markdown_body.strip() in result.context
    raw = MemoryMatchedMemory(
        MemorySearchHit(MemoryURI.from_address(preference.address), 0.9),
        preference,
        ("回答偏好",),
    )
    assert MemoryContextAssembler().assemble((raw,)).context == result.context


def test_context_budget_rejects_top_document_that_cannot_fit() -> None:
    config = MemorySearchServiceConfig(
        max_context_chars=1024,
        retrieval_grader_max_context_chars=1024,
        summary_fallback_max_context_chars=512,
    )
    preference = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "超长偏好", "content": "x" * 2_000},
    )
    memory = MemoryMatchedMemory(
        MemorySearchHit(MemoryURI.from_address(preference.address), 0.9),
        preference,
        ("偏好",),
    )
    with pytest.raises(MemorySearchError, match="top memory"):
        MemoryContextAssembler(config=config).assemble((memory,))


def test_recent_messages_keep_prompt_completion_and_tool_roles_separate() -> None:
    rendered = render_recent_messages(tool_turn(), max_message_chars=100)
    assert "[0][prompt]" in rendered
    assert "[1][tool_call][tool=workspace.inspect][call=call-1]" in rendered
    assert "[2][tool_result][tool=workspace.inspect][call=call-1][status=completed]" in rendered
    assert "[3][completion]" in rendered
