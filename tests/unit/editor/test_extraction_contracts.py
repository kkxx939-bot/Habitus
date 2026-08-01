"""受控 ReAct、Retrieval Grader 和候选解析的严格契约测试。"""

import pytest

from memory.editor import (
    ConversationSegmentQueryBuilder,
    MemoryExtractionConfig,
    MemoryRetrievalAction,
    MemoryRetrievalConfig,
    MemoryRetrievalDecision,
    MemoryRetrievalIncompleteError,
    MemoryRetrievalObservation,
    MemoryRetrievalStatus,
)
from pre.conversation import ConversationMessageRole, ConversationSegment
from tests.helpers import message, tool_turn


def test_retrieval_decision_allows_exactly_one_action_shape() -> None:
    finish = MemoryRetrievalDecision.model_validate(
        {
            "status": "sufficient",
            "action": "finish",
            "query": None,
            "uri": None,
            "reason": "旧记忆已经足够。",
        }
    )
    search = MemoryRetrievalDecision.model_validate(
        {
            "status": "insufficient",
            "action": "memory_search",
            "query": "用户的回答风格",
            "uri": None,
            "reason": "需要补充偏好。",
        }
    )
    read = MemoryRetrievalDecision.model_validate(
        {
            "status": "insufficient",
            "action": "memory_read",
            "query": None,
            "uri": "memory://profile",
            "reason": "需要阅读全文。",
        }
    )
    assert (finish.action, search.action, read.action) == tuple(MemoryRetrievalAction)

    with pytest.raises(ValueError, match="requires query"):
        MemoryRetrievalDecision(
            MemoryRetrievalStatus.INSUFFICIENT,
            MemoryRetrievalAction.SEARCH,
            None,
            None,
            "缺少查询",
        )
    with pytest.raises(ValueError, match="must finish"):
        MemoryRetrievalDecision(
            MemoryRetrievalStatus.SUFFICIENT,
            MemoryRetrievalAction.READ,
            None,
            "memory://profile",
            "错误组合",
        )
    with pytest.raises(ValueError, match="unknown fields"):
        MemoryRetrievalDecision.model_validate(
            {
                "status": "sufficient",
                "action": "finish",
                "query": None,
                "uri": None,
                "reason": "完成",
                "confidence": 1,
            }
        )


def test_final_retrieval_round_cannot_request_another_tool_action() -> None:
    decision = MemoryRetrievalDecision(
        MemoryRetrievalStatus.INSUFFICIENT,
        MemoryRetrievalAction.SEARCH,
        "继续搜索",
        None,
        "上下文不足",
    )
    with pytest.raises(MemoryRetrievalIncompleteError, match="final round"):
        decision.require_action_allowed(allow_action=False)


def test_observation_is_a_bounded_audit_reference_not_a_document_copy() -> None:
    observation = MemoryRetrievalObservation(
        1,
        MemoryRetrievalAction.SEARCH,
        "偏好",
        ("memory://preferences/回答风格",),
        ("memory://preferences/回答风格",),
        (),
    )
    assert set(observation.to_dict()) == {
        "iteration",
        "action",
        "input",
        "result_uris",
        "added_uris",
        "relation_expanded_uris",
        "cached",
    }
    with pytest.raises(ValueError, match="duplicate"):
        MemoryRetrievalObservation(
            1,
            MemoryRetrievalAction.READ,
            "memory://profile",
            ("memory://profile", "memory://profile"),
            (),
            (),
        )


def test_query_builder_preserves_roles_tool_identity_status_and_prompt_priority() -> None:
    source = ConversationSegment("conversation-1", "segment-1", tool_turn())
    query = ConversationSegmentQueryBuilder().build(source)
    assert query.index("[0][prompt]") < query.index("[1][tool_call]")
    assert "[tool=workspace.inspect]" in query
    assert "[call=call-1]" in query
    assert "[status=completed]" in query


def test_query_builder_truncates_each_role_and_total_query_without_losing_role_headers() -> None:
    source = ConversationSegment("conversation-1", "segment-1", tool_turn())
    builder = ConversationSegmentQueryBuilder(
        MemoryRetrievalConfig(
            max_query_chars=80,
            max_prompt_chars=10,
            max_completion_chars=10,
            max_tool_message_chars=10,
            search_limit=5,
            max_tool_uris=5,
        )
    )
    query = builder.build(source)
    assert len(query) <= 80
    assert query.startswith("[0][prompt]")


def test_query_builder_keeps_latest_user_correction_under_total_budget() -> None:
    source = ConversationSegment(
        "conversation-1",
        "segment-1",
        (
            message(0, ConversationMessageRole.PROMPT, "最初要求剪成六十秒" * 20),
            message(1, ConversationMessageRole.COMPLETION, "已按六十秒处理" * 20),
            message(2, ConversationMessageRole.PROMPT, "最终纠正：必须剪成三十秒"),
            message(3, ConversationMessageRole.COMPLETION, "已确认最终要求"),
        ),
    )
    query = ConversationSegmentQueryBuilder(
        MemoryRetrievalConfig(
            max_query_chars=90,
            max_prompt_chars=80,
            max_completion_chars=40,
            max_tool_message_chars=40,
        )
    ).build(source)
    assert "最终纠正：必须剪成三十秒" in query
    assert "最初要求剪成六十秒" not in query


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_retrieval_iterations", 0),
        ("max_retrieval_iterations", 9),
        ("max_old_memory_items", 100),
    ],
)
def test_extraction_config_enforces_cost_and_page_id_bounds(field: str, value: int) -> None:
    with pytest.raises(ValueError):
        MemoryExtractionConfig(**{field: value})


@pytest.mark.parametrize("field", ["max_candidate_regenerations", "reviewer_max_output_tokens"])
def test_extraction_config_does_not_keep_removed_review_controls(field: str) -> None:
    with pytest.raises(TypeError, match="unexpected keyword"):
        MemoryExtractionConfig(**{field: 1})
