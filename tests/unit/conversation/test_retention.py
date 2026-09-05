"""Conversation 完整轮次提交、语义切段与大工具结果降载测试。"""

from dataclasses import replace

import pytest

from habitus.memory.conversation import (
    ConversationBoundaryHints,
    ConversationMessageChunker,
    ConversationRetentionError,
    ConversationRetentionPlanner,
    ConversationSegmentationConfig,
    ConversationSemanticBoundary,
    ConversationToolResultReducer,
)
from habitus.pre.conversation import (
    ConversationBatch,
    ConversationMessageRole,
    ConversationToolResultContentMode,
    ConversationToolResultStatus,
)
from tests.helpers import closed_turn, message, tool_turn


def config(**overrides: object) -> ConversationSegmentationConfig:
    values = {
        "commit_token_threshold": 20,
        "keep_recent_turn_count": 1,
        "retained_message_token_budget": 100,
        "max_live_messages": 100,
        "max_live_bytes": 100_000,
        "max_segment_messages": 100,
        "max_segment_bytes": 100_000,
        "max_inline_tool_result_bytes": 32,
        "max_tool_result_summary_chars": 128,
    }
    values.update(overrides)
    return ConversationSegmentationConfig(**values)


def test_no_after_turn_never_seals_even_when_threshold_is_reached() -> None:
    batch = ConversationBatch("conversation-1", closed_turn())
    planner = ConversationRetentionPlanner(config(commit_token_threshold=1), token_estimator=lambda _: 100)
    plan = planner.plan(batch)
    assert not plan.should_seal
    assert plan.retained_messages == batch.messages
    assert "afterTurn" in plan.reason


def test_after_turn_archives_only_complete_old_turns_and_keeps_recent_turn() -> None:
    messages = (*closed_turn(start_sequence=0), *tool_turn(start_sequence=2))
    batch = ConversationBatch("conversation-1", messages)
    planner = ConversationRetentionPlanner(config(), token_estimator=lambda _: 10)

    plan = planner.plan(batch, after_turn=True)
    assert plan.should_seal
    assert plan.through_sequence == 1
    assert plan.archive_messages == messages[:2]
    assert plan.retained_messages == messages[2:]
    assert plan.pending_tokens == 20


def test_flush_archives_all_complete_turns_and_drain_pending_has_strict_context() -> None:
    batch = ConversationBatch("conversation-1", closed_turn())
    planner = ConversationRetentionPlanner(config(commit_token_threshold=1_000), token_estimator=lambda _: 1)
    plan = planner.plan(batch, after_turn=True, flush=True)
    assert plan.flush and plan.through_sequence == 1 and not plan.retained_messages
    with pytest.raises(ValueError, match="only valid"):
        planner.plan(batch, drain_pending=True)


def test_incomplete_completion_or_tool_pair_is_rejected_at_commit_boundary() -> None:
    planner = ConversationRetentionPlanner(config(commit_token_threshold=1), token_estimator=lambda _: 1)
    incomplete_completion = ConversationBatch(
        "conversation-1",
        (message(0, ConversationMessageRole.PROMPT, "尚未回答"),),
    )
    with pytest.raises(ConversationRetentionError, match="final completion"):
        planner.plan(incomplete_completion, after_turn=True)

    incomplete_tool = ConversationBatch(
        "conversation-1",
        (
            message(0, ConversationMessageRole.PROMPT, "执行工具"),
            message(
                1,
                ConversationMessageRole.TOOL_CALL,
                {"path": "."},
                tool_call_id="call-1",
                tool_name="workspace.inspect",
            ),
            message(2, ConversationMessageRole.COMPLETION, "完成"),
        ),
    )
    with pytest.raises(ConversationRetentionError, match="terminal tool_result"):
        planner.plan(incomplete_tool, after_turn=True)


def test_oversized_completed_turn_is_split_at_a_safe_message_boundary() -> None:
    batch = ConversationBatch("conversation-1", closed_turn(prompt="x" * 100))
    planner = ConversationRetentionPlanner(
        config(keep_recent_turn_count=0, max_segment_tokens=1),
        token_estimator=lambda _: 1,
    )
    plan = planner.plan(batch, after_turn=True, flush=True)
    assert plan.through_sequence == 0
    assert plan.archive_messages == batch.messages[:1]
    assert plan.boundary_kind == "message"


def test_message_chunker_preserves_long_text_and_declares_logical_parts() -> None:
    source = ConversationBatch(
        "conversation-1",
        closed_turn(prompt="第一段。\n\n第二段。\n\n" + "x" * 80),
    )
    chunker = ConversationMessageChunker(
        max_message_tokens=24,
        token_estimator=lambda item: len(item.content) if isinstance(item.content, str) else 1,
    )

    normalized = chunker.normalize(source)

    prompt_parts = normalized.messages[:-1]
    assert len(prompt_parts) > 1
    assert "".join(str(item.content) for item in prompt_parts) == source.messages[0].content
    assert {item.logical_message_id for item in prompt_parts} == {source.messages[0].message_id}
    assert tuple(item.logical_part_index for item in prompt_parts) == tuple(range(len(prompt_parts)))
    assert {item.logical_part_count for item in prompt_parts} == {len(prompt_parts)}
    assert tuple(item.sequence for item in normalized.messages) == tuple(range(len(normalized.messages)))


def test_semantic_segment_never_splits_a_tool_call_from_its_terminal_result() -> None:
    batch = ConversationBatch("conversation-1", tool_turn())
    planner = ConversationRetentionPlanner(
        config(keep_recent_turn_count=0, max_segment_tokens=3),
        token_estimator=lambda _: 1,
    )

    plan = planner.plan(batch, after_turn=True, flush=True)

    assert plan.through_sequence == 2
    assert tuple(item.role for item in plan.archive_messages[-2:]) == (
        ConversationMessageRole.TOOL_CALL,
        ConversationMessageRole.TOOL_RESULT,
    )
    assert plan.boundary_kind == "assistant_step"


def test_intermediate_completion_is_not_mistaken_for_a_turn_boundary() -> None:
    messages = (
        message(0, ConversationMessageRole.PROMPT, "先分析再调用工具"),
        message(1, ConversationMessageRole.COMPLETION, "我先检查。"),
        message(
            2,
            ConversationMessageRole.TOOL_CALL,
            {"path": "."},
            tool_call_id="call-1",
            tool_name="workspace.inspect",
        ),
        message(
            3,
            ConversationMessageRole.TOOL_RESULT,
            "正常",
            tool_call_id="call-1",
            tool_name="workspace.inspect",
            tool_status=ConversationToolResultStatus.COMPLETED,
            content_mode=ConversationToolResultContentMode.INLINE,
        ),
        message(4, ConversationMessageRole.COMPLETION, "检查完成。"),
    )
    planner = ConversationRetentionPlanner(
        config(keep_recent_turn_count=0, max_segment_tokens=2),
        token_estimator=lambda _: 1,
    )

    plan = planner.plan(
        ConversationBatch("conversation-1", messages),
        after_turn=True,
        flush=True,
    )

    assert plan.through_sequence == 0
    assert plan.archive_messages[-1].role is ConversationMessageRole.PROMPT


def test_embedding_distance_ranks_safe_boundaries_within_the_same_structure_level() -> None:
    chunker = ConversationMessageChunker(
        max_message_tokens=10,
        token_estimator=lambda item: len(item.content) if isinstance(item.content, str) else 1,
    )
    live = chunker.normalize(
        ConversationBatch("conversation-1", closed_turn(prompt="A" * 30))
    )
    planner = ConversationRetentionPlanner(
        config(
            keep_recent_turn_count=0,
            max_segment_tokens=2,
            semantic_boundary_min_ratio=0.5,
        ),
        token_estimator=lambda _: 1,
    )
    hints = ConversationBoundaryHints(
        live.digest,
        (
            ConversationSemanticBoundary(0, 2.0),
            ConversationSemanticBoundary(1, 0.0),
        ),
        "embedding-v1",
    )

    structural = planner.plan(live, after_turn=True, flush=True)
    semantic = planner.plan(
        live,
        after_turn=True,
        flush=True,
        boundary_hints=hints,
    )

    assert structural.through_sequence == 1
    assert semantic.through_sequence == 0
    assert semantic.embedding_fingerprint == "embedding-v1"


def test_tool_result_reducer_keeps_small_result_summarizes_large_text_and_omits_media() -> None:
    original = tool_turn()[2]
    reducer = ConversationToolResultReducer(config())
    assert reducer.reduce(original) is original

    large = replace(original, content="x" * 100)
    summarized = reducer.reduce(large)
    assert summarized.content_mode is ConversationToolResultContentMode.SUMMARIZED
    assert summarized.original_size_bytes is not None
    assert len(summarized.original_sha256) == 64
    assert "工具结果已压缩" in summarized.content

    token_limited = reducer.reduce(original, force_summarize=True)
    assert token_limited.content_mode is ConversationToolResultContentMode.SUMMARIZED

    omitted = reducer.reduce(large, force_omit=True, description="视频载荷未长期保存")
    assert omitted.content_mode is ConversationToolResultContentMode.OMITTED
    assert omitted.content == "视频载荷未长期保存"
    with pytest.raises(ValueError, match="only tool_result"):
        reducer.reduce(closed_turn()[0])
    with pytest.raises(ValueError, match="force-omitted"):
        reducer.reduce(original, force_omit=True, force_summarize=True)
