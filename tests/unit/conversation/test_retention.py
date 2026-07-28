"""Conversation 只在完整轮次边界切段，以及大工具结果降载测试。"""

from dataclasses import replace

import pytest

from memory.conversation import (
    ConversationRetentionError,
    ConversationRetentionPlanner,
    ConversationSegmentationConfig,
    ConversationToolResultReducer,
)
from pre.conversation import (
    ConversationBatch,
    ConversationMessageRole,
    ConversationToolResultContentMode,
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


def test_one_oversized_atomic_turn_fails_instead_of_being_split() -> None:
    batch = ConversationBatch("conversation-1", closed_turn(prompt="x" * 100))
    planner = ConversationRetentionPlanner(
        config(keep_recent_turn_count=0, max_segment_tokens=1),
        token_estimator=lambda _: 1,
    )
    with pytest.raises(ConversationRetentionError, match="oldest complete turn"):
        planner.plan(batch, after_turn=True, flush=True)


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

    omitted = reducer.reduce(large, force_omit=True, description="视频载荷未长期保存")
    assert omitted.content_mode is ConversationToolResultContentMode.OMITTED
    assert omitted.content == "视频载荷未长期保存"
    with pytest.raises(ValueError, match="only tool_result"):
        reducer.reduce(closed_turn()[0])

