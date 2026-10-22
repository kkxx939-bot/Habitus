"""Conversation 完整轮次提交、语义切段与大工具结果降载测试。"""

from dataclasses import replace
from datetime import UTC, datetime

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
    ConversationAdapterContext,
    ConversationAdapterRegistry,
    ConversationBatch,
    ConversationMessage,
    ConversationMessageRole,
    ConversationMessageSchemaError,
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
    live = chunker.normalize(ConversationBatch("conversation-1", closed_turn(prompt="A" * 30)))
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


def _codex_message(role: str, text: str) -> dict[str, object]:
    kind = "input_text" if role == "user" else "output_text"
    return {"type": "response_item", "payload": {"type": "message", "role": role, "content": [{"type": kind, "text": text}]}}


def test_a_real_shaped_codex_rollout_passes_the_after_turn_check() -> None:
    """Codex（CLI 与 Desktop）在会话开头和中途以用户消息写进上下文；它们不是人说的话、不开一轮，
    否则整批转写在提交边界被拒（2026-10-22 实测：每个会话的第一批都 400、后面的永远排在它后面）。"""

    records = [
        {"type": "session_meta", "payload": {}},
        {"type": "event_msg", "payload": {"type": "task_started", "turn_id": "t1"}},
        _codex_message("developer", "<permissions instructions>…"),
        _codex_message("user", "<environment_context>\n  <cwd>/w</cwd>\n</environment_context>"),
        _codex_message("user", '<external_codex_apps_open_page>{"page_id":null}</external_codex_apps_open_page>'),
        _codex_message("user", "stats.py 的平均分不对，帮我看看"),
        {"type": "response_item", "payload": {"type": "function_call", "call_id": "call_q", "name": "request_user_input_async",
                                              "arguments": '{"questions":[{"title":"周报还没写。","options":["现在就做"]}]}'}},
        {"type": "response_item", "payload": {"type": "function_call_output", "call_id": "call_q", "output": '{"accepted":true}'}},
        _codex_message("assistant", "多减了 1，已修正。"),
        {"type": "event_msg", "payload": {"type": "task_complete"}},
        {"type": "event_msg", "payload": {"type": "task_started", "turn_id": "t2"}},
        _codex_message("user", "<recommended_plugins>…</recommended_plugins>"),
        _codex_message("user", '<external_codex_apps_open_page>{"page_id":null}</external_codex_apps_open_page>'),
        _codex_message(
            "user",
            '<send_user_message_question_reply>\n[{"questionItemId":"[\\"request_user_input_async\\",\\"call_q\\",0]",'
            '"question":"周报还没写。","answer":"现在就做"}]\n</send_user_message_question_reply>',
        ),
        _codex_message("assistant", "收到。"),
    ]
    adapted = ConversationAdapterRegistry.with_builtins().adapt(
        "codex_rollout", records, ConversationAdapterContext("codex-s1", 0, datetime(2026, 10, 22, tzinfo=UTC))
    )
    roles = tuple(item.role for item in adapted.batch.messages)
    assert roles == (
        ConversationMessageRole.PROMPT,
        ConversationMessageRole.TOOL_CALL,
        ConversationMessageRole.TOOL_RESULT,
        ConversationMessageRole.COMPLETION,
        ConversationMessageRole.PROMPT,
        ConversationMessageRole.COMPLETION,
    )
    # Desktop 的作答仍是一条人说的话（提醒出口与会话 lane 都读它）
    assert adapted.batch.messages[4].content.startswith("<send_user_message_question_reply>")
    assert adapted.after_turn
    planner = ConversationRetentionPlanner(config(commit_token_threshold=1), token_estimator=lambda _: 1)
    planner.plan(adapted.batch, after_turn=True)  # 以前在这里抛 "afterTurn requires every turn to end with a final completion"


def _task_started() -> dict[str, object]:
    return {"type": "event_msg", "payload": {"type": "task_started", "turn_id": "t"}}


def _call(call_id: str, name: str) -> list[dict[str, object]]:
    return [
        {"type": "response_item", "payload": {"type": "function_call", "call_id": call_id, "name": name, "arguments": "{}"}},
        {"type": "response_item", "payload": {"type": "function_call_output", "call_id": call_id, "output": "ok"}},
    ]


def test_what_he_says_while_the_agent_is_still_working_stays_in_that_turn() -> None:
    """照他 10-12 一场真实 Desktop 会话的形状：一轮里他作答两次、又补一句、最后叫停，agent 才给最终回复。
    插在中间的四句是同一轮的，不开新一轮；整批过提交边界，切段也不从这几句前面切开。"""

    records = [
        _task_started(),
        _codex_message("user", "<environment_context>…</environment_context>"),
        _codex_message("user", "看一下能不能写个脚本连我的云电脑"),
        _codex_message("assistant", "我先检查客户端和连接方式。"),
        *_call("call_q1", "request_user_input_async"),
        _codex_message("user", '<send_user_message_question_reply>\n[{"questionItemId":"[\\"request_user_input_async\\",\\"call_q1\\",0]","question":"走哪条路？","answer":"云助手"}]\n</send_user_message_question_reply>'),
        *_call("call_js1", "js"),
        _codex_message("user", "你可以先看一下我是账号"),
        *_call("call_js2", "js"),
        _codex_message("user", "还是暂停吧"),
        _codex_message("assistant", "已暂停。"),
        {"type": "event_msg", "payload": {"type": "task_complete"}},
        _task_started(),
        _codex_message("user", "是不是脚本上传了就可以了？"),
        _codex_message("assistant", "上传脚本还不够。"),
    ]
    adapted = ConversationAdapterRegistry.with_builtins().adapt(
        "codex_rollout", records, ConversationAdapterContext("codex-s1", 0, datetime(2026, 10, 12, tzinfo=UTC))
    )
    prompts = [item for item in adapted.batch.messages if item.role is ConversationMessageRole.PROMPT]
    assert [item.mid_turn for item in prompts] == [False, True, True, True, False]
    assert [item.starts_turn for item in prompts] == [True, False, False, False, True]
    planner = ConversationRetentionPlanner(config(commit_token_threshold=1), token_estimator=lambda _: 1)
    planner.plan(adapted.batch, after_turn=True)  # 以前在作答那一句处切出一轮没有回复的，整批被拒
    # 插话前面不是"一轮结束"：最终回复之前的任何位置都不算轮次边界
    final = next(i for i, item in enumerate(adapted.batch.messages) if item.content == "已暂停。")
    for index, item in enumerate(adapted.batch.messages[:final]):
        assert ConversationRetentionPlanner._boundary_kind(adapted.batch.messages, index) != "turn", item.content


def test_a_mid_turn_mark_survives_the_round_trip_and_only_prompts_carry_it() -> None:
    marked = replace(message(0, ConversationMessageRole.PROMPT, "还是暂停吧"), mid_turn=True)
    assert ConversationMessage.from_dict(marked.to_dict()) == marked
    # 没标的消息序列化不变（已有的身份摘要不受影响）
    plain = message(0, ConversationMessageRole.PROMPT, "x")
    assert "mid_turn" not in plain.to_dict()
    with pytest.raises(ConversationMessageSchemaError, match="only a prompt"):
        replace(message(1, ConversationMessageRole.COMPLETION, "y"), mid_turn=True)


def test_a_batch_cut_in_the_middle_of_a_codex_turn_still_knows_his_words_are_mid_turn() -> None:
    """插件会在一轮中途的答复处、压缩前送一批：下一批里没有 task_started，他的插话照样是插在中间的（四路评审确认的 bug）。"""

    second_batch = [
        *_call("call_js1", "js"),
        _codex_message("user", "你可以先看一下我是账号"),
        *_call("call_js2", "js"),
        _codex_message("assistant", "已暂停。"),
    ]
    adapted = ConversationAdapterRegistry.with_builtins().adapt(
        "codex_rollout", second_batch, ConversationAdapterContext("codex-s1", 40, datetime(2026, 10, 12, tzinfo=UTC))
    )
    (spoken,) = [item for item in adapted.batch.messages if item.role is ConversationMessageRole.PROMPT]
    assert spoken.mid_turn and not spoken.starts_turn
    planner = ConversationRetentionPlanner(config(commit_token_threshold=1), token_estimator=lambda _: 1)
    planner.plan(adapted.batch, after_turn=True, leading_continuation=True)  # 以前这里又是那个 400


def test_a_mid_turn_interjection_right_after_a_progress_reply_is_not_a_turn_boundary() -> None:
    """他 10-12 的真实形状：助手中途说一句、他紧接着作答。那句中途答复不是这一轮的结尾，切段不在它后面切。"""

    messages = (
        message(0, ConversationMessageRole.PROMPT, "看一下能不能写个脚本"),
        message(1, ConversationMessageRole.COMPLETION, "我先检查客户端。"),
        replace(message(2, ConversationMessageRole.PROMPT, "（回答助手的提问）走哪条路？ → 云助手"), mid_turn=True),
        message(3, ConversationMessageRole.COMPLETION, "已暂停。"),
    )
    assert ConversationRetentionPlanner._boundary_kind(messages, 1) != "turn"
    assert 1 not in ConversationRetentionPlanner._safe_boundaries(messages)
    assert ConversationRetentionPlanner._boundary_kind(messages, 3) == "turn"


def test_a_long_turn_with_his_interjection_is_sealed_in_pieces_that_know_they_are_mid_turn(tmp_path) -> None:
    """一轮长到要切成几段、中间有他的插话：整批收下；以插话开头的那一段知道自己是从一轮中间开始的。"""

    from datetime import date

    from habitus.infrastructure.store.contracts import PathLock
    from habitus.infrastructure.store.locks import ProcessLocalLockStore
    from habitus.memory.conversation import ConversationAddress, ConversationMessageJournal
    from habitus.memory.workflow import ConversationMemoryEnqueuer
    from habitus.memory.workflow.jobs import MemoryJobStore

    records = [
        _task_started(),
        _codex_message("user", "看一下能不能写个脚本连我的云电脑"),
        *_call("c1", "js"),
        *_call("c2", "js"),
        _codex_message("user", "你可以先看一下我是账号"),
        *_call("c3", "js"),
        *_call("c4", "js"),
        _codex_message("assistant", "已暂停。"),
    ]
    adapted = ConversationAdapterRegistry.with_builtins().adapt(
        "codex_rollout", records, ConversationAdapterContext("k", 0, datetime(2026, 10, 22, tzinfo=UTC))
    )
    lock = PathLock(ProcessLocalLockStore())
    journal = ConversationMessageJournal(tmp_path / "c", lock)
    planner = ConversationRetentionPlanner(
        config(commit_token_threshold=1, keep_recent_turn_count=0, retained_message_token_budget=4, max_segment_messages=3),
        token_estimator=lambda _: 1,
    )
    enqueuer = ConversationMemoryEnqueuer(journal, MemoryJobStore(tmp_path / "w", lock, memory_root=tmp_path / "m"), planner)
    address = ConversationAddress("k", date(2026, 10, 22))
    enqueuer.append_and_maybe_enqueue(address, adapted.batch, after_turn=True)
    segments = journal.list_history(address)
    assert len(segments) > 1
    for segment in segments:
        head = segment.messages[0]
        assert segment.starts_mid_turn == (not head.starts_turn)
    assert any(segment.messages[0].mid_turn and segment.starts_mid_turn for segment in segments)


def test_semantic_cut_points_never_fall_between_a_progress_reply_and_his_interjection() -> None:
    from habitus.memory.conversation.segmentation import _safe_semantic_units

    messages = (
        message(0, ConversationMessageRole.PROMPT, "看一下能不能写个脚本"),
        message(1, ConversationMessageRole.COMPLETION, "我先检查客户端。"),
        replace(message(2, ConversationMessageRole.PROMPT, "还是暂停吧"), mid_turn=True),
        message(3, ConversationMessageRole.COMPLETION, "已暂停。"),
        message(4, ConversationMessageRole.PROMPT, "下一个问题"),
        message(5, ConversationMessageRole.COMPLETION, "好的。"),
    )
    # 中途那句答复（1）后面紧跟的是他插的话：那里不是切点（以前会在 1 后面切开）；其余照旧
    assert [sequence for sequence, _ in _safe_semantic_units(messages)] == [0, 2, 3, 4, 5]
