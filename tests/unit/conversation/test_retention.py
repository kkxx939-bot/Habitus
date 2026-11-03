"""Conversation 完整轮次提交、语义切段与大工具结果降载测试。"""

import hashlib
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from habitus.foundation.integrity import canonical_json
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


def test_a_strong_semantic_hint_never_beats_a_complete_turn_boundary_in_the_window() -> None:
    chunker = ConversationMessageChunker(max_message_tokens=1_000, token_estimator=lambda _: 10)
    live = chunker.normalize(
        ConversationBatch(
            "conversation-1",
            tuple(item for turn in range(6) for item in closed_turn(start_sequence=2 * turn, prompt=f"第{turn}轮")),
        )
    )
    planner = ConversationRetentionPlanner(
        config(
            keep_recent_turn_count=0,
            max_segment_tokens=100,
            retained_message_token_budget=1_000,
        ),
        token_estimator=lambda _: 10,
    )
    hints = ConversationBoundaryHints(live.digest, (ConversationSemanticBoundary(8, 2.0),), "embedding-v1")

    plan = planner.plan(live, after_turn=True, flush=True, boundary_hints=hints)

    assert plan.through_sequence == 9


def test_a_boundary_inside_the_sixty_percent_window_beats_a_better_structure_before_it() -> None:
    chunker = ConversationMessageChunker(max_message_tokens=1_000, token_estimator=lambda _: 10)
    messages = [*closed_turn(start_sequence=0, prompt="短轮"), message(2, ConversationMessageRole.PROMPT, "长轮")]
    sequence = 3
    for index in range(5):
        messages.append(
            message(
                sequence,
                ConversationMessageRole.TOOL_CALL,
                {"i": index},
                tool_call_id=f"call-{index}",
                tool_name="read",
            )
        )
        messages.append(
            message(
                sequence + 1,
                ConversationMessageRole.TOOL_RESULT,
                "ok",
                tool_call_id=f"call-{index}",
                tool_name="read",
                tool_status=ConversationToolResultStatus.COMPLETED,
                content_mode=ConversationToolResultContentMode.INLINE,
            )
        )
        sequence += 2
    messages.append(message(sequence, ConversationMessageRole.COMPLETION, "完成"))
    live = chunker.normalize(ConversationBatch("conversation-1", tuple(messages)))
    planner = ConversationRetentionPlanner(
        config(
            keep_recent_turn_count=0,
            max_segment_tokens=100,
            retained_message_token_budget=1_000,
        ),
        token_estimator=lambda _: 10,
    )

    plan = planner.plan(live, after_turn=True, flush=True)

    assert plan.through_sequence == 8


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
    from habitus.memory.conversation import ConversationAddress, ConversationMessageJournal
    from habitus.memory.workflow import ConversationMemoryEnqueuer
    from habitus.memory.workflow.jobs import MemoryJobStore
    from tests.lock_store import ProcessLocalLockStore

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


def _exchange_enqueuer(tmp_path, **overrides: object):  # type: ignore[no-untyped-def]
    from datetime import date

    from habitus.infrastructure.store.contracts import PathLock
    from habitus.memory.conversation import ConversationAddress, ConversationMessageJournal
    from habitus.memory.workflow import ConversationMemoryEnqueuer
    from habitus.memory.workflow.jobs import MemoryJobStore
    from tests.lock_store import ProcessLocalLockStore

    values: dict[str, object] = {
        "commit_token_threshold": 1,
        "keep_recent_turn_count": 0,
        "retained_message_token_budget": 10_000,
        "max_segment_tokens": 2_000,
        "max_inline_tool_result_bytes": 64 * 1024,
        "max_tool_result_summary_chars": 1_000,
    }
    values.update(overrides)
    lock = PathLock(ProcessLocalLockStore())
    journal = ConversationMessageJournal(tmp_path / "c", lock)
    planner = ConversationRetentionPlanner(config(**values))
    enqueuer = ConversationMemoryEnqueuer(journal, MemoryJobStore(tmp_path / "w", lock, memory_root=tmp_path / "m"), planner)
    return enqueuer, journal, ConversationAddress("conversation-1", date(2026, 10, 26))


def _result(sequence: int, call_id: str, content: str) -> ConversationMessage:
    return message(
        sequence,
        ConversationMessageRole.TOOL_RESULT,
        content,
        tool_call_id=call_id,
        tool_name="Read",
        tool_status=ConversationToolResultStatus.COMPLETED,
        content_mode=ConversationToolResultContentMode.INLINE,
    )


def test_two_parallel_reads_that_only_together_exceed_a_segment_are_fitted_not_stuck(tmp_path) -> None:
    """真实形状（05c8b451 第 0 批）：两个并行 Read，单条都不超一段，合起来超；整组抽到放得下，会话照常封存。"""

    enqueuer, journal, address = _exchange_enqueuer(tmp_path)
    small = "".join(f"import module_{index}\n" for index in range(180))
    large = "".join(f"line {index} ok\n" for index in range(420))
    turn = (
        message(0, ConversationMessageRole.PROMPT, "看一下这两个文件"),
        message(1, ConversationMessageRole.TOOL_CALL, {"file_path": "/a.py"}, tool_call_id="r1", tool_name="Read"),
        message(2, ConversationMessageRole.TOOL_CALL, {"file_path": "/b.log"}, tool_call_id="r2", tool_name="Read"),
        _result(3, "r1", small),
        _result(4, "r2", large),
        message(5, ConversationMessageRole.COMPLETION, "看完了。"),
    )
    estimate = enqueuer.retention_planner.token_estimator
    assert estimate(turn[3]) < 2_000 and estimate(turn[4]) < 2_000
    assert sum(estimate(item) for item in turn[1:5]) > 2_000

    enqueuer.append_and_maybe_enqueue(address, ConversationBatch("conversation-1", turn), after_turn=True)

    assert journal.read_live(address) is None
    stored = {item.sequence: item for segment in journal.list_history(address) for item in segment.messages}
    assert stored[3].content_mode is ConversationToolResultContentMode.INLINE
    assert stored[4].content_mode is ConversationToolResultContentMode.SUMMARIZED
    assert "工具结果已压缩" in stored[4].content
    for segment in journal.list_history(address):
        calls = {item.tool_call_id for item in segment.messages if item.role is ConversationMessageRole.TOOL_CALL}
        results = {item.tool_call_id for item in segment.messages if item.role is ConversationMessageRole.TOOL_RESULT}
        assert calls == results


def test_one_write_whose_arguments_exceed_a_segment_keeps_the_path_and_seals(tmp_path) -> None:
    """段 3 C1 形状：一次 Write 写入约 9,200 字，参数单独就超一段；路径原样留、内容抽取，记原参数大小与 sha256。"""

    enqueuer, journal, address = _exchange_enqueuer(tmp_path, max_segment_tokens=12_000, max_tool_result_summary_chars=4_000)
    document = "这是一段很长的中文设计文档内容，用来写入文件。" * 400
    arguments = {"file_path": "docs/plan.md", "content": document}
    turn = (
        message(0, ConversationMessageRole.PROMPT, "把方案写进 docs/plan.md"),
        message(1, ConversationMessageRole.TOOL_CALL, arguments, tool_call_id="w1", tool_name="Write"),
        replace(_result(2, "w1", "File created successfully"), tool_name="Write"),
        message(3, ConversationMessageRole.COMPLETION, "已写入。"),
    )
    assert enqueuer.retention_planner.token_estimator(turn[1]) > 12_000

    enqueuer.append_and_maybe_enqueue(address, ConversationBatch("conversation-1", turn), after_turn=True)

    assert journal.read_live(address) is None
    call = journal.list_history(address)[0].messages[1]
    assert call.content_mode is ConversationToolResultContentMode.SUMMARIZED
    assert call.content["file_path"] == "docs/plan.md"
    assert call.content["content"].startswith("工具调用参数已压缩（关键信息抽取，未调用模型）：Write 调用的参数 content")
    assert "省略了" in call.content["content"]
    encoded = canonical_json(arguments).encode("utf-8")
    assert call.original_size_bytes == len(encoded)
    assert call.original_sha256 == hashlib.sha256(encoded).hexdigest()


def test_an_exchange_that_already_fits_a_segment_keeps_its_arguments_and_results_verbatim(tmp_path) -> None:
    enqueuer, journal, address = _exchange_enqueuer(tmp_path)
    turn = tool_turn()

    enqueuer.append_and_maybe_enqueue(address, ConversationBatch("conversation-1", turn), after_turn=True)

    stored = journal.list_history(address)[0].messages
    assert stored == turn


def test_a_summarized_tool_call_must_carry_its_original_size_and_digest() -> None:
    call = message(1, ConversationMessageRole.TOOL_CALL, {"path": "."}, tool_call_id="c", tool_name="t")
    summarized = ConversationToolResultReducer(config()).reduce_tool_call(call)
    assert ConversationMessage.from_dict(summarized.to_dict()) == summarized
    with pytest.raises(ConversationMessageSchemaError, match="original-argument size"):
        replace(summarized, original_sha256=None)
    with pytest.raises(ConversationMessageSchemaError, match="verbatim tool_call"):
        replace(call, original_size_bytes=3)
    with pytest.raises(ConversationMessageSchemaError, match="verbatim or summarized"):
        replace(summarized, content_mode=ConversationToolResultContentMode.OMITTED)


# ── 切不开的工具交换抽到放得下：对半缩、只换更小的、从原文重抽、下限（N1 的防线，U22） ──────────────────────


def _parallel_exchange(count: int, result: str, *, tool: str = "Read") -> list[ConversationMessage]:
    """一句人话、``count`` 个并行调用、各自的结果、一句答复：调用与结果合起来是一组切不开的工具交换。"""

    turn = [message(0, ConversationMessageRole.PROMPT, "把这些都看一下")]
    turn += [
        message(1 + index, ConversationMessageRole.TOOL_CALL, {"file_path": f"/repo/pkg/module_{index}.py"},
                tool_call_id=f"c{index}", tool_name=tool)
        for index in range(count)
    ]
    turn += [
        replace(_result(1 + count + index, f"c{index}", result.replace("{k}", str(index))), tool_name=tool)
        for index in range(count)
    ]
    turn.append(message(1 + 2 * count, ConversationMessageRole.COMPLETION, "看完了。"))
    return turn


def _stored(journal, address) -> list[ConversationMessage]:  # type: ignore[no-untyped-def]
    return [item for segment in journal.list_history(address) for item in segment.messages]


def test_twelve_parallel_reads_are_fitted_by_halving_the_excerpts_and_keep_small_calls_verbatim(tmp_path) -> None:
    """12 个并行 Read、每个约 2,800 token（段上限 12,000、摘录 4,000 字）：每份都抽成 4,000 字一轮还放不下，摘录对半缩到
    2,000 字再抽，放下了就停，会话照常封存；小调用参数抽了反而更大，原样留着。"""

    enqueuer, journal, address = _exchange_enqueuer(tmp_path, max_segment_tokens=12_000, max_tool_result_summary_chars=4_000)
    body = "".join(f"line {line} of module {{k}}: value = compute({line})\n" for line in range(250))
    turn = _parallel_exchange(12, body)

    enqueuer.append_and_maybe_enqueue(address, ConversationBatch("conversation-1", tuple(turn)), after_turn=True)

    assert journal.read_live(address) is None
    stored = _stored(journal, address)
    results = [item for item in stored if item.role is ConversationMessageRole.TOOL_RESULT]
    assert {item.content_mode for item in results} == {ConversationToolResultContentMode.SUMMARIZED}
    assert min(len(item.content) for item in results) <= 2_000  # 缩过一轮
    calls = [item for item in stored if item.role is ConversationMessageRole.TOOL_CALL]
    assert [(item.content, item.content_mode) for item in calls] == [
        ({"file_path": f"/repo/pkg/module_{index}.py"}, None) for index in range(12)
    ]


def test_a_result_asked_to_be_omitted_stays_omitted_while_its_group_is_fitted(tmp_path) -> None:
    """同一组里有一份被要求省略的结果（不可长期保存）：整组抽到放得下时它仍只有说明文字，原文不以摘录落库。"""

    enqueuer, journal, address = _exchange_enqueuer(tmp_path, max_segment_tokens=12_000, max_tool_result_summary_chars=4_000)
    body = "".join(f"line {line} of module {{k}}: value = compute({line})\n" for line in range(250))
    turn = _parallel_exchange(12, body)
    turn[24] = replace(_result(24, "c11", "SECRET_TOKEN=abcd1234 " * 400), tool_name="Read")

    enqueuer.append_and_maybe_enqueue(
        address, ConversationBatch("conversation-1", tuple(turn)), after_turn=True, omit_tool_call_ids=frozenset({"c11"})
    )

    stored = _stored(journal, address)
    omitted = next(item for item in stored if item.role is ConversationMessageRole.TOOL_RESULT and item.tool_call_id == "c11")
    assert omitted.content_mode is ConversationToolResultContentMode.OMITTED
    assert not any("SECRET_TOKEN" in canonical_json(item.content) for item in stored)


def test_results_over_the_inline_bound_are_re_extracted_from_the_original_to_fit(tmp_path) -> None:
    """14 份约 80 KB 的结果：进来时先各降成 4,000 字摘录；整组放不下时从原文重抽成 2,000 字的摘录，放得进一段。
    从已降载的摘录再抽只能抽回原样（约 4,000 字），整组永远放不下。"""

    enqueuer, journal, address = _exchange_enqueuer(
        tmp_path, max_segment_tokens=12_000, max_tool_result_summary_chars=4_000, max_inline_tool_result_bytes=64 * 1024
    )
    body = "".join(f"2026-10-01 log{{k}} line {line} status ok detail {line * 7}\n" for line in range(1_700))
    turn = _parallel_exchange(14, body, tool="Bash")
    assert len(turn[15].content.encode()) > 64 * 1024

    enqueuer.append_and_maybe_enqueue(address, ConversationBatch("conversation-1", tuple(turn)), after_turn=True)

    assert journal.read_live(address) is None
    results = [item for item in _stored(journal, address) if item.role is ConversationMessageRole.TOOL_RESULT]
    assert len(results) == 14 and min(len(item.content) for item in results) <= 2_000


def test_halving_stops_at_the_excerpt_floor_when_the_group_still_cannot_fit(tmp_path) -> None:
    """48 份结果连 128 字的摘录都放不进一段：对半缩到 128 字的下限就停（不无限缩下去），留给规划器报错。"""

    enqueuer, journal, address = _exchange_enqueuer(tmp_path, max_segment_tokens=2_000, max_tool_result_summary_chars=4_000)
    body = "".join(f"line {line} of module {{k}}: value = compute({line})\n" for line in range(200))
    turn = _parallel_exchange(48, body)

    enqueuer.append(address, ConversationBatch("conversation-1", tuple(turn)))

    live = journal.read_live(address)
    assert live is not None
    results = [item for item in live.messages if item.role is ConversationMessageRole.TOOL_RESULT]
    at_floor = [
        enqueuer.tool_result_reducer.reduce(original, force_summarize=True, maximum=128).content
        for original in turn
        if original.role is ConversationMessageRole.TOOL_RESULT
    ]
    assert [item.content for item in results] == at_floor
    with pytest.raises(ConversationRetentionError, match="atomic tool exchange"):
        enqueuer.enqueue_ready_segments(address, after_turn=True, flush=True)


# ── 超长单行的拆块：每块只估算 O(log 单元数) 次（U21） ───────────────────────────────────────────────────


def _prompt(text: str) -> ConversationBatch:
    return ConversationBatch("conversation-1", (message(0, ConversationMessageRole.PROMPT, text),))


def test_a_long_single_line_is_chunked_without_re_estimating_the_block_per_unit() -> None:
    """80 KB 没有换行、空格分隔的单行（32,000 个小单元）：原样拼回、两块；估算器调用次数有界（以前每加一个单元重估
    整块，这一条要跑一百多秒）。"""

    from habitus.memory.conversation.retention import estimate_conversation_message_tokens

    calls = [0]

    def counting(item: ConversationMessage) -> int:
        calls[0] += 1
        return estimate_conversation_message_tokens(item)

    text = "word " * 16_000
    chunker = ConversationMessageChunker(max_message_tokens=12_000, token_estimator=counting)

    parts = chunker.normalize(_prompt(text)).messages

    assert "".join(str(item.content) for item in parts) == text
    assert len(parts) == 2
    assert calls[0] < 200


@pytest.mark.parametrize(
    ("text", "limit"),
    [
        ("第一段。\n\n第二段很长很长。\n\n" + "x" * 80 + "\n短\n" + "句子。" * 30, 24),
        ("word " * 300, 40),
        ("```code\nx = 1\n```\n\n" + "长" * 90 + " a b c " * 20, 30),
        ("a" * 200 + " " + "b" * 3, 25),
    ],
)
def test_chunk_boundaries_are_the_greedy_ones(text: str, limit: int) -> None:
    """块的切法与逐单元贪心并块（每块并到再多一个就放不下）一致：同样的输入、同样的块。"""

    estimate = lambda item: len(item.content) if isinstance(item.content, str) else 1  # noqa: E731

    class _Greedy(ConversationMessageChunker):
        def _greedy_blocks(self, source, units):  # type: ignore[no-untyped-def]
            blocks, current = [], ""
            for unit in units:
                if current and not self._fits(source, current + unit):
                    blocks.append(current)
                    current = unit
                else:
                    current += unit
            return [*blocks, current] if current else blocks

    expected = _Greedy(max_message_tokens=limit, token_estimator=estimate).normalize(_prompt(text)).messages
    actual = ConversationMessageChunker(max_message_tokens=limit, token_estimator=estimate).normalize(_prompt(text)).messages

    assert [item.content for item in actual] == [item.content for item in expected]
    assert len(actual) > 1


@pytest.mark.parametrize(
    ("character", "cost"),
    [
        ("汉", 1.5),  # 中日韩统一表意文字
        ("𠀀", 1.5),  # 扩展 B 区表意文字（基本多文种平面以外，东亚宽字符）
        ("か", 1.5),  # 平假名
        ("カ", 1.5),  # 片假名
        ("한", 1.5),  # 韩文音节
        ("Ａ", 1.5),  # 全角拉丁字母
        ("，", 1.5),  # 全角逗号
        ("。", 1.5),  # 中文句号
        ("😀", 1.5),  # emoji（东亚宽字符）
        ("𝐀", 2.0),  # 基本多文种平面以外、不是东亚宽字符
        ("a", 0.25),
        ("7", 0.25),
        (" ", 0.25),
        ("—", 0.25),  # 东亚宽度不定（A）的破折号按窄字符算
    ],
)
def test_token_estimate_charges_each_character_by_east_asian_width(character: str, cost: float) -> None:
    from habitus.memory.conversation.retention import estimate_conversation_message_tokens

    repeat = 40

    def estimate(text: str) -> int:
        return estimate_conversation_message_tokens(message(0, ConversationMessageRole.PROMPT, text))

    assert estimate("x" + character * repeat) - estimate("x") == cost * repeat
