"""外部 Conversation 协议转换的角色、工具配对和拒绝场景。"""

import json
from datetime import UTC, datetime

import pytest

from habitus.pre.conversation import (
    ConversationAdapterContext,
    ConversationAdapterRegistry,
    ConversationMessageRole,
    ConversationProtocolError,
    ConversationToolResultStatus,
)


def context() -> ConversationAdapterContext:
    return ConversationAdapterContext("conversation-1", 10, datetime(2026, 7, 28, tzinfo=UTC))


def test_openai_chat_preserves_text_tools_and_ignores_system_metadata() -> None:
    registry = ConversationAdapterRegistry.with_builtins()
    payload = {
        "messages": [
            {"role": "system", "content": "不要存为用户事实"},
            {"role": "user", "content": "查天气"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {"name": "get_weather", "arguments": '{"city":"上海"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": "晴，28℃"},
            {"role": "assistant", "content": "上海今天晴。"},
        ]
    }

    result = registry.adapt("openai_chat_completions", payload, context())

    assert result.ignored_items == 1
    assert result.after_turn
    assert tuple(message.role for message in result.batch.messages) == (
        ConversationMessageRole.PROMPT,
        ConversationMessageRole.TOOL_CALL,
        ConversationMessageRole.TOOL_RESULT,
        ConversationMessageRole.COMPLETION,
    )
    assert result.batch.messages[1].tool_name == result.batch.messages[2].tool_name == "get_weather"
    assert result.batch.messages[2].tool_status is ConversationToolResultStatus.UNKNOWN
    assert result.batch.start_sequence == 10
    assert registry.adapt("openai_chat_completions", payload, context()).batch == result.batch


def test_openai_responses_maps_function_items_and_keeps_pending_turn_open() -> None:
    payload = {
        "output": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "搜索"}]},
            {"type": "function_call", "call_id": "call-1", "name": "search", "arguments": "{}"},
        ]
    }
    result = ConversationAdapterRegistry.with_builtins().adapt("openai_responses", payload, context())
    assert not result.after_turn
    assert result.batch.messages[-1].role is ConversationMessageRole.TOOL_CALL


def test_tool_result_requires_explicit_terminal_status_before_it_is_verified() -> None:
    payload = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {"name": "search", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": "完成",
                "status": "completed",
            },
        ]
    }
    result = ConversationAdapterRegistry.with_builtins().adapt(
        "openai_chat_completions",
        payload,
        context(),
    )
    assert result.batch.messages[-1].tool_status is ConversationToolResultStatus.COMPLETED


def test_anthropic_maps_error_tool_result_without_losing_tool_name() -> None:
    payload = {
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "tool-1", "name": "download", "input": {"url": "x"}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "tool-1", "content": "timeout", "is_error": True}],
            },
        ]
    }
    result = ConversationAdapterRegistry.with_builtins().adapt("anthropic_messages", payload, context())
    tool_result = result.batch.messages[-1]
    assert tool_result.tool_name == "download"
    assert tool_result.tool_status is ConversationToolResultStatus.ERROR
    assert not result.after_turn


def test_unmatched_tool_result_and_unknown_protocol_are_rejected() -> None:
    registry = ConversationAdapterRegistry.with_builtins()
    with pytest.raises(ConversationProtocolError, match="matching tool_call"):
        registry.adapt(
            "openai_chat_completions",
            {"messages": [{"role": "tool", "tool_call_id": "missing", "content": "x"}]},
            context(),
        )
    with pytest.raises(ConversationProtocolError, match="unsupported"):
        registry.adapt("unknown", {}, context())


def test_codex_rollout_keeps_function_call_result_and_record_timestamps() -> None:
    records = [
        {"type": "session_meta", "timestamp": "2026-07-28T00:00:00Z", "payload": {}},
        {
            "type": "response_item",
            "timestamp": "2026-07-28T00:00:01Z",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "读取文件"}],
            },
        },
        {
            "type": "response_item",
            "timestamp": "2026-07-28T00:00:02Z",
            "payload": {"type": "function_call", "call_id": "call-1", "name": "read_file", "arguments": "{}"},
        },
        {
            "type": "response_item",
            "timestamp": "2026-07-28T00:00:03Z",
            "payload": {"type": "function_call_output", "call_id": "call-1", "output": "内容"},
        },
        {
            "type": "response_item",
            "timestamp": "2026-07-28T00:00:04Z",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "已读取"}],
            },
        },
    ]
    result = ConversationAdapterRegistry.with_builtins().adapt("codex_rollout", records, context())
    assert result.ignored_items == 1
    assert result.after_turn
    assert result.batch.messages[1].tool_name == result.batch.messages[2].tool_name == "read_file"
    assert result.batch.messages[2].tool_status is ConversationToolResultStatus.UNKNOWN
    assert result.batch.messages[0].occurred_at.isoformat() == "2026-07-28T00:00:01+00:00"


def test_codex_rollout_keeps_custom_tool_call_and_output() -> None:
    records = [
        {
            "type": "response_item",
            "payload": {"type": "custom_tool_call", "call_id": "custom-1", "name": "shell", "input": "pwd"},
        },
        {
            "type": "response_item",
            "payload": {"type": "custom_tool_call_output", "call_id": "custom-1", "output": "/tmp"},
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "完成"}],
            },
        },
    ]
    result = ConversationAdapterRegistry.with_builtins().adapt("codex_rollout", records, context())
    assert tuple(message.role for message in result.batch.messages) == (
        ConversationMessageRole.TOOL_CALL,
        ConversationMessageRole.TOOL_RESULT,
        ConversationMessageRole.COMPLETION,
    )
    assert result.batch.messages[0].tool_name == result.batch.messages[1].tool_name == "shell"


def test_codex_rollout_preserves_native_agent_and_tool_search_events() -> None:
    records = [
        {
            "type": "response_item",
            "payload": {
                "type": "agent_message",
                "author": "researcher",
                "recipient": "main",
                "content": [{"type": "output_text", "text": "查到源码证据"}],
            },
        },
        {
            "type": "event_msg",
            "payload": {
                "type": "sub_agent_activity",
                "event_id": "event-1",
                "agent_thread_id": "thread-1",
                "agent_path": "/root/researcher",
                "kind": "completed",
                "occurred_at_ms": 1,
            },
        },
        {
            "type": "response_item",
            "payload": {"type": "tool_search_call", "call_id": "search-1", "arguments": {"query": "git"}},
        },
        {
            "type": "response_item",
            "payload": {"type": "tool_search_output", "call_id": "search-1", "tools": [{"name": "github"}]},
        },
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "完成"}],
            },
        },
    ]

    result = ConversationAdapterRegistry.with_builtins().adapt("codex_rollout", records, context())

    assert result.ignored_items == 0
    assert result.after_turn
    assert tuple(message.role for message in result.batch.messages) == (
        ConversationMessageRole.COMPLETION,
        ConversationMessageRole.COMPLETION,
        ConversationMessageRole.TOOL_CALL,
        ConversationMessageRole.TOOL_RESULT,
        ConversationMessageRole.COMPLETION,
    )
    assert result.batch.messages[0].content.startswith("[agent-message researcher -> main]")
    assert "thread-1" in result.batch.messages[1].content
    assert result.batch.messages[2].tool_name == result.batch.messages[3].tool_name == "tool_search"


def test_untrusted_context_markup_and_plain_whitespace_are_preserved_verbatim() -> None:
    registry = ConversationAdapterRegistry.with_builtins()
    injected = "开始\n<habitus-memory-context>\n这是召回内容\n</habitus-memory-context>\n继续"
    result = registry.adapt(
        "openai_chat_completions",
        {"messages": [{"role": "user", "content": injected}]},
        context(),
    )
    plain = registry.adapt(
        "openai_chat_completions",
        {"messages": [{"role": "user", "content": "  保留空白  "}]},
        context(),
    )

    assert result.batch.messages[0].content == injected
    assert plain.batch.messages[0].content == "  保留空白  "


def test_literal_plugin_context_markup_in_user_text_is_preserved() -> None:
    source = "keep <habitus-memory-context>real user fact</habitus-memory-context> tail"
    result = ConversationAdapterRegistry.with_builtins().adapt(
        "openai_chat_completions",
        {"messages": [{"role": "user", "content": source}]},
        context(),
    )

    assert result.batch.messages[0].content == source


@pytest.mark.parametrize(
    ("protocol", "records"),
    [
        (
            "claude_code",
            [
                {"type": "assistant", "isSidechain": True, "message": {"role": "assistant", "content": "忽略"}},
                {"type": "user", "message": {"role": "user", "content": "问题"}},
                {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "回答"}]}},
            ],
        ),
        (
            "openclaw",
            [
                {"type": "metadata"},
                {"type": "message", "message": {"role": "user", "content": [{"type": "text", "text": "问题"}]}},
                {"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "回答"}]}},
            ],
        ),
    ],
)
def test_agent_harness_records_preserve_completed_text_turns(protocol: str, records: object) -> None:
    result = ConversationAdapterRegistry.with_builtins().adapt(protocol, records, context())
    assert result.ignored_items == 1
    assert result.after_turn
    assert tuple(item.role for item in result.batch.messages) == (
        ConversationMessageRole.PROMPT,
        ConversationMessageRole.COMPLETION,
    )


def test_opencode_messages_keep_typed_text_finished_tools_and_question_answers() -> None:
    """OpenCode：合成段（我们注入的上下文、它自己读文件的回执）不是他说的话；没结束的工具不收；提问工具的作答按
    ``"<题目>"="<作答>"`` 写成结果，与 Claude Code 同形（方案 36 第四节）。"""

    question = {"questions": [{"question": "周四了，周报还没写。", "header": "提醒", "options": [{"label": "现在就做", "description": ""}]}]}
    records = [
        {
            "info": {"id": "msg_1", "role": "user", "sessionID": "ses_1", "time": {"created": 1792000000000}},
            "parts": [
                {"type": "text", "text": "<habitus-memory-context receipt=\"x\">…</habitus-memory-context>", "synthetic": True},
                {"type": "text", "text": "帮我看看这个报错"},
            ],
        },
        {
            "info": {"id": "msg_2", "role": "assistant", "sessionID": "ses_1", "time": {"created": 1792000001000, "completed": 1792000060000}},
            "parts": [
                {"type": "step-start"},
                {"type": "reasoning", "text": "想一想"},
                {
                    "type": "tool", "callID": "call_q", "tool": "question",
                    "state": {
                        "status": "completed", "input": question, "title": "",
                        "output": 'User has answered your questions: "周四了，周报还没写。"="这次不用". You can now continue.',
                        "metadata": {"answers": [["这次不用"]]},
                        "time": {"start": 1792000002000, "end": 1792000010000},
                    },
                },
                {
                    "type": "tool", "callID": "call_b", "tool": "bash",
                    "state": {"status": "error", "input": {"command": "pytest"}, "error": "exit 1", "time": {"start": 1792000011000, "end": 1792000012000}},
                },
                {"type": "tool", "callID": "call_r", "tool": "read", "state": {"status": "running", "input": {}, "time": {"start": 1792000013000}}},
                {"type": "text", "text": "看完了，是依赖版本不对。", "time": {"start": 1792000050000, "end": 1792000060000}},
            ],
        },
    ]
    result = ConversationAdapterRegistry.with_builtins().adapt("opencode", {"records": records}, context())
    messages = result.batch.messages
    assert [item.role for item in messages] == [
        ConversationMessageRole.PROMPT,
        ConversationMessageRole.TOOL_CALL,
        ConversationMessageRole.TOOL_RESULT,
        ConversationMessageRole.TOOL_CALL,
        ConversationMessageRole.TOOL_RESULT,
        ConversationMessageRole.COMPLETION,
    ]
    assert messages[0].content == "帮我看看这个报错"
    assert messages[1].tool_name == "question" and messages[1].tool_call_id == "call_q"
    assert messages[2].content == '"周四了，周报还没写。"="这次不用"'
    assert messages[4].tool_status is ConversationToolResultStatus.ERROR
    assert messages[0].occurred_at == datetime.fromtimestamp(1792000000, tz=UTC)
    assert messages[5].occurred_at == datetime.fromtimestamp(1792000060, tz=UTC)
    assert result.after_turn and result.ignored_items == 4  # 合成段、步骤、思考、没结束的工具


def test_opencode_compaction_summaries_are_not_agent_replies() -> None:
    """压缩写的摘要（summary 为真）不是 agent 对他说的话。"""

    records = [
        {"info": {"id": "u", "role": "user", "time": {"created": 1792000000000}}, "parts": [{"type": "compaction", "auto": True}]},
        {
            "info": {"id": "s", "role": "assistant", "summary": True, "mode": "compaction", "time": {"created": 1792000001000, "completed": 1792000002000}},
            "parts": [{"type": "text", "text": "## Summary\n之前在改登录页"}],
        },
        {"info": {"id": "u2", "role": "user", "time": {"created": 1792000003000}}, "parts": [{"type": "text", "text": "继续改"}]},
    ]
    result = ConversationAdapterRegistry.with_builtins().adapt("opencode", {"records": records}, context())
    assert [item.content for item in result.batch.messages] == ["继续改"]


def test_opencode_timestamps_out_of_range_are_protocol_errors() -> None:
    """坏时间戳按协议错误拒收（400），不当成服务内部错误（500）。"""

    records = [{"info": {"id": "m", "role": "user", "time": {"created": 10**20}}, "parts": [{"type": "text", "text": "x"}]}]
    with pytest.raises(ConversationProtocolError, match="out of range"):
        ConversationAdapterRegistry.with_builtins().adapt("opencode", {"records": records}, context())


def test_each_harness_protocol_declares_its_host_and_question_tools() -> None:
    """提问工具与宿主名由各协议的适配器声明（方案 36 第八节第 1 条）：通用 SDK 协议两样都没有，接新宿主只加一个适配器。"""

    from habitus.pre.conversation import QuestionTool

    registry = ConversationAdapterRegistry.with_builtins()
    assert registry.host("codex_rollout") == "codex" and registry.host("CLAUDE_CODE") == "claude-code"
    assert registry.host("openai_chat_completions") is None and registry.host("nope") is None
    assert [tool.tool for tool in registry.question_tools("codex_rollout")] == ["request_user_input", "request_user_input_async"]
    assert registry.question_tools("opencode") == (QuestionTool("opencode", "question", "question"),)
    assert registry.question_tools("openai_chat_completions") == ()
    assert {tool.host for tool in registry.question_tools()} == {"codex", "claude-code", "opencode", "pi", "zcode", "dsh"}
    with pytest.raises(ValueError):
        QuestionTool("codex", " ", "title")


def test_pi_session_entries_keep_spoken_text_tools_and_the_ask_user_answer() -> None:
    """pi 的会话 JSONL：只收 message 记录；人说的话、答复、toolCall 块与 toolResult；插件注入的 custom、思考、会话头、
    模型变更与压缩都不收；ask_user 的结果已是 ``"<题目>"="<作答>"``。"""

    records = [
        {"type": "session", "version": 3, "id": "s1", "timestamp": "2026-10-22T07:00:00Z", "cwd": "/w"},
        {"type": "model_change", "id": "a", "parentId": None, "timestamp": "2026-10-22T07:00:00Z", "provider": "x", "modelId": "y"},
        {"type": "message", "id": "b", "parentId": "a", "timestamp": "2026-10-22T07:00:01Z",
         "message": {"role": "user", "content": [{"type": "text", "text": "帮我看看这个报错"}], "timestamp": 1792998001000}},
        {"type": "message", "id": "c", "parentId": "b", "timestamp": "2026-10-22T07:00:01Z",
         "message": {"role": "custom", "customType": "habitus-memory", "content": "<habitus-memory-context>…", "display": False, "timestamp": 1792998001000}},
        {"type": "message", "id": "d", "parentId": "c", "timestamp": "2026-10-22T07:00:05Z",
         "message": {"role": "assistant", "stopReason": "toolUse", "timestamp": 1792998002000, "content": [
             {"type": "thinking", "thinking": "想一想"},
             {"type": "toolCall", "id": "call_q", "name": "ask_user", "arguments": {"questions": [{"question": "周报还没写。", "options": [{"label": "现在就做"}]}]}},
         ]}},
        {"type": "message", "id": "e", "parentId": "d", "timestamp": "2026-10-22T07:00:09Z",
         "message": {"role": "toolResult", "toolCallId": "call_q", "toolName": "ask_user", "isError": False, "timestamp": 1792998009000,
                     "content": [{"type": "text", "text": '"周报还没写。"="这次不用"'}]}},
        {"type": "compaction", "id": "f", "parentId": "e", "timestamp": "2026-10-22T07:00:10Z", "summary": "…", "firstKeptEntryId": "b"},
        {"type": "message", "id": "g", "parentId": "f", "timestamp": "2026-10-22T07:00:20Z",
         "message": {"role": "assistant", "stopReason": "stop", "timestamp": 1792998011000, "content": [{"type": "text", "text": "看完了，是依赖版本不对。"}]}},
    ]
    result = ConversationAdapterRegistry.with_builtins().adapt("pi", {"records": records}, context())
    messages = result.batch.messages
    assert [item.role for item in messages] == [
        ConversationMessageRole.PROMPT,
        ConversationMessageRole.TOOL_CALL,
        ConversationMessageRole.TOOL_RESULT,
        ConversationMessageRole.COMPLETION,
    ]
    assert messages[0].content == "帮我看看这个报错"
    assert (messages[1].tool_name, messages[1].tool_call_id) == ("ask_user", "call_q")
    assert messages[2].content == '"周报还没写。"="这次不用"'
    assert messages[0].occurred_at == datetime.fromtimestamp(1792998001, tz=UTC)
    assert result.after_turn and result.ignored_items == 5  # 会话头、模型变更、注入的 custom、思考、压缩
    registry = ConversationAdapterRegistry.with_builtins()
    assert registry.host("pi") == "pi" and [tool.tool for tool in registry.question_tools("pi")] == ["ask_user"]


def test_zcode_messages_read_like_opencode_with_ask_user_question_and_no_hook_context_line() -> None:
    """ZCode 的消息与 OpenCode 同形；AskUserQuestion 的作答在结果文字里；ZCode 给注入加的那行 hook 上下文前缀不是他说的话。"""

    records = [
        {
            "info": {"id": "m1", "role": "user", "time": {"created": 1792000000000}},
            "parts": [
                {"type": "text", "text": "UserPromptSubmit hook additional context: \n#1"},
                {"type": "text", "text": "帮我看看这个报错"},
            ],
        },
        {
            "info": {"id": "m2", "role": "assistant", "time": {"created": 1792000001000, "completed": 1792000060000}},
            "parts": [
                {
                    "type": "tool", "callID": "call_q", "tool": "AskUserQuestion",
                    "state": {
                        "status": "completed", "input": {"questions": [{"question": "周报还没写。"}]}, "metadata": {},
                        "output": 'User has answered your questions: "周报还没写。"="现在就做". You can now continue with the user\'s answers in mind.',
                        "time": {"start": 1792000002000, "end": 1792000010000},
                    },
                },
                {"type": "text", "text": "好，现在写。", "time": {"start": 1792000050000, "end": 1792000060000}},
            ],
        },
    ]
    registry = ConversationAdapterRegistry.with_builtins()
    messages = registry.adapt("zcode", {"records": records}, context()).batch.messages
    assert [item.content for item in messages if item.role is ConversationMessageRole.PROMPT] == ["帮我看看这个报错"]
    assert '"周报还没写。"="现在就做"' in messages[2].content
    assert registry.host("zcode") == "zcode" and [tool.tool for tool in registry.question_tools("zcode")] == ["AskUserQuestion"]


def test_deepseek_harness_events_keep_human_text_tools_and_structured_question_answers() -> None:
    """DeepSeek Harness 的日志事件：只有人亲手输入（source.kind user）算他说的话，hook 桥注入的上下文不算；tool-call 块是调用、
    tool/call 事件重复不收；ask_user_question 的结构化作答按题目 id 配回成 ``"<题目>"="<作答>"``。"""

    question_args = json.dumps({"questions": [{"id": "q1", "question": "周报还没写。", "options": [{"label": "现在就做"}]}]}, ensure_ascii=False)
    records = [
        {"id": "s1", "createdAt": "2026-10-22T07:00:00Z", "cwd": "/w"},
        {"type": "turn/start", "seq": 1, "time": 1792998000000, "data": {"turn": 1}},
        {"type": "user/message", "seq": 2, "time": 1792998001000, "surfaceOp": "append",
         "data": {"id": "u1", "role": "user", "source": {"kind": "user"}, "content": [{"type": "text", "text": "帮我看看这个报错"}]}},
        {"type": "user/message", "seq": 3, "time": 1792998001000, "surfaceOp": "append",
         "data": {"id": "u2", "role": "user", "source": {"kind": "hooks-claude-code"}, "content": [{"type": "text", "text": "<habitus-memory-context>…"}]}},
        {"type": "assistant/message", "seq": 4, "time": 1792998002000, "surfaceOp": "append",
         "data": {"turn": 1, "step": 1, "stream": [], "message": {"id": "a1", "role": "assistant", "source": {"kind": "model"}, "content": [
             {"type": "reasoning", "text": "想一想"},
             {"type": "tool-call", "id": "call_q", "name": "ask_user_question", "arguments": question_args},
         ]}}},
        {"type": "tool/call", "seq": 5, "time": 1792998002000, "data": {"turn": 1, "step": 1, "callId": "call_q", "name": "ask_user_question", "arguments": question_args}},
        {"type": "tool/result", "seq": 6, "time": 1792998009000, "surfaceOp": "append",
         "data": {"turn": 1, "step": 1, "message": {"id": "t1", "role": "tool", "toolCallId": "call_q", "source": {"kind": "tool"},
                  "content": [{"type": "text", "text": json.dumps({"answers": [{"id": "q1", "selected": ["这次不用"]}]}, ensure_ascii=False)}]}}},
        {"type": "assistant/message", "seq": 7, "time": 1792998011000, "surfaceOp": "append",
         "data": {"turn": 1, "step": 2, "stream": [], "message": {"id": "a2", "role": "assistant", "source": {"kind": "model"}, "content": [{"type": "text", "text": "好，那先不写。"}]}}},
        {"type": "turn/end", "seq": 8, "time": 1792998011000, "data": {"turn": 1, "reason": "completed"}},
    ]
    registry = ConversationAdapterRegistry.with_builtins()
    result = registry.adapt("dsh", {"records": records}, context())
    messages = result.batch.messages
    assert [item.role for item in messages] == [
        ConversationMessageRole.PROMPT,
        ConversationMessageRole.TOOL_CALL,
        ConversationMessageRole.TOOL_RESULT,
        ConversationMessageRole.COMPLETION,
    ]
    assert messages[0].content == "帮我看看这个报错"
    assert messages[1].tool_name == "ask_user_question"
    assert messages[2].content == '"周报还没写。"="这次不用"' and messages[2].tool_name == "ask_user_question"
    assert messages[0].occurred_at == datetime.fromtimestamp(1792998001, tz=UTC)
    assert result.after_turn and result.ignored_items == 6  # 会话头、turn/start、注入的上下文、思考、tool/call、turn/end
    assert registry.host("dsh") == "dsh" and [tool.tool for tool in registry.question_tools("dsh")] == ["ask_user_question"]
