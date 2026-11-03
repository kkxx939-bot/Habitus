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
                {"type": "user", "origin": {"kind": "human"}, "message": {"role": "user", "content": "问题"}},
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


# ── 宿主时刻倒退（U1）：先后以转写顺序为准，时刻夹成不降 ────────────────────────────────────────────

_T = 1_790_000_000_000  # 毫秒


def _opencode_user(text: str = "查一下") -> dict:
    return {"info": {"role": "user", "time": {"created": _T}}, "parts": [{"type": "text", "text": text, "time": {"start": _T, "end": _T}}]}


def _opencode_tool(call_id: str, start: int, end: int) -> dict:
    return {
        "type": "tool", "callID": call_id, "tool": "WebSearch",
        "state": {"status": "completed", "input": {"q": call_id}, "output": f"r-{call_id}", "time": {"start": _T + start, "end": _T + end}},
    }


@pytest.mark.parametrize("protocol", ["opencode", "zcode"])
def test_parallel_opencode_tools_and_a_text_part_that_ends_earlier_keep_transcript_order(protocol: str) -> None:
    """并行的两个工具（调用 1 [100, 5000]、调用 2 [200, 4000]）与"工具结束晚于其后文字段"的真实形状：宿主给的时刻倒退，
    不再整批拒收；时刻夹成不降，序号与调用—结果的配对按转写顺序不变。"""

    parallel = {
        "info": {"role": "assistant", "time": {"created": _T + 50}},
        "parts": [
            _opencode_tool("a", 100, 5000),
            _opencode_tool("b", 200, 4000),
            {"type": "text", "text": "好了", "time": {"start": _T + 5100, "end": _T + 5200}},
        ],
    }
    later_text = {
        "info": {"role": "assistant", "time": {"created": _T + 6000}},
        "parts": [
            _opencode_tool("c", 6694, 6738),
            {"type": "text", "text": "读完了", "time": {"start": _T + 6700, "end": _T + 6709}},
        ],
    }
    messages = ConversationAdapterRegistry.with_builtins().adapt(
        protocol, {"records": [_opencode_user(), parallel, later_text]}, context()
    ).batch.messages

    assert [(item.role.value, item.tool_call_id) for item in messages] == [
        ("prompt", None),
        ("tool_call", "a"), ("tool_result", "a"),
        ("tool_call", "b"), ("tool_result", "b"),
        ("completion", None),
        ("tool_call", "c"), ("tool_result", "c"),
        ("completion", None),
    ]
    assert [item.sequence for item in messages] == list(range(10, 19))
    assert all(left.occurred_at <= right.occurred_at for left, right in zip(messages, messages[1:], strict=False))
    at = lambda offset: datetime.fromtimestamp((_T + offset) / 1000, tz=UTC)  # noqa: E731
    # 早于上一条的夹到上一条，不早的照原样
    assert [item.occurred_at for item in messages] == [
        at(0), at(100), at(5000), at(5000), at(5000), at(5200), at(6694), at(6738), at(6738)
    ]


def test_claude_code_parallel_tool_results_written_a_few_ms_out_of_order_are_accepted() -> None:
    records = [
        {"type": "user", "origin": {"kind": "human"}, "timestamp": "2026-10-01T00:00:00.000Z", "message": {"role": "user", "content": "读两个文件"}},
        {"type": "assistant", "timestamp": "2026-10-01T00:00:01.000Z",
         "message": {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Read", "input": {"f": 1}}]}},
        {"type": "assistant", "timestamp": "2026-10-01T00:00:01.100Z",
         "message": {"role": "assistant", "content": [{"type": "tool_use", "id": "t2", "name": "Read", "input": {"f": 2}}]}},
        {"type": "user", "timestamp": "2026-10-01T00:00:01.261Z",
         "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "a"}]}},
        {"type": "user", "timestamp": "2026-10-01T00:00:01.259Z",
         "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "b"}]}},
        {"type": "assistant", "timestamp": "2026-10-01T00:00:02.000Z",
         "message": {"role": "assistant", "content": [{"type": "text", "text": "完成"}]}},
    ]
    messages = ConversationAdapterRegistry.with_builtins().adapt("claude_code", {"records": records}, context()).batch.messages

    assert [(item.role.value, item.tool_call_id, item.content) for item in messages[3:5]] == [
        ("tool_result", "t1", "a"),
        ("tool_result", "t2", "b"),
    ]
    assert messages[4].occurred_at == messages[3].occurred_at == datetime(2026, 10, 1, 0, 0, 1, 261000, tzinfo=UTC)
    assert all(left.occurred_at <= right.occurred_at for left, right in zip(messages, messages[1:], strict=False))


# ── 消息身份（U6、S2-16b）：不依赖记录在这次请求里的位置，同一会话不同序号不撞 ────────────────────────────


def test_message_identity_does_not_depend_on_ignored_records_in_the_payload() -> None:
    meta = {"type": "user", "isMeta": True, "timestamp": "2026-10-01T00:00:00Z", "message": {"role": "user", "content": "<local-command-caveat>x"}}
    turn = [
        {"type": "user", "origin": {"kind": "human"}, "timestamp": "2026-10-01T00:00:01Z", "message": {"role": "user", "content": "你好"}},
        {"type": "assistant", "timestamp": "2026-10-01T00:00:02Z", "message": {"role": "assistant", "content": [{"type": "text", "text": "在"}]}},
    ]
    registry = ConversationAdapterRegistry.with_builtins()
    with_meta = registry.adapt("claude_code", {"records": [meta, *turn]}, context()).batch.messages
    without = registry.adapt("claude_code", {"records": turn}, context()).batch.messages

    assert [item.message_id for item in with_meta] == [item.message_id for item in without]


def test_the_same_words_at_two_sequences_of_one_conversation_have_two_identities() -> None:
    registry = ConversationAdapterRegistry.with_builtins()
    payload = {"messages": [{"role": "user", "content": "继续"}]}
    first = registry.adapt("openai_chat_completions", payload, ConversationAdapterContext("c", 0, datetime(2026, 10, 1, tzinfo=UTC)))
    later = registry.adapt("openai_chat_completions", payload, ConversationAdapterContext("c", 5, datetime(2026, 10, 1, tzinfo=UTC)))

    assert first.batch.messages[0].message_id != later.batch.messages[0].message_id


# ── 宿主"不是人话"的过滤与状态映射（U9 见证输入） ─────────────────────────────────────────────────


def _spoken(protocol: str, records: list) -> list[tuple[str, str]]:
    messages = ConversationAdapterRegistry.with_builtins().adapt(protocol, {"records": records}, context()).batch.messages
    return [(item.role.value, str(item.content)) for item in messages]


def test_claude_code_meta_records_are_not_his_words() -> None:
    assert _spoken(
        "claude_code",
        [
            {"type": "user", "isMeta": True, "message": {"role": "user", "content": "<local-command-caveat>Caveat"}},
            {"type": "user", "origin": {"kind": "human"}, "message": {"role": "user", "content": "你好"}},
            {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "在"}]}},
        ],
    ) == [("prompt", "你好"), ("completion", "在")]


@pytest.mark.parametrize("event", ["UserPromptSubmit", "SessionStart", "Stop"])
def test_every_zcode_hook_context_line_is_not_his_words(event: str) -> None:
    records = [
        {"info": {"role": "user", "time": {"created": _T}}, "parts": [
            {"type": "text", "text": f"{event} hook additional context: x", "time": {"end": _T}},
            {"type": "text", "text": "真话", "time": {"end": _T}},
        ]},
        {"info": {"role": "assistant", "time": {"created": _T + 1}}, "parts": [{"type": "text", "text": "答", "time": {"end": _T + 2}}]},
    ]
    assert _spoken("zcode", records) == [("prompt", "真话"), ("completion", "答")]


def test_an_opencode_summary_flagged_message_is_not_an_agent_reply() -> None:
    records = [
        _opencode_user("问"),
        {"info": {"role": "assistant", "summary": True, "time": {"created": _T + 1}}, "parts": [{"type": "text", "text": "压缩摘要", "time": {"end": _T + 2}}]},
        {"info": {"role": "assistant", "time": {"created": _T + 3}}, "parts": [{"type": "text", "text": "答", "time": {"end": _T + 4}}]},
    ]
    assert _spoken("opencode", records) == [("prompt", "问"), ("completion", "答")]


def test_a_late_deepseek_harness_question_reply_is_his_words() -> None:
    records = [
        {"type": "user/message", "time": _T, "data": {"source": {"kind": "user"}, "content": "问"}},
        {"type": "assistant/message", "time": _T + 1, "data": {"message": {"content": [{"type": "text", "text": "要不要?"}]}}},
        {"type": "user/message", "time": _T + 2, "data": {"source": {"kind": "user-question-reply"}, "content": "要"}},
    ]
    assert _spoken("dsh", records) == [("prompt", "问"), ("completion", "要不要?"), ("prompt", "要")]


@pytest.mark.parametrize(
    "text", ["<user_shell_command>ls</user_shell_command>", "<turn_aborted>The user interrupted</turn_aborted>"]
)
def test_codex_shell_commands_and_aborted_turn_notes_are_not_his_words(text: str) -> None:
    records = [
        {"type": "event_msg", "timestamp": "2026-10-01T00:00:00Z", "payload": {"type": "task_started"}},
        {"type": "response_item", "timestamp": "2026-10-01T00:00:01Z",
         "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "做事"}]}},
        {"type": "response_item", "timestamp": "2026-10-01T00:00:02Z",
         "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "好"}]}},
        {"type": "response_item", "timestamp": "2026-10-01T00:00:03Z",
         "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": text}]}},
    ]
    assert _spoken("codex_rollout", records) == [("prompt", "做事"), ("completion", "好")]


@pytest.mark.parametrize("status", ["cancelled", "canceled"])
def test_a_cancelled_tool_result_keeps_its_terminal_status(status: str) -> None:
    payload = {"items": [
        {"type": "message", "role": "user", "content": "x"},
        {"type": "function_call", "call_id": "1", "name": "f", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "1", "output": "stop", "status": status},
    ]}
    messages = ConversationAdapterRegistry.with_builtins().adapt("openai_responses", payload, context()).batch.messages
    assert messages[-1].tool_status is ConversationToolResultStatus.CANCELLED


# ── 宿主报错结束的一轮（U2，裁定 76 第 2 条） ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("protocol", ["opencode", "zcode"])
def test_an_assistant_message_that_ended_in_a_host_error_becomes_the_turn_s_reply(protocol: str) -> None:
    error = {"name": "AiSdkModelAdapterError", "data": {"message": "exceed quota limit, api_key=sk-abc123", "code": "model_request_failed"}}
    records = [
        {"info": {"role": "user", "time": {"created": 1_791_469_998_000}}, "parts": [{"type": "text", "text": "继续"}]},
        {"info": {"role": "assistant", "time": {"created": 1_791_469_998_188, "completed": 1_791_469_998_283}, "error": error},
         "parts": [{"type": "step-start"}]},
    ]
    adapted = ConversationAdapterRegistry.with_builtins().adapt(protocol, {"records": records}, context())

    assert [(item.role.value, item.content) for item in adapted.batch.messages] == [
        ("prompt", "继续"),
        ("completion", "[host error] AiSdkModelAdapterError: exceed quota limit, api_key=[REDACTED]"),
    ]
    assert adapted.batch.messages[1].occurred_at == datetime(2026, 10, 8, 14, 33, 18, 283000, tzinfo=UTC)  # 取 info.time.completed
    assert adapted.after_turn


def test_a_host_error_after_tools_closes_the_turn_after_the_last_result() -> None:
    records = [
        {"info": {"role": "user", "time": {"created": 1_000}}, "parts": [{"type": "text", "text": "看下目录"}]},
        {"info": {"role": "assistant", "time": {"created": 1_100}}, "parts": [
            {"type": "tool", "callID": "b1", "tool": "Bash",
             "state": {"status": "completed", "input": {"command": "ls"}, "output": "a", "time": {"start": 1_200, "end": 1_300}}},
        ]},
        {"info": {"role": "assistant", "time": {"created": 1_400}, "error": {"name": "StartPlanBusyAutoRetryExhaustedError"}},
         "parts": [{"type": "step-start"}]},
    ]
    messages = ConversationAdapterRegistry.with_builtins().adapt("zcode", {"records": records}, context()).batch.messages

    assert [item.role.value for item in messages] == ["prompt", "tool_call", "tool_result", "completion"]
    assert messages[-1].content == "[host error] StartPlanBusyAutoRetryExhaustedError"


# ── Claude Code 宿主注入、不是人话的消息（U3，裁定 76 第 3 条） ─────────────────────────────────────────


def claude_code_host_messages() -> list[dict[str, object]]:
    """本机真实转写里的形状：压缩摘要、/model 的两条记录、[Request interrupted、轮中间的后台任务通知、
    <artifact-content…> 压缩摘要、``!`` 模式的 <bash-input> 与它的输出、``claude -p`` 的输入。"""

    def user(text: str, **marks: object) -> dict[str, object]:
        return {"type": "user", **marks, "message": {"role": "user", "content": text}}

    def assistant(*blocks: dict[str, object]) -> dict[str, object]:
        return {"type": "assistant", "message": {"role": "assistant", "content": list(blocks)}}

    human = {"origin": {"kind": "human"}, "turnOrigin": "human"}
    records: list[dict[str, object]] = [
        user("This session is being continued from a previous conversation…", isCompactSummary=True),
        user("<command-name>/model</command-name>\n<command-message>model</command-message>"),
        user("<local-command-stdout>Set model to opus</local-command-stdout>"),
        user("跑一下全量回归", **human),
        assistant({"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "pytest"}}),
        user("<task-notification><status>completed</status></task-notification>",
             origin={"kind": "task-notification"}, turnOrigin="task_notification"),
        {"type": "user", "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]}},
        user("[Request interrupted by user]"),
        user("先别跑了", **human),
        assistant({"type": "text", "text": "已停。"}),
        user("<artifact-content-authoring>…</artifact-content-authoring>", isCompactSummary=True, turnOrigin="human"),
        user("<bash-input>npm login</bash-input>"),
        user("<bash-stdout>Logged in</bash-stdout><bash-stderr></bash-stderr>", turnOrigin="human"),
        user("[system] Return exactly one JSON object", turnOrigin="sdk"),
        assistant({"type": "text", "text": "{}"}),
    ]
    return [{**record, "timestamp": f"2026-10-01T00:{index:02d}:00Z"} for index, record in enumerate(records)]


def test_claude_code_keeps_only_his_words_by_the_host_s_own_marks() -> None:
    messages = ConversationAdapterRegistry.with_builtins().adapt(
        "claude_code", {"records": claude_code_host_messages()}, context()
    ).batch.messages

    assert [(item.role.value, str(item.content)[:20], item.mid_turn) for item in messages] == [
        ("prompt", "跑一下全量回归", False),
        ("tool_call", "{'command': 'pytest'", False),
        ("prompt", "<task-notification><", True),  # 助手还在干活时到的通知：插在这一轮中间
        ("tool_result", "ok", False),
        ("prompt", "先别跑了", True),  # 叫停之后他补的一句：还在这一轮里（裁定 69）
        ("completion", "已停。", False),
        ("prompt", "<bash-input>npm logi", False),
        ("prompt", "[system] Return exac", True),
        ("completion", "{}", False),
    ]


# ── 一条人话里有几个文字块：十个适配器同一个口径，合成一句（U14） ────────────────────────────────────────

_FIRST, _SECOND = "<ide_opened_file>src/app.py</ide_opened_file>", "帮我看下这个报错"


def _two_text_blocks(protocol: str) -> object:
    """同一条人的消息带两个文字块、后面一条答复，按各协议自己的形状写。"""

    anthropic = [{"type": "text", "text": _FIRST}, {"type": "text", "text": _SECOND}]
    opencode = [
        {"info": {"role": "user", "time": {"created": _T}}, "parts": [{"type": "text", "text": _FIRST}, {"type": "text", "text": _SECOND}]},
        {"info": {"role": "assistant", "time": {"created": _T + 1}}, "parts": [{"type": "text", "text": "好的"}]},
    ]
    payloads: dict[str, object] = {
        "openai_chat_completions": {"messages": [{"role": "user", "content": anthropic}, {"role": "assistant", "content": "好的"}]},
        "openai_responses": {"items": [
            {"type": "message", "role": "user", "content": [{"type": "input_text", "text": _FIRST}, {"type": "input_text", "text": _SECOND}]},
            {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "好的"}]},
        ]},
        "anthropic_messages": {"messages": [
            {"role": "user", "content": anthropic},
            {"role": "assistant", "content": [{"type": "text", "text": "好的"}]},
        ]},
        "codex_rollout": {"records": [
            {"type": "event_msg", "timestamp": "2026-10-01T00:00:00Z", "payload": {"type": "task_started"}},
            {"type": "response_item", "timestamp": "2026-10-01T00:00:01Z", "payload": {
                "type": "message", "role": "user",
                "content": [{"type": "input_text", "text": _FIRST}, {"type": "input_text", "text": _SECOND}]}},
            {"type": "response_item", "timestamp": "2026-10-01T00:00:02Z",
             "payload": {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "好的"}]}},
        ]},
        "claude_code": {"records": [
            {"type": "user", "origin": {"kind": "human"}, "message": {"role": "user", "content": anthropic}},
            {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "好的"}]}},
        ]},
        "openclaw": {"records": [
            {"type": "message", "message": {"role": "user", "content": anthropic}},
            {"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "好的"}]}},
        ]},
        "pi": {"records": [
            {"type": "message", "message": {"role": "user", "content": anthropic, "timestamp": _T}},
            {"type": "message", "message": {"role": "assistant", "content": [{"type": "text", "text": "好的"}], "timestamp": _T + 1}},
        ]},
        "opencode": {"records": opencode},
        "zcode": {"records": opencode},
        "dsh": {"records": [
            {"type": "user/message", "time": _T, "data": {"source": {"kind": "user"}, "content": anthropic}},
            {"type": "assistant/message", "time": _T + 1, "data": {"message": {"content": [{"type": "text", "text": "好的"}]}}},
        ]},
    }
    return payloads[protocol]


_ALL_PROTOCOLS = (
    "openai_chat_completions",
    "openai_responses",
    "anthropic_messages",
    "codex_rollout",
    "claude_code",
    "openclaw",
    "pi",
    "opencode",
    "zcode",
    "dsh",
)


def test_the_multi_text_block_matrix_covers_every_registered_adapter() -> None:
    """新加的适配器必须进下面这张矩阵：同一条人话的几个文字块怎么收，各协议不能各说各的。"""

    assert set(ConversationAdapterRegistry.with_builtins().protocols()) == set(_ALL_PROTOCOLS)


@pytest.mark.parametrize("protocol", _ALL_PROTOCOLS)
def test_several_text_blocks_of_one_human_message_are_one_prompt(protocol: str) -> None:
    """一条人话的几个文字块用换行合成一句：拆成几句会各开一轮，afterTurn 时整批被保留规则拒收、会话卡住。"""

    from habitus.memory.conversation.retention import ConversationRetentionPlanner

    result = ConversationAdapterRegistry.with_builtins().adapt(protocol, _two_text_blocks(protocol), context())

    prompts = [item for item in result.batch.messages if item.role is ConversationMessageRole.PROMPT]
    assert [item.content for item in prompts] == [f"{_FIRST}\n{_SECOND}"]
    assert [item.starts_turn for item in prompts] == [True]
    assert result.after_turn
    ConversationRetentionPlanner().plan(result.batch, after_turn=True, flush=True)


def test_a_claude_code_compact_summary_is_not_his_words_even_when_marked_sdk() -> None:
    """压缩写的摘要不收，``turnOrigin`` 为 sdk 也一样（S2R-43）：先看是不是压缩摘要，再看来源标记。"""

    assert _spoken(
        "claude_code",
        [
            {"type": "user", "isCompactSummary": True, "turnOrigin": "sdk",
             "message": {"role": "user", "content": "This session is being continued from a previous conversation."}},
            {"type": "user", "turnOrigin": "sdk", "message": {"role": "user", "content": "[system]\n跑一下"}},
            {"type": "assistant", "message": {"role": "assistant", "content": [{"type": "text", "text": "好"}]}},
        ],
    ) == [("prompt", "[system]\n跑一下"), ("completion", "好")]
