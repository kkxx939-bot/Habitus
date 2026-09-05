"""外部 Conversation 协议转换的角色、工具配对和拒绝场景。"""

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
                "content": [
                    {"type": "tool_result", "tool_use_id": "tool-1", "content": "timeout", "is_error": True}
                ],
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
