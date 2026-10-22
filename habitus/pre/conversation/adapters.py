"""把主流 Agent/模型协议转换为 Habitus 的严格 Conversation 事实。"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from habitus.foundation.integrity import canonical_digest
from habitus.pre.conversation.messages import (
    ConversationBatch,
    ConversationMessage,
    ConversationMessageRole,
    ConversationToolResultContentMode,
    ConversationToolResultStatus,
)
from habitus.pre.conversation.questions import QuestionTool


class ConversationProtocolError(ValueError):
    """外部协议无法无损转换为 Conversation 角色事实。"""


@dataclass(frozen=True)
class ConversationAdapterContext:
    """协议本身没有稳定序号和时间时由应用边界补充的可信上下文。"""

    conversation_id: str
    start_sequence: int
    occurred_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.conversation_id, str) or not self.conversation_id.strip():
            raise ValueError("conversation_id must be non-empty text")
        if isinstance(self.start_sequence, bool) or not isinstance(self.start_sequence, int) or self.start_sequence < 0:
            raise ValueError("start_sequence must be a non-negative integer")
        if not isinstance(self.occurred_at, datetime) or self.occurred_at.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")
        object.__setattr__(self, "occurred_at", self.occurred_at.astimezone(UTC))


@dataclass(frozen=True)
class ConversationAdaptation:
    """转换后的批次及协议层明确忽略的系统元数据数量。"""

    batch: ConversationBatch
    after_turn: bool
    ignored_items: int = 0


class ConversationProtocolAdapter(Protocol):
    """``protocol`` 是协议名。来自某个 agent 宿主的协议另声明 ``host``（宿主名，与插件取提醒时报的一致）与 ``question_tools``
    （它那个宿主的提问工具）；通用 SDK 协议两样都不声明。"""

    protocol: str

    def adapt(self, payload: object, context: ConversationAdapterContext) -> ConversationAdaptation: ...


class ConversationAdapterRegistry:
    """按协议名注册适配器，不按模型或供应商枚举。"""

    def __init__(self) -> None:
        self._adapters: dict[str, ConversationProtocolAdapter] = {}

    def register(self, adapter: ConversationProtocolAdapter) -> None:
        protocol = _clean_text(getattr(adapter, "protocol", None), "protocol").lower()
        if protocol in self._adapters:
            raise ValueError(f"conversation protocol is already registered: {protocol}")
        if not callable(getattr(adapter, "adapt", None)):
            raise TypeError("conversation adapter must implement adapt")
        self._adapters[protocol] = adapter

    def adapt(
        self,
        protocol: str,
        payload: object,
        context: ConversationAdapterContext,
    ) -> ConversationAdaptation:
        normalized = _clean_text(protocol, "protocol").lower()
        adapter = self._adapters.get(normalized)
        if adapter is None:
            raise ConversationProtocolError(f"unsupported conversation protocol: {normalized}")
        return adapter.adapt(payload, context)

    def protocols(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))

    def host(self, protocol: str) -> str | None:
        """这个协议来自哪个 agent 宿主；通用协议或没注册的给 None。"""

        adapter = self._adapters.get(_clean_text(protocol, "protocol").lower())
        value = getattr(adapter, "host", None)
        return value if isinstance(value, str) and value else None

    def question_tools(self, protocol: str | None = None) -> tuple[QuestionTool, ...]:
        """一个协议（省略是全部已注册协议）的宿主用来问人的工具。"""

        if protocol is None:
            adapters = [self._adapters[name] for name in sorted(self._adapters)]
        else:
            adapter = self._adapters.get(_clean_text(protocol, "protocol").lower())
            adapters = [] if adapter is None else [adapter]
        found: list[QuestionTool] = []
        for adapter in adapters:
            for tool in getattr(adapter, "question_tools", ()):
                if not isinstance(tool, QuestionTool):
                    raise TypeError("conversation adapter question_tools must be QuestionTool values")
                if tool not in found:
                    found.append(tool)
        return tuple(found)

    @classmethod
    def with_builtins(cls) -> ConversationAdapterRegistry:
        registry = cls()
        registry.register(OpenAIChatCompletionsConversationAdapter())
        registry.register(OpenAIResponsesConversationAdapter())
        registry.register(AnthropicMessagesConversationAdapter())
        registry.register(CodexRolloutConversationAdapter())
        registry.register(ClaudeCodeConversationAdapter())
        registry.register(OpenClawConversationAdapter())
        registry.register(OpenCodeConversationAdapter())
        registry.register(PiConversationAdapter())
        registry.register(ZCodeConversationAdapter())
        registry.register(DeepSeekHarnessConversationAdapter())
        return registry


class _Builder:
    def __init__(self, protocol: str, context: ConversationAdapterContext) -> None:
        self.protocol = protocol
        self.context = context
        self.messages: list[ConversationMessage] = []
        self.tool_names: dict[str, str] = {}

    def add(
        self,
        role: ConversationMessageRole,
        content: object,
        *,
        source_index: str,
        tool_call_id: str | None = None,
        tool_name: str | None = None,
        tool_status: ConversationToolResultStatus | None = None,
        occurred_at: datetime | None = None,
        mid_turn: bool = False,
    ) -> None:
        sequence = self.context.start_sequence + len(self.messages)
        identity = canonical_digest(
            {
                "protocol": self.protocol,
                "conversation_id": self.context.conversation_id,
                "sequence": sequence,
                "source_index": source_index,
                "role": role.value,
                "content": content,
                "tool_call_id": tool_call_id,
                "tool_name": tool_name,
            }
        )[:32]
        message = ConversationMessage(
            message_id=f"{self.protocol}-{identity}",
            sequence=sequence,
            role=role,
            occurred_at=self.context.occurred_at if occurred_at is None else occurred_at,
            content=content,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            tool_status=tool_status,
            content_mode=(
                ConversationToolResultContentMode.INLINE if role is ConversationMessageRole.TOOL_RESULT else None
            ),
            mid_turn=mid_turn,
        )
        self.messages.append(message)
        if role is ConversationMessageRole.TOOL_CALL:
            assert tool_call_id is not None and tool_name is not None
            self.tool_names[tool_call_id] = tool_name

    def finish(self, *, after_turn: bool, ignored_items: int) -> ConversationAdaptation:
        if not self.messages:
            raise ConversationProtocolError("conversation payload contains no persistable messages")
        return ConversationAdaptation(
            ConversationBatch(self.context.conversation_id, tuple(self.messages)),
            after_turn=after_turn,
            ignored_items=ignored_items,
        )


class OpenAIChatCompletionsConversationAdapter:
    protocol = "openai_chat_completions"

    def adapt(self, payload: object, context: ConversationAdapterContext) -> ConversationAdaptation:
        messages = _message_sequence(payload)
        builder = _Builder(self.protocol, context)
        ignored = 0
        last_role = ""
        for index, raw in enumerate(messages):
            item = _mapping(raw, f"messages[{index}]")
            role = _clean_text(item.get("role"), "message role").lower()
            last_role = role
            if role in {"system", "developer"}:
                ignored += 1
                continue
            if role in {"user", "assistant"}:
                content = _text_content(item.get("content"))
                if content:
                    builder.add(
                        ConversationMessageRole.PROMPT if role == "user" else ConversationMessageRole.COMPLETION,
                        content,
                        source_index=str(index),
                    )
                tool_calls = item.get("tool_calls", ())
                if tool_calls is not None:
                    for call_index, raw_call in enumerate(_sequence(tool_calls, "tool_calls")):
                        call = _mapping(raw_call, "tool_call")
                        function = _mapping(call.get("function"), "tool_call.function")
                        call_id = _clean_text(call.get("id"), "tool_call.id")
                        name = _clean_text(function.get("name"), "tool_call.function.name")
                        builder.add(
                            ConversationMessageRole.TOOL_CALL,
                            _tool_arguments(function.get("arguments")),
                            source_index=f"{index}.tool_calls.{call_index}",
                            tool_call_id=call_id,
                            tool_name=name,
                        )
                continue
            if role == "tool":
                call_id = _clean_text(item.get("tool_call_id"), "tool_result.tool_call_id")
                result_name = builder.tool_names.get(call_id)
                if result_name is None:
                    raise ConversationProtocolError("tool result has no matching tool_call name in payload")
                builder.add(
                    ConversationMessageRole.TOOL_RESULT,
                    _tool_result_content(item.get("content")),
                    source_index=str(index),
                    tool_call_id=call_id,
                    tool_name=result_name,
                    tool_status=_tool_result_status(item.get("status")),
                )
                continue
            raise ConversationProtocolError(f"unsupported Chat Completions role: {role}")
        pending = bool(builder.messages and builder.messages[-1].role is ConversationMessageRole.TOOL_CALL)
        return builder.finish(after_turn=last_role == "assistant" and not pending, ignored_items=ignored)


class OpenAIResponsesConversationAdapter:
    protocol = "openai_responses"

    def adapt(self, payload: object, context: ConversationAdapterContext) -> ConversationAdaptation:
        root = _mapping(payload, "Responses payload")
        if "items" in root:
            items = _sequence(root["items"], "Responses items")
        else:
            input_items = _sequence(root.get("input", ()), "Responses input")
            output_items = _sequence(root.get("output", ()), "Responses output")
            items = (*input_items, *output_items)
        builder = _Builder(self.protocol, context)
        ignored = 0
        for index, raw in enumerate(items):
            item = _mapping(raw, f"items[{index}]")
            item_type = _clean_text(item.get("type", "message"), "Responses item type")
            if item_type == "message":
                role = _clean_text(item.get("role"), "Responses message role").lower()
                if role in {"system", "developer"}:
                    ignored += 1
                    continue
                if role not in {"user", "assistant"}:
                    raise ConversationProtocolError(f"unsupported Responses message role: {role}")
                content = _responses_text(item.get("content"))
                if content:
                    builder.add(
                        ConversationMessageRole.PROMPT if role == "user" else ConversationMessageRole.COMPLETION,
                        content,
                        source_index=str(index),
                    )
            elif item_type == "function_call":
                call_id = _clean_text(item.get("call_id", item.get("id")), "function_call.call_id")
                name = _clean_text(item.get("name"), "function_call.name")
                builder.add(
                    ConversationMessageRole.TOOL_CALL,
                    _tool_arguments(item.get("arguments")),
                    source_index=str(index),
                    tool_call_id=call_id,
                    tool_name=name,
                )
            elif item_type == "function_call_output":
                call_id = _clean_text(item.get("call_id"), "function_call_output.call_id")
                result_name = builder.tool_names.get(call_id)
                if result_name is None:
                    raise ConversationProtocolError("function_call_output has no matching function_call")
                builder.add(
                    ConversationMessageRole.TOOL_RESULT,
                    _tool_result_content(item.get("output")),
                    source_index=str(index),
                    tool_call_id=call_id,
                    tool_name=result_name,
                    tool_status=_tool_result_status(item.get("status")),
                )
            else:
                ignored += 1
        completed = bool(builder.messages and builder.messages[-1].role is ConversationMessageRole.COMPLETION)
        return builder.finish(after_turn=completed, ignored_items=ignored)


class AnthropicMessagesConversationAdapter:
    protocol = "anthropic_messages"

    def adapt(self, payload: object, context: ConversationAdapterContext) -> ConversationAdaptation:
        messages = _message_sequence(payload)
        builder = _Builder(self.protocol, context)
        ignored = 0
        for message_index, raw in enumerate(messages):
            item = _mapping(raw, f"messages[{message_index}]")
            role = _clean_text(item.get("role"), "Anthropic message role").lower()
            if role not in {"user", "assistant"}:
                raise ConversationProtocolError(f"unsupported Anthropic message role: {role}")
            ignored += _add_anthropic_content(
                builder,
                role=role,
                content=item.get("content"),
                source_prefix=str(message_index),
                occurred_at=context.occurred_at,
            )
        completed = bool(builder.messages and builder.messages[-1].role is ConversationMessageRole.COMPLETION)
        return builder.finish(after_turn=completed, ignored_items=ignored)


class CodexRolloutConversationAdapter:
    """适配 Codex rollout JSONL 解码后的记录序列。"""

    protocol = "codex_rollout"
    host = "codex"
    # CLI 的同步提问；Desktop 的异步提问作答以一条 <send_user_message_question_reply> 用户消息晚来（方案 32 第 5.3 条）
    question_tools = (
        QuestionTool("codex", "request_user_input", "title"),
        QuestionTool("codex", "request_user_input_async", "title", replies_later=True),
    )
    # Codex 以 role=user 写进转写、但不是人说的话的上下文消息：会话开头和中途都会写，当成人话会自成一轮、没有回复，
    # 整批转写被保留规则拒收。与 developer 消息一样不收。前三个是本机 CLI 0.147 与 Desktop 0.153–0.162 真实转写里见到的
    # （2026-10-22 扫了 15 份；见到的是 ``<external_codex_apps_open_page``，按会话 lane 的口径收 ``<external_codex`` 整族）；
    # 其余是会话 lane 一直当"不是人话"的 Codex 合成消息（``lanes/session/material.py``）。
    # Desktop 的作答 ``<send_user_message_question_reply>`` 与后台任务叫醒的 ``<subagent_notification>`` 不在此列。
    host_context_prefixes = (
        "<environment_context",
        "<external_codex",
        "<recommended_plugins",
        "<user_instructions",
        "# AGENTS.md instructions",
        "<skill>",
        "<in-app-browser",
        "<turn_aborted",
        "<user_shell_command",
    )

    def adapt(self, payload: object, context: ConversationAdapterContext) -> ConversationAdaptation:
        builder = _Builder(self.protocol, context)
        ignored = 0
        # Codex 自己分轮（每轮以 task_started 开始，本机 25 份真实转写里每句人话前都有）：一轮里第一句人话之后再来的人话
        # （他作答、补一句、叫停）是插在这一轮中间的。一批不一定从一轮开头切起（插件在一轮中途的答复处、压缩前都会送一批），
        # 所以这一批里第一个 task_started 之前的人话，属于开头在更早一批里的那一轮——也是插在中间的。
        human_in_turn = True
        for index, raw in enumerate(_records(payload, "Codex rollout records")):
            record = _mapping(raw, f"records[{index}]")
            record_type = record.get("type")
            if record_type not in {"response_item", "event_msg"}:
                ignored += 1
                continue
            item = _mapping(record.get("payload"), "Codex response_item payload")
            item_type = _clean_text(item.get("type"), "Codex response_item type")
            occurred_at = _timestamp(record.get("timestamp"), context.occurred_at)
            if record_type == "event_msg":
                if item_type == "task_started":
                    human_in_turn = False
                if item_type != "sub_agent_activity":
                    ignored += 1
                    continue
                activity = {
                    key: item[key] for key in ("agent_thread_id", "agent_path", "kind", "occurred_at_ms") if key in item
                }
                builder.add(
                    ConversationMessageRole.COMPLETION,
                    "[sub-agent-activity]\n" + json.dumps(activity, ensure_ascii=False, sort_keys=True),
                    source_index=str(index),
                    occurred_at=occurred_at,
                )
                continue
            if item_type == "message":
                role = _clean_text(item.get("role"), "Codex message role").lower()
                if role in {"system", "developer"}:
                    ignored += 1
                    continue
                if role not in {"user", "assistant"}:
                    raise ConversationProtocolError(f"unsupported Codex message role: {role}")
                content = _responses_text(item.get("content"))
                if not content or (role == "user" and content.lstrip().startswith(self.host_context_prefixes)):
                    ignored += 1
                    continue
                builder.add(
                    ConversationMessageRole.PROMPT if role == "user" else ConversationMessageRole.COMPLETION,
                    content,
                    source_index=str(index),
                    occurred_at=occurred_at,
                    mid_turn=role == "user" and human_in_turn,
                )
                human_in_turn = human_in_turn or role == "user"
            elif item_type == "agent_message":
                content = _responses_text(item.get("content"))
                if not content:
                    ignored += 1
                    continue
                author = str(item.get("author") or "subagent")
                recipient = str(item.get("recipient") or "main")
                builder.add(
                    ConversationMessageRole.COMPLETION,
                    f"[agent-message {author} -> {recipient}]\n{content}",
                    source_index=str(index),
                    occurred_at=occurred_at,
                )
            elif item_type == "sub_agent_activity":
                activity = {
                    key: item[key] for key in ("agent_thread_id", "agent_path", "kind", "occurred_at_ms") if key in item
                }
                builder.add(
                    ConversationMessageRole.COMPLETION,
                    "[sub-agent-activity]\n" + json.dumps(activity, ensure_ascii=False, sort_keys=True),
                    source_index=str(index),
                    occurred_at=occurred_at,
                )
            elif item_type in {"function_call", "custom_tool_call", "tool_search_call"}:
                call_id = _clean_text(item.get("call_id", item.get("id")), "Codex function_call.call_id")
                name = (
                    "tool_search"
                    if item_type == "tool_search_call"
                    else _clean_text(
                        item.get("name"),
                        "Codex function_call.name",
                    )
                )
                builder.add(
                    ConversationMessageRole.TOOL_CALL,
                    _tool_arguments(item.get("arguments", item.get("input", {}))),
                    source_index=str(index),
                    tool_call_id=call_id,
                    tool_name=name,
                    occurred_at=occurred_at,
                )
            elif item_type in {"function_call_output", "custom_tool_call_output", "tool_search_output"}:
                call_id = _clean_text(item.get("call_id", item.get("id")), "Codex function_call_output.call_id")
                result_name = builder.tool_names.get(call_id)
                if result_name is None:
                    raise ConversationProtocolError("Codex function_call_output has no matching function_call")
                builder.add(
                    ConversationMessageRole.TOOL_RESULT,
                    _tool_result_content(
                        {"tools": item.get("tools", [])}
                        if item_type == "tool_search_output"
                        else item.get("output", item.get("result"))
                    ),
                    source_index=str(index),
                    tool_call_id=call_id,
                    tool_name=result_name,
                    tool_status=_tool_result_status(item.get("status")),
                    occurred_at=occurred_at,
                )
            else:
                ignored += 1
        completed = bool(builder.messages and builder.messages[-1].role is ConversationMessageRole.COMPLETION)
        return builder.finish(after_turn=completed, ignored_items=ignored)


class ClaudeCodeConversationAdapter:
    """适配 Claude Code 项目会话 JSONL 解码后的记录序列。"""

    protocol = "claude_code"
    host = "claude-code"
    question_tools = (QuestionTool("claude-code", "AskUserQuestion", "question"),)

    def adapt(self, payload: object, context: ConversationAdapterContext) -> ConversationAdaptation:
        builder = _Builder(self.protocol, context)
        ignored = 0
        for index, raw in enumerate(_records(payload, "Claude Code records")):
            record = _mapping(raw, f"records[{index}]")
            record_type = record.get("type")
            if (
                record_type not in {"user", "assistant"}
                or record.get("isSidechain") is True
                or record.get("isMeta") is True
            ):
                ignored += 1
                continue
            message = _mapping(record.get("message"), "Claude Code message")
            role = _clean_text(message.get("role"), "Claude Code message role").lower()
            if role not in {"user", "assistant"}:
                raise ConversationProtocolError(f"unsupported Claude Code message role: {role}")
            ignored += _add_anthropic_content(
                builder,
                role=role,
                content=message.get("content"),
                source_prefix=str(index),
                occurred_at=_timestamp(record.get("timestamp"), context.occurred_at),
            )
        completed = bool(builder.messages and builder.messages[-1].role is ConversationMessageRole.COMPLETION)
        return builder.finish(after_turn=completed, ignored_items=ignored)


class OpenClawConversationAdapter:
    """适配 OpenClaw Agent session JSONL 解码后的记录序列。"""

    protocol = "openclaw"

    def adapt(self, payload: object, context: ConversationAdapterContext) -> ConversationAdaptation:
        builder = _Builder(self.protocol, context)
        ignored = 0
        for index, raw in enumerate(_records(payload, "OpenClaw records")):
            record = _mapping(raw, f"records[{index}]")
            if record.get("type") != "message":
                ignored += 1
                continue
            message = _mapping(record.get("message"), "OpenClaw message")
            role = _clean_text(message.get("role"), "OpenClaw message role").lower()
            if role not in {"user", "assistant"}:
                ignored += 1
                continue
            ignored += _add_anthropic_content(
                builder,
                role=role,
                content=message.get("content"),
                source_prefix=str(index),
                occurred_at=_timestamp(
                    record.get("timestamp", message.get("timestamp")),
                    context.occurred_at,
                ),
            )
        completed = bool(builder.messages and builder.messages[-1].role is ConversationMessageRole.COMPLETION)
        return builder.finish(after_turn=completed, ignored_items=ignored)


class PiConversationAdapter:
    """适配 pi（earendil-works/pi）自己落的会话 JSONL：一行一条记录 ``{type, id, parentId, timestamp, ...}``，插件按游标把
    增量原样送来。

    - 只收 ``type == "message"`` 的记录；会话头、模型 / 思考档位变更、压缩、分支摘要、标签等都不是对话。
    - ``message.role``：``user`` 的文本 → 人说的话；``assistant`` 的 ``text`` 块 → 答复、``toolCall`` 块 → 工具调用（``thinking`` 不收）；
      ``toolResult`` → 工具结果（``isError`` 为真记失败）；``custom``（插件注入的上下文，``before_agent_start`` 返回的那条）
      与 pi 自己的 ``bashExecution`` / ``branchSummary`` 等都不是他说的话，不收。
    - 提问工具：pi 没有内置的，插件注册 ``ask_user``，结果文字已经是 ``"<题目>"="<作答>"`` 一行一题。
    时刻：消息上的 ``timestamp`` 是毫秒；没有就用记录的 ISO 时刻。
    """

    protocol = "pi"
    host = "pi"
    question_tools = (QuestionTool("pi", "ask_user", "question"),)

    def adapt(self, payload: object, context: ConversationAdapterContext) -> ConversationAdaptation:
        builder = _Builder(self.protocol, context)
        ignored = 0
        for index, raw in enumerate(_records(payload, "pi records")):
            record = _mapping(raw, f"records[{index}]")
            if record.get("type") != "message":
                ignored += 1
                continue
            message = _mapping(record.get("message"), "pi message")
            role = message.get("role")
            at = _epoch_ms(message.get("timestamp"), _timestamp(record.get("timestamp"), context.occurred_at))
            if role == "user":
                ignored += _add_pi_text(builder, ConversationMessageRole.PROMPT, message.get("content"), str(index), at)
            elif role == "assistant":
                ignored += _add_pi_assistant(builder, message.get("content"), str(index), at)
            elif role == "toolResult":
                call_id = _clean_text(message.get("toolCallId"), "pi toolResult toolCallId")
                builder.add(
                    ConversationMessageRole.TOOL_RESULT,
                    _pi_text(message.get("content")) or "",
                    source_index=f"{index}.result",
                    tool_call_id=call_id,
                    tool_name=_clean_text(message.get("toolName"), "pi toolResult toolName"),
                    tool_status=(
                        ConversationToolResultStatus.ERROR
                        if message.get("isError") is True
                        else ConversationToolResultStatus.COMPLETED
                    ),
                    occurred_at=at,
                )
            else:
                ignored += 1
        completed = bool(builder.messages and builder.messages[-1].role is ConversationMessageRole.COMPLETION)
        return builder.finish(after_turn=completed, ignored_items=ignored)


def _pi_text(content: object) -> str:
    """pi 的消息内容：一段字符串，或 ``[{type: "text", text}, {type: "image", ...}]``（只取文字）。"""

    if isinstance(content, str):
        return content.strip()
    texts: list[str] = []
    for block in _sequence(content if content is not None else (), "pi message content"):
        text = block.get("text") if isinstance(block, Mapping) and block.get("type") == "text" else None
        if isinstance(text, str) and text.strip():
            texts.append(text)
    return "\n".join(texts).strip()


def _add_pi_text(
    builder: _Builder, role: ConversationMessageRole, content: object, source: str, at: datetime
) -> int:
    text = _pi_text(content)
    if not text:
        return 1
    builder.add(role, text, source_index=source, occurred_at=at)
    return 0


def _add_pi_assistant(builder: _Builder, content: object, source: str, at: datetime) -> int:
    if isinstance(content, str):
        return _add_pi_text(builder, ConversationMessageRole.COMPLETION, content, source, at)
    ignored = 0
    for block_index, raw in enumerate(_sequence(content if content is not None else (), "pi assistant content")):
        block = _mapping(raw, "pi assistant content block")
        kind = block.get("type")
        text = block.get("text")
        if kind == "text" and isinstance(text, str) and text.strip():
            builder.add(ConversationMessageRole.COMPLETION, text.strip(), source_index=f"{source}.{block_index}", occurred_at=at)
        elif kind == "toolCall":
            builder.add(
                ConversationMessageRole.TOOL_CALL,
                _tool_arguments(block.get("arguments") or {}),
                source_index=f"{source}.{block_index}",
                tool_call_id=_clean_text(block.get("id"), "pi toolCall id"),
                tool_name=_clean_text(block.get("name"), "pi toolCall name"),
                occurred_at=at,
            )
        else:
            ignored += 1  # thinking 等
    return ignored


class OpenCodeConversationAdapter:
    """适配 OpenCode 的会话消息（插件把每条已完成的消息原样镜像成一条记录：``{"info": Message, "parts": [Part]}``，即
    ``client.session.messages`` 的形状）。

    - 人的消息：非合成（``synthetic``）、非忽略的文本段 → 人说的话；OpenCode 自己合成的（读文件的回执）与我们注入的上下文都是合成段，不算。
    - 助手的消息：文本段 → 答复；工具段只收已结束的（``completed`` / ``error``），一段拆成调用 + 结果；没结束的连调用也不收（调用与结果
      不闭合就进不了 Conversation）。思考、步骤、快照、补丁、文件等段不收。
    - 提问工具 ``question`` 的作答：工具自己把他点了什么记在 ``state.metadata.answers``（每题一组标签，与 ``state.input.questions``
      对齐）；据此把结果写成 ``"<题目>"="<作答>"`` 一行一题，和 Claude Code 的作答同一个形状，提醒出口与会话 lane 都按这个认
      （方案 32 第 4.2 条、方案 36 第四节）。
    - 压缩写的摘要（``summary`` 为真 / ``mode`` 是 compaction 的助手消息）不收；压缩后重放的那句人话插件镜像时就去掉了。
    时刻：OpenCode 记毫秒时间戳。
    """

    protocol = "opencode"
    host = "opencode"
    # question 工具同步等作答；作答按工具自记的结构化作答写成 "<题目>"="<作答>"（见下）
    question_tools = (QuestionTool("opencode", "question", "question"),)

    def adapt(self, payload: object, context: ConversationAdapterContext) -> ConversationAdaptation:
        builder = _Builder(self.protocol, context)
        ignored = 0
        for index, raw in enumerate(_records(payload, "OpenCode records")):
            record = _mapping(raw, f"records[{index}]")
            info = _mapping(record.get("info"), "OpenCode message info")
            role = _clean_text(info.get("role"), "OpenCode message role").lower()
            if role not in {"user", "assistant"} or info.get("summary") is True or info.get("mode") == "compaction":
                ignored += 1  # 压缩写的摘要不是 agent 对他说的话
                continue
            time = info.get("time")
            created = _epoch_ms(time.get("created") if isinstance(time, Mapping) else None, context.occurred_at)
            for part_index, raw_part in enumerate(_sequence(record.get("parts", ()), "OpenCode parts")):
                part = _mapping(raw_part, "OpenCode part")
                ignored += _add_opencode_part(builder, role, part, f"{index}.{part_index}", created)
        completed = bool(builder.messages and builder.messages[-1].role is ConversationMessageRole.COMPLETION)
        return builder.finish(after_turn=completed, ignored_items=ignored)


class DeepSeekHarnessConversationAdapter:
    """适配 DeepSeek Harness（deepseek-ai/deepseek-harness，MIT）的会话日志事件（插件解出 ``$DSH_HOME/sessions`` 下的日志、按 ``seq``
    镜像送来）：一行一个事件 ``{type, seq, time, data}``。

    - ``user/message``：只有 ``source.kind == "user"``（人亲手输入）与 ``"user-question-reply"``（提问超时后他迟到的作答）是他说的话；
      注入的上下文（``hooks-claude-code`` 等）、压缩替换的摘要（``compact-checkpoint``）、续轮都不算。
    - ``assistant/message``：``text`` 块 → 答复，``tool-call`` 块（``arguments`` 是 JSON 文字）→ 工具调用，``reasoning`` 不收；
      ``tool/call`` 事件与它重复，不收。``tool/result`` → 工具结果（``isError`` 记失败）。
    - 提问工具 ``ask_user_question``：结果是 ``{"answers": [{id, selected, custom?}]}``，按调用里 ``questions[].id`` 配回题目，写成
      ``"<题目>"="<作答>"`` 一行一题。
    时刻：事件上的 ``time`` 是毫秒。
    """

    protocol = "dsh"
    host = "dsh"
    question_tools = (QuestionTool("dsh", "ask_user_question", "question"),)
    _SPOKEN = frozenset({"user", "user-question-reply"})

    def adapt(self, payload: object, context: ConversationAdapterContext) -> ConversationAdaptation:
        builder = _Builder(self.protocol, context)
        ignored = 0
        asked: dict[str, Mapping[str, object]] = {}
        for index, raw in enumerate(_records(payload, "DeepSeek Harness records")):
            event = _mapping(raw, f"records[{index}]")
            kind = event.get("type")
            data = event.get("data")
            at = _epoch_ms(event.get("time"), context.occurred_at)
            if kind == "user/message" and isinstance(data, Mapping):
                source = data.get("source")
                spoken = isinstance(source, Mapping) and source.get("kind") in self._SPOKEN
                text = _dsh_text(data.get("content")) if spoken else ""
                if text:
                    builder.add(ConversationMessageRole.PROMPT, text, source_index=str(index), occurred_at=at)
                else:
                    ignored += 1
            elif kind == "assistant/message" and isinstance(data, Mapping):
                message = _mapping(data.get("message"), "DeepSeek Harness assistant message")
                ignored += _add_dsh_assistant(builder, message.get("content"), str(index), at, asked)
            elif kind == "tool/result" and isinstance(data, Mapping):
                message = _mapping(data.get("message"), "DeepSeek Harness tool result")
                call_id = _clean_text(message.get("toolCallId"), "DeepSeek Harness toolCallId")
                text = _dsh_text(message.get("content"))
                answers = _dsh_answers(asked.get(call_id), text)
                builder.add(
                    ConversationMessageRole.TOOL_RESULT,
                    answers if answers is not None else text,
                    source_index=f"{index}.result",
                    tool_call_id=call_id,
                    tool_name=builder.tool_names.get(call_id, "unknown"),
                    tool_status=(
                        ConversationToolResultStatus.ERROR
                        if message.get("isError") is True
                        else ConversationToolResultStatus.COMPLETED
                    ),
                    occurred_at=at,
                )
            else:
                ignored += 1
        completed = bool(builder.messages and builder.messages[-1].role is ConversationMessageRole.COMPLETION)
        return builder.finish(after_turn=completed, ignored_items=ignored)


def _dsh_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    texts: list[str] = []
    for block in _sequence(content if content is not None else (), "DeepSeek Harness content"):
        text = block.get("text") if isinstance(block, Mapping) and block.get("type") == "text" else None
        if isinstance(text, str) and text.strip():
            texts.append(text)
    return "\n".join(texts).strip()


def _add_dsh_assistant(
    builder: _Builder, content: object, source: str, at: datetime, asked: dict[str, Mapping[str, object]]
) -> int:
    ignored = 0
    for block_index, raw in enumerate(_sequence(content if content is not None else (), "DeepSeek Harness assistant content")):
        block = _mapping(raw, "DeepSeek Harness content block")
        kind = block.get("type")
        text = block.get("text")
        if kind == "text" and isinstance(text, str) and text.strip():
            builder.add(ConversationMessageRole.COMPLETION, text.strip(), source_index=f"{source}.{block_index}", occurred_at=at)
        elif kind == "tool-call":
            call_id = _clean_text(block.get("id"), "DeepSeek Harness tool-call id")
            arguments = _tool_arguments(block.get("arguments") or {})
            if isinstance(arguments, Mapping):
                asked[call_id] = arguments
            builder.add(
                ConversationMessageRole.TOOL_CALL,
                arguments,
                source_index=f"{source}.{block_index}",
                tool_call_id=call_id,
                tool_name=_clean_text(block.get("name"), "DeepSeek Harness tool-call name"),
                occurred_at=at,
            )
        else:
            ignored += 1  # reasoning 等
    return ignored


def _dsh_answers(arguments: Mapping[str, object] | None, result: str) -> str | None:
    """``ask_user_question`` 的结构化作答 → 一行一题 ``"<题目>"="<作答>"``；不是它、或形状不对给 None（用原文）。"""

    questions = arguments.get("questions") if arguments is not None else None
    if not isinstance(questions, Sequence) or isinstance(questions, str):
        return None
    try:
        parsed = json.loads(result)
    except ValueError:
        return None
    answers = parsed.get("answers") if isinstance(parsed, Mapping) else None
    if not isinstance(answers, Sequence) or isinstance(answers, str):
        return None
    titles = {
        item.get("id"): item.get("question")
        for item in questions
        if isinstance(item, Mapping) and isinstance(item.get("question"), str)
    }
    lines: list[str] = []
    for answer in answers:
        if not isinstance(answer, Mapping) or answer.get("id") not in titles:
            return None
        selected = answer.get("selected")
        labels = [label for label in selected if isinstance(label, str) and label] if isinstance(selected, Sequence) else []
        custom = answer.get("custom")
        if isinstance(custom, str) and custom.strip():
            labels.append(custom.strip())
        lines.append(f'"{titles[answer.get("id")]}"="{"、".join(labels)}"')
    return "\n".join(lines) if lines else None


class ZCodeConversationAdapter(OpenCodeConversationAdapter):
    """适配 ZCode（zai-org/ZCode）：真会话在它的 SQLite 库里，一条消息 ``{info, parts}`` 与 OpenCode 同形，插件从库里镜像出来送来。
    与 OpenCode 的两处不同：提问工具是 ``AskUserQuestion``，作答直接写在结果文字里（``"<题目>"="<作答>"``，与 Claude Code 同形）；
    hook 注入的上下文 ZCode 前面加一行 ``<事件> hook additional context:``——插件剥掉回执块之后剩下的这行不是他说的话，不收
    （它在库里存成什么样待真实 ZCode 实测）。"""

    protocol = "zcode"
    host = "zcode"
    question_tools = (QuestionTool("zcode", "AskUserQuestion", "question"),)

    def adapt(self, payload: object, context: ConversationAdapterContext) -> ConversationAdaptation:
        records = [_without_hook_context(raw) for raw in _records(payload, "ZCode records")]
        return super().adapt({"records": records}, context)


_HOOK_CONTEXT_PREFIX = re.compile(r"^\s*(?:UserPromptSubmit|SessionStart|Stop) hook additional context:")


def _without_hook_context(raw: object) -> object:
    if not isinstance(raw, Mapping) or not isinstance(raw.get("parts"), Sequence):
        return raw
    parts = [
        {**part, "synthetic": True}
        if isinstance(part, Mapping)
        and part.get("type") == "text"
        and isinstance(part.get("text"), str)
        and _HOOK_CONTEXT_PREFIX.match(part["text"])
        else part
        for part in raw["parts"]
    ]
    return {**raw, "parts": parts}


def _add_opencode_part(
    builder: _Builder, role: str, part: Mapping[str, object], source: str, fallback: datetime
) -> int:
    part_type = part.get("type")
    if part_type == "text":
        text = part.get("text")
        if (
            part.get("synthetic") is True
            or part.get("ignored") is True
            or not isinstance(text, str)
            or not text.strip()
        ):
            return 1
        span = part.get("time")
        at = _epoch_ms(span.get("end") or span.get("start") if isinstance(span, Mapping) else None, fallback)
        builder.add(
            ConversationMessageRole.PROMPT if role == "user" else ConversationMessageRole.COMPLETION,
            text.strip(),
            source_index=source,
            occurred_at=at,
        )
        return 0
    if part_type != "tool" or role != "assistant":
        return 1
    state = _mapping(part.get("state"), "OpenCode tool state")
    status = state.get("status")
    if status not in {"completed", "error"}:
        return 1
    call_id = _clean_text(part.get("callID"), "OpenCode tool callID")
    name = _clean_text(part.get("tool"), "OpenCode tool name")
    span = state.get("time")
    started = _epoch_ms(span.get("start") if isinstance(span, Mapping) else None, fallback)
    ended = _epoch_ms(span.get("end") if isinstance(span, Mapping) else None, started)
    builder.add(
        ConversationMessageRole.TOOL_CALL,
        _tool_arguments(state.get("input") or {}),
        source_index=f"{source}.call",
        tool_call_id=call_id,
        tool_name=name,
        occurred_at=started,
    )
    answers = _opencode_answers(state.get("input"), state.get("metadata")) if name == "question" else None
    if status == "error":
        content: object = str(state.get("error") or "error")
    elif answers is not None:
        content = answers
    else:
        content = _tool_result_content(state.get("output"))
    builder.add(
        ConversationMessageRole.TOOL_RESULT,
        content,
        source_index=f"{source}.result",
        tool_call_id=call_id,
        tool_name=name,
        tool_status=ConversationToolResultStatus.ERROR if status == "error" else ConversationToolResultStatus.COMPLETED,
        occurred_at=ended,
    )
    return 0


def _opencode_answers(arguments: object, metadata: object) -> str | None:
    """question 工具自己记的结构化作答：``input.questions[i].question`` 配 ``metadata.answers[i]``（每题一组选中的标签）
    → 一行一题 ``"<题目>"="<作答>"``（多选用顿号连）。形状不对给 None（用工具原输出）。"""

    questions = arguments.get("questions") if isinstance(arguments, Mapping) else None
    answers = metadata.get("answers") if isinstance(metadata, Mapping) else None
    if (
        not isinstance(questions, Sequence)
        or isinstance(questions, str)
        or not isinstance(answers, Sequence)
        or isinstance(answers, str)
        or not questions
        or len(questions) != len(answers)
    ):
        return None
    lines: list[str] = []
    for item, chosen in zip(questions, answers, strict=True):
        question = item.get("question") if isinstance(item, Mapping) else None
        if not isinstance(question, str) or not isinstance(chosen, Sequence) or isinstance(chosen, str):
            return None
        labels = [label for label in chosen if isinstance(label, str) and label]
        lines.append(f'"{question}"="{"、".join(labels)}"')
    return "\n".join(lines)


def _epoch_ms(value: object, fallback: datetime) -> datetime:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        return fallback
    try:
        return datetime.fromtimestamp(value / 1000, tz=UTC)
    except (OverflowError, OSError, ValueError) as error:
        raise ConversationProtocolError(f"OpenCode timestamp is out of range: {value}") from error


def _message_sequence(payload: object) -> tuple[object, ...]:
    if isinstance(payload, Mapping):
        payload = payload.get("messages")
    return _sequence(payload, "messages")


def _records(payload: object, label: str) -> tuple[object, ...]:
    if isinstance(payload, Mapping):
        payload = payload.get("records")
    return _sequence(payload, label)


def _timestamp(value: object, fallback: datetime) -> datetime:
    if value is None:
        return fallback
    if not isinstance(value, str):
        raise ConversationProtocolError("record timestamp must be ISO-8601 text")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ConversationProtocolError("record timestamp must be ISO-8601 text") from exc
    if parsed.tzinfo is None:
        raise ConversationProtocolError("record timestamp must include timezone")
    return parsed.astimezone(UTC)


def _add_anthropic_content(
    builder: _Builder,
    *,
    role: str,
    content: object,
    source_prefix: str,
    occurred_at: datetime,
) -> int:
    if isinstance(content, str):
        builder.add(
            ConversationMessageRole.PROMPT if role == "user" else ConversationMessageRole.COMPLETION,
            _clean_text(content, "message content"),
            source_index=source_prefix,
            occurred_at=occurred_at,
        )
        return 0
    ignored = 0
    for block_index, raw_block in enumerate(_sequence(content, "message content")):
        block = _mapping(raw_block, "message content block")
        block_type = _clean_text(block.get("type"), "message content block type")
        source_index = f"{source_prefix}.{block_index}"
        if block_type == "text":
            builder.add(
                ConversationMessageRole.PROMPT if role == "user" else ConversationMessageRole.COMPLETION,
                _clean_text(block.get("text"), "message text"),
                source_index=source_index,
                occurred_at=occurred_at,
            )
        elif block_type == "tool_use":
            call_id = _clean_text(block.get("id"), "tool_use.id")
            name = _clean_text(block.get("name"), "tool_use.name")
            builder.add(
                ConversationMessageRole.TOOL_CALL,
                block.get("input", {}),
                source_index=source_index,
                tool_call_id=call_id,
                tool_name=name,
                occurred_at=occurred_at,
            )
        elif block_type == "tool_result":
            call_id = _clean_text(block.get("tool_use_id"), "tool_result.tool_use_id")
            result_name = builder.tool_names.get(call_id)
            if result_name is None:
                raise ConversationProtocolError("tool_result has no matching tool_use")
            builder.add(
                ConversationMessageRole.TOOL_RESULT,
                _tool_result_content(block.get("content")),
                source_index=source_index,
                tool_call_id=call_id,
                tool_name=result_name,
                tool_status=_tool_result_status_from_error_flag(block.get("is_error")),
                occurred_at=occurred_at,
            )
        else:
            ignored += 1
    return ignored


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ConversationProtocolError(f"{label} must be an object")
    return value


def _sequence(value: object, label: str) -> tuple[object, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ConversationProtocolError(f"{label} must be a sequence")
    return tuple(value)


def _clean_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConversationProtocolError(f"{label} must be non-empty text")
    return value.strip()


def _text_content(value: object) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    parts: list[str] = []
    for raw in _sequence(value, "message content"):
        block = _mapping(raw, "message content block")
        text = block.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
    return "\n".join(parts)


def _responses_text(value: object) -> str:
    return _text_content(value)


def _tool_arguments(value: object) -> object:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return value
        return dict(parsed) if isinstance(parsed, Mapping) else value
    raise ConversationProtocolError("tool arguments must be an object or non-empty JSON text")


def _tool_result_content(value: object) -> object:
    if value is None:
        return "null"
    if isinstance(value, str):
        return value or ""
    if isinstance(value, Sequence) and not isinstance(value, str):
        texts: list[str] = []
        for raw in value:
            if isinstance(raw, Mapping) and isinstance(raw.get("text"), str):
                texts.append(str(raw["text"]))
        if texts:
            return "\n".join(texts)
    return value


def _tool_result_status(value: object) -> ConversationToolResultStatus:
    """只接受协议显式给出的终态；缺失或非终态不能伪装成成功。"""

    if value is None:
        return ConversationToolResultStatus.UNKNOWN
    if not isinstance(value, str):
        raise ConversationProtocolError("tool result status must be text when provided")
    normalized = value.strip().casefold()
    if normalized in {"completed", "succeeded", "success"}:
        return ConversationToolResultStatus.COMPLETED
    if normalized in {"error", "failed", "failure"}:
        return ConversationToolResultStatus.ERROR
    if normalized in {"cancelled", "canceled"}:
        return ConversationToolResultStatus.CANCELLED
    return ConversationToolResultStatus.UNKNOWN


def _tool_result_status_from_error_flag(value: object) -> ConversationToolResultStatus:
    if value is True:
        return ConversationToolResultStatus.ERROR
    if value is False:
        return ConversationToolResultStatus.COMPLETED
    if value is None:
        return ConversationToolResultStatus.UNKNOWN
    raise ConversationProtocolError("tool_result.is_error must be boolean when provided")


__all__ = [
    "AnthropicMessagesConversationAdapter",
    "ClaudeCodeConversationAdapter",
    "CodexRolloutConversationAdapter",
    "ConversationAdaptation",
    "ConversationAdapterContext",
    "ConversationAdapterRegistry",
    "ConversationProtocolAdapter",
    "ConversationProtocolError",
    "OpenAIChatCompletionsConversationAdapter",
    "OpenAIResponsesConversationAdapter",
    "OpenClawConversationAdapter",
    "OpenCodeConversationAdapter",
]
