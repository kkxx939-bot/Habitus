"""把主流 Agent/模型协议转换为 m2bOS 的严格 Conversation 事实。"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from foundation.integrity import canonical_digest
from pre.conversation.messages import (
    ConversationBatch,
    ConversationMessage,
    ConversationMessageRole,
    ConversationToolResultContentMode,
    ConversationToolResultStatus,
)


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
        object.__setattr__(self, "occurred_at", self.occurred_at.astimezone(timezone.utc))


@dataclass(frozen=True)
class ConversationAdaptation:
    """转换后的批次及协议层明确忽略的系统元数据数量。"""

    batch: ConversationBatch
    after_turn: bool
    ignored_items: int = 0


class ConversationProtocolAdapter(Protocol):
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

    @classmethod
    def with_builtins(cls) -> ConversationAdapterRegistry:
        registry = cls()
        registry.register(OpenAIChatCompletionsConversationAdapter())
        registry.register(OpenAIResponsesConversationAdapter())
        registry.register(AnthropicMessagesConversationAdapter())
        registry.register(CodexRolloutConversationAdapter())
        registry.register(ClaudeCodeConversationAdapter())
        registry.register(OpenClawConversationAdapter())
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
                ConversationToolResultContentMode.INLINE
                if role is ConversationMessageRole.TOOL_RESULT
                else None
            ),
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

    def adapt(self, payload: object, context: ConversationAdapterContext) -> ConversationAdaptation:
        builder = _Builder(self.protocol, context)
        ignored = 0
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
                if item_type != "sub_agent_activity":
                    ignored += 1
                    continue
                activity = {
                    key: item[key]
                    for key in ("agent_thread_id", "agent_path", "kind", "occurred_at_ms")
                    if key in item
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
                if not content:
                    ignored += 1
                    continue
                builder.add(
                    ConversationMessageRole.PROMPT if role == "user" else ConversationMessageRole.COMPLETION,
                    content,
                    source_index=str(index),
                    occurred_at=occurred_at,
                )
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
                    key: item[key]
                    for key in ("agent_thread_id", "agent_path", "kind", "occurred_at_ms")
                    if key in item
                }
                builder.add(
                    ConversationMessageRole.COMPLETION,
                    "[sub-agent-activity]\n" + json.dumps(activity, ensure_ascii=False, sort_keys=True),
                    source_index=str(index),
                    occurred_at=occurred_at,
                )
            elif item_type in {"function_call", "custom_tool_call", "tool_search_call"}:
                call_id = _clean_text(item.get("call_id", item.get("id")), "Codex function_call.call_id")
                name = "tool_search" if item_type == "tool_search_call" else _clean_text(
                    item.get("name"),
                    "Codex function_call.name",
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

    def adapt(self, payload: object, context: ConversationAdapterContext) -> ConversationAdaptation:
        builder = _Builder(self.protocol, context)
        ignored = 0
        for index, raw in enumerate(_records(payload, "Claude Code records")):
            record = _mapping(raw, f"records[{index}]")
            record_type = record.get("type")
            if record_type not in {"user", "assistant"} or record.get("isSidechain") is True or record.get("isMeta") is True:
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
    return parsed.astimezone(timezone.utc)


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
]
