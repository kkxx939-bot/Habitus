"""Conversation 完整轮次切段与超大工具结果保留策略。"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from foundation.integrity import canonical_json
from pre.conversation.messages import (
    ConversationBatch,
    ConversationMessage,
    ConversationMessageRole,
    ConversationToolResultContentMode,
)


class ConversationRetentionError(ValueError):
    """Conversation 无法形成安全、有界的保留计划。"""


ConversationTokenEstimator = Callable[[ConversationMessage], int]


def _token_weight(character: str) -> float:
    code_point = ord(character)
    if (
        0x3400 <= code_point <= 0x4DBF
        or 0x4E00 <= code_point <= 0x9FFF
        or 0xF900 <= code_point <= 0xFAFF
        or 0x20000 <= code_point <= 0x2EBEF
        or 0x3040 <= code_point <= 0x30FF
        or 0x31F0 <= code_point <= 0x31FF
        or 0xAC00 <= code_point <= 0xD7AF
        or 0x1100 <= code_point <= 0x11FF
        or 0x3130 <= code_point <= 0x318F
        or 0xFF00 <= code_point <= 0xFFEF
        or 0x3000 <= code_point <= 0x303F
    ):
        return 1.5
    if code_point > 0xFFFF:
        return 2.0
    return 0.25


def estimate_conversation_message_tokens(message: ConversationMessage) -> int:
    """提供与供应商无关的保守 Token 估算；运行时仍可注入精确 tokenizer。"""

    if not isinstance(message, ConversationMessage):
        raise TypeError("message must be a ConversationMessage")
    rendered = canonical_json(message.to_dict())
    return max(1, math.ceil(sum(_token_weight(character) for character in rendered)))


@dataclass(frozen=True)
class ConversationSegmentationConfig:
    """自动切段和工具结果降载的显式边界。"""

    commit_token_threshold: int = 12_000
    keep_recent_turn_count: int = 3
    retained_message_token_budget: int = 12_000
    max_live_messages: int = 200
    max_live_bytes: int = 2 * 1024 * 1024
    max_segment_messages: int = 500
    max_segment_bytes: int = 1024 * 1024
    max_segment_tokens: int | None = None
    max_inline_tool_result_bytes: int = 64 * 1024
    max_tool_result_summary_chars: int = 4_000

    def __post_init__(self) -> None:
        for name, value, minimum, maximum in (
            ("commit_token_threshold", self.commit_token_threshold, 1, 10_000_000),
            ("keep_recent_turn_count", self.keep_recent_turn_count, 0, 10_000),
            (
                "retained_message_token_budget",
                self.retained_message_token_budget,
                1,
                10_000_000,
            ),
            ("max_live_messages", self.max_live_messages, 1, 100_000),
            ("max_live_bytes", self.max_live_bytes, 1, 64 * 1024 * 1024),
            ("max_segment_messages", self.max_segment_messages, 1, 100_000),
            ("max_segment_bytes", self.max_segment_bytes, 1, 64 * 1024 * 1024),
            (
                "max_inline_tool_result_bytes",
                self.max_inline_tool_result_bytes,
                1,
                16 * 1024 * 1024,
            ),
            (
                "max_tool_result_summary_chars",
                self.max_tool_result_summary_chars,
                128,
                100_000,
            ),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}")
        for token_name, token_value in (("max_segment_tokens", self.max_segment_tokens),):
            if token_value is not None and (
                isinstance(token_value, bool) or not isinstance(token_value, int) or not 1 <= token_value <= 10_000_000
            ):
                raise ValueError(f"{token_name} must be null or a positive bounded integer")


@dataclass(frozen=True)
class ConversationTurn:
    """由 prompt 锚定且不会在工具调用中间切开的逻辑轮次。"""

    messages: tuple[ConversationMessage, ...]

    def __post_init__(self) -> None:
        if not self.messages:
            raise ConversationRetentionError("conversation turn cannot be empty")
        if any(not isinstance(message, ConversationMessage) for message in self.messages):
            raise TypeError("conversation turn must contain ConversationMessage values")

    @property
    def start_sequence(self) -> int:
        return self.messages[0].sequence

    @property
    def end_sequence(self) -> int:
        return self.messages[-1].sequence


@dataclass(frozen=True)
class ConversationRetentionPlan:
    """一次纯切段判断的完整结果。"""

    through_sequence: int | None
    archive_messages: tuple[ConversationMessage, ...]
    retained_messages: tuple[ConversationMessage, ...]
    triggered: bool
    flush: bool
    pending_tokens: int
    budget_exceeded: bool
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.triggered, bool) or not isinstance(self.flush, bool):
            raise TypeError("retention flags must be boolean")
        if isinstance(self.pending_tokens, bool) or not isinstance(self.pending_tokens, int) or self.pending_tokens < 0:
            raise ValueError("pending_tokens must be a non-negative integer")
        if not isinstance(self.budget_exceeded, bool):
            raise TypeError("budget_exceeded must be boolean")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("retention reason must be non-empty text")
        if self.through_sequence is None:
            if self.archive_messages:
                raise ValueError("retention plan without a boundary cannot archive messages")
        elif not self.archive_messages or self.archive_messages[-1].sequence != self.through_sequence:
            raise ValueError("retention boundary must match the archived prefix")

    @property
    def should_seal(self) -> bool:
        return self.through_sequence is not None


class ConversationRetentionPlanner:
    """参考 OpenViking 的纯规划方式，但只输出非重叠的安全前缀。"""

    def __init__(
        self,
        config: ConversationSegmentationConfig | None = None,
        *,
        token_estimator: ConversationTokenEstimator | None = None,
    ) -> None:
        if config is not None and not isinstance(config, ConversationSegmentationConfig):
            raise TypeError("config must be ConversationSegmentationConfig")
        if token_estimator is not None and not callable(token_estimator):
            raise TypeError("token_estimator must be callable or None")
        self.config = config or ConversationSegmentationConfig()
        self.token_estimator = token_estimator or estimate_conversation_message_tokens

    def plan(
        self,
        live: ConversationBatch | None,
        *,
        after_turn: bool = False,
        flush: bool = False,
        drain_pending: bool = False,
    ) -> ConversationRetentionPlan:
        """在 afterTurn 或显式 flush 时选择完整、连续且不重复的封存前缀。"""

        if any(not isinstance(value, bool) for value in (after_turn, flush, drain_pending)):
            raise TypeError("retention control flags must be boolean")
        if drain_pending and (not after_turn or flush):
            raise ValueError("drain_pending is only valid while completing one afterTurn commit")
        if live is None:
            return self._empty(flush=flush, reason="live conversation is empty")
        if not isinstance(live, ConversationBatch):
            raise TypeError("live must be ConversationBatch or None")

        messages = live.messages
        if not after_turn and not flush:
            return self._empty(
                flush=False,
                reason="conversation remains live until the afterTurn boundary",
                retained_messages=messages,
            )

        turns = self._turns(messages)
        self._require_complete_turns(turns)
        retained_turns = () if flush else self._retained_turns(turns)
        archive_turn_count = len(turns) - len(retained_turns)
        pending_turns = turns[:archive_turn_count]
        pending_messages = self._flatten_turns(pending_turns)
        pending_tokens = self._tokens(pending_messages)
        triggered, trigger_reason = self._trigger(
            messages,
            pending_tokens=pending_tokens,
            flush=flush,
            drain_pending=drain_pending,
        )
        if not triggered:
            return ConversationRetentionPlan(
                through_sequence=None,
                archive_messages=(),
                retained_messages=messages,
                triggered=False,
                flush=False,
                pending_tokens=pending_tokens,
                budget_exceeded=self._live_safety_exceeded(messages),
                reason=trigger_reason,
            )
        if not pending_turns:
            return ConversationRetentionPlan(
                through_sequence=None,
                archive_messages=(),
                retained_messages=messages,
                triggered=True,
                flush=flush,
                pending_tokens=0,
                budget_exceeded=self._live_safety_exceeded(messages),
                reason=f"{trigger_reason}; newest complete turn is retained as one atomic unit",
            )

        selected_turns = self._bounded_archive_prefix(pending_turns)
        archived = self._flatten_turns(selected_turns)
        retained = messages[len(archived) :]
        return ConversationRetentionPlan(
            through_sequence=archived[-1].sequence,
            archive_messages=archived,
            retained_messages=retained,
            triggered=True,
            flush=flush,
            pending_tokens=pending_tokens,
            budget_exceeded=self._live_safety_exceeded(retained),
            reason=trigger_reason,
        )

    def _trigger(
        self,
        messages: tuple[ConversationMessage, ...],
        *,
        pending_tokens: int,
        flush: bool,
        drain_pending: bool,
    ) -> tuple[bool, str]:
        if flush:
            return True, "explicit conversation flush requested"
        if drain_pending and pending_tokens > 0:
            return True, "draining the remaining eligible messages from one afterTurn commit"
        if pending_tokens >= self.config.commit_token_threshold:
            return True, "pending conversation token threshold reached"
        if len(messages) > self.config.max_live_messages:
            return True, "live message count exceeded"
        if self._bytes(messages) > self.config.max_live_bytes:
            return True, "live byte budget exceeded"
        return False, "pending conversation remains below the commit threshold"

    def _retained_turns(
        self,
        turns: tuple[ConversationTurn, ...],
    ) -> tuple[ConversationTurn, ...]:
        maximum = self.config.keep_recent_turn_count
        if not turns or maximum == 0:
            return ()
        selected = [turns[-1]]
        selected_tokens = self._tokens(turns[-1].messages)
        for turn in reversed(turns[:-1]):
            if len(selected) >= maximum:
                break
            turn_tokens = self._tokens(turn.messages)
            if selected_tokens + turn_tokens > self.config.retained_message_token_budget:
                break
            selected.insert(0, turn)
            selected_tokens += turn_tokens
        return tuple(selected)

    def _bounded_archive_prefix(
        self,
        turns: tuple[ConversationTurn, ...],
    ) -> tuple[ConversationTurn, ...]:
        selected: list[ConversationTurn] = []
        for turn in turns:
            candidate = self._flatten_turns((*selected, turn))
            if not self._within_segment_budget(candidate):
                if not selected:
                    raise ConversationRetentionError("the oldest complete turn exceeds the configured segment bound")
                break
            selected.append(turn)
        return tuple(selected)

    def _live_safety_exceeded(self, messages: tuple[ConversationMessage, ...]) -> bool:
        if len(messages) > self.config.max_live_messages:
            return True
        if self._bytes(messages) > self.config.max_live_bytes:
            return True
        turns = self._turns(messages) if messages else ()
        retained = self._retained_turns(turns)
        return (
            bool(retained) and self._tokens(self._flatten_turns(retained)) > self.config.retained_message_token_budget
        )

    def _within_segment_budget(self, messages: tuple[ConversationMessage, ...]) -> bool:
        if len(messages) > self.config.max_segment_messages:
            return False
        if self._bytes(messages) > self.config.max_segment_bytes:
            return False
        return not (
            self.config.max_segment_tokens is not None and self._tokens(messages) > self.config.max_segment_tokens
        )

    def _tokens(self, messages: tuple[ConversationMessage, ...]) -> int:
        total = 0
        for message in messages:
            value = self.token_estimator(message)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ConversationRetentionError("token_estimator must return a non-negative integer")
            total += value
        return total

    @staticmethod
    def _bytes(messages: tuple[ConversationMessage, ...]) -> int:
        return sum(len(canonical_json(message.to_dict()).encode("utf-8")) + 1 for message in messages)

    @staticmethod
    def _turns(messages: tuple[ConversationMessage, ...]) -> tuple[ConversationTurn, ...]:
        grouped: list[list[ConversationMessage]] = []
        for message in messages:
            if message.role is ConversationMessageRole.PROMPT or not grouped:
                grouped.append([message])
            else:
                grouped[-1].append(message)
        return tuple(ConversationTurn(tuple(group)) for group in grouped)

    @staticmethod
    def _flatten_turns(
        turns: tuple[ConversationTurn, ...],
    ) -> tuple[ConversationMessage, ...]:
        return tuple(message for turn in turns for message in turn.messages)

    @staticmethod
    def _require_complete_turns(turns: tuple[ConversationTurn, ...]) -> None:
        for turn in turns:
            if turn.messages[0].role is not ConversationMessageRole.PROMPT:
                raise ConversationRetentionError("every conversation turn must start with a prompt")
            if turn.messages[-1].role is not ConversationMessageRole.COMPLETION:
                raise ConversationRetentionError("afterTurn requires every turn to end with a final completion")
            calls = {
                message.tool_call_id for message in turn.messages if message.role is ConversationMessageRole.TOOL_CALL
            }
            results = {
                message.tool_call_id for message in turn.messages if message.role is ConversationMessageRole.TOOL_RESULT
            }
            if calls != results:
                raise ConversationRetentionError(
                    "afterTurn requires every tool_call to have one terminal tool_result in the same turn"
                )

    @staticmethod
    def _empty(
        *,
        flush: bool,
        reason: str,
        retained_messages: tuple[ConversationMessage, ...] = (),
    ) -> ConversationRetentionPlan:
        return ConversationRetentionPlan(
            through_sequence=None,
            archive_messages=(),
            retained_messages=retained_messages,
            triggered=False,
            flush=flush,
            pending_tokens=0,
            budget_exceeded=False,
            reason=reason,
        )


class ConversationToolResultReducer:
    """在写入 Conversation 前压缩超大工具结果，不建立旁路原文存储。"""

    def __init__(self, config: ConversationSegmentationConfig | None = None) -> None:
        if config is not None and not isinstance(config, ConversationSegmentationConfig):
            raise TypeError("config must be ConversationSegmentationConfig")
        self.config = config or ConversationSegmentationConfig()

    def reduce(
        self,
        message: ConversationMessage,
        *,
        force_omit: bool = False,
        description: str | None = None,
    ) -> ConversationMessage:
        """保留小结果；把大文本概括，把媒体或不可保留结果明确省略。"""

        if not isinstance(message, ConversationMessage):
            raise TypeError("message must be ConversationMessage")
        if message.role is not ConversationMessageRole.TOOL_RESULT:
            raise ValueError("only tool_result messages can be reduced")
        if not isinstance(force_omit, bool):
            raise TypeError("force_omit must be boolean")
        encoded = canonical_json(message.content).encode("utf-8")
        size = len(encoded)
        digest = hashlib.sha256(encoded).hexdigest()
        if message.content_mode is not ConversationToolResultContentMode.INLINE:
            return message
        if not force_omit and size <= self.config.max_inline_tool_result_bytes:
            return message

        if force_omit:
            content = self._description(
                description,
                fallback="工具结果属于媒体、二进制、下载载荷或不可长期保存内容，原始载荷未保存。",
            )
            mode = ConversationToolResultContentMode.OMITTED
        else:
            content = self._summarize(message.content)
            mode = ConversationToolResultContentMode.SUMMARIZED
        return ConversationMessage(
            message_id=message.message_id,
            sequence=message.sequence,
            role=message.role,
            occurred_at=message.occurred_at,
            content=content,
            tool_call_id=message.tool_call_id,
            tool_name=message.tool_name,
            tool_status=message.tool_status,
            content_mode=mode,
            source_ref=message.source_ref,
            original_size_bytes=size,
            original_sha256=digest,
        )

    def _summarize(self, value: Any) -> str:
        maximum = self.config.max_tool_result_summary_chars
        if isinstance(value, Mapping):
            keys = [str(key) for key in list(value)[:32]]
            prefix = f"工具结果已压缩：JSON 对象，共 {len(value)} 个顶层字段；字段：{', '.join(keys)}。\n"
        elif isinstance(value, list | tuple):
            prefix = f"工具结果已压缩：JSON 数组，共 {len(value)} 项。\n"
        elif isinstance(value, str):
            prefix = f"工具结果已压缩：文本，共 {len(value)} 个字符。\n"
        else:
            prefix = f"工具结果已压缩：{type(value).__name__}。\n"
        rendered = json.dumps(value, ensure_ascii=False, sort_keys=True) if not isinstance(value, str) else value
        remaining = max(0, maximum - len(prefix))
        if len(rendered) <= remaining:
            sample = rendered
        elif remaining >= 64:
            head = max(1, remaining * 2 // 3)
            tail = max(1, remaining - head - 5)
            sample = f"{rendered[:head]}\n...\n{rendered[-tail:]}"
        else:
            sample = rendered[:remaining]
        return (prefix + sample)[:maximum]

    @staticmethod
    def _description(value: str | None, *, fallback: str) -> str:
        if value is None:
            return fallback
        if not isinstance(value, str) or not value.strip():
            raise ValueError("description must be non-empty text or None")
        return value.strip()


__all__ = [
    "ConversationSegmentationConfig",
    "ConversationRetentionError",
    "ConversationRetentionPlan",
    "ConversationRetentionPlanner",
    "ConversationTokenEstimator",
    "ConversationToolResultReducer",
    "ConversationTurn",
    "estimate_conversation_message_tokens",
]
