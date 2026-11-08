"""Conversation 完整轮次切段保留策略。"""

from __future__ import annotations

import math
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass

from habitus.foundation.integrity import canonical_json
from habitus.memory.conversation.segmentation import (
    SEMANTIC_CHUNKER_VERSION,
    ConversationBoundaryHints,
)
from habitus.pre.conversation.messages import (
    ConversationBatch,
    ConversationMessage,
    ConversationMessageRole,
)


class ConversationRetentionError(ValueError):
    """Conversation 无法形成安全、有界的保留计划。"""


ConversationTokenEstimator = Callable[[ConversationMessage], int]


_WIDE_EAST_ASIAN_WIDTHS = frozenset({"W", "F"})


def _character_token_cost(character: str) -> float:
    """单个字符折算的 Token 数，按 Unicode 东亚宽度（UAX #11）分三档。

    宽字符与全角字符（W / F：汉字、假名、韩文音节、全角标点与字母、多数 emoji）记 1.5；其余落在基本多文种平面
    以外的字符记 2.0；剩下的拉丁字母、数字、半角标点和空白记 0.25。
    """

    if unicodedata.east_asian_width(character) in _WIDE_EAST_ASIAN_WIDTHS:
        return 1.5
    if ord(character) > 0xFFFF:
        return 2.0
    return 0.25


def estimate_conversation_message_tokens(message: ConversationMessage) -> int:
    """提供与供应商无关的保守 Token 估算；运行时仍可注入精确 tokenizer。"""

    rendered = canonical_json(message.to_dict())
    return max(1, math.ceil(sum(_character_token_cost(character) for character in rendered)))


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
    max_segment_tokens: int = 12_000
    semantic_boundary_min_ratio: float = 0.60
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
            if not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}")
        if not 1 <= self.max_segment_tokens <= 10_000_000:
            raise ValueError("max_segment_tokens must be a positive bounded integer")
        if not 0.1 <= float(self.semantic_boundary_min_ratio) <= 1.0:
            raise ValueError("semantic_boundary_min_ratio must be between 0.1 and 1.0")
        object.__setattr__(
            self,
            "semantic_boundary_min_ratio",
            float(self.semantic_boundary_min_ratio),
        )


@dataclass(frozen=True)
class ConversationTurn:
    """由 prompt 锚定且不会在工具调用中间切开的逻辑轮次。"""

    messages: tuple[ConversationMessage, ...]

    def __post_init__(self) -> None:
        if not self.messages:
            raise ConversationRetentionError("conversation turn cannot be empty")

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
    boundary_kind: str | None = None
    embedding_fingerprint: str | None = None
    chunker_version: str = SEMANTIC_CHUNKER_VERSION

    def __post_init__(self) -> None:
        if self.pending_tokens < 0:
            raise ValueError("pending_tokens must be a non-negative integer")
        if not self.reason.strip():
            raise ValueError("retention reason must be non-empty text")
        if self.boundary_kind is not None and not self.boundary_kind.strip():
            raise ValueError("boundary_kind must be non-empty text or None")
        if self.embedding_fingerprint is not None and not self.embedding_fingerprint.strip():
            raise ValueError("embedding_fingerprint must be non-empty text or None")
        if not self.chunker_version.strip():
            raise ValueError("chunker_version must be non-empty text")
        if self.through_sequence is None:
            if self.archive_messages:
                raise ValueError("retention plan without a boundary cannot archive messages")
        elif not self.archive_messages or self.archive_messages[-1].sequence != self.through_sequence:
            raise ValueError("retention boundary must match the archived prefix")

    @property
    def should_seal(self) -> bool:
        return self.through_sequence is not None


# TODO(conversation-source): 修改此规划算法语义时，必须同步提升
# memory/workflow/conversation_consumer.py 中 retention_planner_version。
class ConversationRetentionPlanner:
    """只输出非重叠且可安全回收的历史前缀规划。"""

    def __init__(
        self,
        config: ConversationSegmentationConfig | None = None,
        *,
        token_estimator: ConversationTokenEstimator | None = None,
    ) -> None:
        self.config = config or ConversationSegmentationConfig()
        self.token_estimator = token_estimator or estimate_conversation_message_tokens

    def plan(
        self,
        live: ConversationBatch | None,
        *,
        after_turn: bool = False,
        flush: bool = False,
        drain_pending: bool = False,
        boundary_hints: ConversationBoundaryHints | None = None,
        leading_continuation: bool = False,
    ) -> ConversationRetentionPlan:
        """在完整轮次已闭合后选择连续、安全且不超过硬上限的语义前缀。"""

        if drain_pending and (not after_turn or flush):
            raise ValueError("drain_pending is only valid while completing one afterTurn commit")
        if live is None:
            return self._empty(flush=flush, reason="live conversation is empty")
        # 消息序号一经写入不可改；后续封存只裁剪前缀，因此完整 live 快照产生的
        # 边界提示仍可安全用于它的连续后缀。新追加内容没有提示时自动使用结构分数。

        messages = live.messages
        if not after_turn and not flush:
            return self._empty(
                flush=False,
                reason="conversation remains live until the afterTurn boundary",
                retained_messages=messages,
            )

        turns = self._turns(messages)
        self._require_complete_turns(
            turns,
            leading_continuation=leading_continuation,
        )
        retained_turns = () if flush else self._retained_turns(turns)
        archive_turn_count = len(turns) - len(retained_turns)
        pending_messages = self._flatten_turns(turns[:archive_turn_count])
        if not flush:
            pending_messages = self._extend_pending_for_oversized_tail(
                messages,
                pending_messages,
            )
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
        if not pending_messages:
            return ConversationRetentionPlan(
                through_sequence=None,
                archive_messages=(),
                retained_messages=messages,
                triggered=True,
                flush=flush,
                pending_tokens=0,
                budget_exceeded=self._live_safety_exceeded(messages),
                reason=f"{trigger_reason}; retained window does not require a safe prefix",
            )

        archived, boundary_kind = self._bounded_archive_prefix(
            pending_messages,
            boundary_hints=boundary_hints,
        )
        retained = messages[len(archived) :]
        embedding_fingerprint = boundary_hints.embedding_fingerprint if boundary_hints is not None else None
        return ConversationRetentionPlan(
            through_sequence=archived[-1].sequence,
            archive_messages=archived,
            retained_messages=retained,
            triggered=True,
            flush=flush,
            pending_tokens=pending_tokens,
            budget_exceeded=self._live_safety_exceeded(retained),
            reason=f"{trigger_reason}; selected {boundary_kind} boundary",
            boundary_kind=boundary_kind,
            embedding_fingerprint=embedding_fingerprint,
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
        if self._tokens(messages) > self.config.retained_message_token_budget:
            return True, "retained conversation token budget exceeded"
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

    def _extend_pending_for_oversized_tail(
        self,
        messages: tuple[ConversationMessage, ...],
        pending: tuple[ConversationMessage, ...],
    ) -> tuple[ConversationMessage, ...]:
        """当最近轮次本身过大时，把其最小安全前缀也纳入待封存范围。"""

        if len(pending) == len(messages):
            return pending
        tail_start = len(pending)
        tail = messages[tail_start:]
        if self._within_live_window(tail):
            return pending
        safe_sequences = set(self._safe_boundaries(tail))
        for offset, message in enumerate(tail, start=tail_start):
            if message.sequence not in safe_sequences:
                continue
            remainder = messages[offset + 1 :]
            if not remainder or self._within_live_window(remainder):
                return messages[: offset + 1]
        raise ConversationRetentionError(
            "the retained conversation tail has no safe boundary within its configured budget"
        )

    def _bounded_archive_prefix(
        self,
        messages: tuple[ConversationMessage, ...],
        *,
        boundary_hints: ConversationBoundaryHints | None,
    ) -> tuple[tuple[ConversationMessage, ...], str]:
        if self._within_segment_budget(messages):
            return messages, self._boundary_kind(messages, len(messages) - 1)

        safe_sequences = set(self._safe_boundaries(messages))
        candidates: list[tuple[float, int, str]] = []
        minimum_tokens = int(self.config.max_segment_tokens * self.config.semantic_boundary_min_ratio)
        for index, message in enumerate(messages):
            if message.sequence not in safe_sequences:
                continue
            prefix = messages[: index + 1]
            if not self._within_segment_budget(prefix):
                continue
            tokens = self._tokens(prefix)
            boundary_kind = self._boundary_kind(messages, index)
            structural = {
                "turn": 40.0,
                "assistant_step": 30.0,
                "message": 20.0,
                "content": 10.0,
            }[boundary_kind]
            semantic = 0.0
            if boundary_hints is not None:
                semantic = boundary_hints.distance_after(message.sequence) or 0.0
            budget_closeness = tokens / self.config.max_segment_tokens
            score = structural + semantic * 2.0 + budget_closeness
            candidates.append((score, index, boundary_kind))

        if not candidates:
            raise ConversationRetentionError("one atomic tool exchange exceeds the configured segment bound")
        preferred = [item for item in candidates if self._tokens(messages[: item[1] + 1]) >= minimum_tokens]
        selected = max(preferred or candidates, key=lambda item: (item[0], item[1]))
        return messages[: selected[1] + 1], selected[2]

    def atomic_spans(self, messages: tuple[ConversationMessage, ...]) -> tuple[tuple[int, int], ...]:
        """按安全切点把消息分成 ``[start, end)`` 下标区间：区间里面切不开，一组工具交换及连着它的消息只能整组进段。"""

        safe_sequences = set(self._safe_boundaries(messages))
        spans: list[tuple[int, int]] = []
        start = 0
        for index, message in enumerate(messages):
            if message.sequence in safe_sequences:
                spans.append((start, index + 1))
                start = index + 1
        if start < len(messages):
            spans.append((start, len(messages)))
        return tuple(spans)

    def fits_segment(self, messages: tuple[ConversationMessage, ...]) -> bool:
        """这些消息能否放进一段（消息数、字节、Token 三个上限）。"""

        return self._within_segment_budget(messages)

    def _live_safety_exceeded(self, messages: tuple[ConversationMessage, ...]) -> bool:
        return not self._within_live_window(messages)

    def _within_live_window(self, messages: tuple[ConversationMessage, ...]) -> bool:
        return (
            len(messages) <= self.config.max_live_messages
            and self._bytes(messages) <= self.config.max_live_bytes
            and self._tokens(messages) <= self.config.retained_message_token_budget
        )

    def _within_segment_budget(self, messages: tuple[ConversationMessage, ...]) -> bool:
        if len(messages) > self.config.max_segment_messages:
            return False
        if self._bytes(messages) > self.config.max_segment_bytes:
            return False
        return self._tokens(messages) <= self.config.max_segment_tokens

    def _tokens(self, messages: tuple[ConversationMessage, ...]) -> int:
        total = 0
        for message in messages:
            value = self.token_estimator(message)
            if value < 0:
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
            if message.starts_turn or not grouped:
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
    def _require_complete_turns(
        turns: tuple[ConversationTurn, ...],
        *,
        leading_continuation: bool,
    ) -> None:
        for index, turn in enumerate(turns):
            if not turn.messages[0].starts_turn and not (index == 0 and leading_continuation):
                raise ConversationRetentionError("every conversation turn must start with a prompt")
            if turn.messages[-1].role is not ConversationMessageRole.COMPLETION:
                raise ConversationRetentionError("afterTurn requires every turn to end with a final completion")
            if not turn.messages[-1].completes_logical_message:
                raise ConversationRetentionError(
                    "afterTurn requires the final completion logical message to be complete"
                )
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
    def _safe_boundaries(messages: tuple[ConversationMessage, ...]) -> tuple[int, ...]:
        """返回不会切开 tool_call/tool_result 的消息序号。"""

        open_calls: set[str] = set()
        boundaries: list[int] = []
        for index, message in enumerate(messages):
            if message.role is ConversationMessageRole.TOOL_CALL:
                assert message.tool_call_id is not None
                open_calls.add(message.tool_call_id)
            elif message.role is ConversationMessageRole.TOOL_RESULT:
                assert message.tool_call_id is not None
                open_calls.discard(message.tool_call_id)
            if open_calls:
                continue
            following = messages[index + 1] if index + 1 < len(messages) else None
            if (
                message.role is ConversationMessageRole.COMPLETION
                and message.completes_logical_message
                and following is not None
                and not following.starts_turn
            ):
                # 中间 completion 后仍有同轮工具步骤时不在此切割，确保 Segment
                # 仅凭首尾角色就能确定 starts/ends_mid_turn。
                continue
            boundaries.append(message.sequence)
        return tuple(boundaries)

    @staticmethod
    def _boundary_kind(
        messages: tuple[ConversationMessage, ...],
        index: int,
    ) -> str:
        current = messages[index]
        following = messages[index + 1] if index + 1 < len(messages) else None
        if (
            current.role is ConversationMessageRole.COMPLETION
            and current.completes_logical_message
            and (following is None or following.starts_turn)
        ):
            return "turn"
        if current.role is ConversationMessageRole.TOOL_RESULT:
            return "assistant_step"
        if (
            current.logical_message_id is not None
            and following is not None
            and following.logical_message_id == current.logical_message_id
        ):
            return "content"
        return "message"

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


__all__ = [
    "ConversationSegmentationConfig",
    "ConversationRetentionError",
    "ConversationRetentionPlan",
    "ConversationRetentionPlanner",
    "ConversationTokenEstimator",
    "ConversationTurn",
    "estimate_conversation_message_tokens",
]
