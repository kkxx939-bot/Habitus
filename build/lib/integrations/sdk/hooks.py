"""可注册到不同 Agent 运行时的低耦合记忆生命周期 Hook。"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from integrations.sdk.contracts import (
    AgentFlushResult,
    AgentMemoryPort,
    AgentRecallResult,
    AgentRememberResult,
    ConversationRef,
)

_SESSION_SCHEMA = "agent_hook_session_v1"
_PREPARED_TURN_SCHEMA = "prepared_agent_turn_v1"


def _canonical_payload(value: object) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("payload must be canonical JSON data") from exc


@dataclass(frozen=True)
class AgentHookSession:
    """由 Agent 自己持久化的最小会话状态；不隐藏服务端序号。"""

    conversation_id: str
    started_on: date
    protocol: str
    next_sequence: int = 0

    def __post_init__(self) -> None:
        conversation = ConversationRef(self.conversation_id, self.started_on)
        if not isinstance(self.protocol, str) or not self.protocol or self.protocol != self.protocol.strip().lower():
            raise ValueError("protocol must be non-empty normalized lowercase text")
        if (
            isinstance(self.next_sequence, bool)
            or not isinstance(self.next_sequence, int)
            or self.next_sequence < 0
        ):
            raise ValueError("next_sequence must be a non-negative integer")
        object.__setattr__(self, "conversation_id", conversation.conversation_id)

    @property
    def conversation(self) -> ConversationRef:
        return ConversationRef(self.conversation_id, self.started_on)

    def advance(self, next_sequence: int) -> AgentHookSession:
        """仅在服务确认写入后生成新的不可变游标状态。"""

        if (
            isinstance(next_sequence, bool)
            or not isinstance(next_sequence, int)
            or next_sequence <= self.next_sequence
        ):
            raise ValueError("confirmed next_sequence must advance the session cursor")
        return AgentHookSession(
            conversation_id=self.conversation_id,
            started_on=self.started_on,
            protocol=self.protocol,
            next_sequence=next_sequence,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": _SESSION_SCHEMA,
            "conversation_id": self.conversation_id,
            "started_on": self.started_on.isoformat(),
            "protocol": self.protocol,
            "next_sequence": self.next_sequence,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> AgentHookSession:
        """从 Agent 自己的 checkpoint 恢复显式 Hook 状态。"""

        if not isinstance(value, Mapping):
            raise TypeError("session state must be a mapping")
        expected = {"schema_version", "conversation_id", "started_on", "protocol", "next_sequence"}
        if set(value) != expected or value.get("schema_version") != _SESSION_SCHEMA:
            raise ValueError("session state has an unsupported schema")
        started_on = value["started_on"]
        if not isinstance(started_on, str):
            raise TypeError("session started_on must be ISO-8601 text")
        try:
            parsed_date = date.fromisoformat(started_on)
        except ValueError as exc:
            raise ValueError("session started_on must be an ISO-8601 date") from exc
        conversation_id = value["conversation_id"]
        protocol = value["protocol"]
        next_sequence = value["next_sequence"]
        if not isinstance(conversation_id, str) or not isinstance(protocol, str):
            raise TypeError("session conversation_id and protocol must be text")
        if isinstance(next_sequence, bool) or not isinstance(next_sequence, int):
            raise TypeError("session next_sequence must be an integer")
        return cls(conversation_id, parsed_date, protocol, next_sequence)


@dataclass(frozen=True)
class PreparedAgentTurn:
    """网络调用前固定并可持久化的 Agent turn 写入事实。"""

    session: AgentHookSession
    payload_json: str
    occurred_at: datetime
    after_turn: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.session, AgentHookSession):
            raise TypeError("session must be AgentHookSession")
        if not isinstance(self.payload_json, str):
            raise TypeError("payload_json must be text")
        try:
            payload = json.loads(self.payload_json)
        except json.JSONDecodeError as exc:
            raise ValueError("payload_json must contain canonical JSON") from exc
        if _canonical_payload(payload) != self.payload_json:
            raise ValueError("payload_json must use canonical JSON encoding")
        if not isinstance(self.occurred_at, datetime) or self.occurred_at.tzinfo is None:
            raise ValueError("occurred_at must be timezone-aware")
        object.__setattr__(self, "occurred_at", self.occurred_at.astimezone(timezone.utc))
        if self.after_turn is not None and not isinstance(self.after_turn, bool):
            raise TypeError("after_turn must be boolean or None")

    @property
    def payload(self) -> Any:
        """每次发送都从不可变规范 JSON 重建完全相同的载荷。"""

        return json.loads(self.payload_json)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": _PREPARED_TURN_SCHEMA,
            "session": self.session.to_dict(),
            "payload": self.payload,
            "occurred_at": self.occurred_at.isoformat(),
            "after_turn": self.after_turn,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> PreparedAgentTurn:
        """从 checkpoint 恢复同一份可安全重试的待发送 turn。"""

        if not isinstance(value, Mapping):
            raise TypeError("prepared turn state must be a mapping")
        expected = {"schema_version", "session", "payload", "occurred_at", "after_turn"}
        if set(value) != expected or value.get("schema_version") != _PREPARED_TURN_SCHEMA:
            raise ValueError("prepared turn state has an unsupported schema")
        session = value["session"]
        occurred_at = value["occurred_at"]
        if not isinstance(session, Mapping):
            raise TypeError("prepared turn session must be a mapping")
        if not isinstance(occurred_at, str):
            raise TypeError("prepared turn occurred_at must be ISO-8601 text")
        try:
            parsed_time = datetime.fromisoformat(occurred_at.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("prepared turn occurred_at must be ISO-8601") from exc
        after_turn = value["after_turn"]
        if after_turn is not None and not isinstance(after_turn, bool):
            raise TypeError("prepared turn after_turn must be boolean or None")
        return cls(
            session=AgentHookSession.from_dict(session),
            payload_json=_canonical_payload(value["payload"]),
            occurred_at=parsed_time,
            after_turn=after_turn,
        )


@dataclass(frozen=True)
class AgentBeforeTurnResult:
    """before_turn 可直接注入提示词的上下文及完整召回事实。"""

    session: AgentHookSession
    recall: AgentRecallResult

    @property
    def context(self) -> str:
        return self.recall.context


@dataclass(frozen=True)
class AgentAfterTurnResult:
    """after_turn 成功后的新游标与记忆写入事实。"""

    session: AgentHookSession
    remember: AgentRememberResult


@dataclass(frozen=True)
class AgentSessionCloseResult:
    """session_close 提交剩余完整轮次后的结果。"""

    session: AgentHookSession
    flush: AgentFlushResult


class AgentMemoryHooks:
    """实现召回、事件捕获与会话提交，不接管 Agent 调度器。"""

    def __init__(self, memory: AgentMemoryPort) -> None:
        required = ("remember", "recall", "flush", "cursor")
        if any(not callable(getattr(memory, name, None)) for name in required):
            raise TypeError("memory must implement AgentMemoryPort")
        self.memory = memory

    @staticmethod
    def new_session(conversation_id: str, started_on: date, protocol: str) -> AgentHookSession:
        """为确定没有服务端历史的新 Conversation 创建零游标状态。"""

        return AgentHookSession(conversation_id, started_on, protocol, 0)

    async def resume_session(self, conversation_id: str, started_on: date, protocol: str) -> AgentHookSession:
        """从服务端耐久游标恢复进程重启后的 Conversation。"""

        conversation = ConversationRef(conversation_id, started_on)
        next_sequence = await self.memory.cursor(conversation)
        return AgentHookSession(conversation.conversation_id, conversation.started_on, protocol, next_sequence)

    async def before_turn(
        self,
        session: AgentHookSession,
        query: str,
        *,
        limit: int | None = None,
        kinds: tuple[str, ...] = (),
        intention_scope: str = "active",
    ) -> AgentBeforeTurnResult:
        """在模型调用前召回并组装带来源、受预算约束的 Memory 上下文。"""

        _require_session(session)
        recalled = await self.memory.recall(
            query,
            conversation=session.conversation,
            limit=limit,
            kinds=kinds,
            intention_scope=intention_scope,
        )
        return AgentBeforeTurnResult(session=session, recall=recalled)

    @staticmethod
    def prepare_after_turn(
        session: AgentHookSession,
        payload: object,
        *,
        occurred_at: datetime,
        after_turn: bool | None = None,
    ) -> PreparedAgentTurn:
        """在 I/O 前固定 turn 身份；调用方应先持久化再发送。"""

        _require_session(session)
        return PreparedAgentTurn(
            session=session,
            payload_json=_canonical_payload(payload),
            occurred_at=occurred_at,
            after_turn=after_turn,
        )

    async def after_turn(
        self,
        prepared: PreparedAgentTurn,
        *,
        wait_timeout_seconds: float | None = None,
    ) -> AgentAfterTurnResult:
        """发送已经固定的 turn；失败后可重发同一个 PreparedAgentTurn。"""

        if not isinstance(prepared, PreparedAgentTurn):
            raise TypeError("prepared must be PreparedAgentTurn")
        session = prepared.session
        remembered = await self.memory.remember(
            session.conversation,
            protocol=session.protocol,
            payload=prepared.payload,
            start_sequence=session.next_sequence,
            occurred_at=prepared.occurred_at,
            after_turn=prepared.after_turn,
            wait_timeout_seconds=wait_timeout_seconds,
        )
        return AgentAfterTurnResult(
            session=session.advance(remembered.next_sequence),
            remember=remembered,
        )

    async def on_session_close(
        self,
        session: AgentHookSession,
        *,
        wait_timeout_seconds: float | None = None,
    ) -> AgentSessionCloseResult:
        """仅在会话关闭边界 flush；不把不完整工具调用强制写入长期记忆。"""

        _require_session(session)
        flushed = await self.memory.flush(
            session.conversation,
            wait_timeout_seconds=wait_timeout_seconds,
        )
        return AgentSessionCloseResult(session=session, flush=flushed)


def _require_session(value: AgentHookSession) -> None:
    if not isinstance(value, AgentHookSession):
        raise TypeError("session must be AgentHookSession")


__all__ = [
    "AgentAfterTurnResult",
    "AgentBeforeTurnResult",
    "AgentHookSession",
    "AgentMemoryHooks",
    "AgentSessionCloseResult",
    "PreparedAgentTurn",
]
