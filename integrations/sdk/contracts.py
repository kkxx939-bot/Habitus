"""外部 Agent SDK 使用的纯 Python 生命周期与记忆服务契约。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol


def _clean_text(value: str, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be non-empty normalized text")
    return value


@dataclass(frozen=True)
class ConversationRef:
    """SDK 层稳定会话身份，不依赖 m2bOS 内部存储地址类型。"""

    conversation_id: str
    started_on: date

    def __post_init__(self) -> None:
        _clean_text(self.conversation_id, "conversation_id")
        if isinstance(self.started_on, datetime) or not isinstance(self.started_on, date):
            raise TypeError("started_on must be a calendar date")


@dataclass(frozen=True)
class AgentMemoryJob:
    """Agent 只需持有的公开异步记忆任务身份。"""

    memory_sequence: int
    conversation_id: str
    started_on: date
    status: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.memory_sequence, bool)
            or not isinstance(self.memory_sequence, int)
            or self.memory_sequence <= 0
        ):
            raise ValueError("memory_sequence must be a positive integer")
        ConversationRef(self.conversation_id, self.started_on)
        _clean_text(self.status, "status")


@dataclass(frozen=True)
class AgentMemoryConsistency:
    """Agent 等待记忆任务后可观察的稳定一致性状态。"""

    memory_sequence: int
    state: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.memory_sequence, bool)
            or not isinstance(self.memory_sequence, int)
            or self.memory_sequence <= 0
        ):
            raise ValueError("memory_sequence must be a positive integer")
        _clean_text(self.state, "state")


@dataclass(frozen=True)
class AgentRememberResult:
    """一个 Agent 事件写入服务后得到的传输无关结果。"""

    ignored_items: int
    after_turn: bool
    next_sequence: int
    jobs: tuple[AgentMemoryJob, ...] = ()
    consistency: tuple[AgentMemoryConsistency, ...] = ()

    def __post_init__(self) -> None:
        for label, value in (
            ("ignored_items", self.ignored_items),
            ("next_sequence", self.next_sequence),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer")
        if not isinstance(self.after_turn, bool):
            raise TypeError("after_turn must be boolean")
        if not isinstance(self.jobs, tuple) or any(not isinstance(item, AgentMemoryJob) for item in self.jobs):
            raise TypeError("jobs must contain AgentMemoryJob values")
        if not isinstance(self.consistency, tuple) or any(
            not isinstance(item, AgentMemoryConsistency) for item in self.consistency
        ):
            raise TypeError("consistency must contain AgentMemoryConsistency values")


@dataclass(frozen=True)
class AgentFlushResult:
    """会话关闭时显式提交完整剩余轮次的结果。"""

    jobs: tuple[AgentMemoryJob, ...] = ()
    consistency: tuple[AgentMemoryConsistency, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.jobs, tuple) or any(not isinstance(item, AgentMemoryJob) for item in self.jobs):
            raise TypeError("jobs must contain AgentMemoryJob values")
        if not isinstance(self.consistency, tuple) or any(
            not isinstance(item, AgentMemoryConsistency) for item in self.consistency
        ):
            raise TypeError("consistency must contain AgentMemoryConsistency values")


@dataclass(frozen=True)
class AgentRecallMemory:
    """一条可溯源的直接记忆命中。"""

    uri: str
    score: float
    matched_queries: tuple[str, ...]

    def __post_init__(self) -> None:
        _clean_text(self.uri, "uri")
        if isinstance(self.score, bool) or not isinstance(self.score, int | float) or not math.isfinite(self.score):
            raise ValueError("score must be finite")
        if not isinstance(self.matched_queries, tuple) or any(
            not isinstance(item, str) or not item for item in self.matched_queries
        ):
            raise TypeError("matched_queries must contain non-empty text")


@dataclass(frozen=True)
class AgentRecallSummary:
    """一次历史摘要回退命中。"""

    reference: str
    score: float

    def __post_init__(self) -> None:
        _clean_text(self.reference, "reference")
        if isinstance(self.score, bool) or not isinstance(self.score, int | float) or not math.isfinite(self.score):
            raise ValueError("score must be finite")


@dataclass(frozen=True)
class AgentRecallDegradation:
    """可选召回增强失败后的安全降级事实。"""

    stage: str
    error_type: str

    def __post_init__(self) -> None:
        _clean_text(self.stage, "stage")
        _clean_text(self.error_type, "error_type")


@dataclass(frozen=True)
class AgentRecallResult:
    """可直接注入 Agent 上下文的传输无关召回结果。"""

    query: str
    queries: tuple[str, ...]
    context: str
    memories: tuple[AgentRecallMemory, ...]
    summaries: tuple[AgentRecallSummary, ...]
    degradations: tuple[AgentRecallDegradation, ...]
    budget_exhausted: bool

    def __post_init__(self) -> None:
        _clean_text(self.query, "query")
        if not isinstance(self.queries, tuple) or any(not isinstance(item, str) or not item for item in self.queries):
            raise TypeError("queries must contain non-empty text")
        if not isinstance(self.context, str):
            raise TypeError("context must be text")
        if not isinstance(self.memories, tuple) or any(not isinstance(item, AgentRecallMemory) for item in self.memories):
            raise TypeError("memories must contain AgentRecallMemory values")
        if not isinstance(self.summaries, tuple) or any(
            not isinstance(item, AgentRecallSummary) for item in self.summaries
        ):
            raise TypeError("summaries must contain AgentRecallSummary values")
        if not isinstance(self.degradations, tuple) or any(
            not isinstance(item, AgentRecallDegradation) for item in self.degradations
        ):
            raise TypeError("degradations must contain AgentRecallDegradation values")
        if not isinstance(self.budget_exhausted, bool):
            raise TypeError("budget_exhausted must be boolean")


class AgentMemoryPort(Protocol):
    """Hook 所依赖的最小能力口；可由本地 Runtime 或 HTTP 服务实现。"""

    async def remember(
        self,
        conversation: ConversationRef,
        *,
        protocol: str,
        payload: object,
        start_sequence: int,
        occurred_at: datetime,
        after_turn: bool | None = None,
        wait_timeout_seconds: float | None = None,
    ) -> AgentRememberResult: ...

    async def recall(
        self,
        query: str,
        *,
        conversation: ConversationRef | None = None,
        limit: int | None = None,
        kinds: tuple[str, ...] = (),
        intention_scope: str = "active",
    ) -> AgentRecallResult: ...

    async def flush(
        self,
        conversation: ConversationRef,
        *,
        wait_timeout_seconds: float | None = None,
    ) -> AgentFlushResult: ...

    async def cursor(self, conversation: ConversationRef) -> int: ...


__all__ = [
    "AgentFlushResult",
    "AgentMemoryConsistency",
    "AgentMemoryJob",
    "AgentMemoryPort",
    "AgentRecallDegradation",
    "AgentRecallMemory",
    "AgentRecallResult",
    "AgentRecallSummary",
    "AgentRememberResult",
    "ConversationRef",
]
