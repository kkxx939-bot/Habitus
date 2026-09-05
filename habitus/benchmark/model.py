"""数据集驱动记忆基准的公共领域模型。"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from types import MappingProxyType
from typing import Literal

BenchmarkMessageRole = Literal["prompt", "completion", "tool_call", "tool_result"]
BenchmarkToolStatus = Literal["completed", "error", "cancelled"]

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,255}$")


class BenchmarkDatasetName(str, Enum):
    """当前正式支持的记忆评测数据协议。"""

    LOCOMO = "locomo"
    LONGMEMEVAL = "longmemeval"
    HABITUS = "habitus"


def _text(value: object, label: str, *, maximum: int = 2_000_000) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text")
    if len(value) > maximum:
        raise ValueError(f"{label} exceeds its size bound")
    return value


def _identifier(value: object, label: str) -> str:
    if not isinstance(value, str) or _SAFE_IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a safe identifier")
    return value


def _aware(value: datetime, label: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(f"{label} must be a timezone-aware datetime")
    return value.astimezone(UTC)


@dataclass(frozen=True)
class BenchmarkMessage:
    """数据集中的一条原始消息，不包含预期内部操作。"""

    message_id: str
    role: BenchmarkMessageRole
    content: object
    occurred_at: datetime
    tool_call_id: str | None = None
    tool_name: str | None = None
    tool_status: BenchmarkToolStatus | None = None
    source_ref: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "message_id", _identifier(self.message_id, "message_id"))
        if self.role not in {"prompt", "completion", "tool_call", "tool_result"}:
            raise ValueError("unsupported benchmark message role")
        object.__setattr__(self, "occurred_at", _aware(self.occurred_at, "message occurred_at"))
        try:
            json.dumps(self.content, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("benchmark message content must be JSON-serializable") from exc
        for name in ("tool_call_id", "tool_name", "source_ref"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise ValueError(f"{name} must be non-empty when provided")
        if self.role in {"prompt", "completion"}:
            _text(self.content, "message content")
            if any((self.tool_call_id, self.tool_name, self.tool_status)):
                raise ValueError("text messages cannot carry tool fields")
        elif self.role == "tool_call":
            if self.tool_call_id is None or self.tool_name is None:
                raise ValueError("tool_call requires tool_call_id and tool_name")
            if self.tool_status is not None:
                raise ValueError("tool_call cannot carry tool_status")
        else:
            if self.tool_call_id is None or self.tool_name is None or self.tool_status is None:
                raise ValueError("tool_result requires call identity, tool name, and status")
            if self.tool_status not in {"completed", "error", "cancelled"}:
                raise ValueError("unsupported tool result status")


@dataclass(frozen=True)
class BenchmarkSession:
    """按原始时间顺序导入 Habitus 的一个完整会话。"""

    session_id: str
    started_at: datetime
    messages: tuple[BenchmarkMessage, ...]
    source_label: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "session_id", _identifier(self.session_id, "session_id"))
        object.__setattr__(self, "started_at", _aware(self.started_at, "session started_at"))
        object.__setattr__(self, "messages", tuple(self.messages))
        if not self.messages or any(not isinstance(item, BenchmarkMessage) for item in self.messages):
            raise ValueError("benchmark session requires messages")
        if tuple(sorted(self.messages, key=lambda item: item.occurred_at)) != self.messages:
            raise ValueError("benchmark session messages must be chronological")
        if len({item.message_id for item in self.messages}) != len(self.messages):
            raise ValueError("benchmark session message IDs must be unique")
        if self.source_label and not isinstance(self.source_label, str):
            raise TypeError("session source_label must be text")


@dataclass(frozen=True)
class BenchmarkQuestion:
    """公开数据集给出的自然语言问题和参考答案。"""

    question_id: str
    question: str
    reference_answer: str
    question_type: str
    question_time: datetime | None = None
    evidence_refs: tuple[str, ...] = ()
    evidence_texts: tuple[str, ...] = ()
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "question_id", _identifier(self.question_id, "question_id"))
        object.__setattr__(self, "question", _text(self.question, "question"))
        object.__setattr__(self, "reference_answer", _text(self.reference_answer, "reference_answer"))
        object.__setattr__(self, "question_type", _text(self.question_type, "question_type", maximum=256))
        if self.question_time is not None:
            object.__setattr__(self, "question_time", _aware(self.question_time, "question_time"))
        for name in ("evidence_refs", "evidence_texts"):
            values = tuple(getattr(self, name))
            if any(not isinstance(item, str) or not item.strip() for item in values):
                raise ValueError(f"{name} must contain non-empty text")
            object.__setattr__(self, name, values)
        if not isinstance(self.metadata, Mapping):
            raise TypeError("question metadata must be an object")
        metadata = dict(self.metadata)
        try:
            json.dumps(metadata, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("question metadata must be JSON-serializable") from exc
        object.__setattr__(self, "metadata", MappingProxyType(metadata))


@dataclass(frozen=True)
class BenchmarkSample:
    """相互隔离的一份长期对话历史及其评测问题。"""

    sample_id: str
    source_id: str
    sessions: tuple[BenchmarkSession, ...]
    questions: tuple[BenchmarkQuestion, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "sample_id", _identifier(self.sample_id, "sample_id"))
        object.__setattr__(self, "source_id", _text(self.source_id, "source_id", maximum=1_000))
        object.__setattr__(self, "sessions", tuple(self.sessions))
        object.__setattr__(self, "questions", tuple(self.questions))
        if not self.sessions or any(not isinstance(item, BenchmarkSession) for item in self.sessions):
            raise ValueError("benchmark sample requires sessions")
        if not self.questions or any(not isinstance(item, BenchmarkQuestion) for item in self.questions):
            raise ValueError("benchmark sample requires questions")
        if len({item.session_id for item in self.sessions}) != len(self.sessions):
            raise ValueError("sample session IDs must be unique")
        if len({item.question_id for item in self.questions}) != len(self.questions):
            raise ValueError("sample question IDs must be unique")

    @property
    def digest(self) -> str:
        payload = {
            "sample_id": self.sample_id,
            "source_id": self.source_id,
            "sessions": [
                {
                    "session_id": session.session_id,
                    "started_at": session.started_at.isoformat(),
                    "messages": [
                        {
                            "message_id": message.message_id,
                            "role": message.role,
                            "content": message.content,
                            "occurred_at": message.occurred_at.isoformat(),
                            "tool_call_id": message.tool_call_id,
                            "tool_name": message.tool_name,
                            "tool_status": message.tool_status,
                            "source_ref": message.source_ref,
                        }
                        for message in session.messages
                    ],
                    "source_label": session.source_label,
                }
                for session in self.sessions
            ],
            "questions": [
                {
                    "question_id": question.question_id,
                    "question": question.question,
                    "reference_answer": question.reference_answer,
                    "question_type": question.question_type,
                    "question_time": None if question.question_time is None else question.question_time.isoformat(),
                    "evidence_refs": list(question.evidence_refs),
                    "evidence_texts": list(question.evidence_texts),
                    "metadata": dict(question.metadata),
                }
                for question in self.questions
            ],
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BenchmarkDataset:
    """直接从公开数据集解析出的样本集合。"""

    name: BenchmarkDatasetName
    source_path: str
    samples: tuple[BenchmarkSample, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", BenchmarkDatasetName(self.name))
        object.__setattr__(self, "source_path", _text(self.source_path, "source_path", maximum=10_000))
        object.__setattr__(self, "samples", tuple(self.samples))
        if not self.samples or any(not isinstance(item, BenchmarkSample) for item in self.samples):
            raise ValueError("benchmark dataset requires samples")
        if len({item.sample_id for item in self.samples}) != len(self.samples):
            raise ValueError("benchmark sample IDs must be unique")


@dataclass(frozen=True)
class BenchmarkAnswerRecord:
    """一条问题经过真实检索和回答后的可复查记录。"""

    dataset: str
    sample_id: str
    source_id: str
    question_id: str
    question_type: str
    question: str
    reference_answer: str
    response: str
    question_time: str
    retrieved_uris: tuple[str, ...]
    related_uris: tuple[str, ...]
    summary_fallback_ids: tuple[str, ...]
    answer_memory_uris: tuple[str, ...]
    answer_related_uris: tuple[str, ...]
    answer_summary_ids: tuple[str, ...]
    skipped_context_ids: tuple[str, ...]
    answer_context_limit_chars: int
    summary_fallback_attempted: bool
    retrieval_sufficient: bool | None
    evidence_count: int
    evidence_recall: float | None
    expected_summary_fallback: bool | None
    expected_related_memory: bool | None
    summary_fallback_route_correct: bool | None
    related_memory_route_correct: bool | None
    context_chars: int
    context_tokens_approx: int
    answer_input_tokens: int
    answer_output_tokens: int
    answer_total_tokens: int
    retrieval_latency_ms: float
    answer_latency_ms: float
    error: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": "habitus_benchmark_answer_v2",
            **self.__dict__,
            "retrieved_uris": list(self.retrieved_uris),
            "related_uris": list(self.related_uris),
            "summary_fallback_ids": list(self.summary_fallback_ids),
            "answer_memory_uris": list(self.answer_memory_uris),
            "answer_related_uris": list(self.answer_related_uris),
            "answer_summary_ids": list(self.answer_summary_ids),
            "skipped_context_ids": list(self.skipped_context_ids),
        }


@dataclass(frozen=True)
class BenchmarkJudgeRecord:
    """独立 Judge 对一条回答给出的二元质量判定。"""

    answer: BenchmarkAnswerRecord
    verdict: Literal["correct", "wrong", "judge_error"]
    reasoning: str
    judge_input_tokens: int = 0
    judge_output_tokens: int = 0
    judge_total_tokens: int = 0
    judge_latency_ms: float = 0.0

    @property
    def graded(self) -> bool:
        return self.verdict in {"correct", "wrong"}

    def to_dict(self) -> dict[str, object]:
        return {
            **self.answer.to_dict(),
            "schema_version": "habitus_benchmark_judge_v2",
            "verdict": self.verdict,
            "judge_reasoning": self.reasoning,
            "judge_input_tokens": self.judge_input_tokens,
            "judge_output_tokens": self.judge_output_tokens,
            "judge_total_tokens": self.judge_total_tokens,
            "judge_latency_ms": self.judge_latency_ms,
        }


__all__ = [
    "BenchmarkAnswerRecord",
    "BenchmarkDataset",
    "BenchmarkDatasetName",
    "BenchmarkJudgeRecord",
    "BenchmarkMessage",
    "BenchmarkMessageRole",
    "BenchmarkQuestion",
    "BenchmarkSample",
    "BenchmarkSession",
    "BenchmarkToolStatus",
]
