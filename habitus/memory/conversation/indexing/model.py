"""Conversation Summary 索引源、身份和后备召回结果。"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from habitus.foundation.ids import same_path_identity
from habitus.memory.conversation.layout import ConversationAddress
from habitus.pre.conversation import ConversationRangeSummary, ConversationSegmentSummary

ConversationSummary = ConversationSegmentSummary | ConversationRangeSummary


class ConversationSummaryIndexError(RuntimeError):
    """Summary 索引无法在完整性或资源边界内完成操作。"""


class ConversationSummaryStage(str, Enum):
    """用于索引身份的三种不可变 Summary 阶段。"""

    SEGMENT = "segment"
    RANGE = "range"
    ARCHIVE = "archive"


@dataclass(frozen=True)
class ConversationSummaryReference:
    """无需把 Summary 冒充 memory:// 节点的严格来源引用。"""

    address: ConversationAddress
    stage: ConversationSummaryStage
    summary_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.address, ConversationAddress):
            raise TypeError("summary reference address must be ConversationAddress")
        object.__setattr__(self, "stage", ConversationSummaryStage(self.stage))
        if not isinstance(self.summary_id, str) or not self.summary_id or self.summary_id != self.summary_id.strip():
            raise ValueError("summary reference summary_id must be normalized non-empty text")

    @property
    def identity(self) -> str:
        return (
            "conversation-summary:"
            f"{self.address.started_on.isoformat()}:"
            f"{self.address.conversation_id}:"
            f"{self.stage.value}:"
            f"{self.summary_id}"
        )

    @property
    def conversation_scope(self) -> str:
        return f"conversation-summary-scope:{self.address.started_on.isoformat()}:{self.address.conversation_id}"

    @classmethod
    def parse(cls, value: str) -> ConversationSummaryReference:
        if not isinstance(value, str) or not value.startswith("conversation-summary:"):
            raise ValueError("summary reference identity is invalid")
        payload = value.removeprefix("conversation-summary:")
        if len(payload) < 12 or payload[10] != ":":
            raise ValueError("summary reference identity lacks its date")
        try:
            started_on = datetime.fromisoformat(payload[:10]).date()
            conversation_id, stage, summary_id = payload[11:].rsplit(":", 2)
            reference = cls(
                ConversationAddress(conversation_id, started_on),
                ConversationSummaryStage(stage),
                summary_id,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("summary reference identity is invalid") from exc
        if reference.identity != value:
            raise ValueError("summary reference identity is not canonical")
        return reference


@dataclass(frozen=True)
class ConversationSummaryIndexSource:
    """由当前活跃摘要前沿生成的一条规范向量源记录。"""

    reference: ConversationSummaryReference
    summary: ConversationSummary
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.reference, ConversationSummaryReference):
            raise TypeError("summary index source reference is invalid")
        if not isinstance(self.summary, ConversationSegmentSummary | ConversationRangeSummary):
            raise TypeError("summary index source summary is invalid")
        if not same_path_identity(
            self.summary.conversation_id,
            self.reference.address.conversation_id,
            "conversation_id",
        ):
            raise ValueError("summary index source Conversation identity differs")
        if _summary_id(self.summary) != self.reference.summary_id:
            raise ValueError("summary index source summary identity differs")
        if _summary_stage(self.summary) is not self.reference.stage:
            raise ValueError("summary index source stage differs")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("summary index source content must be non-empty text")

    @property
    def identity(self) -> str:
        return self.reference.identity

    @property
    def content_digest(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ConversationSummaryVectorConsistencyReport:
    """活跃 Summary 前沿与远程向量记录的完整差异。"""

    expected_count: int
    indexed_count: int
    missing_identities: tuple[str, ...] = ()
    stale_identities: tuple[str, ...] = ()
    orphan_identities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("expected_count", "indexed_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def ok(self) -> bool:
        return not (self.missing_identities or self.stale_identities or self.orphan_identities)


@dataclass(frozen=True)
class ConversationSummaryMatch:
    """Memory 不充分时返回的一条可溯源历史摘要。"""

    reference: ConversationSummaryReference
    summary: ConversationSummary
    content: str
    score: float
    vector_score: float
    rerank_score: float | None = None

    def __post_init__(self) -> None:
        ConversationSummaryIndexSource(self.reference, self.summary, self.content)
        score = _finite_score(self.score, "summary score")
        vector_score = _vector_score(self.vector_score)
        if self.rerank_score is None:
            if score != vector_score:
                raise ValueError("summary final score must equal vector score without reranking")
        else:
            rerank_score = _finite_score(self.rerank_score, "summary rerank score")
            if score != rerank_score:
                raise ValueError("summary final score must equal rerank score")
            object.__setattr__(self, "rerank_score", rerank_score)
        object.__setattr__(self, "score", score)
        object.__setattr__(self, "vector_score", vector_score)

    @property
    def started_at(self) -> datetime:
        return self.summary.started_at

    @property
    def ended_at(self) -> datetime:
        return self.summary.ended_at


def summary_reference(
    address: ConversationAddress,
    summary: ConversationSummary,
) -> ConversationSummaryReference:
    """由可信文件位置和严格 Summary Schema 生成稳定索引身份。"""

    return ConversationSummaryReference(
        address=address,
        stage=_summary_stage(summary),
        summary_id=_summary_id(summary),
    )


def _summary_stage(summary: ConversationSummary) -> ConversationSummaryStage:
    if isinstance(summary, ConversationSegmentSummary):
        return ConversationSummaryStage.SEGMENT
    value = summary.stage.value
    if value == "range":
        return ConversationSummaryStage.RANGE
    if value == "archive":
        return ConversationSummaryStage.ARCHIVE
    raise ValueError("unsupported Conversation Summary stage")


def _summary_id(summary: ConversationSummary) -> str:
    if isinstance(summary, ConversationSegmentSummary):
        return summary.segment_id
    return summary.range_id


def _finite_score(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{label} must be numeric")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError(f"{label} must be finite")
    return score


def _vector_score(value: object) -> float:
    score = _finite_score(value, "summary vector score")
    if not -1.0 <= score <= 1.0:
        raise ValueError("summary vector score must be between -1 and 1")
    return score


__all__ = [
    "ConversationSummary",
    "ConversationSummaryIndexError",
    "ConversationSummaryIndexSource",
    "ConversationSummaryVectorConsistencyReport",
    "ConversationSummaryMatch",
    "ConversationSummaryReference",
    "ConversationSummaryStage",
    "summary_reference",
]
