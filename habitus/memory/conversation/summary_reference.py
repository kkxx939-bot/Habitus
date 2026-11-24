"""Conversation Summary 的稳定来源引用与检索结果（不冒充树上的卡片节点）。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from habitus.foundation.ids import same_path_identity
from habitus.memory.conversation.layout import ConversationAddress
from habitus.pre.conversation import ConversationRangeSummary, ConversationSegmentSummary

ConversationSummary = ConversationSegmentSummary | ConversationRangeSummary


class ConversationSummaryStage(str, Enum):
    """用于索引身份的三种不可变 Summary 阶段。"""

    SEGMENT = "segment"
    RANGE = "range"
    ARCHIVE = "archive"


@dataclass(frozen=True)
class ConversationSummaryReference:
    """无需把 Summary 冒充树上卡片节点的严格来源引用。"""

    address: ConversationAddress
    stage: ConversationSummaryStage
    summary_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage", ConversationSummaryStage(self.stage))
        if not self.summary_id or self.summary_id != self.summary_id.strip():
            raise ValueError("summary reference summary_id must be normalized non-empty text")

    @property
    def identity(self) -> str:
        return (
            "conversation-summary:"
            f"{self.address.started_on.isoformat()}:"
            f"{self.address.identity}:"
            f"{self.stage.value}:"
            f"{self.summary_id}"
        )

    @property
    def conversation_scope(self) -> str:
        return f"conversation-summary-scope:{self.address.started_on.isoformat()}:{self.address.identity}"

    @classmethod
    def parse(cls, value: str) -> ConversationSummaryReference:
        if not value.startswith("conversation-summary:"):
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
class ConversationSummaryMatch:
    """检索按日期或关键词找到的一份当前有效摘要，``content`` 是交给模型看的规范文本。"""

    reference: ConversationSummaryReference
    summary: ConversationSummary
    content: str

    def __post_init__(self) -> None:
        if not same_path_identity(
            self.summary.conversation_id,
            self.reference.address.conversation_id,
            "conversation_id",
        ):
            raise ValueError("summary match Conversation identity differs")
        if _summary_id(self.summary) != self.reference.summary_id:
            raise ValueError("summary match summary identity differs")
        if _summary_stage(self.summary) is not self.reference.stage:
            raise ValueError("summary match stage differs")
        if not self.content.strip():
            raise ValueError("summary match content must be non-empty text")

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


__all__ = [
    "ConversationSummary",
    "ConversationSummaryMatch",
    "ConversationSummaryReference",
    "ConversationSummaryStage",
    "summary_reference",
]
