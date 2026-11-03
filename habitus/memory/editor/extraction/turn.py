"""一轮太长、按段分开抽取时，给这一段带上的同轮上下文。"""

from __future__ import annotations

from dataclasses import dataclass, replace

from habitus.pre.conversation import (
    ConversationMessage,
    ConversationMessageRole,
    ConversationSegmentSummary,
)


@dataclass(frozen=True)
class MemoryTurnContext:
    """同一轮里本段以外的材料，只帮模型理解这一段在做什么，不是事实来源。

    - ``opening_prompts``：这一轮开头他说的话（本段就从开轮处起时为空，那几句已在本段里）；
    - ``earlier_summaries``：这一轮前面各段的段摘要（各段 Job 生成的那份），按先后排；
    - ``earlier_segment_count``：这一轮前面一共几段，摘要没读到或因预算没带上的按差额如实告诉模型。

    前面各段的事实由它们各自的 Job 抽取。带上开头的人话借鉴了 OpenViking ``session/retention.py`` 的
    ``_partial_latest_turn_plan`` 的设计（一轮超预算时前半归档，并附开轮那条用户消息）；带上前面各段的段摘要是
    Habitus 自己加的，OpenViking 的抽取提示里不带归档概览。
    """

    opening_prompts: tuple[ConversationMessage, ...]
    earlier_summaries: tuple[ConversationSegmentSummary, ...]
    earlier_segment_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.opening_prompts, tuple) or any(
            not isinstance(message, ConversationMessage) or message.role is not ConversationMessageRole.PROMPT
            for message in self.opening_prompts
        ):
            raise TypeError("opening_prompts must be a tuple of PROMPT messages")
        if not isinstance(self.earlier_summaries, tuple) or any(
            not isinstance(summary, ConversationSegmentSummary) for summary in self.earlier_summaries
        ):
            raise TypeError("earlier_summaries must be a tuple of ConversationSegmentSummary values")
        if (
            isinstance(self.earlier_segment_count, bool)
            or not isinstance(self.earlier_segment_count, int)
            or self.earlier_segment_count < len(self.earlier_summaries)
        ):
            raise ValueError("earlier_segment_count must cover every earlier summary")

    @property
    def omitted_earlier_segment_count(self) -> int:
        return self.earlier_segment_count - len(self.earlier_summaries)

    def without_oldest_summary(self) -> MemoryTurnContext:
        """去掉最早的一份段摘要：预算放不下时先舍远的，离本段最近的留到最后。"""

        if not self.earlier_summaries:
            raise ValueError("turn context has no earlier summary left to omit")
        return replace(self, earlier_summaries=self.earlier_summaries[1:])


__all__ = ["MemoryTurnContext"]
