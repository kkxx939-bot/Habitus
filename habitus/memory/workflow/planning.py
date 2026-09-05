"""从同一不可变 ConversationSegment 并行生成工作流产品。"""

from __future__ import annotations

from asyncio import gather
from dataclasses import dataclass

from habitus.memory.conversation import ConversationAddress
from habitus.memory.conversation.summary import ConversationSummaryService
from habitus.memory.editor.engine import MemoryEditor, MemoryEditorPlan
from habitus.pre.conversation import ConversationSegment, ConversationSegmentSummary

_USE_SOURCE_SEGMENT = object()


@dataclass(frozen=True)
class MemorySegmentProducts:
    """一次 Segment 对应的过程摘要和长期记忆纯计划。"""

    summary: ConversationSegmentSummary
    editor_plan: MemoryEditorPlan

    def __post_init__(self) -> None:
        if not isinstance(self.summary, ConversationSegmentSummary):
            raise TypeError("summary must be ConversationSegmentSummary")
        if not isinstance(self.editor_plan, MemoryEditorPlan):
            raise TypeError("editor_plan must be MemoryEditorPlan")


class MemorySegmentProductBuilder:
    """并行计算两个只读同一不可变 Segment 的独立产品。"""

    def __init__(
        self,
        summary_service: ConversationSummaryService,
        editor: MemoryEditor,
    ) -> None:
        if not isinstance(summary_service, ConversationSummaryService):
            raise TypeError("summary_service must be ConversationSummaryService")
        if not isinstance(editor, MemoryEditor):
            raise TypeError("editor must be MemoryEditor")
        self.summary_service = summary_service
        self.editor = editor

    async def build(
        self,
        address: ConversationAddress,
        segment: ConversationSegment,
        *,
        editor_segment: ConversationSegment | None | object = _USE_SOURCE_SEGMENT,
    ) -> MemorySegmentProducts:
        """等待两个分支全部收束后返回，任一失败都不产生可提交结果。"""

        if not isinstance(address, ConversationAddress):
            raise TypeError("address must be ConversationAddress")
        if not isinstance(segment, ConversationSegment):
            raise TypeError("segment must be ConversationSegment")
        if editor_segment is _USE_SOURCE_SEGMENT:
            resolved_editor_segment = segment
        elif editor_segment is None:
            resolved_editor_segment = None
        elif isinstance(editor_segment, ConversationSegment):
            resolved_editor_segment = editor_segment
        else:
            raise TypeError("editor_segment must be ConversationSegment or None")
        if resolved_editor_segment is None:
            summary = await self.summary_service.get_or_create(address, segment)
            return MemorySegmentProducts(
                summary=summary,
                editor_plan=MemoryEditorPlan.deferred_until_turn_boundary(),
            )
        summary_result, plan_result = await gather(
            self.summary_service.get_or_create(address, segment),
            self.editor.plan(resolved_editor_segment),
            return_exceptions=True,
        )
        if isinstance(summary_result, BaseException):
            raise summary_result
        if isinstance(plan_result, BaseException):
            raise plan_result
        return MemorySegmentProducts(summary=summary_result, editor_plan=plan_result)


__all__ = ["MemorySegmentProductBuilder", "MemorySegmentProducts"]
