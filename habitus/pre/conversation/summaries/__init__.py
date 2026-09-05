"""绑定不可变 ConversationSegment 的历史过程摘要 Schema。"""

from habitus.pre.conversation.summaries.model import (
    ConversationRangeSummary,
    ConversationRangeSummaryStage,
    ConversationSegmentSummary,
    ConversationSummaryContent,
    ConversationSummarySchemaError,
    ConversationSummarySourceKind,
    ConversationSummarySourceRef,
)

__all__ = [
    "ConversationRangeSummary",
    "ConversationRangeSummaryStage",
    "ConversationSegmentSummary",
    "ConversationSummaryContent",
    "ConversationSummarySchemaError",
    "ConversationSummarySourceKind",
    "ConversationSummarySourceRef",
]
