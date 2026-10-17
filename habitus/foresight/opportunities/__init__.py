"""机会表（``19-预测层方案.md`` 第九节 N1）：形状、编解码、夜里的构建。落盘在组合根（``runtime/foresight_opportunities.py``）。"""

from habitus.foresight.opportunities.build import build_table
from habitus.foresight.opportunities.codec import OPPORTUNITY_SCHEMA, decode_table, encode_table
from habitus.foresight.opportunities.model import (
    SEGMENTS,
    STATUSES,
    KindDay,
    KindDays,
    Opportunity,
    OpportunityTable,
    RelationOpportunities,
    TreeReading,
)

__all__ = [
    "OPPORTUNITY_SCHEMA",
    "SEGMENTS",
    "STATUSES",
    "KindDay",
    "KindDays",
    "Opportunity",
    "OpportunityTable",
    "RelationOpportunities",
    "TreeReading",
    "build_table",
    "decode_table",
    "encode_table",
]
