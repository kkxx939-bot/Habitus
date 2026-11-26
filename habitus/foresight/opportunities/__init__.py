"""机会表：形状、编解码、换代时的构建。落盘在组合根（``runtime/foresight_opportunities.py``）。"""

from habitus.foresight.opportunities.build import build_table
from habitus.foresight.opportunities.codec import OPPORTUNITY_SCHEMA, decode_table, encode_table
from habitus.foresight.opportunities.model import (
    SEGMENTS,
    STATUSES,
    ConditionSpec,
    DayWatch,
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
    "ConditionSpec",
    "DayWatch",
    "KindDays",
    "Opportunity",
    "OpportunityTable",
    "RelationOpportunities",
    "TreeReading",
    "build_table",
    "decode_table",
    "encode_table",
]
