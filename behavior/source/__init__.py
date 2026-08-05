"""Behavior 原始来源规范化公共边界。"""

from behavior.source.adapter import BehaviorSourceAdapter, BehaviorSourceAdapterRegistry
from behavior.source.identity import SourceRecordIdentityFactory
from behavior.source.model import CaptureState, Modality, SourceRecord, SourceRecordBatch, SourceType
from behavior.source.service import SourceRecordService

__all__ = [
    "BehaviorSourceAdapter",
    "BehaviorSourceAdapterRegistry",
    "CaptureState",
    "Modality",
    "SourceRecord",
    "SourceRecordBatch",
    "SourceRecordIdentityFactory",
    "SourceRecordService",
    "SourceType",
]
