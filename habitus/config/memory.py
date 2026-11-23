"""长期记忆文档、检索、解析与语义层配置。"""

from __future__ import annotations

from dataclasses import dataclass, field

from habitus.config.loader import construct_config, group_fields
from habitus.infrastructure.editor.snapshot import SnapshotReadConfig
from habitus.memory.compaction import (
    MemoryFieldCompactionConfig,
    MemoryLifecycleMaintenanceConfig,
)
from habitus.memory.document import MemoryDocumentConfig
from habitus.memory.editor import (
    MemoryCommitConfig,
    MemoryExtractionConfig,
    MemoryRetrievalConfig,
    MemoryTransactionJournalConfig,
)
from habitus.memory.recall import (
    MemoryRecallLifecycleConfig,
)
from habitus.memory.retrieval import MemorySearchServiceConfig
from habitus.memory.semantic import MemorySemanticConfig
from habitus.memory.tree import MemoryTreeConfig


@dataclass(frozen=True)
class MemoryConfig:
    """记忆领域全部强类型子配置的聚合。"""

    document: MemoryDocumentConfig = field(default_factory=MemoryDocumentConfig)
    tree: MemoryTreeConfig = field(default_factory=MemoryTreeConfig)
    snapshot: SnapshotReadConfig = field(default_factory=SnapshotReadConfig)
    search_service: MemorySearchServiceConfig = field(default_factory=MemorySearchServiceConfig)
    recall_lifecycle: MemoryRecallLifecycleConfig = field(default_factory=MemoryRecallLifecycleConfig)
    field_compaction: MemoryFieldCompactionConfig = field(default_factory=MemoryFieldCompactionConfig)
    lifecycle_maintenance: MemoryLifecycleMaintenanceConfig = field(default_factory=MemoryLifecycleMaintenanceConfig)
    editor_retrieval: MemoryRetrievalConfig = field(default_factory=MemoryRetrievalConfig)
    extraction: MemoryExtractionConfig = field(default_factory=MemoryExtractionConfig)
    commit: MemoryCommitConfig = field(default_factory=MemoryCommitConfig)
    transaction_journal: MemoryTransactionJournalConfig = field(default_factory=MemoryTransactionJournalConfig)
    semantic: MemorySemanticConfig = field(default_factory=MemorySemanticConfig)

    @classmethod
    def from_mapping(cls, value: object) -> MemoryConfig:
        data = group_fields(cls, value, "config.memory")
        return cls(
            document=construct_config(
                MemoryDocumentConfig,
                data.get("document", {}),
                "config.memory.document",
            ),
            tree=construct_config(
                MemoryTreeConfig,
                data.get("tree", {}),
                "config.memory.tree",
            ),
            snapshot=construct_config(
                SnapshotReadConfig,
                data.get("snapshot", {}),
                "config.memory.snapshot",
            ),
            search_service=construct_config(
                MemorySearchServiceConfig,
                data.get("search_service", {}),
                "config.memory.search_service",
            ),
            recall_lifecycle=construct_config(
                MemoryRecallLifecycleConfig,
                data.get("recall_lifecycle", {}),
                "config.memory.recall_lifecycle",
            ),
            field_compaction=construct_config(
                MemoryFieldCompactionConfig,
                data.get("field_compaction", {}),
                "config.memory.field_compaction",
            ),
            lifecycle_maintenance=construct_config(
                MemoryLifecycleMaintenanceConfig,
                data.get("lifecycle_maintenance", {}),
                "config.memory.lifecycle_maintenance",
            ),
            editor_retrieval=construct_config(
                MemoryRetrievalConfig,
                data.get("editor_retrieval", {}),
                "config.memory.editor_retrieval",
            ),
            extraction=construct_config(
                MemoryExtractionConfig,
                data.get("extraction", {}),
                "config.memory.extraction",
            ),
            commit=construct_config(
                MemoryCommitConfig,
                data.get("commit", {}),
                "config.memory.commit",
            ),
            transaction_journal=construct_config(
                MemoryTransactionJournalConfig,
                data.get("transaction_journal", {}),
                "config.memory.transaction_journal",
            ),
            semantic=construct_config(
                MemorySemanticConfig,
                data.get("semantic", {}),
                "config.memory.semantic",
            ),
        )


__all__ = ["MemoryConfig"]
