"""长期记忆文档、检索、索引、解析与语义层配置。"""

from __future__ import annotations

from dataclasses import dataclass, field

from habitus.config.loader import construct_config, group_fields
from habitus.infrastructure.editor.snapshot import SnapshotReadConfig
from habitus.infrastructure.vector import VectorStoreConfig
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
from habitus.memory.indexing import MemoryVectorIndexConfig
from habitus.memory.intention import MemoryIntentionReviewConfig
from habitus.memory.retrieval import (
    MemoryRecallLifecycleConfig,
    MemorySearchServiceConfig,
    MemorySemanticSearchConfig,
)
from habitus.memory.semantic import MemorySemanticConfig
from habitus.memory.tree import MemoryTreeConfig


@dataclass(frozen=True)
class MemoryConfig:
    """记忆领域全部强类型子配置的聚合。"""

    document: MemoryDocumentConfig = field(default_factory=MemoryDocumentConfig)
    intention_review: MemoryIntentionReviewConfig = field(default_factory=MemoryIntentionReviewConfig)
    tree: MemoryTreeConfig = field(default_factory=MemoryTreeConfig)
    snapshot: SnapshotReadConfig = field(default_factory=SnapshotReadConfig)
    search_service: MemorySearchServiceConfig = field(default_factory=MemorySearchServiceConfig)
    recall_lifecycle: MemoryRecallLifecycleConfig = field(default_factory=MemoryRecallLifecycleConfig)
    field_compaction: MemoryFieldCompactionConfig = field(default_factory=MemoryFieldCompactionConfig)
    lifecycle_maintenance: MemoryLifecycleMaintenanceConfig = field(
        default_factory=MemoryLifecycleMaintenanceConfig
    )
    retrieval: MemoryRetrievalConfig = field(default_factory=MemoryRetrievalConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    vector_index: MemoryVectorIndexConfig = field(default_factory=MemoryVectorIndexConfig)
    semantic_search: MemorySemanticSearchConfig = field(default_factory=MemorySemanticSearchConfig)
    extraction: MemoryExtractionConfig = field(default_factory=MemoryExtractionConfig)
    commit: MemoryCommitConfig = field(default_factory=MemoryCommitConfig)
    transaction_journal: MemoryTransactionJournalConfig = field(default_factory=MemoryTransactionJournalConfig)
    semantic: MemorySemanticConfig = field(default_factory=MemorySemanticConfig)

    def __post_init__(self) -> None:
        expected = (
            ("document", self.document, MemoryDocumentConfig),
            ("intention_review", self.intention_review, MemoryIntentionReviewConfig),
            ("tree", self.tree, MemoryTreeConfig),
            ("snapshot", self.snapshot, SnapshotReadConfig),
            ("search_service", self.search_service, MemorySearchServiceConfig),
            ("recall_lifecycle", self.recall_lifecycle, MemoryRecallLifecycleConfig),
            ("field_compaction", self.field_compaction, MemoryFieldCompactionConfig),
            ("lifecycle_maintenance", self.lifecycle_maintenance, MemoryLifecycleMaintenanceConfig),
            ("retrieval", self.retrieval, MemoryRetrievalConfig),
            ("vector_store", self.vector_store, VectorStoreConfig),
            ("vector_index", self.vector_index, MemoryVectorIndexConfig),
            (
                "semantic_search",
                self.semantic_search,
                MemorySemanticSearchConfig,
            ),
            ("extraction", self.extraction, MemoryExtractionConfig),
            ("commit", self.commit, MemoryCommitConfig),
            (
                "transaction_journal",
                self.transaction_journal,
                MemoryTransactionJournalConfig,
            ),
            ("semantic", self.semantic, MemorySemanticConfig),
        )
        for name, value, expected_type in expected:
            if not isinstance(value, expected_type):
                raise TypeError(f"memory.{name} must be {expected_type.__name__}")

    @classmethod
    def from_mapping(cls, value: object) -> MemoryConfig:
        data = group_fields(cls, value, "config.memory")
        return cls(
            document=construct_config(
                MemoryDocumentConfig,
                data.get("document", {}),
                "config.memory.document",
            ),
            intention_review=construct_config(
                MemoryIntentionReviewConfig,
                data.get("intention_review", {}),
                "config.memory.intention_review",
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
            retrieval=construct_config(
                MemoryRetrievalConfig,
                data.get("retrieval", {}),
                "config.memory.retrieval",
            ),
            vector_store=_vector_store_config(data.get("vector_store", {})),
            vector_index=construct_config(
                MemoryVectorIndexConfig,
                data.get("vector_index", {}),
                "config.memory.vector_index",
            ),
            semantic_search=construct_config(
                MemorySemanticSearchConfig,
                data.get("semantic_search", {}),
                "config.memory.semantic_search",
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


def _vector_store_config(value: object) -> VectorStoreConfig:
    try:
        return VectorStoreConfig.from_mapping(value)
    except (TypeError, ValueError) as exc:
        from habitus.config.loader import ConfigError

        raise ConfigError(f"invalid 'config.memory.vector_store': {exc}") from exc
