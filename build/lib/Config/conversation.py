"""Conversation 切段、单段摘要与长期生命周期配置。"""

from __future__ import annotations

from dataclasses import dataclass, field

from Config.loader import ConfigError, construct_config, group_fields
from infrastructure.vector import VectorStoreConfig
from memory.conversation import (
    ConversationJournalConfig,
    ConversationRangeSummaryCompactionConfig,
    ConversationSegmentationConfig,
    ConversationSegmentSummaryCompactionConfig,
    ConversationSummaryCompactionConfig,
    ConversationSummaryConfig,
    ConversationSummaryVectorIndexConfig,
)


@dataclass(frozen=True)
class ConversationLifecycleConfig:
    """Conversation 原文释放和两阶段 Summary 压缩的统一生命周期配置。"""

    maintenance_interval_seconds: int = 3_600
    max_conversations_per_cycle: int = 100
    lease_ttl_seconds: int = 300
    heartbeat_interval_seconds: float = 60.0
    shutdown_timeout_seconds: float = 30.0
    summary_compaction: ConversationSummaryCompactionConfig = field(default_factory=ConversationSummaryCompactionConfig)

    def __post_init__(self) -> None:
        _bounded_int(
            self.maintenance_interval_seconds,
            "conversation lifecycle maintenance_interval_seconds",
            minimum=60,
            maximum=86_400,
        )
        _bounded_int(
            self.max_conversations_per_cycle,
            "conversation lifecycle max_conversations_per_cycle",
            minimum=1,
            maximum=10_000,
        )
        _bounded_int(
            self.lease_ttl_seconds,
            "conversation lifecycle lease_ttl_seconds",
            minimum=30,
            maximum=3_600,
        )
        for name, value, maximum in (
            ("heartbeat_interval_seconds", self.heartbeat_interval_seconds, 1_200.0),
            ("shutdown_timeout_seconds", self.shutdown_timeout_seconds, 3_600.0),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float) or not 0 < float(value) <= maximum:
                raise ValueError(f"conversation lifecycle {name} must be greater than zero and at most {maximum:g}")
        if self.heartbeat_interval_seconds > self.lease_ttl_seconds / 3:
            raise ValueError("conversation lifecycle heartbeat interval must be at most one third of the lease TTL")
        if not isinstance(
            self.summary_compaction,
            ConversationSummaryCompactionConfig,
        ):
            raise TypeError("conversation lifecycle summary_compaction must be ConversationSummaryCompactionConfig")

    @classmethod
    def from_mapping(cls, value: object) -> ConversationLifecycleConfig:
        data = group_fields(cls, value, "config.conversation.lifecycle")
        return cls(
            maintenance_interval_seconds=_integer_value(
                data.get("maintenance_interval_seconds", 3_600),
                "config.conversation.lifecycle.maintenance_interval_seconds",
            ),
            max_conversations_per_cycle=_integer_value(
                data.get("max_conversations_per_cycle", 100),
                "config.conversation.lifecycle.max_conversations_per_cycle",
            ),
            lease_ttl_seconds=_integer_value(
                data.get("lease_ttl_seconds", 300),
                "config.conversation.lifecycle.lease_ttl_seconds",
            ),
            heartbeat_interval_seconds=_number_value(
                data.get("heartbeat_interval_seconds", 60.0),
                "config.conversation.lifecycle.heartbeat_interval_seconds",
            ),
            shutdown_timeout_seconds=_number_value(
                data.get("shutdown_timeout_seconds", 30.0),
                "config.conversation.lifecycle.shutdown_timeout_seconds",
            ),
            summary_compaction=_summary_compaction_config(data.get("summary_compaction", {})),
        )


@dataclass(frozen=True)
class ConversationConfig:
    """Conversation journal、切段、摘要和生命周期配置分组。"""

    journal: ConversationJournalConfig = field(default_factory=ConversationJournalConfig)
    segmentation: ConversationSegmentationConfig = field(default_factory=ConversationSegmentationConfig)
    summary: ConversationSummaryConfig = field(default_factory=ConversationSummaryConfig)
    summary_vector_store: VectorStoreConfig = field(
        default_factory=lambda: VectorStoreConfig(collection="conversation_summaries")
    )
    summary_vector_index: ConversationSummaryVectorIndexConfig = field(
        default_factory=ConversationSummaryVectorIndexConfig
    )
    lifecycle: ConversationLifecycleConfig = field(default_factory=ConversationLifecycleConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.journal, ConversationJournalConfig):
            raise TypeError("conversation.journal must be ConversationJournalConfig")
        if not isinstance(self.segmentation, ConversationSegmentationConfig):
            raise TypeError("conversation.segmentation must be ConversationSegmentationConfig")
        if not isinstance(self.summary, ConversationSummaryConfig):
            raise TypeError("conversation.summary must be ConversationSummaryConfig")
        if not isinstance(self.summary_vector_store, VectorStoreConfig):
            raise TypeError("conversation.summary_vector_store must be VectorStoreConfig")
        if not isinstance(self.summary_vector_index, ConversationSummaryVectorIndexConfig):
            raise TypeError(
                "conversation.summary_vector_index must be ConversationSummaryVectorIndexConfig"
            )
        if not isinstance(self.lifecycle, ConversationLifecycleConfig):
            raise TypeError("conversation.lifecycle must be ConversationLifecycleConfig")
        for stage in (
            self.lifecycle.summary_compaction.segment_to_range,
            self.lifecycle.summary_compaction.range_to_archive,
        ):
            if stage.max_source_chars > self.summary.max_input_chars:
                raise ValueError("summary compaction source chars cannot exceed summary max_input_chars")
            if stage.max_source_count > self.summary.max_files_per_conversation:
                raise ValueError("summary compaction source count cannot exceed summary file enumeration bound")

    @classmethod
    def from_mapping(cls, value: object) -> ConversationConfig:
        data = group_fields(cls, value, "config.conversation")
        return cls(
            journal=construct_config(
                ConversationJournalConfig,
                data.get("journal", {}),
                "config.conversation.journal",
            ),
            segmentation=construct_config(
                ConversationSegmentationConfig,
                data.get("segmentation", {}),
                "config.conversation.segmentation",
            ),
            summary=construct_config(
                ConversationSummaryConfig,
                data.get("summary", {}),
                "config.conversation.summary",
            ),
            summary_vector_store=_summary_vector_store_config(
                data.get("summary_vector_store", {})
            ),
            summary_vector_index=construct_config(
                ConversationSummaryVectorIndexConfig,
                data.get("summary_vector_index", {}),
                "config.conversation.summary_vector_index",
            ),
            lifecycle=ConversationLifecycleConfig.from_mapping(data.get("lifecycle", {})),
        )


def _bounded_int(value: object, label: str, *, minimum: int, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ValueError(f"{label} must be between {minimum} and {maximum}")


def _integer_value(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"'{path}' must be an integer")
    return value


def _number_value(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ConfigError(f"'{path}' must be numeric")
    return float(value)


def _summary_compaction_config(value: object) -> ConversationSummaryCompactionConfig:
    path = "config.conversation.lifecycle.summary_compaction"
    data = group_fields(ConversationSummaryCompactionConfig, value, path)
    defaults = ConversationSummaryCompactionConfig()
    try:
        return ConversationSummaryCompactionConfig(
            enabled=_boolean_value(data.get("enabled", defaults.enabled), f"{path}.enabled"),
            segment_to_range=construct_config(
                ConversationSegmentSummaryCompactionConfig,
                data.get("segment_to_range", {}),
                f"{path}.segment_to_range",
            ),
            range_to_archive=construct_config(
                ConversationRangeSummaryCompactionConfig,
                data.get("range_to_archive", {}),
                f"{path}.range_to_archive",
            ),
            superseded_source_retention_days=_integer_value(
                data.get(
                    "superseded_source_retention_days",
                    defaults.superseded_source_retention_days,
                ),
                f"{path}.superseded_source_retention_days",
            ),
            cleanup_batch_size=_integer_value(
                data.get("cleanup_batch_size", defaults.cleanup_batch_size),
                f"{path}.cleanup_batch_size",
            ),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid '{path}': {exc}") from exc


def _boolean_value(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"'{path}' must be a boolean")
    return value


def _summary_vector_store_config(value: object) -> VectorStoreConfig:
    try:
        if value == {}:
            return VectorStoreConfig(collection="conversation_summaries")
        return VectorStoreConfig.from_mapping(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"invalid 'config.conversation.summary_vector_store': {exc}") from exc


__all__ = [
    "ConversationConfig",
    "ConversationLifecycleConfig",
    "ConversationRangeSummaryCompactionConfig",
    "ConversationSegmentSummaryCompactionConfig",
    "ConversationSummaryCompactionConfig",
    "ConversationSummaryVectorIndexConfig",
]
