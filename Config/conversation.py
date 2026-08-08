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
class ConversationSourceConfig:
    """Source、Output、Outcome、Recovery 与执行租约的独立容量边界。"""

    max_envelope_bytes: int = 16 * 1024 * 1024
    max_source_files: int = 100_000
    max_output_files_per_consumer: int = 4
    max_outcome_bytes: int = 64 * 1024
    max_memory_output_bytes: int = 32 * 1024 * 1024
    recovery_batch_size: int = 100
    execution_lock_ttl_seconds: int = 300
    execution_lock_heartbeat_seconds: float = 60.0
    execution_lock_wait_seconds: float = 30.0
    shutdown_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        for integer_name, integer_value, integer_maximum in (
            ("max_envelope_bytes", self.max_envelope_bytes, 256 * 1024 * 1024),
            ("max_source_files", self.max_source_files, 10_000_000),
            ("max_output_files_per_consumer", self.max_output_files_per_consumer, 1_000),
            ("max_outcome_bytes", self.max_outcome_bytes, 16 * 1024 * 1024),
            ("max_memory_output_bytes", self.max_memory_output_bytes, 512 * 1024 * 1024),
            ("recovery_batch_size", self.recovery_batch_size, 100_000),
            ("execution_lock_ttl_seconds", self.execution_lock_ttl_seconds, 86_400),
        ):
            _bounded_int(
                integer_value,
                f"conversation source {integer_name}",
                minimum=1,
                maximum=integer_maximum,
            )
        for number_name, number_value, number_maximum in (
            ("execution_lock_heartbeat_seconds", self.execution_lock_heartbeat_seconds, 28_800.0),
            ("execution_lock_wait_seconds", self.execution_lock_wait_seconds, 86_400.0),
            ("shutdown_timeout_seconds", self.shutdown_timeout_seconds, 3_600.0),
        ):
            if (
                isinstance(number_value, bool)
                or not isinstance(number_value, int | float)
                or not 0 < float(number_value) <= number_maximum
            ):
                raise ValueError(
                    f"conversation source {number_name} must be greater than zero "
                    f"and at most {number_maximum:g}"
                )
        if self.execution_lock_heartbeat_seconds > self.execution_lock_ttl_seconds / 3:
            raise ValueError("conversation source heartbeat interval must be at most one third of lock TTL")


@dataclass(frozen=True)
class ConversationBehaviorProjectionConfig:
    """Behavior Projection Output 的独立容量边界。"""

    max_projection_output_bytes: int = 16 * 1024 * 1024
    max_projection_items: int = 100_000

    def __post_init__(self) -> None:
        _bounded_int(
            self.max_projection_output_bytes,
            "conversation behavior projection max_projection_output_bytes",
            minimum=1,
            maximum=256 * 1024 * 1024,
        )
        _bounded_int(
            self.max_projection_items,
            "conversation behavior projection max_projection_items",
            minimum=1,
            maximum=10_000_000,
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
    source: ConversationSourceConfig = field(default_factory=ConversationSourceConfig)
    behavior_projection: ConversationBehaviorProjectionConfig = field(
        default_factory=ConversationBehaviorProjectionConfig
    )
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
        if not isinstance(self.source, ConversationSourceConfig):
            raise TypeError("conversation.source must be ConversationSourceConfig")
        if not isinstance(self.behavior_projection, ConversationBehaviorProjectionConfig):
            raise TypeError(
                "conversation.behavior_projection must be ConversationBehaviorProjectionConfig"
            )
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
        expansion_reads = self.lifecycle.summary_compaction.range_to_archive.max_source_count * (
            1 + self.lifecycle.summary_compaction.segment_to_range.max_source_count
        )
        if expansion_reads > 100_000:
            raise ValueError("summary compaction fanout exceeds the bounded fallback expansion limit")

    @classmethod
    def from_mapping(cls, value: object) -> ConversationConfig:
        data = group_fields(cls, value, "config.conversation")
        return cls(
            journal=construct_config(
                ConversationJournalConfig,
                data.get("journal", {}),
                "config.conversation.journal",
            ),
            source=construct_config(
                ConversationSourceConfig,
                data.get("source", {}),
                "config.conversation.source",
            ),
            behavior_projection=construct_config(
                ConversationBehaviorProjectionConfig,
                data.get("behavior_projection", {}),
                "config.conversation.behavior_projection",
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
            recent_use_protection_days=_integer_value(
                data.get(
                    "recent_use_protection_days",
                    defaults.recent_use_protection_days,
                ),
                f"{path}.recent_use_protection_days",
            ),
            archive_retire_days=_integer_value(
                data.get("archive_retire_days", defaults.archive_retire_days),
                f"{path}.archive_retire_days",
            ),
            archive_retire_grace_days=_integer_value(
                data.get("archive_retire_grace_days", defaults.archive_retire_grace_days),
                f"{path}.archive_retire_grace_days",
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
    "ConversationBehaviorProjectionConfig",
    "ConversationConfig",
    "ConversationLifecycleConfig",
    "ConversationRangeSummaryCompactionConfig",
    "ConversationSegmentSummaryCompactionConfig",
    "ConversationSourceConfig",
    "ConversationSummaryCompactionConfig",
    "ConversationSummaryVectorIndexConfig",
]
