"""Habitus 唯一外部配置根与跨领域校验。"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from habitus.config.behavior import BehaviorConfig
from habitus.config.conversation import ConversationConfig
from habitus.config.credentials import CredentialRegistry
from habitus.config.foresight import ForesightConfig
from habitus.config.http import HTTPAPIConfig
from habitus.config.lifecycle import LifecycleConfig
from habitus.config.loader import ConfigError, group_fields, load_config_object, required_field
from habitus.config.locale import LocaleConfig
from habitus.config.memory import MemoryConfig
from habitus.config.models import ModelConfig
from habitus.config.network import outbound_settings_from_mapping
from habitus.config.observability import observability_config_from_mapping
from habitus.config.scene import SceneConfig
from habitus.config.storage import StorageConfig
from habitus.config.temporal import TemporalConfig
from habitus.config.workflow import WorkflowConfig
from habitus.foundation.network import OutboundSettings
from habitus.foundation.validation import is_nonblank_str
from habitus.infrastructure.observability.config import ObservabilityConfig
from habitus.infrastructure.store.filesystem import grants_group_or_other_access

_FIXED_RETRIEVAL_DOCUMENTS = 1
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


_REMOVED_SECTIONS = (
    ("models", "embedding"),
    ("models", "rerank"),
    ("memory", "vector_store"),
    ("memory", "vector_index"),
    ("memory", "semantic_search"),
    ("conversation", "summary_vector_store"),
    ("conversation", "summary_vector_index"),
)


def _reject_removed_sections(value: object) -> None:
    """向量与重排已从 Habitus 移除：旧配置里还留着这些段时，指明是哪一段、该怎么改，不当作拼写错误。"""

    if not isinstance(value, Mapping):
        return
    found = [
        f"config.{group}.{name}"
        for group, name in _REMOVED_SECTIONS
        if isinstance(value.get(group), Mapping) and name in value[group]
    ]
    if found:
        raise ConfigError(
            "这些配置段对应的向量检索与重排已经从 Habitus 移除，请从配置文件里删掉它们（连同 credentials 里只给它们用的"
            "凭据）后重启：" + "、".join(found)
        )


@dataclass(frozen=True)
class HabitusConfig:
    """以一棵严格分组配置树描述完整 Habitus 记忆主链。"""

    storage: StorageConfig
    models: ModelConfig
    credentials: CredentialRegistry = field(repr=False)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    foresight: ForesightConfig = field(default_factory=ForesightConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    # 统一生命周期：Worker 调度、重试账、派生物代数、容量告警（裁定 60）。
    lifecycle: LifecycleConfig = field(default_factory=LifecycleConfig)
    # 主体所在地：情景读侧标日型、预测层造"此刻"时钟，两处必须同一个地方，所以单独一组。
    locale: LocaleConfig = field(default_factory=LocaleConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    http: HTTPAPIConfig = field(default_factory=HTTPAPIConfig)
    network: OutboundSettings = field(default_factory=OutboundSettings)  # 调模型服务的证书库、额外 CA 与代理（W18）
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)

    def __post_init__(self) -> None:
        self._validate_cross_domain_limits()

    @property
    def storage_root(self) -> Path:
        """返回 Conversation、Memory 和 Workflow 的共同根目录。"""

        return Path(self.storage.root)

    @property
    def memory_root(self) -> Path:
        """返回唯一 L2 记忆树根目录。"""

        return self.storage_root / "memory"

    @property
    def conversation_root(self) -> Path:
        """返回 Conversation messages/summaries 的共同根目录。"""

        return self.storage_root / "conversation"

    @property
    def behavior_root(self) -> Path:
        """返回行为管线（观测/判断/回执/作业/行为树/归约账本）的共同根目录。"""

        return self.storage_root / "behavior"

    @property
    def scene_root(self) -> Path:
        """语义关联层（情景树与刷新进度）的根；与行为树并列，是行为树的派生物。"""

        return self.storage_root / "scene"

    @property
    def foresight_root(self) -> Path:
        """预测层的结算账本根（承诺与结算）；派生物，与三棵树并列。"""

        return self.storage_root / "foresight"

    @property
    def workflow_root(self) -> Path:
        """返回 Job、Receipt 和事务日志的树外工作目录。"""

        return self.storage_root / "workflow"

    @property
    def transaction_root(self) -> Path:
        """返回多文档提交恢复日志目录。"""

        return self.workflow_root / "transactions"

    @classmethod
    def from_mapping(cls, value: object) -> HabitusConfig:
        """从一个严格对象创建全部配置；未知字段立即失败。"""

        _reject_removed_sections(value)
        data = group_fields(cls, value, "config")
        return cls(
            storage=StorageConfig.from_mapping(required_field(data, "storage", path="config")),
            models=ModelConfig.from_mapping(required_field(data, "models", path="config")),
            credentials=CredentialRegistry.from_mapping(required_field(data, "credentials", path="config")),
            http=HTTPAPIConfig.from_mapping(data.get("http", {})),
            network=outbound_settings_from_mapping(data.get("network", {})),
            observability=observability_config_from_mapping(data.get("observability", {})),
            conversation=ConversationConfig.from_mapping(data.get("conversation", {})),
            memory=MemoryConfig.from_mapping(data.get("memory", {})),
            behavior=BehaviorConfig.from_mapping(data.get("behavior", {})),
            scene=SceneConfig.from_mapping(data.get("scene", {})),
            foresight=ForesightConfig.from_mapping(data.get("foresight", {})),
            temporal=TemporalConfig.from_mapping(data.get("temporal", {})),
            lifecycle=LifecycleConfig.from_mapping(data.get("lifecycle", {})),
            locale=LocaleConfig.from_mapping(data.get("locale", {})),
            workflow=WorkflowConfig.from_mapping(data.get("workflow", {})),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> HabitusConfig:
        """从单一有界 YAML 文件加载全部 Habitus 配置。"""

        resolved = Path(path).expanduser().absolute()
        config = cls.from_mapping(load_config_object(resolved))
        if config.credentials.contains_secret_values and grants_group_or_other_access(resolved):
            raise ConfigError("secret-bearing config file must not grant group or other permissions")
        return config

    @classmethod
    def from_env(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        config_path_env: str = "HABITUS_CONFIG_FILE",
    ) -> HabitusConfig:
        """环境只选择统一 YAML 文件；全部运行配置和秘密值都来自该文件。"""

        values = os.environ if environ is None else environ
        if _ENV_NAME.fullmatch(config_path_env) is None:
            raise ValueError("config_path_env must be a normalized environment name")
        raw_path = values.get(config_path_env)
        if not is_nonblank_str(raw_path):
            raise ConfigError(f"config file path is missing: {config_path_env}")
        return cls.from_file(raw_path.strip())

    def _validate_cross_domain_limits(self) -> None:
        self._validate_credential_references()
        # 各组按原顺序依次校验：同时违反几条时先报哪一条不变
        _require_layer_dependencies(self)
        _require_storage_bounds(self)
        _require_search_service_bounds(self)
        _require_extraction_bounds(self)
        _require_workflow_bounds(self)

    def _validate_credential_references(self) -> None:
        for reference in (self.models.chat.route.credential_ref, self.observability.tracing.credential_ref):
            if reference:
                self.credentials.resolve(reference)


def _require_layer_dependencies(config: HabitusConfig) -> None:
    # 语义树从行为树派生（每封口日一次封口），行为侧没开就没有输入。这是**配置自相矛盾**（我们自己的两组配置
    # 互相打架），不是现实世界的形状，所以硬拒：放过去的下场是启动一切正常、什么都没发生、无处可查。
    if config.scene.enabled and not config.behavior.enabled:
        raise ConfigError(
            "config.scene is enabled but config.behavior.primary_subject is empty; "
            "the semantic layer reads sealed days of the behaviour tree and has no input without it"
        )
    # 预测层的机会表、复现与关系全从语义树那一晚来，结算也是整晚那一趟的第一步（裁定 59：预测树删了）。
    # 语义树没开就没有整晚、没有机会表——同一类配置自相矛盾，硬拒。
    if config.foresight.enabled and not config.scene.enabled:
        raise ConfigError(
            "config.foresight is enabled but config.scene is not; "
            "the foresight layer reads its opportunity table from the semantic layer's generation "
            "and has nothing to build expectations from without it"
        )
    # 预测拍一拍一槽、对齐槽边界（实时化方案第 5 节）：两个数各配一份迟早对不上，对不上就硬拒。
    if config.foresight.enabled and config.foresight.tick_interval_seconds != config.temporal.slot_minutes * 60:
        raise ConfigError(
            f"config.foresight.tick_interval_seconds ({config.foresight.tick_interval_seconds:g}) must equal "
            f"config.temporal.slot_minutes × 60 ({config.temporal.slot_minutes * 60}): a tick is one slot"
        )


def _require_storage_bounds(config: HabitusConfig) -> None:
    memory = config.memory
    conversation = config.conversation
    segmentation = conversation.segmentation
    if memory.document.max_encoded_bytes > memory.snapshot.max_item_bytes:
        raise ConfigError("memory.document.max_encoded_bytes cannot exceed memory.snapshot.max_item_bytes")
    if max(segmentation.max_live_bytes, segmentation.max_segment_bytes) > conversation.journal.max_file_bytes:
        raise ConfigError("conversation segmentation byte limits cannot exceed conversation.journal.max_file_bytes")
    if memory.semantic.max_direct_entries + 2 > memory.tree.max_children_per_directory:
        raise ConfigError("memory direct-entry limits cannot exceed memory.tree.max_children_per_directory")
    if memory.transaction_journal.max_record_bytes < memory.snapshot.max_total_bytes * 2:
        raise ConfigError(
            "memory.transaction_journal.max_record_bytes must be at least twice memory.snapshot.max_total_bytes"
        )
    initial_items = _FIXED_RETRIEVAL_DOCUMENTS + memory.editor_retrieval.max_tool_uris
    if initial_items > memory.snapshot.max_items:
        raise ConfigError("memory.editor_retrieval can exceed memory.snapshot.max_items")
    if initial_items > memory.extraction.max_old_memory_items:
        raise ConfigError("memory.editor_retrieval can exceed memory.extraction.max_old_memory_items")


def _require_search_service_bounds(config: HabitusConfig) -> None:
    memory = config.memory
    conversation = config.conversation
    search = memory.search_service
    if search.max_limit > memory.snapshot.max_items:
        raise ConfigError("memory.search_service.max_limit cannot exceed memory.snapshot.max_items")
    if search.max_reads_per_round > memory.snapshot.max_items:
        raise ConfigError("memory.search_service.max_reads_per_round cannot exceed memory.snapshot.max_items")
    if search.max_relation_neighbors_total > memory.snapshot.max_items:
        raise ConfigError("memory.search_service relation expansion cannot exceed memory.snapshot.max_items")
    if search.max_context_chars < memory.document.max_markdown_body_chars + 2_048:
        raise ConfigError("memory.search_service.max_context_chars cannot fit one maximum-size memory document")
    if search.max_recent_messages > conversation.segmentation.max_live_messages:
        raise ConfigError("memory.search_service recent messages cannot exceed the live message bound")
    if (
        search.max_material_chars + search.max_catalog_chars + search.max_summary_context_chars
        > search.max_prompt_chars
    ):
        raise ConfigError(
            "memory.search_service materials, catalog and conversation context can exceed max_prompt_chars"
        )
    if conversation.summary_reader.max_records_per_conversation < conversation.summary.max_files_per_conversation * 3:
        raise ConfigError("Summary reader per-Conversation bound cannot enumerate all three physical stages")


def _require_extraction_bounds(config: HabitusConfig) -> None:
    memory = config.memory
    models = config.models
    segmentation = config.conversation.segmentation
    if memory.extraction.max_old_memory_items > memory.snapshot.max_items:
        raise ConfigError("memory.extraction.max_old_memory_items cannot exceed memory.snapshot.max_items")
    if memory.extraction.max_old_memory_bytes > memory.snapshot.max_total_bytes:
        raise ConfigError("memory.extraction.max_old_memory_bytes cannot exceed memory.snapshot.max_total_bytes")
    maximum_extraction_output = max(
        memory.extraction.grader_max_output_tokens,
        memory.extraction.candidate_max_output_tokens,
    )
    if memory.extraction.max_input_tokens + maximum_extraction_output > models.chat.context_window_tokens:
        raise ConfigError("memory.extraction input and output budgets exceed models.chat.context_window_tokens")
    if (
        memory.extraction.max_old_memory_tokens + memory.extraction.max_listing_tokens
        >= memory.extraction.max_input_tokens
    ):
        raise ConfigError(
            "memory.extraction.max_old_memory_tokens plus max_listing_tokens must leave room for Conversation and prompts"
        )
    if segmentation.max_inline_tool_result_bytes > segmentation.max_segment_bytes:
        raise ConfigError("conversation.segmentation.max_inline_tool_result_bytes cannot exceed max_segment_bytes")


def _require_workflow_bounds(config: HabitusConfig) -> None:
    workflow = config.workflow
    conversation = config.conversation
    if workflow.worker.heartbeat_interval_seconds > workflow.jobs.lease_ttl_seconds / 3:
        raise ConfigError("workflow.worker.heartbeat_interval_seconds must be at most one third of the job lease TTL")
    if workflow.lifecycle.cleanup_batch_size > workflow.jobs.max_files:
        raise ConfigError("workflow lifecycle cleanup batch cannot exceed workflow.jobs.max_files")
    if workflow.lifecycle.cleanup_batch_size > workflow.receipts.max_files:
        raise ConfigError("workflow lifecycle cleanup batch cannot exceed workflow.receipts.max_files")
    if conversation.lifecycle.max_conversations_per_cycle > conversation.journal.max_conversation_tree_entries:
        raise ConfigError("conversation lifecycle batch cannot exceed the Conversation tree enumeration bound")
    if conversation.lifecycle.idle_scan_batch_conversations > conversation.journal.max_conversation_tree_entries:
        raise ConfigError("conversation idle scan batch cannot exceed the Conversation tree enumeration bound")
    if conversation.lifecycle.idle_scan_interval_seconds > config.lifecycle.worker.maintenance_interval_seconds:
        raise ConfigError(
            "conversation.lifecycle.idle_scan_interval_seconds cannot exceed lifecycle.worker.maintenance_interval_seconds"
        )


__all__ = ["HabitusConfig"]
