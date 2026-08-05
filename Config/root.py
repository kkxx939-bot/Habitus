"""m2bOS 唯一外部配置根与跨领域校验。"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from behavior.config import BehaviorConfig
from Config.behavior import behavior_config_from_mapping
from Config.conversation import ConversationConfig
from Config.credentials import CredentialRegistry
from Config.http import HTTPAPIConfig
from Config.loader import ConfigError, group_fields, load_config_object, required_field
from Config.memory import MemoryConfig
from Config.models import ModelConfig
from Config.observability import ObservabilityConfig
from Config.storage import StorageConfig
from Config.workflow import WorkflowConfig

_FIXED_RETRIEVAL_DOCUMENTS = 1
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class M2BOSConfig:
    """以一棵严格分组配置树描述完整 m2bOS 记忆主链。"""

    storage: StorageConfig
    models: ModelConfig
    credentials: CredentialRegistry = field(repr=False)
    conversation: ConversationConfig = field(default_factory=ConversationConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    behavior: BehaviorConfig = field(default_factory=BehaviorConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    http: HTTPAPIConfig = field(default_factory=HTTPAPIConfig)
    observability: ObservabilityConfig = field(default_factory=ObservabilityConfig)

    def __post_init__(self) -> None:
        expected = (
            ("storage", self.storage, StorageConfig),
            ("models", self.models, ModelConfig),
            ("credentials", self.credentials, CredentialRegistry),
            ("http", self.http, HTTPAPIConfig),
            ("observability", self.observability, ObservabilityConfig),
            ("conversation", self.conversation, ConversationConfig),
            ("memory", self.memory, MemoryConfig),
            ("behavior", self.behavior, BehaviorConfig),
            ("workflow", self.workflow, WorkflowConfig),
        )
        for name, value, expected_type in expected:
            if not isinstance(value, expected_type):
                raise TypeError(f"{name} must be {expected_type.__name__}")
        self._validate_cross_domain_limits()

    @property
    def storage_root(self) -> Path:
        """返回 Conversation、Memory、Behavior 和 Workflow 的共同根目录。"""

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
        """返回 Evidence & Claim Layer 的独立耐久根目录。"""

        return self.storage_root / "behavior"

    @property
    def workflow_root(self) -> Path:
        """返回 Job、Receipt 和事务日志的树外工作目录。"""

        return self.storage_root / "workflow"

    @property
    def transaction_root(self) -> Path:
        """返回多文档提交恢复日志目录。"""

        return self.workflow_root / "transactions"

    @classmethod
    def from_mapping(cls, value: object) -> M2BOSConfig:
        """从一个严格对象创建全部配置；未知字段立即失败。"""

        data = group_fields(cls, value, "config")
        return cls(
            storage=StorageConfig.from_mapping(required_field(data, "storage", path="config")),
            models=ModelConfig.from_mapping(required_field(data, "models", path="config")),
            credentials=CredentialRegistry.from_mapping(
                required_field(data, "credentials", path="config")
            ),
            http=HTTPAPIConfig.from_mapping(data.get("http", {})),
            observability=ObservabilityConfig.from_mapping(data.get("observability", {})),
            conversation=ConversationConfig.from_mapping(data.get("conversation", {})),
            memory=MemoryConfig.from_mapping(data.get("memory", {})),
            behavior=behavior_config_from_mapping(data.get("behavior", {})),
            workflow=WorkflowConfig.from_mapping(data.get("workflow", {})),
        )

    @classmethod
    def from_file(cls, path: str | Path) -> M2BOSConfig:
        """从单一有界 YAML 文件加载全部 m2bOS 配置。"""

        resolved = Path(path).expanduser().absolute()
        config = cls.from_mapping(load_config_object(resolved))
        if config.credentials.contains_secret_values and resolved.stat().st_mode & 0o077:
            raise ConfigError("secret-bearing config file must not grant group or other permissions")
        return config

    @classmethod
    def from_env(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
        config_path_env: str = "M2BOS_CONFIG_FILE",
    ) -> M2BOSConfig:
        """环境只选择统一 YAML 文件；全部运行配置和秘密值都来自该文件。"""

        values = os.environ if environ is None else environ
        if not isinstance(values, Mapping):
            raise TypeError("environ must be a string mapping")
        if not isinstance(config_path_env, str) or _ENV_NAME.fullmatch(config_path_env) is None:
            raise ValueError("config_path_env must be a normalized environment name")
        raw_path = values.get(config_path_env)
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ConfigError(f"config file path is missing: {config_path_env}")
        return cls.from_file(raw_path.strip())

    def _validate_cross_domain_limits(self) -> None:
        self._validate_credential_references()
        memory = self.memory
        models = self.models
        workflow = self.workflow
        conversation = self.conversation
        behavior = self.behavior
        segmentation = conversation.segmentation

        if (
            behavior.claim.max_model_input_chars + behavior.claim.max_model_output_tokens
            > models.chat.context_window_tokens
        ):
            raise ConfigError("behavior Claim model budgets exceed models.chat.context_window_tokens")

        if memory.document.max_encoded_bytes > memory.snapshot.max_item_bytes:
            raise ConfigError("memory.document.max_encoded_bytes cannot exceed memory.snapshot.max_item_bytes")
        if max(segmentation.max_live_bytes, segmentation.max_segment_bytes) > conversation.journal.max_file_bytes:
            raise ConfigError("conversation segmentation byte limits cannot exceed conversation.journal.max_file_bytes")
        if (
            max(
                memory.vector_index.max_direct_entries,
                memory.semantic.max_direct_entries,
            )
            + 2
            > memory.tree.max_children_per_directory
        ):
            raise ConfigError("memory direct-entry limits cannot exceed memory.tree.max_children_per_directory")
        if memory.transaction_journal.max_record_bytes < memory.snapshot.max_total_bytes * 2:
            raise ConfigError(
                "memory.transaction_journal.max_record_bytes must be at least twice memory.snapshot.max_total_bytes"
            )
        initial_items = _FIXED_RETRIEVAL_DOCUMENTS + memory.retrieval.max_tool_uris + memory.retrieval.search_limit
        if initial_items > memory.snapshot.max_items:
            raise ConfigError("memory retrieval can exceed memory.snapshot.max_items")
        if initial_items > memory.extraction.max_old_memory_items:
            raise ConfigError("memory retrieval can exceed memory.extraction.max_old_memory_items")
        if memory.search_service.max_limit > memory.snapshot.max_items:
            raise ConfigError("memory.search_service.max_limit cannot exceed memory.snapshot.max_items")
        if memory.search_service.max_relation_neighbors_total > memory.snapshot.max_items:
            raise ConfigError("memory.search_service relation expansion cannot exceed memory.snapshot.max_items")
        lifecycle_candidate_limit = memory.search_service.max_limit * memory.search_service.candidate_multiplier
        if (
            memory.recall_lifecycle.enabled
            and lifecycle_candidate_limit > memory.recall_lifecycle.max_batch_size
        ):
            raise ConfigError("memory recall lifecycle cannot rank the maximum search candidate batch")
        if memory.search_service.max_context_chars < memory.document.max_markdown_body_chars + 2_048:
            raise ConfigError("memory.search_service.max_context_chars cannot fit one maximum-size memory document")
        planner_context_bound = (
            memory.search_service.max_query_chars
            + memory.search_service.max_summary_context_chars
            + memory.search_service.max_recent_messages * memory.search_service.max_recent_message_chars
            + memory.search_service.max_target_context_chars
            + 4_096
        )
        if planner_context_bound > memory.search_service.max_planner_context_chars:
            raise ConfigError("memory.search_service planner inputs can exceed max_planner_context_chars")
        if memory.search_service.max_recent_messages > segmentation.max_live_messages:
            raise ConfigError("memory.search_service recent messages cannot exceed the live message bound")
        if conversation.summary_vector_store.collection == memory.vector_store.collection:
            raise ConfigError("Memory and Conversation Summary must use different vector collections")
        if (
            conversation.summary_vector_index.max_record_chars
            > models.embedding.max_input_chars
        ):
            raise ConfigError(
                "conversation.summary_vector_index.max_record_chars cannot exceed models.embedding.max_input_chars"
            )
        summary_candidate_limit = max(
            conversation.summary_vector_index.min_vector_candidates,
            memory.search_service.summary_fallback_limit
            * conversation.summary_vector_index.candidate_multiplier,
        )
        if summary_candidate_limit > conversation.summary_vector_index.max_search_hits:
            raise ConfigError(
                "Summary fallback vector candidates can exceed its index search bound"
            )
        if (
            conversation.summary_vector_index.max_records_per_conversation
            < conversation.summary.max_files_per_conversation * 3
        ):
            raise ConfigError(
                "Summary index per-Conversation bound cannot enumerate all three physical stages"
            )
        if memory.extraction.max_old_memory_items > memory.snapshot.max_items:
            raise ConfigError("memory.extraction.max_old_memory_items cannot exceed memory.snapshot.max_items")
        if memory.extraction.max_old_memory_bytes > memory.snapshot.max_total_bytes:
            raise ConfigError("memory.extraction.max_old_memory_bytes cannot exceed memory.snapshot.max_total_bytes")
        maximum_extraction_output = max(
            memory.extraction.grader_max_output_tokens,
            memory.extraction.candidate_max_output_tokens,
        )
        if (
            memory.extraction.max_input_tokens + maximum_extraction_output
            > models.chat.context_window_tokens
        ):
            raise ConfigError(
                "memory.extraction input and output budgets exceed models.chat.context_window_tokens"
            )
        if memory.extraction.max_old_memory_tokens >= memory.extraction.max_input_tokens:
            raise ConfigError(
                "memory.extraction.max_old_memory_tokens must leave room for Conversation and prompts"
            )
        if segmentation.max_inline_tool_result_bytes > segmentation.max_segment_bytes:
            raise ConfigError("conversation.segmentation.max_inline_tool_result_bytes cannot exceed max_segment_bytes")
        if memory.vector_index.max_record_chars > models.embedding.max_input_chars:
            raise ConfigError("memory.vector_index.max_record_chars cannot exceed models.embedding.max_input_chars")
        maximum_query_chars = max(
            memory.search_service.max_query_chars,
            memory.search_service.max_planned_query_chars,
            memory.retrieval.max_query_chars,
            memory.extraction.max_query_chars,
        )
        if maximum_query_chars > models.embedding.max_input_chars:
            raise ConfigError("memory query character limits cannot exceed models.embedding.max_input_chars")
        maximum_search_request = max(
            memory.search_service.max_limit * memory.search_service.candidate_multiplier,
            memory.retrieval.search_limit,
            memory.extraction.additional_search_limit,
        )
        maximum_vector_candidates = max(
            memory.semantic_search.min_vector_candidates,
            maximum_search_request * memory.semantic_search.candidate_multiplier,
        )
        index_hit_limit = memory.vector_index.max_search_hits
        if maximum_vector_candidates > index_hit_limit:
            raise ConfigError("semantic search vector candidates can exceed memory.vector_index.max_search_hits")
        if (
            max(
                memory.semantic_search.directory_candidates,
                memory.semantic_search.child_candidates,
            )
            > index_hit_limit
        ):
            raise ConfigError("semantic directory candidates can exceed memory.vector_index.max_search_hits")
        if memory.semantic.max_direct_entries > memory.vector_index.max_direct_entries:
            raise ConfigError("memory.semantic.max_direct_entries cannot exceed memory.vector_index.max_direct_entries")
        if workflow.worker.heartbeat_interval_seconds > workflow.jobs.lease_ttl_seconds / 3:
            raise ConfigError(
                "workflow.worker.heartbeat_interval_seconds must be at most one third of the job lease TTL"
            )
        if workflow.lifecycle.cleanup_batch_size > workflow.jobs.max_files:
            raise ConfigError("workflow lifecycle cleanup batch cannot exceed workflow.jobs.max_files")
        if workflow.lifecycle.cleanup_batch_size > workflow.receipts.max_files:
            raise ConfigError("workflow lifecycle cleanup batch cannot exceed workflow.receipts.max_files")
        if conversation.lifecycle.max_conversations_per_cycle > conversation.journal.max_conversation_tree_entries:
            raise ConfigError("conversation lifecycle batch cannot exceed the Conversation tree enumeration bound")
        if models.rerank is not None:
            if memory.semantic_search.max_rerank_candidates > models.rerank.max_documents:
                raise ConfigError(
                    "memory.semantic_search.max_rerank_candidates cannot exceed models.rerank.max_documents"
                )
            if memory.semantic_search.max_rerank_document_chars > models.rerank.max_document_chars:
                raise ConfigError(
                    "memory.semantic_search.max_rerank_document_chars cannot exceed models.rerank.max_document_chars"
                )
            if maximum_query_chars > models.rerank.max_query_chars:
                raise ConfigError("memory query character limits cannot exceed models.rerank.max_query_chars")
            if (
                conversation.summary_vector_index.max_rerank_candidates
                > models.rerank.max_documents
            ):
                raise ConfigError(
                    "conversation Summary rerank candidates cannot exceed models.rerank.max_documents"
                )
            if (
                conversation.summary_vector_index.max_rerank_document_chars
                > models.rerank.max_document_chars
            ):
                raise ConfigError(
                    "conversation Summary rerank content cannot exceed models.rerank.max_document_chars"
                )
            if (
                memory.search_service.summary_fallback_limit
                > conversation.summary_vector_index.max_rerank_candidates
            ):
                raise ConfigError(
                    "Summary fallback limit cannot exceed its rerank candidate bound"
                )

    def _validate_credential_references(self) -> None:
        model_routes = [
            ("config.models.chat.route", self.models.chat.route.credential_ref),
            ("config.models.embedding.route", self.models.embedding.route.credential_ref),
        ]
        if self.models.rerank is not None:
            model_routes.append(
                ("config.models.rerank.route", self.models.rerank.route.credential_ref)
            )
        for _path, reference in model_routes:
            if reference:
                self.credentials.resolve(reference)
        for reference in (
            self.memory.vector_store.route.credential_ref,
            self.conversation.summary_vector_store.route.credential_ref,
            self.observability.tracing.credential_ref,
        ):
            if reference:
                self.credentials.resolve(reference)


__all__ = ["M2BOSConfig"]
