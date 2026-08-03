"""只负责生成和校验 L2 业务字段的语义压缩，不负责生命周期或提交。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from foundation.integrity import canonical_json
from memory.compaction.field_ops import (
    SemanticFieldMergePolicy,
    SemanticFieldOperationBatch,
    SemanticFieldOperationError,
    merge_semantic_fields,
)
from memory.document import MemoryDocument
from memory.schema import MemoryFieldRole, MemoryFieldSchema, MemoryFieldType, MemorySchemaRegistry
from ModelClient import ChatCallContext, ChatMessage, ChatRequest, StructuredChatClient


class MemoryFieldCompactionError(RuntimeError):
    """模型字段操作没有形成更短且 Schema 有效的同身份内容。"""


@dataclass(frozen=True)
class MemoryFieldCompactionConfig:
    max_input_chars: int = 2_000_000
    max_output_tokens: int = 4_096
    max_field_chars: int = 1_000_000

    def __post_init__(self) -> None:
        for name, value, minimum, maximum in (
            ("max_input_chars", self.max_input_chars, 1_024, 16_000_000),
            ("max_output_tokens", self.max_output_tokens, 256, 65_536),
            ("max_field_chars", self.max_field_chars, 1_024, 4_000_000),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be between {minimum} and {maximum}")


@dataclass(frozen=True)
class MemoryFieldCompactionResult:
    source: MemoryDocument
    fields: Mapping[str, Any]
    operations: SemanticFieldOperationBatch
    changed_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source, MemoryDocument):
            raise TypeError("source must be MemoryDocument")
        if not isinstance(self.fields, Mapping):
            raise TypeError("fields must be a mapping")
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))
        if not isinstance(self.operations, SemanticFieldOperationBatch):
            raise TypeError("operations must be SemanticFieldOperationBatch")
        if not isinstance(self.changed_fields, tuple):
            raise TypeError("changed_fields must be a tuple")

    @property
    def changed(self) -> bool:
        return bool(self.changed_fields)


class MemoryFieldCompactor:
    """参考 OpenViking 的字段操作协议，把详细 L2 压缩为可检索概念。"""

    def __init__(
        self,
        client: StructuredChatClient,
        *,
        registry: MemorySchemaRegistry | None = None,
        config: MemoryFieldCompactionConfig | None = None,
    ) -> None:
        if not isinstance(client, StructuredChatClient):
            raise TypeError("client must be StructuredChatClient")
        if registry is not None and not isinstance(registry, MemorySchemaRegistry):
            raise TypeError("registry must be MemorySchemaRegistry")
        if config is not None and not isinstance(config, MemoryFieldCompactionConfig):
            raise TypeError("config must be MemoryFieldCompactionConfig")
        self.client = client
        self.registry = registry or MemorySchemaRegistry.load_default()
        self.config = config or MemoryFieldCompactionConfig()

    async def compact(self, document: MemoryDocument) -> MemoryFieldCompactionResult:
        if not isinstance(document, MemoryDocument):
            raise TypeError("document must be MemoryDocument")
        schema = self.registry.get(document.kind)
        content_fields = tuple(
            field
            for field in schema.fields
            if field.role is MemoryFieldRole.CONTENT and field.field_type is MemoryFieldType.STRING
        )
        policies = tuple(
            SemanticFieldMergePolicy(
                field=field.name,
                allow_update=not bool(field.allowed_values),
                allow_append=not bool(field.allowed_values),
                max_chars=self.config.max_field_chars,
            )
            for field in content_fields
        )
        source_fields = {field.name: document.fields.get(field.name, "") for field in content_fields}
        payload = canonical_json(
            {
                "memory_type": document.kind.value,
                "identity_fields": {
                    field.name: document.fields[field.name]
                    for field in schema.fields
                    if field.role is MemoryFieldRole.ADDRESS
                },
                "content_fields": source_fields,
                "field_rules": {
                    field.name: field.description
                    for field in content_fields
                },
            }
        )
        if len(payload) > self.config.max_input_chars:
            raise MemoryFieldCompactionError("L2 memory exceeds the semantic compaction input bound")
        response = await self.client.complete_model_async(
            ChatRequest(
                messages=(
                    ChatMessage(
                        role="system",
                        content=(
                            "你是长期记忆字段压缩器。输入是一条已经通过证据审查的 L2 记忆，不是新来源。"
                            "对每个允许处理的 content field 只输出 KEEP、UPDATE 或 APPEND。缺失字段由系统"
                            "按 KEEP 处理。目标是把详细叙述逐步压缩成自包含、可检索的概念摘要，同时保留"
                            "仍可能影响未来回答的名字、日期、数量、约束、例外、纠正和最终状态。不得修改"
                            "identity_fields，不得新增字段、系统字段、事实或推断，不得因追求短而删除唯一"
                            "信息。UPDATE 必须给出该字段完整替代内容；APPEND 只列要追加且去重的项目。"
                            "若当前字段已经足够紧凑或无法安全压缩，输出 KEEP。"
                        ),
                    ),
                    ChatMessage(role="user", content="请输出严格字段操作 JSON：\n" + payload),
                ),
                temperature=0.0,
                max_output_tokens=self.config.max_output_tokens,
            ),
            model_class=SemanticFieldOperationBatch,
            name="l2_semantic_field_compaction",
            context=ChatCallContext(prompt_version="l2_semantic_field_compaction_v1"),
        )
        operations = response.value
        try:
            merged_content = merge_semantic_fields(source_fields, operations, policies)
            merged = dict(document.fields)
            merged.update(merged_content)
            validated = self.registry.validate(document.kind, merged)
            if self.registry.address_for(document.kind, validated) != document.address:
                raise SemanticFieldOperationError("semantic compaction changed the L2 identity")
        except (TypeError, ValueError) as exc:
            raise MemoryFieldCompactionError("L2 semantic field operations failed server validation") from exc
        changed = tuple(
            field.name
            for field in content_fields
            if validated.get(field.name) != document.fields.get(field.name)
        )
        if changed and _content_size(validated, content_fields) >= _content_size(document.fields, content_fields):
            raise MemoryFieldCompactionError("L2 semantic compaction did not reduce business content")
        return MemoryFieldCompactionResult(document, validated, operations, changed)


def _content_size(fields: Mapping[str, Any], schemas: tuple[MemoryFieldSchema, ...]) -> int:
    return sum(len(str(fields.get(field.name, ""))) for field in schemas)


__all__ = [
    "MemoryFieldCompactionConfig",
    "MemoryFieldCompactionError",
    "MemoryFieldCompactionResult",
    "MemoryFieldCompactor",
]
