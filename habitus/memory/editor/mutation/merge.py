"""按照记忆 YAML 声明计算完整且可校验的最终业务字段。"""

from __future__ import annotations

from typing import Any

from habitus.memory.document import MemoryDocument
from habitus.memory.editor.mutation.model import (
    MemoryFieldMergeResult,
    MemoryNodeMatch,
    MemoryNodeMatchStatus,
)
from habitus.memory.schema import (
    MemoryMergeStrategy,
    MemoryOperationMode,
    MemorySchemaRegistry,
    MemoryTypeSchema,
)

_MISSING = object()


class MemoryFieldMergeError(ValueError):
    """候选字段与旧字段无法按严格 Schema 合并。"""


class MemoryFieldMerger:
    """执行字段级策略，不解释自然语言，也不产生存储副作用。"""

    def __init__(self, registry: MemorySchemaRegistry | None = None) -> None:
        if registry is not None and not isinstance(registry, MemorySchemaRegistry):
            raise TypeError("registry must be a MemorySchemaRegistry")
        self.registry = registry or MemorySchemaRegistry.load_default()

    def merge(self, match: MemoryNodeMatch) -> MemoryFieldMergeResult:
        """为新节点保留候选字段，为旧节点计算完整最终字段。"""

        if not isinstance(match, MemoryNodeMatch):
            raise TypeError("match must be a MemoryNodeMatch")
        schema = self.registry.get(match.candidate.kind)
        try:
            incoming = schema.validate_payload(match.candidate.fields)
        except (TypeError, ValueError) as exc:
            raise MemoryFieldMergeError("candidate fields failed memory Schema validation") from exc
        if match.status is MemoryNodeMatchStatus.NEW:
            return MemoryFieldMergeResult(
                fields=incoming,
                changed_fields=tuple(field.name for field in schema.fields if field.name in incoming),
            )

        document = match.snapshot.value
        if not isinstance(document, MemoryDocument):  # pragma: no cover - 匹配模型已保证。
            raise MemoryFieldMergeError("existing node match has no memory document")
        try:
            current = schema.validate_payload(document.fields)
        except (TypeError, ValueError) as exc:
            raise MemoryFieldMergeError("old memory fields failed current Schema validation") from exc

        if schema.operation_mode is MemoryOperationMode.ADD_ONLY:
            if current != incoming:
                raise MemoryFieldMergeError("add_only memory already exists with different business fields")
            return MemoryFieldMergeResult(fields=current, changed_fields=())

        merged = self._merge_upsert(schema, current=current, incoming=incoming)
        try:
            normalized = schema.validate_payload(merged)
        except (TypeError, ValueError) as exc:
            raise MemoryFieldMergeError("merged fields failed memory Schema validation") from exc
        changed_fields = tuple(
            field.name
            for field in schema.fields
            if current.get(field.name, _MISSING) != normalized.get(field.name, _MISSING)
        )
        return MemoryFieldMergeResult(
            fields=normalized,
            changed_fields=changed_fields,
        )

    @staticmethod
    def _merge_upsert(
        schema: MemoryTypeSchema,
        *,
        current: dict[str, Any],
        incoming: dict[str, Any],
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for field in schema.fields:
            has_current = field.name in current
            has_incoming = field.name in incoming
            current_value = current.get(field.name)
            incoming_value = incoming.get(field.name)

            if field.merge_strategy is MemoryMergeStrategy.IMMUTABLE:
                if has_current:
                    if has_incoming and current_value != incoming_value:
                        raise MemoryFieldMergeError(f"immutable memory field changed: {field.name}")
                    result[field.name] = current_value
                elif has_incoming:
                    result[field.name] = incoming_value
                continue

            if field.merge_strategy is MemoryMergeStrategy.PATCH:
                if has_incoming:
                    # 候选输出完整目标值；这里不接受或执行 LLM SEARCH/REPLACE。
                    result[field.name] = incoming_value
                elif has_current:
                    result[field.name] = current_value
                continue

            if field.merge_strategy is MemoryMergeStrategy.REPLACE:
                # REPLACE 表示最新完整字段状态；可选字段缺席即从结果中移除。
                if has_incoming:
                    result[field.name] = incoming_value
                continue

            raise MemoryFieldMergeError(f"unsupported memory merge strategy: {field.merge_strategy}")
        return result


__all__ = ["MemoryFieldMergeError", "MemoryFieldMerger"]
