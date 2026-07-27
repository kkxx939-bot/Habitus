"""Intention 在 Agent 召回和远程向量分区中的严格语义。"""

from __future__ import annotations

from enum import Enum

from memory.model import MemoryKind

_COMPLETED_STATUS = "completed"
_COMPLETED_INDEX_KIND = "intention_completed"
_ACTIVE_STATUSES = frozenset({"open", "waiting", "blocked"})
_ALL_STATUSES = _ACTIVE_STATUSES | {_COMPLETED_STATUS}


class MemoryIntentionRecallScope(str, Enum):
    """决定直接召回允许哪些 Intention；不改变树中保存的业务状态。"""

    ACTIVE = "active"
    COMPLETED = "completed"
    ALL = "all"


def memory_index_kind(kind: MemoryKind, *, intention_status: str | None = None) -> str:
    """生成现有 `kind` 标量索引使用的确定性检索分区。"""

    normalized = MemoryKind(kind)
    if normalized is not MemoryKind.INTENTION:
        if intention_status is not None:
            raise ValueError("only Intention memory accepts intention_status")
        return normalized.value
    if intention_status not in _ALL_STATUSES:
        raise ValueError("Intention memory status is outside its controlled values")
    return _COMPLETED_INDEX_KIND if intention_status == _COMPLETED_STATUS else normalized.value


def allowed_memory_index_kinds(
    kinds: tuple[MemoryKind, ...],
    intention_scope: MemoryIntentionRecallScope,
) -> tuple[str, ...]:
    """把公开类型条件转换成 Top-K 前使用的唯一向量分区集合。"""

    if not isinstance(kinds, tuple):
        raise TypeError("memory kinds must be a tuple")
    normalized_kinds = tuple(MemoryKind(kind) for kind in kinds) or tuple(MemoryKind)
    if len(normalized_kinds) != len(set(normalized_kinds)):
        raise ValueError("memory kinds must be unique")
    scope = MemoryIntentionRecallScope(intention_scope)
    values: list[str] = []
    for kind in normalized_kinds:
        if kind is not MemoryKind.INTENTION:
            values.append(kind.value)
            continue
        if scope in {MemoryIntentionRecallScope.ACTIVE, MemoryIntentionRecallScope.ALL}:
            values.append(MemoryKind.INTENTION.value)
        if scope in {MemoryIntentionRecallScope.COMPLETED, MemoryIntentionRecallScope.ALL}:
            values.append(_COMPLETED_INDEX_KIND)
    if not values:
        raise ValueError("memory recall filters exclude every index partition")
    return tuple(values)


def intention_matches_scope(status: object, scope: MemoryIntentionRecallScope) -> bool:
    """验证完整 L2 Intention 是否符合直接候选召回范围。"""

    if status not in _ALL_STATUSES:
        raise ValueError("Intention status is outside its controlled values")
    normalized = MemoryIntentionRecallScope(scope)
    completed = status == _COMPLETED_STATUS
    if normalized is MemoryIntentionRecallScope.ALL:
        return True
    if normalized is MemoryIntentionRecallScope.COMPLETED:
        return completed
    return not completed


__all__ = [
    "MemoryIntentionRecallScope",
    "allowed_memory_index_kinds",
    "intention_matches_scope",
    "memory_index_kind",
]
