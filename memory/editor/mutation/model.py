"""记忆节点匹配与字段变更的纯规划模型。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

from infrastructure.editor.snapshot import SnapshotBatch, VersionedSnapshot
from memory.document import MemoryDocument
from memory.editor.candidate import MemoryCandidate
from memory.model import MemoryKind
from memory.snapshot import MemorySnapshotBatch
from memory.uri import MemoryURI


class MemoryNodeMatchStatus(str, Enum):
    """候选目标在精确快照中是否已经存在。"""

    NEW = "new"
    EXISTING = "existing"


class MemoryMutationAction(str, Enum):
    """当前纯规划阶段支持的节点内容动作。"""

    CREATE = "create"
    UPDATE = "update"
    NOOP = "noop"


@dataclass(frozen=True)
class MemoryMutationReadSet:
    """候选解析旧上下文与候选目标精确快照的不可变读集。"""

    old_memories: MemorySnapshotBatch
    target_memories: MemorySnapshotBatch

    def __post_init__(self) -> None:
        if not isinstance(self.old_memories, SnapshotBatch):
            raise TypeError("old_memories must be a MemorySnapshotBatch")
        if not isinstance(self.target_memories, SnapshotBatch):
            raise TypeError("target_memories must be a MemorySnapshotBatch")
        for target in self.target_memories.snapshots:
            previous = self.old_memories.get(target.identity)
            if previous is None:
                continue
            if (
                target.state is not previous.state
                or target.revision != previous.revision
                or target.source_digest != previous.source_digest
            ):
                raise ValueError(f"old memory changed before mutation planning: {target.identity}")


@dataclass(frozen=True)
class MemoryNodeMatch:
    """一条候选与规范 L2 URI 及精确目标快照的绑定。"""

    candidate: MemoryCandidate
    uri: MemoryURI
    status: MemoryNodeMatchStatus
    snapshot: VersionedSnapshot[MemoryDocument]

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, MemoryCandidate):
            raise TypeError("candidate must be a MemoryCandidate")
        if not isinstance(self.uri, MemoryURI):
            raise TypeError("node match URI must be a MemoryURI")
        if self.uri.to_address() != self.candidate.address:
            raise ValueError("node match URI must equal the candidate-derived address")
        status = MemoryNodeMatchStatus(self.status)
        object.__setattr__(self, "status", status)
        if not isinstance(self.snapshot, VersionedSnapshot):
            raise TypeError("node match snapshot must be a VersionedSnapshot")
        if self.snapshot.identity != str(self.uri):
            raise ValueError("node match snapshot identity does not match its URI")
        if status is MemoryNodeMatchStatus.NEW:
            if self.snapshot.exists:
                raise ValueError("new node match requires an explicit missing snapshot")
            return
        if not self.snapshot.exists or not isinstance(self.snapshot.value, MemoryDocument):
            raise ValueError("existing node match requires a complete memory document")
        document = self.snapshot.value
        if document.kind is not self.candidate.kind or document.address != self.candidate.address:
            raise ValueError("existing node document does not match the candidate identity")


@dataclass(frozen=True)
class MemoryFieldMergeResult:
    """一个节点经过 YAML 字段策略计算后的完整最终业务字段。"""

    fields: Mapping[str, Any]
    changed_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.fields, Mapping) or any(not isinstance(name, str) for name in self.fields):
            raise TypeError("merged memory fields must be a mapping with string keys")
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))
        if not isinstance(self.changed_fields, tuple) or any(
            not isinstance(name, str) or not name for name in self.changed_fields
        ):
            raise TypeError("changed_fields must be a tuple of non-empty strings")
        if len(self.changed_fields) != len(set(self.changed_fields)):
            raise ValueError("changed_fields cannot contain duplicates")


@dataclass(frozen=True)
class MemoryMutation:
    """一个不含时间、revision 或落盘副作用的节点内容变更。"""

    match: MemoryNodeMatch
    action: MemoryMutationAction
    fields: Mapping[str, Any]
    changed_fields: tuple[str, ...]
    confirms_intention: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.match, MemoryNodeMatch):
            raise TypeError("match must be a MemoryNodeMatch")
        action = MemoryMutationAction(self.action)
        object.__setattr__(self, "action", action)
        if not isinstance(self.fields, Mapping) or any(not isinstance(name, str) for name in self.fields):
            raise TypeError("mutation fields must be a mapping with string keys")
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))
        if not isinstance(self.changed_fields, tuple) or any(
            not isinstance(name, str) or not name for name in self.changed_fields
        ):
            raise TypeError("mutation changed_fields must contain non-empty strings")
        if len(self.changed_fields) != len(set(self.changed_fields)):
            raise ValueError("mutation changed_fields cannot contain duplicates")
        if not isinstance(self.confirms_intention, bool):
            raise TypeError("confirms_intention must be boolean")
        if self.confirms_intention and self.match.candidate.kind is not MemoryKind.INTENTION:
            raise ValueError("only an Intention mutation can refresh confirmation time")
        if action is MemoryMutationAction.CREATE:
            if self.match.status is not MemoryNodeMatchStatus.NEW:
                raise ValueError("create mutation requires a new-node match")
            if not self.changed_fields:
                raise ValueError("create mutation must contain changed fields")
            return
        if self.match.status is not MemoryNodeMatchStatus.EXISTING:
            raise ValueError("update and noop mutations require an existing-node match")
        if action is MemoryMutationAction.UPDATE and not self.changed_fields:
            raise ValueError("update mutation must contain changed fields")
        if action is MemoryMutationAction.NOOP and self.changed_fields:
            raise ValueError("noop mutation cannot contain changed fields")

    @property
    def uri(self) -> MemoryURI:
        return self.match.uri


@dataclass(frozen=True)
class MemoryMutationPlan:
    """基于同一精确读集生成且可以交给后续提交器的纯变更计划。"""

    read_set: MemoryMutationReadSet
    mutations: tuple[MemoryMutation, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.read_set, MemoryMutationReadSet):
            raise TypeError("read_set must be a MemoryMutationReadSet")
        if not isinstance(self.mutations, tuple) or any(
            not isinstance(mutation, MemoryMutation) for mutation in self.mutations
        ):
            raise TypeError("mutations must be a tuple of MemoryMutation values")
        identities = tuple((str(item.uri), item.match.candidate.page_id) for item in self.mutations)
        if identities != tuple(sorted(identities)):
            raise ValueError("memory mutations must be sorted by URI and page_id")
        uris = tuple(identity[0] for identity in identities)
        page_ids = tuple(identity[1] for identity in identities)
        if len(uris) != len(set(uris)):
            raise ValueError("memory mutation plan cannot target one URI more than once")
        if len(page_ids) != len(set(page_ids)):
            raise ValueError("memory mutation plan cannot repeat a candidate page_id")
        target_identities = tuple(snapshot.identity for snapshot in self.read_set.target_memories.snapshots)
        if tuple(sorted(uris)) != target_identities:
            raise ValueError("memory mutation plan does not cover its exact target snapshots")

    @property
    def changed_mutations(self) -> tuple[MemoryMutation, ...]:
        """排除不需要推进 revision 或刷新确认时间的幂等节点。"""

        return tuple(
            mutation
            for mutation in self.mutations
            if mutation.action is not MemoryMutationAction.NOOP or mutation.confirms_intention
        )


__all__ = [
    "MemoryFieldMergeResult",
    "MemoryMutation",
    "MemoryMutationAction",
    "MemoryMutationPlan",
    "MemoryMutationReadSet",
    "MemoryNodeMatch",
    "MemoryNodeMatchStatus",
]
