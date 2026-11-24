"""把规划好的节点内容、删除和双向关系逐个原子写进记忆树。

写法照 OpenViking：一批要碰的地址（含关联的另一端）排好序一次拿齐锁，锁里核对每个文件的版本与内容摘要，
再逐个文件原子写。没有多文件事务的预备日志、回滚和崩溃恢复：中途某个文件写失败就停下，已经写成的留着，
由调用方按失败处理、整条重做（裁定 111）。写的顺序固定为"新建和更新 → 只改关联的邻居 → 删除"，
合并的目标没写成就不会删来源。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from types import MappingProxyType
from typing import Any

from habitus.infrastructure.editor.snapshot import SnapshotBatch, VersionedSnapshot
from habitus.infrastructure.store.contracts import PathLock
from habitus.infrastructure.tree.metadata import DocumentMetadata
from habitus.memory.document import MemoryDocument, MemoryStoredLink
from habitus.memory.editor.identity import (
    MemoryFinalIdentityMap,
    MemoryNodeDisposition,
)
from habitus.memory.editor.link_plan import MemoryRelationPlan
from habitus.memory.editor.mutation.model import (
    MemoryMutationAction,
    MemoryMutationPlan,
)
from habitus.memory.model import MemoryKind
from habitus.memory.snapshot import MemorySnapshotBatch, MemorySnapshotReader
from habitus.memory.tree import MemoryDocumentLockKeyspace, MemoryTree
from habitus.memory.uri import MemoryURI


class MemoryCommitError(RuntimeError):
    """一批记忆没能全部写成；已经写成的文件留在盘上，调用方按失败处理、整条重做。"""


class MemoryCommitConflictError(MemoryCommitError):
    """规划时读到的某个文件在写之前被别人改了；一个文件都还没写，调用方整条重做。"""


class MemoryCommitStatus(str, Enum):
    """一次写入的结果。"""

    UPDATED = "updated"
    UNCHANGED = "unchanged"


@dataclass(frozen=True)
class MemoryCommitConfig:
    """写入时每把地址锁的租约。"""

    lock_ttl_seconds: int = 30

    def __post_init__(self) -> None:
        if not 1 <= self.lock_ttl_seconds <= 3_600:
            raise ValueError("lock_ttl_seconds must be between 1 and 3600")


@dataclass(frozen=True)
class _LogicalWrite:
    """写之前确定的最终业务内容，不含系统生成的时间与版本。"""

    before: VersionedSnapshot[MemoryDocument]
    uri: MemoryURI
    kind: MemoryKind
    fields: Mapping[str, Any]
    links: tuple[MemoryStoredLink, ...]
    backlinks: tuple[MemoryStoredLink, ...]
    #: 这个地址有没有字段变化（新建、更新）；没有的是只改关联的邻居
    content_changed: bool

    def __post_init__(self) -> None:
        if self.before.identity != str(self.uri):
            raise ValueError("logical write snapshot identity does not match its URI")
        kind = MemoryKind(self.kind)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "fields", MappingProxyType(dict(self.fields)))
        if self.before.exists:
            assert isinstance(self.before.value, MemoryDocument)
            if self.before.value.kind is not kind or MemoryURI.from_address(self.before.value.address) != self.uri:
                raise ValueError("logical write cannot change an existing node identity or kind")


@dataclass(frozen=True)
class MemoryCommitPlan:
    """内容、最终身份与关系计划在写之前形成的统一纯计划。"""

    mutation_plan: MemoryMutationPlan
    identities: MemoryFinalIdentityMap
    relation_plan: MemoryRelationPlan
    read_set: MemorySnapshotBatch

    def __post_init__(self) -> None:
        self._validate_mutation_identities()
        self._validate_snapshot_coverage()

    @classmethod
    def build(
        cls,
        mutation_plan: MemoryMutationPlan,
        identities: MemoryFinalIdentityMap,
        relation_plan: MemoryRelationPlan,
    ) -> MemoryCommitPlan:
        """合并所有语义读取、精确目标和一跳关系快照。"""

        batches = (
            mutation_plan.read_set.old_memories,
            mutation_plan.read_set.target_memories,
            relation_plan.read_set.snapshots,
        )
        snapshots: dict[str, VersionedSnapshot[MemoryDocument]] = {}
        for batch in batches:
            for snapshot in batch.snapshots:
                previous = snapshots.get(snapshot.identity)
                if previous is not None and previous != snapshot:
                    raise MemoryCommitConflictError(f"commit plan contains conflicting snapshots: {snapshot.identity}")
                snapshots[snapshot.identity] = snapshot
        ordered = tuple(snapshots[key] for key in sorted(snapshots))
        return cls(
            mutation_plan=mutation_plan,
            identities=identities,
            relation_plan=relation_plan,
            read_set=SnapshotBatch(
                snapshots=ordered,
                total_bytes=sum(snapshot.size_bytes for snapshot in ordered),
            ),
        )

    @property
    def retired_uris(self) -> tuple[MemoryURI, ...]:
        return self.identities.retired_uris

    @property
    def changed_uris(self) -> tuple[MemoryURI, ...]:
        """这一批写到或删掉的全部地址（刷概览用）。"""

        mutation_uris = {str(mutation.uri) for mutation in self.mutation_plan.changed_mutations}
        relation_uris = {str(update.uri) for update in self.relation_plan.updates}
        retired = {str(uri) for uri in self.retired_uris}
        return tuple(MemoryURI.parse(identity) for identity in sorted(mutation_uris | relation_uris | retired))

    def logical_writes(self) -> tuple[_LogicalWrite, ...]:
        """从统一计划生成每个要写地址的最终业务内容，按地址排序。"""

        mutations = {str(mutation.uri): mutation for mutation in self.mutation_plan.mutations}
        content_targets = {str(mutation.uri) for mutation in self.mutation_plan.changed_mutations}
        relation_updates = {str(update.uri): update for update in self.relation_plan.updates}
        targets = content_targets | set(relation_updates)
        retired = {str(uri) for uri in self.retired_uris}
        if targets & retired:
            raise MemoryCommitError("one URI cannot be written and retired in the same commit")

        writes: list[_LogicalWrite] = []
        for identity in sorted(targets):
            before = self.read_set.get(identity)
            if before is None:
                raise MemoryCommitError("write target is missing from the unified read set")
            mutation = mutations.get(identity)
            relation_update = relation_updates.get(identity)
            if mutation is not None:
                kind = mutation.match.candidate.kind
                fields = mutation.fields
            else:
                if not before.exists or before.value is None:
                    raise MemoryCommitError("relation-only write requires an existing memory document")
                kind = before.value.kind
                fields = before.value.fields

            if relation_update is not None:
                links = relation_update.links
                backlinks = relation_update.backlinks
            elif before.exists:
                assert isinstance(before.value, MemoryDocument)
                links = before.value.links
                backlinks = before.value.backlinks
            else:
                links = ()
                backlinks = ()

            if not before.exists and (mutation is None or mutation.action is not MemoryMutationAction.CREATE):
                raise MemoryCommitError("missing write target requires a CREATE mutation")
            writes.append(
                _LogicalWrite(
                    before=before,
                    uri=MemoryURI.parse(identity),
                    kind=kind,
                    fields=fields,
                    links=links,
                    backlinks=backlinks,
                    content_changed=identity in content_targets,
                )
            )
        return tuple(writes)

    def _validate_mutation_identities(self) -> None:
        expected_disposition = {
            MemoryMutationAction.CREATE: MemoryNodeDisposition.CREATE,
            MemoryMutationAction.UPDATE: MemoryNodeDisposition.UPDATE,
            MemoryMutationAction.NOOP: MemoryNodeDisposition.NOOP,
        }
        retired = {str(uri) for uri in self.identities.retired_uris}
        mutation_targets = {str(mutation.uri) for mutation in self.mutation_plan.mutations}
        old_memories = self.mutation_plan.read_set.old_memories
        final_uris = {str(entry.final_uri) for entry in self.identities.entries if entry.final_uri is not None}
        if retired & final_uris:
            raise ValueError("merge/delete sources must be flattened before final identity publication")
        for entry in self.identities.entries:
            if entry.source_uri is not None:
                source = old_memories.get(str(entry.source_uri))
                if source is None or not source.exists:
                    raise ValueError("final identity source requires a complete extracted old snapshot")
            if (
                entry.disposition is MemoryNodeDisposition.MERGE
                and entry.final_uri is not None
                and str(entry.final_uri) not in mutation_targets
            ):
                raise ValueError("merge target must be covered by deterministic field planning")
        for mutation in self.mutation_plan.mutations:
            entry = self.identities.entry(mutation.match.candidate.page_id)
            if entry.disposition is not expected_disposition[mutation.action]:
                raise ValueError("mutation action does not match its final identity disposition")
            if entry.final_uri != mutation.uri:
                raise ValueError("mutation URI does not match its final identity")
            if str(mutation.uri) in retired:
                raise ValueError("retired node cannot also receive a content mutation")

    def _validate_snapshot_coverage(self) -> None:
        for uri in self.changed_uris:
            snapshot = self.read_set.get(str(uri))
            if snapshot is None:
                raise ValueError(f"commit read set does not contain changed URI: {uri}")
        creates = {
            str(mutation.uri)
            for mutation in self.mutation_plan.mutations
            if mutation.action is MemoryMutationAction.CREATE
        }
        for update in self.relation_plan.updates:
            current = self.read_set.get(str(update.uri))
            if current != update.before:
                raise ValueError("relation update snapshot does not match unified commit read set")
            if not update.before.exists and str(update.uri) not in creates:
                raise ValueError("relation update cannot create a node without a content CREATE mutation")
        for uri in self.retired_uris:
            snapshot = self.read_set.get(str(uri))
            if snapshot is None or not snapshot.exists:
                raise ValueError("retired node requires a complete old snapshot")


@dataclass(frozen=True)
class MemoryCommitResult:
    """一次写入实际写到、删掉的地址与关系变化。"""

    status: MemoryCommitStatus
    created_uris: tuple[MemoryURI, ...]
    updated_uris: tuple[MemoryURI, ...]
    deleted_uris: tuple[MemoryURI, ...]
    unchanged_uris: tuple[MemoryURI, ...]
    added_relations: tuple[MemoryStoredLink, ...]
    removed_relations: tuple[MemoryStoredLink, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "status", MemoryCommitStatus(self.status))
        for name in ("created_uris", "updated_uris", "deleted_uris", "unchanged_uris"):
            identities = tuple(str(uri) for uri in getattr(self, name))
            if identities != tuple(sorted(set(identities))):
                raise ValueError(f"{name} must be unique and sorted")

    @property
    def written_uris(self) -> tuple[MemoryURI, ...]:
        """新建和改写过的地址（含只改关联的邻居）。"""

        return tuple(sorted((*self.created_uris, *self.updated_uris), key=str))


class MemoryCommitWriter:
    """拿齐锁、核对版本、按固定顺序逐个原子写。"""

    def __init__(
        self,
        tree: MemoryTree,
        snapshot_reader: MemorySnapshotReader,
        path_lock: PathLock,
        *,
        config: MemoryCommitConfig | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if snapshot_reader.tree.root != tree.root:
            raise ValueError("snapshot_reader and tree must use the same memory root")
        self.tree = tree
        self.snapshot_reader = snapshot_reader
        self.path_lock = path_lock
        self.config = config or MemoryCommitConfig()
        self.clock = clock or (lambda: datetime.now(UTC))
        self.lock_keys = MemoryDocumentLockKeyspace(tree.root)

    def commit(self, plan: MemoryCommitPlan) -> MemoryCommitResult:
        """锁里核对读集没变，再按"新建和更新 → 只改关联的邻居 → 删除"逐个原子写。"""

        result = self._result(plan)
        if not plan.changed_uris:
            return result
        timestamp = self._timestamp()
        logical = plan.logical_writes()
        content = tuple(self._document(item, timestamp) for item in logical if item.content_changed)
        neighbours = tuple(self._document(item, timestamp) for item in logical if not item.content_changed)
        with ExitStack() as stack:
            guards = tuple(
                stack.enter_context(
                    self.path_lock.acquire(
                        self.lock_keys.key(snapshot.identity),
                        ttl_seconds=self.config.lock_ttl_seconds,
                    )
                )
                for snapshot in plan.read_set.snapshots
            )
            with self.path_lock.fenced(guards):
                self._assert_current(plan.read_set)
                try:
                    for document in (*content, *neighbours):
                        self.tree.write(document)
                    for uri in plan.retired_uris:
                        self.tree.delete(uri.to_address())
                except Exception as exc:
                    raise MemoryCommitError("memory write stopped part-way; written files are kept") from exc
        return result

    def _document(self, logical: _LogicalWrite, timestamp: datetime) -> MemoryDocument:
        if logical.before.exists:
            assert isinstance(logical.before.value, MemoryDocument)
            metadata = logical.before.value.metadata.next_revision(timestamp)
        else:
            metadata = DocumentMetadata.initial(timestamp)
        return self.tree.document_codec.build(
            logical.kind,
            logical.fields,
            metadata=metadata,
            links=logical.links,
            backlinks=logical.backlinks,
        )

    def _assert_current(self, expected: MemorySnapshotBatch) -> None:
        current = self.snapshot_reader.read_many(snapshot.identity for snapshot in expected.snapshots)
        for expected_snapshot in expected.snapshots:
            snapshot = current.get(expected_snapshot.identity)
            if snapshot is None or not (
                snapshot.state is expected_snapshot.state
                and snapshot.revision == expected_snapshot.revision
                and snapshot.source_digest == expected_snapshot.source_digest
            ):
                raise MemoryCommitConflictError(f"memory changed before it was written: {expected_snapshot.identity}")

    @staticmethod
    def _result(plan: MemoryCommitPlan) -> MemoryCommitResult:
        created = {
            str(mutation.uri): mutation.uri
            for mutation in plan.mutation_plan.mutations
            if mutation.action is MemoryMutationAction.CREATE
        }
        changed = {str(uri) for uri in plan.changed_uris}
        deleted = {str(uri) for uri in plan.retired_uris}
        updated = changed - set(created) - deleted
        unchanged = {
            str(mutation.uri): mutation.uri
            for mutation in plan.mutation_plan.mutations
            if mutation.action is MemoryMutationAction.NOOP and str(mutation.uri) not in updated
        }
        return MemoryCommitResult(
            status=(MemoryCommitStatus.UPDATED if changed else MemoryCommitStatus.UNCHANGED),
            created_uris=tuple(created[key] for key in sorted(created)),
            updated_uris=tuple(MemoryURI.parse(key) for key in sorted(updated)),
            deleted_uris=tuple(MemoryURI.parse(key) for key in sorted(deleted)),
            unchanged_uris=tuple(unchanged[key] for key in sorted(unchanged)),
            added_relations=plan.relation_plan.added,
            removed_relations=plan.relation_plan.removed,
        )

    def _timestamp(self) -> datetime:
        value = self.clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("memory commit clock must return a timezone-aware datetime")
        return value.astimezone(UTC)


__all__ = [
    "MemoryCommitConfig",
    "MemoryCommitConflictError",
    "MemoryCommitError",
    "MemoryCommitPlan",
    "MemoryCommitResult",
    "MemoryCommitStatus",
    "MemoryCommitWriter",
]
