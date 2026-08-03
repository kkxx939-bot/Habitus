"""把生命周期压缩、恢复和删除统一翻译成现有 Memory Editor 事务计划。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from infrastructure.editor.snapshot import SnapshotBatch
from memory.document import MemoryDocument
from memory.editor import (
    MemoryCandidate,
    MemoryCommitPlan,
    MemoryCommitResult,
    MemoryCommitTransaction,
    MemoryFinalIdentity,
    MemoryFinalIdentityMap,
    MemoryMutation,
    MemoryMutationAction,
    MemoryMutationPlan,
    MemoryMutationReadSet,
    MemoryNodeDisposition,
    MemoryNodeMatch,
    MemoryNodeMatchStatus,
    MemoryRelationPlan,
    MemoryRelationPlanner,
    MemoryRelationReadSet,
    MemoryRelationReadSetLoader,
)
from memory.model import MemoryKind
from memory.snapshot import MemorySnapshot, MemorySnapshotBatch, MemorySnapshotReader
from memory.uri import MemoryURI


class MemoryLifecycleCommitter:
    """生命周期只造纯计划，实际 L2 修改继续复用统一 CAS/Journal 事务。"""

    def __init__(
        self,
        transaction: MemoryCommitTransaction,
        snapshot_reader: MemorySnapshotReader,
    ) -> None:
        if not isinstance(transaction, MemoryCommitTransaction):
            raise TypeError("transaction must be MemoryCommitTransaction")
        if not isinstance(snapshot_reader, MemorySnapshotReader):
            raise TypeError("snapshot_reader must be MemorySnapshotReader")
        if transaction.snapshot_reader is not snapshot_reader:
            raise ValueError("lifecycle committer and transaction must share one snapshot reader")
        self.transaction = transaction
        self.snapshot_reader = snapshot_reader
        self.relation_loader = MemoryRelationReadSetLoader(snapshot_reader)
        self.relation_planner = MemoryRelationPlanner()

    def replace_fields(
        self,
        snapshot: MemorySnapshot,
        fields: Mapping[str, Any],
    ) -> tuple[MemoryDocument, MemoryCommitResult]:
        if not snapshot.exists or not isinstance(snapshot.value, MemoryDocument):
            raise ValueError("lifecycle field replacement requires an existing L2 document")
        parsed = MemoryURI.parse(snapshot.identity)
        source = snapshot.value
        candidate = MemoryCandidate(
            page_id=1,
            kind=source.kind,
            fields=fields,
            confirmed=False if source.kind is MemoryKind.INTENTION else None,
        )
        if candidate.address != source.address:
            raise ValueError("lifecycle field replacement cannot change memory identity")
        changed_fields = tuple(
            sorted(name for name in set(source.fields) | set(candidate.fields) if source.fields.get(name) != candidate.fields.get(name))
        )
        if not changed_fields:
            raise ValueError("lifecycle field replacement requires changed business fields")
        batch = _batch(snapshot)
        match = MemoryNodeMatch(candidate, parsed, MemoryNodeMatchStatus.EXISTING, snapshot)
        mutation_plan = MemoryMutationPlan(
            MemoryMutationReadSet(batch, batch),
            (
                MemoryMutation(
                    match,
                    MemoryMutationAction.UPDATE,
                    candidate.fields,
                    changed_fields,
                    confirms_intention=False,
                ),
            ),
        )
        identities = MemoryFinalIdentityMap(
            (MemoryFinalIdentity(1, MemoryNodeDisposition.UPDATE, parsed, parsed),)
        )
        relation_plan = _empty_relation_plan()
        result = self.transaction.commit(MemoryCommitPlan.build(mutation_plan, identities, relation_plan))
        current = self.snapshot_reader.read(parsed)
        if not current.exists or not isinstance(current.value, MemoryDocument):
            raise RuntimeError("lifecycle field replacement committed but L2 document is missing")
        if (
            current.revision != source.metadata.revision + 1
            or current.value.metadata.created_at != source.metadata.created_at
        ):
            raise RuntimeError("lifecycle field replacement read-back fingerprint is invalid")
        return current.value, result

    def delete(self, snapshot: MemorySnapshot) -> MemoryCommitResult:
        if not snapshot.exists or not isinstance(snapshot.value, MemoryDocument):
            raise ValueError("lifecycle deletion requires an existing L2 document")
        parsed = MemoryURI.parse(snapshot.identity)
        known = _batch(snapshot)
        mutation_plan = MemoryMutationPlan(
            MemoryMutationReadSet(known, _empty_batch()),
            (),
        )
        identities = MemoryFinalIdentityMap(
            (MemoryFinalIdentity(1, MemoryNodeDisposition.DELETE, parsed, None),)
        )
        relation_read_set = self.relation_loader.load(known, identities, ())
        relation_plan = self.relation_planner.plan(identities, (), relation_read_set)
        return self.transaction.commit(MemoryCommitPlan.build(mutation_plan, identities, relation_plan))


def _batch(snapshot) -> MemorySnapshotBatch:
    return SnapshotBatch((snapshot,), snapshot.size_bytes)


def _empty_batch() -> MemorySnapshotBatch:
    return SnapshotBatch((), 0)


def _empty_relation_plan() -> MemoryRelationPlan:
    return MemoryRelationPlan(
        MemoryRelationReadSet((), _empty_batch()),
        (),
        (),
        (),
        (),
    )


__all__ = ["MemoryLifecycleCommitter"]
