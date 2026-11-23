"""把生命周期压缩、恢复和删除翻译成 Memory Editor 的写入计划，与记忆任务走同一个写入过程。"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from habitus.infrastructure.editor.snapshot import SnapshotBatch
from habitus.memory.document import MemoryDocument
from habitus.memory.editor import (
    MemoryCandidate,
    MemoryCommitPlan,
    MemoryCommitResult,
    MemoryCommitWriter,
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
from habitus.memory.snapshot import MemorySnapshot, MemorySnapshotBatch, MemorySnapshotReader
from habitus.memory.uri import MemoryURI


class MemoryLifecycleCommitter:
    """生命周期只造纯计划，实际 L2 修改复用同一个写入器：拿齐锁、核对版本、逐个原子写；
    退休一张卡片时先清邻居的关联再删它，中途失败留下的悬空关联由关联规划的修复兜住。"""

    def __init__(
        self,
        writer: MemoryCommitWriter,
        snapshot_reader: MemorySnapshotReader,
    ) -> None:
        if writer.snapshot_reader is not snapshot_reader:
            raise ValueError("lifecycle committer and writer must share one snapshot reader")
        self.writer = writer
        self.snapshot_reader = snapshot_reader
        self.relation_loader = MemoryRelationReadSetLoader(snapshot_reader)
        self.relation_planner = MemoryRelationPlanner()

    def replace_fields(
        self,
        snapshot: MemorySnapshot,
        fields: Mapping[str, Any],
    ) -> tuple[MemoryDocument, MemoryCommitResult]:
        if not snapshot.exists or snapshot.value is None:
            raise ValueError("lifecycle field replacement requires an existing L2 document")
        parsed = MemoryURI.parse(snapshot.identity)
        source = snapshot.value
        candidate = MemoryCandidate(page_id=1, kind=source.kind, fields=fields)
        if candidate.address != source.address:
            raise ValueError("lifecycle field replacement cannot change memory identity")
        changed_fields = tuple(
            sorted(
                name
                for name in set(source.fields) | set(candidate.fields)
                if source.fields.get(name) != candidate.fields.get(name)
            )
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
                ),
            ),
        )
        identities = MemoryFinalIdentityMap((MemoryFinalIdentity(1, MemoryNodeDisposition.UPDATE, parsed, parsed),))
        relation_plan = _empty_relation_plan()
        result = self.writer.commit(MemoryCommitPlan.build(mutation_plan, identities, relation_plan))
        current = self.snapshot_reader.read(parsed)
        if not current.exists or current.value is None:
            raise RuntimeError("lifecycle field replacement committed but L2 document is missing")
        if (
            current.revision != source.metadata.revision + 1
            or current.value.metadata.created_at != source.metadata.created_at
        ):
            raise RuntimeError("lifecycle field replacement read-back fingerprint is invalid")
        return current.value, result

    def delete(self, snapshot: MemorySnapshot) -> MemoryCommitResult:
        if not snapshot.exists or snapshot.value is None:
            raise ValueError("lifecycle deletion requires an existing L2 document")
        parsed = MemoryURI.parse(snapshot.identity)
        known = _batch(snapshot)
        mutation_plan = MemoryMutationPlan(
            MemoryMutationReadSet(known, _empty_batch()),
            (),
        )
        identities = MemoryFinalIdentityMap((MemoryFinalIdentity(1, MemoryNodeDisposition.DELETE, parsed, None),))
        relation_read_set = self.relation_loader.load(known, identities, ())
        relation_plan = self.relation_planner.plan(identities, (), relation_read_set)
        return self.writer.commit(MemoryCommitPlan.build(mutation_plan, identities, relation_plan))


def _batch(snapshot: MemorySnapshot) -> MemorySnapshotBatch:
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
