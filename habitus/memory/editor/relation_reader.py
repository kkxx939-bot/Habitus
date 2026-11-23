"""在关系规划前补齐最终处置所需的完整一跳快照。"""

from __future__ import annotations

from habitus.infrastructure.editor.snapshot import SnapshotBatch, VersionedSnapshot
from habitus.memory.document import MemoryDocument
from habitus.memory.editor.identity import MemoryFinalIdentityMap
from habitus.memory.editor.link import MemoryResolvedRelation
from habitus.memory.editor.link_plan import MemoryRelationReadSet, one_hop, relation_endpoints
from habitus.memory.snapshot import MemorySnapshotBatch, MemorySnapshotReader


class MemoryRelationReadConflictError(ValueError):
    """关系闭包读取发现上游完整快照已经变化。"""


class MemoryRelationReadSetLoader:
    """读取操作端点、要写的节点、结构变更节点及其全部一跳邻居。"""

    def __init__(self, reader: MemorySnapshotReader) -> None:
        self.reader = reader

    def load(
        self,
        known_snapshots: MemorySnapshotBatch,
        identities: MemoryFinalIdentityMap,
        operations: tuple[MemoryResolvedRelation, ...],
    ) -> MemoryRelationReadSet:
        """重读要写节点及其一跳邻居，并校验与抽取、字段规划快照没有发生漂移。"""

        required = relation_endpoints(identities, operations)
        if not required:
            return MemoryRelationReadSet.build(
                SnapshotBatch(snapshots=(), total_bytes=0),
                identities,
                operations,
            )

        endpoints = self.reader.read_many(required)
        self._require_unchanged(known_snapshots, endpoints)
        for snapshot in endpoints.snapshots:
            if snapshot.exists:
                assert isinstance(snapshot.value, MemoryDocument)
                required |= one_hop(snapshot.value)

        closure = self.reader.read_many(required)
        self._require_unchanged(known_snapshots, closure)
        self._require_unchanged(endpoints, closure)
        return MemoryRelationReadSet.build(closure, identities, operations)

    @staticmethod
    def _require_unchanged(
        expected: MemorySnapshotBatch,
        current: MemorySnapshotBatch,
    ) -> None:
        for snapshot in current.snapshots:
            previous = expected.get(snapshot.identity)
            if previous is None:
                continue
            if not MemoryRelationReadSetLoader._same_snapshot(previous, snapshot):
                raise MemoryRelationReadConflictError(f"memory changed before relation planning: {snapshot.identity}")

    @staticmethod
    def _same_snapshot(
        left: VersionedSnapshot[MemoryDocument],
        right: VersionedSnapshot[MemoryDocument],
    ) -> bool:
        return (
            left.state is right.state and left.revision == right.revision and left.source_digest == right.source_digest
        )


__all__ = [
    "MemoryRelationReadConflictError",
    "MemoryRelationReadSetLoader",
]
