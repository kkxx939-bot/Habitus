"""在关系规划前补齐最终处置所需的完整一跳快照。"""

from __future__ import annotations

from infrastructure.editor.snapshot import SnapshotBatch, VersionedSnapshot
from memory.document import MemoryDocument
from memory.editor.identity import (
    MemoryFinalIdentityMap,
    MemoryNodeDisposition,
)
from memory.editor.link import MemoryResolvedRelation
from memory.editor.link_plan import MemoryRelationReadSet
from memory.snapshot import MemorySnapshotBatch, MemorySnapshotReader


class MemoryRelationReadConflictError(ValueError):
    """关系闭包读取发现上游完整快照已经变化。"""


class MemoryRelationReadSetLoader:
    """读取操作端点、结构变更节点及其全部一跳邻居。"""

    def __init__(self, reader: MemorySnapshotReader) -> None:
        if not isinstance(reader, MemorySnapshotReader):
            raise TypeError("reader must be a MemorySnapshotReader")
        self.reader = reader

    def load(
        self,
        known_snapshots: MemorySnapshotBatch,
        identities: MemoryFinalIdentityMap,
        operations: tuple[MemoryResolvedRelation, ...],
    ) -> MemoryRelationReadSet:
        """重读完整闭包并校验与抽取、字段规划快照没有发生漂移。"""

        if not isinstance(known_snapshots, SnapshotBatch):
            raise TypeError("known_snapshots must be a MemorySnapshotBatch")
        if not isinstance(identities, MemoryFinalIdentityMap):
            raise TypeError("identities must be a MemoryFinalIdentityMap")
        if not isinstance(operations, tuple) or any(
            not isinstance(operation, MemoryResolvedRelation) for operation in operations
        ):
            raise TypeError("operations must contain MemoryResolvedRelation values")

        structural_sources = {
            str(entry.source_uri)
            for entry in identities.entries
            if entry.disposition in {MemoryNodeDisposition.MERGE, MemoryNodeDisposition.DELETE}
            and entry.source_uri is not None
        }
        required = set(structural_sources)
        required.update(
            str(entry.final_uri)
            for entry in identities.entries
            if entry.disposition is MemoryNodeDisposition.MERGE and entry.final_uri is not None
        )
        required.update(str(uri) for operation in operations for uri in (operation.from_uri, operation.to_uri))
        if not required:
            return MemoryRelationReadSet.build(
                SnapshotBatch(snapshots=(), total_bytes=0),
                identities,
                operations,
            )

        endpoints = self.reader.read_many(required)
        self._require_unchanged(known_snapshots, endpoints)
        for source in structural_sources:
            snapshot = endpoints.get(source)
            if snapshot is None or not snapshot.exists:
                raise MemoryRelationReadConflictError("structural relation source disappeared before planning")
            if not isinstance(snapshot.value, MemoryDocument):
                raise MemoryRelationReadConflictError("structural relation source is not a complete memory document")
            required.update(str(link.to_uri) for link in snapshot.value.links)
            required.update(str(backlink.from_uri) for backlink in snapshot.value.backlinks)

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
