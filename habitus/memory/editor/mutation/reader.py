"""在纯节点规划前读取每个候选规范 URI 的精确快照。"""

from __future__ import annotations

from habitus.infrastructure.editor.snapshot import SnapshotBatch
from habitus.memory.editor.candidate import MemoryCandidateBatch
from habitus.memory.editor.mutation.model import MemoryMutationReadSet
from habitus.memory.snapshot import MemorySnapshotBatch, MemorySnapshotReader
from habitus.memory.uri import MemoryURI


class MemoryMutationReadConflictError(ValueError):
    """候选解析使用的旧记忆在节点规划前已经改变。"""


class MemoryMutationReadSetLoader:
    """补齐候选目标的存在或缺失快照，并验证重叠旧快照仍然有效。"""

    def __init__(self, reader: MemorySnapshotReader) -> None:
        if not isinstance(reader, MemorySnapshotReader):
            raise TypeError("reader must be a MemorySnapshotReader")
        self.reader = reader

    def load(
        self,
        batch: MemoryCandidateBatch,
        old_memories: MemorySnapshotBatch,
    ) -> MemoryMutationReadSet:
        """精确读取所有候选目标 URI；空候选返回空目标读集。"""

        if not isinstance(batch, MemoryCandidateBatch):
            raise TypeError("batch must be a MemoryCandidateBatch")
        if not isinstance(old_memories, SnapshotBatch):
            raise TypeError("old_memories must be a MemorySnapshotBatch")
        target_uris = tuple(
            sorted({str(MemoryURI.from_address(candidate.address)) for candidate in batch.iter_candidates()})
        )
        targets = self.reader.read_many(target_uris)
        try:
            return MemoryMutationReadSet(
                old_memories=old_memories,
                target_memories=targets,
            )
        except ValueError as exc:
            raise MemoryMutationReadConflictError(str(exc)) from exc


__all__ = [
    "MemoryMutationReadConflictError",
    "MemoryMutationReadSetLoader",
]
