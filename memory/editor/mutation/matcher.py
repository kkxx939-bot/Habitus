"""使用临时 page_id、规范 URI 和精确快照匹配记忆节点。"""

from __future__ import annotations

from infrastructure.editor.snapshot import SnapshotBatch
from memory.document import MemoryDocument
from memory.editor.candidate import MemoryCandidateBatch
from memory.editor.mutation.model import (
    MemoryMutationReadSet,
    MemoryNodeMatch,
    MemoryNodeMatchStatus,
)
from memory.editor.page_id import EXISTING_PAGE_ID_MAX, MemoryPageIdMap
from memory.uri import MemoryURI


class MemoryNodeMatchError(ValueError):
    """候选编号、地址或目标快照不能形成唯一节点判断。"""


class MemoryNodeMatcher:
    """只接受精确身份匹配，不在代码中猜测语义相似节点。"""

    def match(
        self,
        batch: MemoryCandidateBatch,
        read_set: MemoryMutationReadSet,
        page_ids: MemoryPageIdMap,
    ) -> tuple[MemoryNodeMatch, ...]:
        """把每条候选严格分类为已有节点或待新建节点。"""

        if not isinstance(batch, MemoryCandidateBatch):
            raise TypeError("batch must be a MemoryCandidateBatch")
        if not isinstance(read_set, MemoryMutationReadSet):
            raise TypeError("read_set must be a MemoryMutationReadSet")
        if not isinstance(page_ids, MemoryPageIdMap):
            raise TypeError("page_ids must be a MemoryPageIdMap")
        self._validate_old_context(read_set.old_memories, page_ids)

        candidates = batch.iter_candidates()
        expected_targets = tuple(sorted(str(MemoryURI.from_address(candidate.address)) for candidate in candidates))
        actual_targets = tuple(snapshot.identity for snapshot in read_set.target_memories.snapshots)
        if actual_targets != expected_targets:
            raise MemoryNodeMatchError("exact target snapshots must cover every candidate URI exactly once")

        matches = [self._match_one(candidate, read_set, page_ids) for candidate in candidates]
        return tuple(sorted(matches, key=lambda item: (str(item.uri), item.candidate.page_id)))

    @staticmethod
    def _validate_old_context(
        old_memories: SnapshotBatch[MemoryDocument],
        page_ids: MemoryPageIdMap,
    ) -> None:
        if page_ids.items() != page_ids.existing_items():
            raise MemoryNodeMatchError("node matching requires a page_id map containing only extracted old nodes")
        found_uris: set[str] = set()
        for snapshot in old_memories.snapshots:
            try:
                uri = MemoryURI.parse(snapshot.identity)
                uri.to_address()
            except (TypeError, ValueError) as exc:
                raise MemoryNodeMatchError("old-memory snapshot identity is not a strict L2 memory URI") from exc
            if not snapshot.exists:
                continue
            if not isinstance(snapshot.value, MemoryDocument):
                raise MemoryNodeMatchError("old-memory snapshot contains an invalid document")
            if str(MemoryURI.from_address(snapshot.value.address)) != snapshot.identity:
                raise MemoryNodeMatchError("old-memory snapshot identity does not match its document")
            found_uris.add(snapshot.identity)
        mapped_uris = {uri for _page_id, uri in page_ids.existing_items()}
        if mapped_uris != found_uris:
            raise MemoryNodeMatchError("page_id map does not match the complete extracted old-memory snapshots")

    @staticmethod
    def _match_one(
        candidate: object,
        read_set: MemoryMutationReadSet,
        page_ids: MemoryPageIdMap,
    ) -> MemoryNodeMatch:
        from memory.editor.candidate import MemoryCandidate

        if not isinstance(candidate, MemoryCandidate):  # pragma: no cover - 批次模型已保证。
            raise TypeError("candidate must be a MemoryCandidate")
        uri = MemoryURI.from_address(candidate.address)
        identity = str(uri)
        target = read_set.target_memories.get(identity)
        if target is None:  # pragma: no cover - 上层覆盖检查已保证。
            raise MemoryNodeMatchError("candidate target snapshot is missing from the read set")

        resolved_uri = page_ids.resolve(candidate.page_id)
        old_snapshot = read_set.old_memories.get(identity)
        if candidate.page_id <= EXISTING_PAGE_ID_MAX:
            if resolved_uri != identity:
                raise MemoryNodeMatchError("existing candidate must reuse the page_id bound to its exact URI")
            if old_snapshot is None or not old_snapshot.exists:
                raise MemoryNodeMatchError("existing candidate page_id has no complete extracted old memory")
            if not target.exists:
                raise MemoryNodeMatchError("existing candidate target disappeared before planning")
            return MemoryNodeMatch(
                candidate=candidate,
                uri=uri,
                status=MemoryNodeMatchStatus.EXISTING,
                snapshot=target,
            )

        if resolved_uri is not None:
            raise MemoryNodeMatchError("new candidate page_id was already bound before planning")
        if page_ids.page_id_for(identity) is not None:
            raise MemoryNodeMatchError("candidate for an extracted old URI must reuse its existing page_id")
        if old_snapshot is not None and old_snapshot.exists:
            raise MemoryNodeMatchError("new candidate URI conflicts with a complete extracted old memory")
        if target.exists:
            raise MemoryNodeMatchError("new candidate target already exists but was not read as an old node")
        return MemoryNodeMatch(
            candidate=candidate,
            uri=uri,
            status=MemoryNodeMatchStatus.NEW,
            snapshot=target,
        )


__all__ = ["MemoryNodeMatchError", "MemoryNodeMatcher"]
