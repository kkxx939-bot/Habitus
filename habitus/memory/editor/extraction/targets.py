"""候选写到的地址上已有旧卡片、这次却没读过：读出来交回模型修补一次，修补后仍不合格的只剔掉那一条。

旧卡片靠模型看名字清单自己去读，漏读一次就会在字段规划里永久失败、堵住后面的队列。这里不替模型按字段合并
（它写这条候选时没看过旧内容，``replace`` 类字段会把旧内容冲掉），只把旧卡片补进上下文、交回修补轮。
"""

from __future__ import annotations

from habitus.infrastructure.editor.snapshot import SnapshotBatch
from habitus.memory.editor.candidate import MemoryCandidate, MemoryCandidateBatch, MemoryCandidateViolation
from habitus.memory.editor.extraction.context import MemoryExtractionContext
from habitus.memory.editor.extraction.model import MemoryExtractionCapacityError
from habitus.memory.editor.page_id import EXISTING_PAGE_ID_MAX
from habitus.memory.uri import MemoryURI


def read_unread_targets(
    context: MemoryExtractionContext, candidates: MemoryCandidateBatch
) -> tuple[tuple[MemoryCandidate, str], ...]:
    """新候选（page_id 100 起）写到的地址上已有旧卡片、这次没读过：把旧卡片完整读进上下文（拿到 page_id）。

    返回这些候选和它们的地址。旧记忆预算装不下时不读，留给修补后的检查剔除。
    """

    read = {snapshot.identity for snapshot in context.snapshots.snapshots}
    wanted = {
        str(MemoryURI.from_address(candidate.address)): candidate
        for candidate in candidates.iter_candidates()
        if candidate.page_id > EXISTING_PAGE_ID_MAX and str(MemoryURI.from_address(candidate.address)) not in read
    }
    if not wanted:
        return ()
    existing = tuple(
        snapshot for snapshot in context.snapshot_reader.read_many(tuple(sorted(wanted))).snapshots if snapshot.exists
    )
    if not existing:
        return ()
    try:
        context.ingest(SnapshotBatch(snapshots=existing, total_bytes=sum(item.size_bytes for item in existing)))
    except MemoryExtractionCapacityError:
        return ()
    return tuple((wanted[snapshot.identity], snapshot.identity) for snapshot in existing)


def repair_notes(context: MemoryExtractionContext, unread: tuple[tuple[MemoryCandidate, str], ...]) -> tuple[str, ...]:
    """修补轮里每条一行：哪条候选、写到哪个地址、那里旧卡片的 page_id。"""

    page_ids = context.page_ids
    return tuple(
        f"page_id={candidate.page_id} 写到 {uri}，这个地址已有旧卡片（已放进 old_memory_snapshots，page_id="
        f"{page_ids.page_id_for(uri)}）：请基于旧卡片重写这一条并复用旧 page_id"
        for candidate, uri in unread
    )


def stale_new_targets(
    context: MemoryExtractionContext, candidates: MemoryCandidateBatch
) -> tuple[MemoryCandidateViolation, ...]:
    """修补之后仍把新 page_id 写到已有旧卡片的地址上（读过的或没读过的）：这些条目要剔掉。"""

    page_ids = context.page_ids
    fresh = [candidate for candidate in candidates.iter_candidates() if candidate.page_id > EXISTING_PAGE_ID_MAX]
    unread = {
        str(MemoryURI.from_address(candidate.address))
        for candidate in fresh
        if page_ids.page_id_for(str(MemoryURI.from_address(candidate.address))) is None
    }
    on_disk = {
        snapshot.identity
        for snapshot in (context.snapshot_reader.read_many(tuple(sorted(unread))).snapshots if unread else ())
        if snapshot.exists
    }
    violations: list[MemoryCandidateViolation] = []
    for candidate in fresh:
        uri = str(MemoryURI.from_address(candidate.address))
        if page_ids.page_id_for(uri) is not None:
            violations.append(
                MemoryCandidateViolation(candidate, "target already holds an old card; reuse its page_id")
            )
        elif uri in on_disk:
            violations.append(MemoryCandidateViolation(candidate, "target already holds an old card that was not read"))
    return tuple(violations)


__all__ = ["read_unread_targets", "repair_notes", "stale_new_targets"]
