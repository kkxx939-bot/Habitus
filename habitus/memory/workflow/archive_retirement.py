"""Archive 退休清单的四相位幂等续跑：下线索引、释放 History、删除来源链、完成清单。

每个相位只在本相位的全部落盘动作完成后推进清单；中途失败或安全门未满足时停在当前相位，
下一次维护从同一相位重新进入。相位顺序、推进时机与报错文字都不得改变。
"""

from __future__ import annotations

from datetime import datetime

from habitus.memory.conversation import (
    ConversationMessageJournal,
    ConversationRangeSummaryStore,
    ConversationSummaryCompactor,
    ConversationSummaryRetirementManifest,
    ConversationSummaryRetirementPhase,
    ConversationSummaryRetirementStore,
    ConversationSummaryStore,
)
from habitus.memory.conversation.compaction import ConversationSummaryUseReader
from habitus.memory.conversation.indexing import PersistentConversationSummaryVectorIndex
from habitus.memory.workflow.committed_cleanup import CommittedWorkflowCleanup
from habitus.memory.workflow.lifecycle_errors import ConversationLifecycleError
from habitus.pre.conversation import ConversationRangeSummaryStage

ArchiveRetirementOutcome = tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]
"""依次为：物理清除的 History、释放的 History、删除的 Segment/Range/Archive Summary。"""

_NOTHING: ArchiveRetirementOutcome = ((), (), (), (), ())


class ConversationArchiveRetirementRunner:
    """严格按下线、History、子摘要、Archive 的阶段幂等续跑一个已准备的退休清单。"""

    def __init__(
        self,
        compactor: ConversationSummaryCompactor,
        journal: ConversationMessageJournal,
        segment_store: ConversationSummaryStore,
        range_store: ConversationRangeSummaryStore,
        summary_vector_index: PersistentConversationSummaryVectorIndex,
        retirement_store: ConversationSummaryRetirementStore,
        workflow: CommittedWorkflowCleanup,
        *,
        cleanup_batch_size: int,
    ) -> None:
        self.compactor = compactor
        self.journal = journal
        self.segment_store = segment_store
        self.range_store = range_store
        self.summary_vector_index = summary_vector_index
        self.retirement_store = retirement_store
        self.workflow = workflow
        self.cleanup_batch_size = cleanup_batch_size

    async def resume(
        self,
        manifest: ConversationSummaryRetirementManifest,
        now: datetime,
    ) -> ArchiveRetirementOutcome:
        """从清单当前相位开始依次续跑；某相位未完成即返回已做的物理清理结果。"""

        current = manifest
        use_store = self.compactor.use_store
        if use_store is None:
            raise ConversationLifecycleError("retiring Archive lacks a Summary use store")
        if current.phase is ConversationSummaryRetirementPhase.RETIRING:
            removed = await self._remove_from_index(current, use_store, now)
            if removed is None:
                return _NOTHING
            current = removed
        purged: tuple[str, ...] = ()
        released: tuple[str, ...] = ()
        if current.phase is ConversationSummaryRetirementPhase.INDEX_REMOVED:
            advanced, purged, released = self._release_history(current, now)
            if advanced is None:
                return purged, released, (), (), ()
            current = advanced
        segments: tuple[str, ...] = ()
        ranges: tuple[str, ...] = ()
        archives: tuple[str, ...] = ()
        if current.phase is ConversationSummaryRetirementPhase.HISTORY_RELEASED:
            current, (segments, ranges, archives) = self._delete_sources(current, use_store, now)
        if current.phase is ConversationSummaryRetirementPhase.SOURCES_DELETED:
            self.retirement_store.complete(current)
        return purged, released, segments, ranges, archives

    async def _remove_from_index(
        self,
        current: ConversationSummaryRetirementManifest,
        use_store: ConversationSummaryUseReader,
        now: datetime,
    ) -> ConversationSummaryRetirementManifest | None:
        """RETIRING：认领使用状态并从 Summary 索引精确删除；使用状态已变化则放弃本清单。"""

        states = use_store.read_many((current.archive.reference,))
        use_state = states[0] if states else None
        claimed = (
            use_state is not None
            and use_state.version == current.expected_use_version + 1
            and use_state.retiring_at is not None
        )
        if not claimed:
            if use_state is None or use_state.version != current.expected_use_version:
                self.retirement_store.complete(current)
                return None
            use_store.claim_retirement(
                current.archive.reference,
                expected_version=current.expected_use_version,
                claimed_at=now,
            )
        await self.summary_vector_index.synchronize(
            current.address,
            removed_references=(current.archive.reference,),
        )
        return self.retirement_store.advance(
            current,
            ConversationSummaryRetirementPhase.INDEX_REMOVED,
            updated_at=now,
        )

    def _release_history(
        self,
        current: ConversationSummaryRetirementManifest,
        now: datetime,
    ) -> tuple[ConversationSummaryRetirementManifest | None, tuple[str, ...], tuple[str, ...]]:
        """INDEX_REMOVED：按批释放并清除覆盖范围内的 History；全部释放后才推进相位。"""

        address = current.address
        purged: tuple[str, ...] = ()
        released: tuple[str, ...] = ()
        journal_state = self.journal.read_state(address)
        if journal_state.released_through < current.end_sequence:
            segment_sources = {item.reference.summary_id: item.digest for item in current.segments}
            active = tuple(
                segment
                for segment in self.journal.list_history(address)
                if current.start_sequence <= segment.start_sequence and segment.end_sequence <= current.end_sequence
            )[: self.cleanup_batch_size]
            for segment in active:
                expected_digest = segment_sources.get(segment.segment_id)
                summary = self.segment_store.try_read_by_id(address, segment.segment_id)
                if expected_digest is None or summary is None or summary.digest != expected_digest:
                    raise ConversationLifecycleError("retiring Archive does not match retained History Segment")
                summary.require_matches_source(segment)
                if not self.workflow.is_committed(address, segment):
                    return None, (), ()
            if not active:
                raise ConversationLifecycleError("retiring Archive History source is missing")
            released = self.journal.release_history_prefix(address, active)
            purged = self.journal.purge_released_history(
                address,
                max_items=self.cleanup_batch_size,
            )
            journal_state = self.journal.read_state(address)
            if journal_state.released_through < current.end_sequence:
                return None, purged, released
        advanced = self.retirement_store.advance(
            current,
            ConversationSummaryRetirementPhase.HISTORY_RELEASED,
            updated_at=now,
        )
        return advanced, purged, released

    def _delete_sources(
        self,
        current: ConversationSummaryRetirementManifest,
        use_store: ConversationSummaryUseReader,
        now: datetime,
    ) -> tuple[ConversationSummaryRetirementManifest, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]]:
        """HISTORY_RELEASED：依次删除 Segment、Range、Archive Summary 与使用覆盖，再推进相位。"""

        deleted_segments = self._delete_segment_summaries(current)
        deleted_ranges = self._delete_range_summaries(current)
        deleted_archives = self._delete_archive_summary(current)
        use_store.delete_coverage(
            current.address,
            start_sequence=current.start_sequence,
            end_sequence=current.end_sequence,
        )
        advanced = self.retirement_store.advance(
            current,
            ConversationSummaryRetirementPhase.SOURCES_DELETED,
            updated_at=now,
        )
        return advanced, (deleted_segments, deleted_ranges, deleted_archives)

    def _delete_segment_summaries(self, current: ConversationSummaryRetirementManifest) -> tuple[str, ...]:
        deleted: list[str] = []
        for source in current.segments:
            existing = self.segment_store.try_read_by_id(current.address, source.reference.summary_id)
            if existing is not None and existing.digest != source.digest:
                raise ConversationLifecycleError("retiring Segment Summary digest changed")
            if existing is not None and self.segment_store.delete_by_id(
                current.address,
                source.reference.summary_id,
            ):
                deleted.append(source.reference.summary_id)
        return tuple(deleted)

    def _delete_range_summaries(self, current: ConversationSummaryRetirementManifest) -> tuple[str, ...]:
        deleted: list[str] = []
        for source in current.ranges:
            range_existing = self.range_store.try_read(
                current.address,
                ConversationRangeSummaryStage.RANGE,
                source.reference.summary_id,
            )
            if range_existing is not None and range_existing.digest != source.digest:
                raise ConversationLifecycleError("retiring Range Summary digest changed")
            if range_existing is not None and self.range_store.delete(
                current.address,
                ConversationRangeSummaryStage.RANGE,
                source.reference.summary_id,
            ):
                deleted.append(source.reference.summary_id)
        return tuple(deleted)

    def _delete_archive_summary(self, current: ConversationSummaryRetirementManifest) -> tuple[str, ...]:
        archive = self.range_store.try_read(
            current.address,
            ConversationRangeSummaryStage.ARCHIVE,
            current.archive.reference.summary_id,
        )
        if archive is not None and archive.digest != current.archive.digest:
            raise ConversationLifecycleError("retiring Archive Summary digest changed")
        if archive is not None and self.range_store.delete(
            current.address,
            ConversationRangeSummaryStage.ARCHIVE,
            current.archive.reference.summary_id,
        ):
            return (current.archive.reference.summary_id,)
        return ()


__all__ = ["ArchiveRetirementOutcome", "ConversationArchiveRetirementRunner"]
