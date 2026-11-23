"""Archive 退休清单的三相位幂等续跑：认领并确认 History 已释放、删除来源链、完成清单。

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
from habitus.memory.workflow.lifecycle_errors import ConversationLifecycleError
from habitus.pre.conversation import ConversationRangeSummaryStage

ArchiveRetirementOutcome = tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]
"""依次为：删除的 Segment/Range/Archive Summary。"""

_NOTHING: ArchiveRetirementOutcome = ((), (), ())


class ConversationArchiveRetirementRunner:
    """严格按认领与确认 History 已释放、子摘要、Archive 的阶段幂等续跑一个已准备的退休清单。"""

    def __init__(
        self,
        compactor: ConversationSummaryCompactor,
        journal: ConversationMessageJournal,
        segment_store: ConversationSummaryStore,
        range_store: ConversationRangeSummaryStore,
        retirement_store: ConversationSummaryRetirementStore,
    ) -> None:
        self.compactor = compactor
        self.journal = journal
        self.segment_store = segment_store
        self.range_store = range_store
        self.retirement_store = retirement_store

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
            if not self._claim(current, use_store, now):
                return _NOTHING
            advanced = self._confirm_history_released(current, now)
            if advanced is None:
                return _NOTHING
            current = advanced
        segments: tuple[str, ...] = ()
        ranges: tuple[str, ...] = ()
        archives: tuple[str, ...] = ()
        if current.phase is ConversationSummaryRetirementPhase.HISTORY_RELEASED:
            current, (segments, ranges, archives) = self._delete_sources(current, use_store, now)
        if current.phase is ConversationSummaryRetirementPhase.SOURCES_DELETED:
            self.retirement_store.complete(current)
        return segments, ranges, archives

    def _claim(
        self,
        current: ConversationSummaryRetirementManifest,
        use_store: ConversationSummaryUseReader,
        now: datetime,
    ) -> bool:
        """RETIRING：认领使用状态；使用状态已变化则放弃本清单。认领过的（中途被打断）直接继续。"""

        states = use_store.read_many((current.archive.reference,))
        use_state = states[0] if states else None
        claimed = (
            use_state is not None
            and use_state.version == current.expected_use_version + 1
            and use_state.retiring_at is not None
        )
        if claimed:
            return True
        if use_state is None or use_state.version != current.expected_use_version:
            self.retirement_store.complete(current)
            return False
        use_store.claim_retirement(
            current.archive.reference,
            expected_version=current.expected_use_version,
            claimed_at=now,
        )
        return True

    def _confirm_history_released(
        self,
        current: ConversationSummaryRetirementManifest,
        now: datetime,
    ) -> ConversationSummaryRetirementManifest | None:
        """RETIRING（已认领）：原文按段到期各自释放（裁定 111），这里只确认覆盖范围内一段都不剩才推进相位。"""

        retained = self.journal.retained_history_ranges(current.address)
        if any(start <= current.end_sequence and current.start_sequence <= end for start, end in retained):
            return None
        return self.retirement_store.advance(
            current,
            ConversationSummaryRetirementPhase.HISTORY_RELEASED,
            updated_at=now,
        )

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
