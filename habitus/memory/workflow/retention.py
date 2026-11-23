"""记忆任务与 History 原文的保留期（裁定 111）。

一段的记忆任务到了终态（做成、跳过、判失败）或被人工放弃，过 ``history_retention_days`` 天就释放这一段的原文；
每段各算各的，不要求从最旧的开始连续，也不等摘要压缩或退休。任务到终态满 ``job_retention_days`` 天、且原文已经释放
就删；原文还在的失败任务不按时间删（人工还能重试）。人工放弃记录只为给原文计时，放弃满 ``history_retention_days`` 天、
这一段的原文文件已经不在（刚释放，或从没发布成 history）就跟着删。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from habitus.foundation.ids import same_path_identity
from habitus.memory.conversation import ConversationAddress, ConversationLayout, ConversationMessageJournal
from habitus.memory.workflow.jobs import MemoryJobStore

if TYPE_CHECKING:
    from habitus.memory.workflow.lifecycle import MemoryWorkflowLifecycleConfig


class MemoryWorkflowRetention:
    """按任务终态时间释放原文、删过期任务；每次维护按配置批次有界执行。"""

    def __init__(
        self,
        journal: ConversationMessageJournal,
        jobs: MemoryJobStore,
        config: MemoryWorkflowLifecycleConfig,
    ) -> None:
        self.journal = journal
        self.jobs = jobs
        self.config = config

    def release_expired_history(self, address: ConversationAddress, now: datetime) -> tuple[str, ...]:
        """释放任务到终态（或人工放弃）已满保留期的各段原文。"""

        cutoff = now - timedelta(days=self.config.history_retention_days)
        due = [
            (job.segment_id, job.source_segment_digest)
            for job in self.jobs.list_for_conversation(address)
            if job.status.terminal and job.updated_at <= cutoff
        ]
        due.extend(
            (record.segment_id, record.source_segment_digest)
            for record in self.jobs.list_abandonments()
            if same_path_identity(record.conversation_id, address.conversation_id, "conversation_id")
            and record.started_on == address.started_on
            and record.abandoned_at <= cutoff
        )
        released: list[str] = []
        for segment_id, digest in due:
            if len(released) >= self.config.cleanup_batch_size:
                break
            if self.journal.history_retained(address, segment_id) and self.journal.release_history(
                address, segment_id, digest
            ):
                released.append(segment_id)
        return tuple(released)

    def delete_expired_jobs(self, address: ConversationAddress, now: datetime) -> tuple[int, ...]:
        """删到终态满任务保留期、原文已经释放的 Job，以及原文已经不在、放弃满原文保留期的人工放弃记录。"""

        cutoff = now - timedelta(days=self.config.job_retention_days)
        archived = self.journal.read_state(address).archived_through_sequence
        deleted: list[int] = []
        for job in self.jobs.list_for_conversation(address):
            if not job.status.terminal or job.updated_at > cutoff:
                continue
            if not self.history_released(address, job.segment_id, archived):
                continue
            if self.jobs.discard_committed(job):
                deleted.append(job.memory_sequence)
            if len(deleted) >= self.config.cleanup_batch_size:
                return tuple(deleted)
        history_cutoff = now - timedelta(days=self.config.history_retention_days)
        for record in self.jobs.list_abandonments():
            if (
                same_path_identity(record.conversation_id, address.conversation_id, "conversation_id")
                and record.started_on == address.started_on
                and record.abandoned_at <= history_cutoff
                and not self.journal.history_retained(address, record.segment_id)
                and self.jobs.abandonment_files.discard(record)
            ):
                deleted.append(record.memory_sequence)
                if len(deleted) >= self.config.cleanup_batch_size:
                    break
        return tuple(deleted)

    def history_released(self, address: ConversationAddress, segment_id: str, archived: int | None) -> bool:
        """这一段已经封存过、原文文件已经不在了。"""

        _start, end = ConversationLayout.segment_range(segment_id)
        return archived is not None and end <= archived and not self.journal.history_retained(address, segment_id)


__all__ = ["MemoryWorkflowRetention"]
