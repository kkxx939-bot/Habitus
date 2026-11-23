"""对话原文按段到期释放、任务保留期与摘要压缩跨过空档（阶段 3 裁定）。

一段的记忆任务到终态（做成、跳过、判失败）满 ``history_retention_days`` 天就释放这一段原文，每段各算各的；
任务到了终态、原文也释放了，再过 ``job_retention_days`` 天删。任务存储的时钟在测试里手动推进，终态时刻可控。
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta
from pathlib import Path

from habitus.memory.conversation import ConversationAddress
from habitus.memory.conversation.compaction import (
    ConversationSegmentSummaryCompactionConfig,
    ConversationSummaryCompactionConfig,
)
from habitus.memory.workflow import MemoryJobStatus
from habitus.pre.conversation import ConversationBatch, ConversationSegment
from habitus.runtime.consistency import MemoryConsistencyService, MemoryConsistencyState
from tests.helpers import BASE_TIME, closed_turn, segment_summary
from tests.unit.workflow.test_lifecycle_manager import lifecycle_manager

ADDRESS = ConversationAddress("conversation-1", date(2026, 7, 1))


def _at(manager, moment: datetime) -> None:  # type: ignore[no-untyped-def]
    manager.jobs.clock = lambda: moment


def _sealed(manager, start: int) -> ConversationSegment:  # type: ignore[no-untyped-def]
    manager.journal.append(ADDRESS, ConversationBatch("conversation-1", closed_turn(start_sequence=start)))
    return manager.journal.seal(ADDRESS, through_sequence=start + 1).segment


def _queued(manager, source: ConversationSegment):  # type: ignore[no-untyped-def]
    return manager.jobs.activate(manager.jobs.stage(ADDRESS, source))


def test_each_segment_is_released_when_its_own_job_has_been_terminal_long_enough(tmp_path: Path) -> None:
    """（1）更早的段到终态更晚、还没到期，更晚的段先到期先释放；不要求从最旧的开始连续。"""

    manager = lifecycle_manager(tmp_path, compaction_enabled=False)
    _at(manager, BASE_TIME)
    earlier, later = _sealed(manager, 0), _sealed(manager, 2)
    earlier_job, later_job = _queued(manager, earlier), _queued(manager, later)
    _at(manager, BASE_TIME + timedelta(days=5))
    manager.jobs.complete(manager.jobs.claim(earlier_job, "worker"))
    _at(manager, BASE_TIME + timedelta(days=1))
    manager.jobs.complete(manager.jobs.claim(later_job, "worker"))

    result = asyncio.run(manager.maintain_once(ADDRESS, now=BASE_TIME + timedelta(days=11)))

    assert result.released_history_segment_ids == (later.segment_id,)
    assert manager.journal.list_history(ADDRESS) == (earlier,)

    result = asyncio.run(manager.maintain_once(ADDRESS, now=BASE_TIME + timedelta(days=15)))
    assert result.released_history_segment_ids == (earlier.segment_id,)
    assert manager.journal.list_history(ADDRESS) == ()


def test_failed_and_skipped_segments_without_summary_are_released_by_the_same_rule(tmp_path: Path) -> None:
    """（2）判了失败的段、跳过且没有摘要的段，到终态满 N 天同样释放；还没到期、还没到终态的不动。"""

    manager = lifecycle_manager(tmp_path, compaction_enabled=False)
    _at(manager, BASE_TIME)
    failed_source, skipped_source, pending_source = _sealed(manager, 0), _sealed(manager, 2), _sealed(manager, 4)
    failed_job = _queued(manager, failed_source)
    manager.jobs.fail(manager.jobs.claim(failed_job, "worker"), RuntimeError("broken"), retryable=False)
    skipped_job = _queued(manager, skipped_source)
    manager.jobs.skip(manager.jobs.claim(skipped_job, "worker"), "invalid_request: too long")
    _queued(manager, pending_source)
    assert manager.segment_store.list(ADDRESS) == ()

    early = asyncio.run(manager.maintain_once(ADDRESS, now=BASE_TIME + timedelta(days=9)))
    assert early.released_history_segment_ids == ()

    due = asyncio.run(manager.maintain_once(ADDRESS, now=BASE_TIME + timedelta(days=10)))
    assert set(due.released_history_segment_ids) == {failed_source.segment_id, skipped_source.segment_id}
    assert manager.journal.list_history(ADDRESS) == (pending_source,)


def test_failed_job_is_kept_while_its_history_remains_and_deleted_after_release_plus_retention(
    tmp_path: Path,
) -> None:
    """（4）原文还在的失败任务不按时间删（人工还能重试）；原文释放后过任务保留期才删。"""

    manager = lifecycle_manager(tmp_path, compaction_enabled=False)
    _at(manager, BASE_TIME)
    source = _sealed(manager, 0)
    failed = manager.jobs.fail(
        manager.jobs.claim(_queued(manager, source), "worker"), RuntimeError("broken"), retryable=False
    )
    assert failed.status is MemoryJobStatus.FAILED
    retention = manager.retention

    assert retention.delete_expired_jobs(ADDRESS, BASE_TIME + timedelta(days=400)) == ()
    assert manager.jobs.list_for_conversation(ADDRESS) == (failed,)

    assert retention.release_expired_history(ADDRESS, BASE_TIME + timedelta(days=10)) == (source.segment_id,)
    assert retention.delete_expired_jobs(ADDRESS, BASE_TIME + timedelta(days=29)) == ()
    assert retention.delete_expired_jobs(ADDRESS, BASE_TIME + timedelta(days=30)) == (failed.memory_sequence,)
    assert manager.jobs.list_for_conversation(ADDRESS) == ()


def test_abandoned_job_releases_history_counted_from_its_abandonment(tmp_path: Path) -> None:
    """人工放弃的失败任务没有 Job 文件了，原文从放弃的时刻起算保留期。"""

    manager = lifecycle_manager(tmp_path, compaction_enabled=False)
    _at(manager, BASE_TIME)
    source = _sealed(manager, 0)
    failed = manager.jobs.fail(
        manager.jobs.claim(_queued(manager, source), "worker"), RuntimeError("broken"), retryable=False
    )
    _at(manager, BASE_TIME + timedelta(days=3))
    manager.jobs.abandon_failed(failed, reason="人工确认这一段不要了")

    assert manager.retention.release_expired_history(ADDRESS, BASE_TIME + timedelta(days=12)) == ()
    assert manager.retention.release_expired_history(ADDRESS, BASE_TIME + timedelta(days=13)) == (source.segment_id,)


def test_compaction_crosses_a_released_summaryless_gap_but_not_a_retained_one(tmp_path: Path) -> None:
    """（3）两份摘要之间隔着一段永久没有摘要的段：它的原文还在时不能跨（以后可能还会有摘要），释放后能接成一组，
    父摘要的来源照实只有那两份。"""

    config = ConversationSummaryCompactionConfig(
        enabled=True,
        segment_to_range=ConversationSegmentSummaryCompactionConfig(min_age_days=1, min_source_count=2),
    )
    manager = lifecycle_manager(tmp_path, compaction_enabled=True, compaction_config=config)
    first, gap, last = _sealed(manager, 0), _sealed(manager, 2), _sealed(manager, 4)
    for source in (first, last):
        manager.segment_store.create(ADDRESS, source, segment_summary(source))
    now = BASE_TIME + timedelta(days=30)

    assert manager.compactor._plan_under_lock(ADDRESS, now) is None

    assert manager.journal.release_history(ADDRESS, gap.segment_id, gap.digest)
    plan = manager.compactor._plan_under_lock(ADDRESS, now)
    assert plan is not None
    assert tuple(ref.summary_id for ref in plan.source_refs) == (first.segment_id, last.segment_id)


def test_abandonment_record_goes_with_its_released_history(tmp_path: Path) -> None:
    """人工放弃记录只为给原文计时：放弃满 history_retention_days 天、原文释放的同一次维护里一起删，不等 30 天；
    之前一致性状态是"已放弃"，删掉之后是 UNKNOWN（用户 2026-11-23：「感觉可以 10 天删除呀」）。"""

    manager = lifecycle_manager(tmp_path, compaction_enabled=False)
    _at(manager, BASE_TIME)
    source = _sealed(manager, 0)
    failed = manager.jobs.fail(
        manager.jobs.claim(_queued(manager, source), "worker"), RuntimeError("broken"), retryable=False
    )
    record = manager.jobs.abandon_failed(failed, reason="人工确认这一段不要了")
    consistency = MemoryConsistencyService(manager.jobs)

    early = asyncio.run(manager.maintain_once(ADDRESS, now=BASE_TIME + timedelta(days=9)))
    assert early.released_history_segment_ids == () and early.deleted_memory_job_sequences == ()
    assert manager.jobs.list_abandonments() == (record,)
    assert consistency.inspect(failed).state is MemoryConsistencyState.ABANDONED

    due = asyncio.run(manager.maintain_once(ADDRESS, now=BASE_TIME + timedelta(days=10)))
    assert due.released_history_segment_ids == (source.segment_id,)
    assert due.deleted_memory_job_sequences == (record.memory_sequence,)
    assert manager.jobs.list_abandonments() == ()
    assert consistency.inspect(failed).state is MemoryConsistencyState.UNKNOWN


def test_abandonment_of_a_segment_never_published_as_history_is_deleted_after_the_same_days(tmp_path: Path) -> None:
    """STAGED 阶段就失败、原文从没发布成 history 的段被放弃：没有原文文件可释放，放弃满同样的天数就删放弃记录。"""

    manager = lifecycle_manager(tmp_path, compaction_enabled=False)
    _at(manager, BASE_TIME)
    manager.journal.append(ADDRESS, ConversationBatch("conversation-1", closed_turn(start_sequence=0)))
    staged = manager.jobs.stage(ADDRESS, ConversationSegment("conversation-1", "000000000000-000000000001", closed_turn()))
    while staged.status is MemoryJobStatus.STAGED:
        staged = manager.jobs.record_staged_failure(staged, OSError(28, "No space left on device"))
    record = manager.jobs.abandon_failed(staged, reason="发布不出来，不要了")
    assert not manager.journal.history_retained(ADDRESS, record.segment_id)

    assert manager.retention.delete_expired_jobs(ADDRESS, BASE_TIME + timedelta(days=9)) == ()
    assert manager.retention.delete_expired_jobs(ADDRESS, BASE_TIME + timedelta(days=10)) == (record.memory_sequence,)
    assert manager.jobs.list_abandonments() == ()
