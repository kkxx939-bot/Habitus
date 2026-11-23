"""阶段 3 的轻写入：锁里核对版本、按固定顺序逐个原子写、中途失败不回滚，以及任务文件里的"卡片已写完"阶段。

崩溃与失败都用确定的办法造：在第几次写文件时抛错、在刷概览时抛错、或抛一个 ``BaseException`` 冒充进程被杀
（Runner 只接 ``Exception``，被杀时任务停在 RUNNING，等租约过期由下一次认领接管）。
"""

from __future__ import annotations

import asyncio
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from habitus.infrastructure.editor.snapshot import SnapshotBatch
from habitus.infrastructure.store.contracts import PathLock
from habitus.memory.conversation import ConversationAddress
from habitus.memory.editor import (
    MemoryCandidate,
    MemoryCandidateBatch,
    MemoryCommitConflictError,
    MemoryCommitError,
    MemoryCommitPlan,
    MemoryCommitWriter,
    MemoryFinalIdentity,
    MemoryFinalIdentityMap,
    MemoryMutationPlanner,
    MemoryMutationReadSet,
    MemoryNodeDisposition,
    MemoryPageIdMap,
    MemoryRelationPlanner,
    MemoryRelationReadSet,
)
from habitus.memory.model import MemoryDirectory, MemoryKind, MemoryLevel
from habitus.memory.semantic import MemorySemanticRefresher
from habitus.memory.snapshot import MemorySnapshotReader
from habitus.memory.tree import MemoryTree
from habitus.memory.uri import MemoryURI
from habitus.memory.workflow import MemoryJobClaim, MemoryJobExecutionError, MemoryJobRunner, MemoryJobStatus
from habitus.memory.workflow.completion import MemoryCommittedJobFinalizer
from habitus.memory.workflow.jobs import MemoryJobConfig, MemoryJobStore
from tests.helpers import BASE_TIME, closed_turn, registry, segment
from tests.lock_store import ProcessLocalLockStore


class Clock:
    def __init__(self) -> None:
        self.now = BASE_TIME

    def __call__(self):
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += timedelta(seconds=seconds)


class Killed(BaseException):
    """冒充进程被杀：不是 ``Exception``，Runner 不会把它记成失败。"""


class Site:
    """一棵空记忆树、一个写入器、一个任务队列和一个用假执行器拼起来的 Runner。"""

    def __init__(self, tmp_path: Path, *, max_attempts: int = 3) -> None:
        lock = PathLock(ProcessLocalLockStore())
        self.clock = Clock()
        self.tree = MemoryTree(tmp_path / "memory")
        self.tree.initialize()
        self.reader = MemorySnapshotReader(self.tree)
        self.writer = MemoryCommitWriter(self.tree, self.reader, lock)
        self.store = MemoryJobStore(
            tmp_path / "workflow",
            lock,
            memory_root=self.tree.root,
            config=MemoryJobConfig(
                max_attempts=max_attempts,
                lease_ttl_seconds=3,
                retry_base_delay_seconds=1,
                retry_max_delay_seconds=4,
            ),
            clock=self.clock,
        )
        self.refresher = MemorySemanticRefresher(self.tree, registry(), lock)
        self.extractions = 0
        self.contents: dict[str, str] = {}
        self.runner = object.__new__(MemoryJobRunner)
        self.runner.store = self.store
        self.runner.executor = SimpleNamespace(execute=self._execute)
        self.runner.committed_finalizer = MemoryCommittedJobFinalizer(
            self.store,
            self.refresher,
            SimpleNamespace(forget=lambda uris: len(uris)),
        )

    async def _execute(self, lease):
        """假执行器：每次都从当前的树重新"提取"（数一次），写卡片，再记下"卡片已写完"。"""

        self.extractions += 1
        commit = self.writer.commit(preference_plan(self.reader, self.contents))
        job = self.store.record_written(
            lease,
            changed_uris=tuple(str(uri) for uri in commit.written_uris),
            deleted_uris=tuple(str(uri) for uri in commit.deleted_uris),
        )
        return SimpleNamespace(commit=commit, job=job, summary_generated=False)

    def enqueue(self, conversation_id: str = "conversation-1"):
        address = ConversationAddress(conversation_id, date(2026, 7, 1))
        source = segment(conversation_id=conversation_id, segment_id="000000000000-000000000001", messages=closed_turn())
        return self.store.activate(self.store.stage(address, source))

    def run_next(self):
        job = self.store.oldest_unfinished()
        assert job is not None
        claim = MemoryJobClaim(self.store.claim(job, "worker-1"))
        return asyncio.run(self.runner.run_claimed(claim))

    def preferences(self) -> tuple[str, ...]:
        return tuple(sorted(str(MemoryURI.from_address(item)) for item in self.tree.list_addresses(MemoryKind.PREFERENCE)))


def preference_plan(reader: MemorySnapshotReader, contents: dict[str, str]) -> MemoryCommitPlan:
    """按当前的树给一组偏好做写入计划：已经在的按同址更新（同内容就是 NOOP），不在的新建。"""

    drafts = [MemoryCandidate(100 + index, MemoryKind.PREFERENCE, {"topic": topic, "content": content})
              for index, (topic, content) in enumerate(sorted(contents.items()))]
    uris = [MemoryURI.from_address(item.address) for item in drafts]
    targets = reader.read_many(uris)
    existing = tuple(snapshot for snapshot in targets.snapshots if snapshot.exists)
    old = SnapshotBatch(existing, sum(snapshot.size_bytes for snapshot in existing))
    pages = MemoryPageIdMap.from_snapshots(old)
    candidates = tuple(
        MemoryCandidate(pages.page_id_for(uri) or draft.page_id, draft.kind, draft.fields)
        for draft, uri in zip(drafts, uris, strict=True)
    )
    batch = MemoryCandidateBatch(preferences=candidates)
    mutations = MemoryMutationPlanner().plan(batch, MemoryMutationReadSet(old, targets), pages)
    identities = MemoryFinalIdentityMap.from_mutation_plan(mutations, pages)
    relations = MemoryRelationPlanner().plan(identities, (), MemoryRelationReadSet.build(targets, identities, ()))
    return MemoryCommitPlan.build(mutations, identities, relations)


def failing_write(monkeypatch: pytest.MonkeyPatch, tree: MemoryTree, *, on_call: int, error: BaseException) -> list[int]:
    """让 ``tree.write`` 在第 ``on_call`` 次调用时抛错（只抛一次），返回调用计数。"""

    original = tree.write
    calls = [0]

    def write(document):
        calls[0] += 1
        if calls[0] == on_call:
            raise error
        return original(document)

    monkeypatch.setattr(tree, "write", write)
    return calls


THREE = {"甲话题": "- 偏好甲", "乙话题": "- 偏好乙", "丙话题": "- 偏好丙"}


def test_a_file_changed_before_writing_raises_conflict_writes_nothing_and_the_job_fails_after_its_attempts(
    tmp_path: Path,
) -> None:
    """（1）规划之后、写之前文件被别人改了：锁里核对版本不一致，一个文件都不写；Runner 按可重试失败整条重做，
    超过 max_attempts 判 FAILED。"""

    site = Site(tmp_path, max_attempts=2)
    site.writer.commit(preference_plan(site.reader, {"甲话题": "- 原来的说法"}))
    stale = preference_plan(site.reader, {"甲话题": "- 这次的说法", "乙话题": "- 新卡片"})
    site.writer.commit(preference_plan(site.reader, {"甲话题": "- 别人刚改的说法"}))  # 例如冷热压缩

    with pytest.raises(MemoryCommitConflictError):
        site.writer.commit(stale)
    assert site.preferences() == ("memory://preferences/甲话题.md",)
    current = site.reader.read("memory://preferences/甲话题.md")
    assert current.value is not None and current.value.fields["content"] == "- 别人刚改的说法"

    async def always_conflicts(_lease):
        site.extractions += 1
        site.writer.commit(stale)

    site.runner.executor = SimpleNamespace(execute=always_conflicts)
    site.enqueue()
    with pytest.raises(MemoryJobExecutionError) as first:
        site.run_next()
    assert first.value.job is not None and first.value.job.status is MemoryJobStatus.QUEUED
    site.clock.advance(1)
    with pytest.raises(MemoryJobExecutionError) as second:
        site.run_next()
    assert second.value.job is not None and second.value.job.status is MemoryJobStatus.FAILED
    assert site.extractions == 2


def test_a_write_that_stops_at_the_second_of_three_files_keeps_the_first_and_a_retry_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """（2）一批三个新卡片第二个写失败：第一个留在盘上、第三个没写，不回滚；任务失败可重试，重试做成。"""

    site = Site(tmp_path)
    site.contents = dict(THREE)
    site.enqueue()
    calls = failing_write(monkeypatch, site.tree, on_call=2, error=OSError("disk full"))

    with pytest.raises(MemoryJobExecutionError) as failed:
        site.run_next()
    assert isinstance(failed.value.__cause__, MemoryCommitError)
    assert site.preferences() == ("memory://preferences/丙话题.md",)  # 按地址排序写，第一个已经落盘
    assert calls[0] == 2
    job = failed.value.job
    assert job is not None and job.status is MemoryJobStatus.QUEUED and not job.cards_written

    site.clock.advance(1)
    result = site.run_next()
    assert result.job is not None and result.job.status is MemoryJobStatus.COMMITTED
    assert site.preferences() == tuple(sorted(f"memory://preferences/{topic}.md" for topic in THREE))
    first = site.reader.read("memory://preferences/丙话题.md")
    assert first.revision == 1  # 已经写成的那张这次是 NOOP，没有再写一遍


def test_a_merge_whose_target_fails_to_write_keeps_its_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """（3）合并：顺序是先写目标、最后删来源；目标没写成就停下，来源不删。"""

    site = Site(tmp_path)
    site.writer.commit(preference_plan(site.reader, {"回答风格": "- 简洁", "输出格式": "- 先给结论"}))
    source_uri = MemoryURI.parse("memory://preferences/回答风格.md")
    target_uri = MemoryURI.parse("memory://preferences/输出格式.md")
    old = site.reader.read_many((source_uri, target_uri))
    pages = MemoryPageIdMap.from_snapshots(old)
    source_page = pages.page_id_for(source_uri)
    target_page = pages.page_id_for(target_uri)
    assert source_page is not None and target_page is not None
    batch = MemoryCandidateBatch(
        preferences=(
            MemoryCandidate(target_page, MemoryKind.PREFERENCE, {"topic": "输出格式", "content": "- 简洁，先给结论"}),
        )
    )
    targets = site.reader.read_many((target_uri,))
    mutations = MemoryMutationPlanner().plan(batch, MemoryMutationReadSet(old, targets), pages)
    identities = MemoryFinalIdentityMap(
        tuple(
            sorted(
                (
                    MemoryFinalIdentity(source_page, MemoryNodeDisposition.MERGE, source_uri, target_uri),
                    MemoryFinalIdentity(target_page, MemoryNodeDisposition.UPDATE, target_uri, target_uri),
                ),
                key=lambda entry: entry.page_id,
            )
        )
    )
    relations = MemoryRelationPlanner().plan(identities, (), MemoryRelationReadSet.build(old, identities, ()))
    plan = MemoryCommitPlan.build(mutations, identities, relations)
    assert plan.retired_uris == (source_uri,)
    failing_write(monkeypatch, site.tree, on_call=1, error=OSError("disk full"))

    with pytest.raises(MemoryCommitError):
        site.writer.commit(plan)

    assert site.reader.read(source_uri).exists
    unchanged = site.reader.read(target_uri)
    assert unchanged.value is not None and unchanged.value.fields["content"] == "- 先给结论"


def test_a_failure_after_cards_written_resumes_only_the_later_steps_without_extracting_again(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """（4）卡片写完、记下"卡片已写完"之后、标完成之前出错（这里是刷概览时抛错）：再跑一次只补刷概览和标完成，
    不再提取。"""

    site = Site(tmp_path)
    site.contents = {"回答风格": "- 简洁"}
    site.enqueue()
    original = site.refresher.refresh_for_many
    refreshes = [0]

    def refresh_once_broken(addresses):
        refreshes[0] += 1
        if refreshes[0] == 1:
            raise OSError("overview write failed")
        return original(addresses)

    monkeypatch.setattr(site.refresher, "refresh_for_many", refresh_once_broken)

    with pytest.raises(MemoryJobExecutionError) as failed:
        site.run_next()
    job = failed.value.job
    assert job is not None and job.status is MemoryJobStatus.QUEUED
    assert job.cards_written and job.changed_uris == ("memory://preferences/回答风格.md",)
    assert not site.tree.layer_exists(MemoryDirectory.preferences(), MemoryLevel.OVERVIEW)

    site.clock.advance(1)
    result = site.run_next()
    assert result.recovered and result.commit is None
    assert result.job is not None and result.job.status is MemoryJobStatus.COMMITTED
    assert site.extractions == 1
    assert site.tree.layer_exists(MemoryDirectory.preferences(), MemoryLevel.OVERVIEW)


def test_a_crash_half_way_through_writing_cards_redoes_the_whole_job_without_duplicates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """（5）写卡片写到一半进程被杀（任务文件里还没有"卡片已写完"）：租约过期后接管，整条重做并做成，卡片不重复。"""

    site = Site(tmp_path)
    site.contents = dict(THREE)
    site.enqueue()
    failing_write(monkeypatch, site.tree, on_call=2, error=Killed())

    with pytest.raises(Killed):
        site.run_next()
    stuck = site.store.oldest_unfinished()
    assert stuck is not None and stuck.status is MemoryJobStatus.RUNNING and not stuck.cards_written
    assert site.preferences() == ("memory://preferences/丙话题.md",)

    site.clock.advance(4)  # 租约 3 秒，过期后才能接管
    result = site.run_next()
    assert result.job is not None and result.job.status is MemoryJobStatus.COMMITTED
    assert result.job.claim_generation == 2
    assert not result.recovered and site.extractions == 2
    assert site.preferences() == tuple(sorted(f"memory://preferences/{topic}.md" for topic in THREE))
