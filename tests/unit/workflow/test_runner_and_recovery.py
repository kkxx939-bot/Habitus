"""MemoryJob 产品并行构建、STAGED 恢复、Runner 正常 / 只补后续步骤 / 失败分支测试。"""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from habitus.infrastructure.store.contracts import PathLock
from habitus.memory.conversation import ConversationAddress, ConversationMessageJournal
from habitus.memory.editor import MemoryEditor
from habitus.memory.workflow import (
    MemoryJobClaim,
    MemoryJobExecutionError,
    MemoryJobLease,
    MemoryJobLeaseLostError,
    MemoryJobRunner,
)
from habitus.memory.workflow.jobs import MemoryJobStatus, MemoryJobStore
from habitus.memory.workflow.planning import MemorySegmentProductBuilder
from habitus.memory.workflow.recovery import MemoryStagedJobRecovery
from habitus.pre.conversation import ConversationBatch
from tests.helpers import closed_turn, segment, segment_summary
from tests.lock_store import ProcessLocalLockStore


def job_store(tmp_path: Path) -> MemoryJobStore:
    return MemoryJobStore(
        tmp_path / "workflow",
        PathLock(ProcessLocalLockStore()),
        memory_root=tmp_path / "memory",
    )


def claimed_job(tmp_path: Path):
    store = job_store(tmp_path)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    source = segment(
        segment_id="000000000000-000000000001",
        messages=closed_turn(),
    )
    queued = store.activate(store.stage(address, source))
    return store, store.claim(queued, "runner-test")


def test_segment_products_run_both_read_only_branches_concurrently_and_wait_for_both() -> None:
    source = segment()
    address = ConversationAddress(source.conversation_id, date(2026, 7, 1))
    summary_started = asyncio.Event()
    editor_started = asyncio.Event()
    summary_finished = False
    editor_finished = False
    plan = SimpleNamespace(name="editor-plan")

    async def scenario():
        nonlocal summary_finished, editor_finished
        summary_service = SimpleNamespace()
        editor = object.__new__(MemoryEditor)

        async def get_or_create(_address, current):
            nonlocal summary_finished
            summary_started.set()
            await editor_started.wait()
            summary_finished = True
            return segment_summary(current)

        async def build_plan(current, *, turn_context=None):
            nonlocal editor_finished
            editor_started.set()
            await summary_started.wait()
            editor_finished = True
            assert current is source
            return plan

        summary_service.get_or_create = get_or_create
        editor.plan = build_plan
        builder = object.__new__(MemorySegmentProductBuilder)
        builder.summary_service = summary_service
        builder.editor = editor
        return await builder.build(address, source)

    products = asyncio.run(scenario())
    assert products.summary == segment_summary(source)
    assert products.editor_plan is plan
    assert summary_finished and editor_finished


def test_segment_product_failure_waits_for_sibling_but_never_returns_partial_product() -> None:
    source = segment()
    address = ConversationAddress(source.conversation_id, date(2026, 7, 1))
    sibling_finished = False

    async def scenario() -> None:
        nonlocal sibling_finished
        summary_service = SimpleNamespace()
        editor = object.__new__(MemoryEditor)

        async def fail_summary(_address, _source):
            await asyncio.sleep(0)
            raise RuntimeError("summary failed")

        async def finish_editor(_source, *, turn_context=None):
            nonlocal sibling_finished
            await asyncio.sleep(0.001)
            sibling_finished = True
            return SimpleNamespace(name="editor-plan")

        summary_service.get_or_create = fail_summary
        editor.plan = finish_editor
        builder = object.__new__(MemorySegmentProductBuilder)
        builder.summary_service = summary_service
        builder.editor = editor
        with pytest.raises(RuntimeError, match="summary failed"):
            await builder.build(address, source)

    asyncio.run(scenario())
    assert sibling_finished


def test_segment_products_defer_editor_until_a_complete_turn_boundary() -> None:
    source = segment()
    address = ConversationAddress(source.conversation_id, date(2026, 7, 1))

    async def scenario():
        summary_service = SimpleNamespace()
        editor = object.__new__(MemoryEditor)

        async def get_or_create(_address, current):
            return segment_summary(current)

        async def unexpected_plan(_current):
            raise AssertionError("deferred segment must not invoke the editor")

        summary_service.get_or_create = get_or_create
        editor.plan = unexpected_plan
        builder = object.__new__(MemorySegmentProductBuilder)
        builder.summary_service = summary_service
        builder.editor = editor
        return await builder.build(address, source, editor_segment=None)

    products = asyncio.run(scenario())
    assert products.summary == segment_summary(source)
    assert products.editor_plan.deferred
    assert products.editor_plan.commit.changed_uris == ()


def test_staged_recovery_publishes_exact_history_then_activates_job(tmp_path: Path) -> None:
    path_lock = PathLock(ProcessLocalLockStore())
    journal = ConversationMessageJournal(tmp_path / "conversation", path_lock)
    jobs = MemoryJobStore(
        tmp_path / "workflow",
        path_lock,
        memory_root=tmp_path / "memory",
    )
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    source = segment(
        segment_id="000000000000-000000000001",
        messages=closed_turn(),
    )
    journal.append(address, ConversationBatch(source.conversation_id, source.messages))
    staged = jobs.stage(address, source)

    recovered = MemoryStagedJobRecovery(journal, jobs).recover(staged)

    assert recovered.status is MemoryJobStatus.QUEUED
    assert journal.read_segment(address, source.segment_id) == source
    assert journal.read_live(address) is None


def runner_with_fakes(store: MemoryJobStore, *, executor_error: BaseException | None = None):
    runner = object.__new__(MemoryJobRunner)
    runner.store = store
    events: list[str] = []

    class Executor:
        async def execute(self, lease):
            events.append("execute")
            if executor_error is not None:
                raise executor_error
            job = store.record_written(lease, changed_uris=("habitus://profile.md",), deleted_uris=())
            return SimpleNamespace(commit="commit", job=job, summary_generated=True)

    class Finalizer:
        async def finalize(self, lease):
            events.append(f"finalize:{store.assert_current(lease).changed_uris}")
            return store.complete(lease)

    runner.executor = Executor()
    runner.committed_finalizer = Finalizer()
    return runner, events


def test_runner_resumes_a_job_whose_cards_are_written_without_replanning_or_rewriting(tmp_path: Path) -> None:
    store, lease = claimed_job(tmp_path)
    written = store.record_written(lease, changed_uris=("habitus://profile.md",), deleted_uris=())
    runner, events = runner_with_fakes(store)

    result = asyncio.run(runner.run_claimed(MemoryJobClaim(MemoryJobLease(written))))

    assert result.recovered
    assert result.commit is None
    assert result.job.status is MemoryJobStatus.COMMITTED
    assert events == ["finalize:('habitus://profile.md',)"]


def test_runner_normal_path_executes_once_then_finalizes_with_summary_generation_flag(tmp_path: Path) -> None:
    store, lease = claimed_job(tmp_path)
    runner, events = runner_with_fakes(store)

    result = asyncio.run(runner.run_claimed(MemoryJobClaim(lease)))

    assert not result.recovered
    assert result.summary_generated
    assert result.commit == "commit"
    assert result.job.status is MemoryJobStatus.COMMITTED
    assert events == ["execute", "finalize:('habitus://profile.md',)"]


def test_runner_failure_records_a_retryable_job(tmp_path: Path) -> None:
    store, lease = claimed_job(tmp_path)
    runner, _events = runner_with_fakes(store, executor_error=TimeoutError("provider slow"))

    with pytest.raises(MemoryJobExecutionError) as raised:
        asyncio.run(runner.run_claimed(MemoryJobClaim(lease)))

    assert raised.value.job is not None
    assert raised.value.job.status is MemoryJobStatus.QUEUED
    assert store.oldest_unfinished() == raised.value.job


def test_runner_reclaiming_an_expired_lease_with_exhausted_attempts_fails_unless_cards_are_written(
    tmp_path: Path,
) -> None:
    """租约过期接管时尝试次数已经用完：卡片没写完就判失败，不再重做。"""

    store, lease = claimed_job(tmp_path)
    runner, events = runner_with_fakes(store)

    with pytest.raises(MemoryJobExecutionError) as raised:
        asyncio.run(runner.run_claimed(MemoryJobClaim(lease, expired_attempts_exhausted=True)))

    assert raised.value.job is not None and raised.value.job.status is MemoryJobStatus.FAILED
    assert events == []


def test_runner_propagates_lease_loss_without_marking_current_job_failed(tmp_path: Path) -> None:
    store, lease = claimed_job(tmp_path)
    runner, _events = runner_with_fakes(store, executor_error=MemoryJobLeaseLostError("lost"))

    with pytest.raises(MemoryJobLeaseLostError):
        asyncio.run(runner.run_claimed(MemoryJobClaim(lease)))

    assert store.oldest_unfinished().status is MemoryJobStatus.RUNNING


# ── 前台补完 STAGED Job 的两处段身份核对（M9 的防线，U22） ──────────────────────────────────────────────


def _foreground(tmp_path: Path):  # type: ignore[no-untyped-def]
    from habitus.memory.conversation import ConversationRetentionPlanner, ConversationSegmentationConfig
    from habitus.memory.workflow import ConversationMemoryEnqueuer

    lock = PathLock(ProcessLocalLockStore())
    journal = ConversationMessageJournal(tmp_path / "conversation", lock)
    jobs = MemoryJobStore(tmp_path / "workflow", lock, memory_root=tmp_path / "memory")
    planner = ConversationRetentionPlanner(
        ConversationSegmentationConfig(commit_token_threshold=1, keep_recent_turn_count=1, retained_message_token_budget=10_000),
        token_estimator=lambda _message: 1,
    )
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    journal.append(
        address,
        ConversationBatch("conversation-1", (*closed_turn(start_sequence=0), *closed_turn(start_sequence=2))),
    )
    other = segment(segment_id="000000000000-000000000001", messages=closed_turn(prompt="不是这一段说的话"))
    return journal, jobs, ConversationMemoryEnqueuer(journal, jobs, planner), address, jobs.stage(address, other)


def test_the_foreground_does_not_publish_history_for_a_staged_job_that_does_not_match_its_boundary(tmp_path: Path) -> None:
    """STAGED Job 记的段与 live 在它边界上的那一段对不上：发布 history 之前就停，不发布、不激活。"""

    from habitus.memory.workflow import MemoryJobError

    journal, jobs, enqueuer, address, staged = _foreground(tmp_path)

    with pytest.raises(MemoryJobError, match="does not match"):
        enqueuer.flush(address)

    assert journal.list_history(address) == ()
    assert jobs.oldest_unfinished().status is MemoryJobStatus.STAGED
    assert jobs.oldest_unfinished().memory_sequence == staged.memory_sequence


def test_the_foreground_does_not_activate_a_staged_job_whose_published_segment_differs(tmp_path: Path) -> None:
    """边界上已经有一份发布过的 history（封存返回原样、不走发布前那一处核对），但不是 Job 记的那一段：不激活。"""

    from habitus.memory.workflow import MemoryJobError

    journal, jobs, enqueuer, address, _staged = _foreground(tmp_path)
    journal.seal(address, through_sequence=1)

    with pytest.raises(MemoryJobError, match="does not match"):
        enqueuer.flush(address)

    assert tuple(item.segment_id for item in journal.list_history(address)) == ("000000000000-000000000001",)
    assert jobs.oldest_unfinished().status is MemoryJobStatus.STAGED
