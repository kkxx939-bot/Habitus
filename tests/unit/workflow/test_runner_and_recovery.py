"""MemoryJob 产品并行构建、STAGED 恢复、Runner 正常/恢复/失败分支测试。"""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from infrastructure.store.contracts import PathLock
from infrastructure.store.locks import ProcessLocalLockStore
from memory.conversation import ConversationAddress, ConversationMessageJournal
from memory.editor import MemoryEditor, MemoryEditorPlan
from memory.workflow import (
    MemoryJobClaim,
    MemoryJobExecutionError,
    MemoryJobLeaseLostError,
    MemoryJobRunner,
)
from memory.workflow.jobs import MemoryJobStatus, MemoryJobStore
from memory.workflow.planning import MemorySegmentProductBuilder
from memory.workflow.recovery import MemoryStagedJobRecovery
from pre.conversation import ConversationBatch
from tests.helpers import closed_turn, segment, segment_summary


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
    plan = object.__new__(MemoryEditorPlan)

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

        async def build_plan(current):
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

        async def finish_editor(_source):
            nonlocal sibling_finished
            await asyncio.sleep(0.001)
            sibling_finished = True
            return object.__new__(MemoryEditorPlan)

        summary_service.get_or_create = fail_summary
        editor.plan = finish_editor
        builder = object.__new__(MemorySegmentProductBuilder)
        builder.summary_service = summary_service
        builder.editor = editor
        with pytest.raises(RuntimeError, match="summary failed"):
            await builder.build(address, source)

    asyncio.run(scenario())
    assert sibling_finished


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


def runner_with_fakes(store: MemoryJobStore, *, journal: object | None, executor_error: BaseException | None = None):
    runner = object.__new__(MemoryJobRunner)
    runner.store = store
    events: list[str] = []

    class Recovery:
        def recover_pending(self):
            events.append("recover_pending")
            return ()

        def inspect(self, _job):
            events.append("inspect")
            return journal

        def discard_uncommitted(self, _source):
            events.append("discard_uncommitted")

    class Executor:
        async def execute(self, _lease):
            events.append("execute")
            if executor_error is not None:
                raise executor_error
            return SimpleNamespace(commit=None, journal="committed-journal", summary_generated=True)

    class Finalizer:
        async def finalize(self, lease, current_journal, *, summary_generated=None):
            events.append(f"finalize:{current_journal}:{summary_generated}")
            committed = store.complete(lease)
            return SimpleNamespace(
                job=committed,
                change_receipt=None,
                summary_generated=bool(summary_generated),
                summary_indexed=True,
                vector_indexed=True,
                journal_cleaned=True,
            )

    runner.transaction_recovery = Recovery()
    runner.executor = Executor()
    runner.committed_finalizer = Finalizer()
    return runner, events


def test_runner_resumes_committed_journal_without_replanning_or_recommitting(tmp_path: Path) -> None:
    store, lease = claimed_job(tmp_path)
    runner, events = runner_with_fakes(store, journal="recovered-journal")

    result = asyncio.run(runner.run_claimed(MemoryJobClaim(lease)))

    assert result.recovered
    assert result.commit is None
    assert result.job.status is MemoryJobStatus.COMMITTED
    assert events == ["recover_pending", "inspect", "finalize:recovered-journal:None"]


def test_runner_normal_path_executes_once_then_finalizes_with_summary_generation_flag(tmp_path: Path) -> None:
    store, lease = claimed_job(tmp_path)
    runner, events = runner_with_fakes(store, journal=None)

    result = asyncio.run(runner.run_claimed(MemoryJobClaim(lease)))

    assert not result.recovered
    assert result.summary_generated
    assert result.job.status is MemoryJobStatus.COMMITTED
    assert events == [
        "recover_pending",
        "inspect",
        "execute",
        "finalize:committed-journal:True",
    ]


def test_runner_failure_cleans_only_uncommitted_state_and_records_retryable_job(tmp_path: Path) -> None:
    store, lease = claimed_job(tmp_path)
    runner, events = runner_with_fakes(store, journal=None, executor_error=TimeoutError("provider slow"))

    with pytest.raises(MemoryJobExecutionError) as raised:
        asyncio.run(runner.run_claimed(MemoryJobClaim(lease)))

    assert raised.value.job is not None
    assert raised.value.job.status is MemoryJobStatus.QUEUED
    assert "discard_uncommitted" in events
    assert store.oldest_uncommitted() == raised.value.job


def test_runner_propagates_lease_loss_without_marking_current_job_failed(tmp_path: Path) -> None:
    store, lease = claimed_job(tmp_path)
    runner, events = runner_with_fakes(
        store,
        journal=None,
        executor_error=MemoryJobLeaseLostError("lost"),
    )

    with pytest.raises(MemoryJobLeaseLostError):
        asyncio.run(runner.run_claimed(MemoryJobClaim(lease)))

    assert store.oldest_uncommitted().status is MemoryJobStatus.RUNNING
    assert "discard_uncommitted" not in events
