"""MemoryJob 规划、执行与完成三阶段的编排和错误边界矩阵。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from habitus.infrastructure.store.contracts import PathLock
from habitus.memory.conversation import ConversationAddress
from habitus.memory.editor import MemoryTransactionJournalState
from habitus.memory.model import MemoryKind
from habitus.memory.recall import (
    MemoryRecallTarget,
)
from habitus.memory.uri import MemoryURI
from habitus.memory.workflow import MemoryChangeReceiptState, MemoryChangeSource, MemoryJobError, MemoryJobStatus
from habitus.memory.workflow.completion import MemoryJobCompletion
from habitus.memory.workflow.execution import MemoryJobCommit
from habitus.pre.conversation import ConversationBatch
from habitus.runtime import build_runtime
from tests.helpers import BASE_TIME, closed_turn, document, segment, segment_summary
from tests.integration.test_memory_job_full_chain import dependencies, runtime_config
from tests.lock_store import ProcessLocalLockStore


def runtime(tmp_path: Path):
    providers, vectors, _chats, _backends = dependencies()
    value = build_runtime(
        runtime_config(tmp_path),
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
    )
    value.initialize()
    return value


def queued_job(value, *, conversation_id: str = "conversation-1"):
    address = ConversationAddress(conversation_id, date(2026, 7, 1))
    value.components.workflow.enqueuer.append_and_maybe_enqueue(
        address,
        ConversationBatch(conversation_id, closed_turn()),
        after_turn=True,
    )
    job = value.components.workflow.enqueuer.flush(address).jobs[0]
    return address, job


def execution_result(tmp_path: Path):
    value = runtime(tmp_path)
    _address, job = queued_job(value)
    lease = value.components.workflow.jobs.claim(job, "matrix-worker")
    commit = asyncio.run(value.components.workflow.runner.executor.execute(lease))
    return value, lease, commit


def completion_result(tmp_path: Path):
    value, lease, commit = execution_result(tmp_path)
    completed = asyncio.run(
        value.components.workflow.runner.committed_finalizer.finalize(
            lease,
            commit.journal,
            summary_generated=commit.summary_generated,
        )
    )
    return value, completed


def test_product_builder_propagates_editor_failure_after_summary_branch_finishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = segment()
    address = ConversationAddress(source.conversation_id, date(2026, 7, 1))
    summary_finished = False
    builder = runtime(tmp_path).components.workflow.runner.executor.segment_products

    async def scenario() -> None:
        nonlocal summary_finished

        async def summarize(_address, current):
            nonlocal summary_finished
            await asyncio.sleep(0)
            summary_finished = True
            return segment_summary(current)

        async def fail_plan(_current, *, turn_context=None):
            await asyncio.sleep(0)
            raise RuntimeError("editor failed")

        monkeypatch.setattr(builder.summary_service, "get_or_create", summarize)
        monkeypatch.setattr(builder.editor, "plan", fail_plan)
        with pytest.raises(RuntimeError, match="editor failed"):
            await builder.build(address, source)

    asyncio.run(scenario())
    assert summary_finished


@pytest.mark.parametrize("clock_value", [BASE_TIME.replace(tzinfo=None)])
def test_executor_clock_must_return_timezone_aware_datetime(tmp_path: Path, clock_value: object) -> None:
    executor = runtime(tmp_path).components.workflow.runner.executor
    executor.clock = lambda: clock_value
    with pytest.raises((TypeError, ValueError), match="clock must return"):
        executor._timestamp()


def test_executor_commits_l2_but_leaves_job_running_for_completion_phase(tmp_path: Path) -> None:
    value, lease, result = execution_result(tmp_path)
    assert isinstance(result, MemoryJobCommit)
    assert result.journal.state is MemoryTransactionJournalState.COMMITTED
    assert result.commit.transaction_id == lease.job.transaction_id
    assert value.components.workflow.jobs.assert_current(lease).status is MemoryJobStatus.RUNNING
    assert result.summary_generated


def test_job_commit_rejects_non_committed_or_foreign_journal(tmp_path: Path) -> None:
    _value, _lease, valid = execution_result(tmp_path)
    with pytest.raises(ValueError, match="COMMITTED"):
        MemoryJobCommit(
            valid.commit,
            replace(valid.journal, state=MemoryTransactionJournalState.PREPARED),
            valid.summary_generated,
        )
    with pytest.raises(ValueError, match="identities differ"):
        MemoryJobCommit(
            valid.commit,
            replace(valid.journal, transaction_id="f" * 32),
            valid.summary_generated,
        )


def test_finalizer_rejects_non_committed_or_foreign_journal(tmp_path: Path) -> None:
    value, lease, commit = execution_result(tmp_path)
    finalizer = value.components.workflow.runner.committed_finalizer
    with pytest.raises(MemoryJobError, match="COMMITTED"):
        asyncio.run(
            finalizer.finalize(
                lease,
                replace(commit.journal, state=MemoryTransactionJournalState.PREPARED),
            )
        )
    with pytest.raises(MemoryJobError, match="does not belong"):
        asyncio.run(finalizer.finalize(lease, replace(commit.journal, transaction_id="f" * 32)))


def test_semantic_refresh_requires_committed_receipt(tmp_path: Path) -> None:
    value, lease, commit = execution_result(tmp_path)
    finalizer = value.components.workflow.runner.committed_finalizer
    receipt = value.components.workflow.receipts.read(MemoryChangeSource.from_job(lease.job))
    assert receipt.state is MemoryChangeReceiptState.PREPARED
    with pytest.raises(MemoryJobError, match="COMMITTED"):
        finalizer._refresh_semantic_layers(receipt)


def test_full_completion_result_is_strict_and_all_durable_steps_are_true(tmp_path: Path) -> None:
    _value, result = completion_result(tmp_path)
    assert isinstance(result, MemoryJobCompletion)
    assert result.job.status is MemoryJobStatus.COMMITTED
    assert result.change_receipt.state is MemoryChangeReceiptState.COMMITTED
    assert result.summary_indexed and result.vector_indexed and result.journal_cleaned


def test_execution_receipt_audits_the_exact_editor_segment(tmp_path: Path) -> None:
    value, lease, _commit = execution_result(tmp_path)
    source = MemoryChangeSource.from_job(lease.job)
    receipt = value.components.workflow.receipts.read(source)
    address = ConversationAddress(lease.job.conversation_id, lease.job.started_on)
    conversations = value.components.workflow.runner.executor.conversations
    trigger = conversations.read_segment(address, lease.job.segment_id)
    editor_segment = None if trigger.ends_mid_turn else conversations.read_editor_turn(address, trigger).whole

    assert editor_segment is not None
    assert receipt.source.editor_segment_id == editor_segment.segment_id
    assert receipt.source.editor_segment_digest == editor_segment.digest


def test_finalizer_forgets_retired_recall_state_and_cleanup_failure_is_non_blocking(tmp_path: Path) -> None:
    value = runtime(tmp_path)
    lifecycle = value.components.memory.search.recall_lifecycle
    item = document(MemoryKind.PREFERENCE)
    uri = MemoryURI.from_address(item.address)
    target = MemoryRecallTarget(uri, item.metadata.revision, item.metadata.created_at)
    lifecycle.record_use((target,), used_at=BASE_TIME)
    finalizer = value.components.workflow.runner.committed_finalizer

    asyncio.run(finalizer._forget_retired_recall_state((uri,)))
    assert lifecycle.store.read_many((uri,)) == ()

    class FailingCleaner:
        def forget(self, uris):
            raise OSError("state store unavailable")

    finalizer.recall_state_cleaner = FailingCleaner()
    asyncio.run(finalizer._forget_retired_recall_state((uri,)))
