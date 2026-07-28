"""MemoryJob 规划、执行与完成三阶段的编排和错误边界矩阵。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from infrastructure.store.contracts import PathLock
from infrastructure.store.locks import ProcessLocalLockStore
from memory.conversation import ConversationAddress
from memory.editor import MemoryTransactionJournalState
from memory.workflow import MemoryChangeReceiptState, MemoryChangeSource, MemoryJobError, MemoryJobStatus
from memory.workflow.completion import MemoryCommittedJobFinalizer, MemoryJobCompletion
from memory.workflow.execution import MemoryJobCommit, MemoryJobExecutor
from memory.workflow.planning import MemorySegmentProductBuilder, MemorySegmentProducts
from pre.conversation import ConversationBatch
from Runtime import build_runtime
from tests.helpers import BASE_TIME, closed_turn, segment, segment_summary
from tests.integration.test_change_receipt_chain import editor_plan
from tests.integration.test_memory_job_full_chain import dependencies, runtime_config


def runtime(tmp_path: Path):
    providers, vectors, _chats, _backends = dependencies()
    value = build_runtime(
        runtime_config(tmp_path),
        providers=providers,
        vector_stores=vectors,
        path_lock=PathLock(ProcessLocalLockStore()),
        environ={},
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


@pytest.mark.parametrize(
    ("summary", "plan", "message"),
    [
        (object(), editor_plan(), "summary must be"),
        (segment_summary(), object(), "editor_plan must be"),
    ],
)
def test_segment_products_require_both_typed_products(summary: object, plan: object, message: str) -> None:
    with pytest.raises(TypeError, match=message):
        MemorySegmentProducts(summary, plan)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("summary_service", object(), "summary_service must be"),
        ("editor", object(), "editor must be"),
    ],
)
def test_product_builder_constructor_rejects_invalid_dependencies(
    tmp_path: Path,
    field: str,
    invalid: object,
    message: str,
) -> None:
    workflow = runtime(tmp_path).components.workflow
    arguments = {
        "summary_service": workflow.runner.executor.summary_service,
        "editor": workflow.runner.executor.editor,
    }
    arguments[field] = invalid
    with pytest.raises(TypeError, match=message):
        MemorySegmentProductBuilder(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("address", "source", "message"),
    [
        (object(), segment(), "address must be"),
        (ConversationAddress("conversation-1", date(2026, 7, 1)), object(), "segment must be"),
    ],
)
def test_product_builder_rejects_invalid_build_inputs(
    tmp_path: Path,
    address: object,
    source: object,
    message: str,
) -> None:
    builder = runtime(tmp_path).components.workflow.runner.executor.segment_products
    with pytest.raises(TypeError, match=message):
        asyncio.run(builder.build(address, source))  # type: ignore[arg-type]


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

        async def fail_plan(_current):
            await asyncio.sleep(0)
            raise RuntimeError("editor failed")

        monkeypatch.setattr(builder.summary_service, "get_or_create", summarize)
        monkeypatch.setattr(builder.editor, "plan", fail_plan)
        with pytest.raises(RuntimeError, match="editor failed"):
            await builder.build(address, source)

    asyncio.run(scenario())
    assert summary_finished


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("conversations", object(), "conversations must be"),
        ("editor", object(), "editor must be"),
        ("summary_service", object(), "summary_service must be"),
        ("change_receipts", object(), "change_receipts must be"),
        ("jobs", object(), "jobs must be"),
        ("clock", 1, "clock must be callable"),
    ],
)
def test_executor_constructor_rejects_invalid_dependencies(
    tmp_path: Path,
    field: str,
    invalid: object,
    message: str,
) -> None:
    executor = runtime(tmp_path).components.workflow.runner.executor
    arguments = {
        "conversations": executor.conversations,
        "editor": executor.editor,
        "summary_service": executor.summary_service,
        "change_receipts": executor.change_receipts,
        "jobs": executor.jobs,
        "clock": executor.clock,
    }
    arguments[field] = invalid
    with pytest.raises(TypeError, match=message):
        MemoryJobExecutor(**arguments)  # type: ignore[arg-type]


def test_executor_rejects_non_lease(tmp_path: Path) -> None:
    executor = runtime(tmp_path).components.workflow.runner.executor
    with pytest.raises(TypeError, match="lease must be"):
        asyncio.run(executor.execute(object()))  # type: ignore[arg-type]


@pytest.mark.parametrize("clock_value", ["now", BASE_TIME.replace(tzinfo=None)])
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


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("commit", object(), "commit must be"),
        ("journal", object(), "journal must be"),
        ("summary_generated", 1, "summary_generated must be"),
    ],
)
def test_job_commit_rejects_invalid_fields(
    tmp_path: Path,
    field: str,
    invalid: object,
    message: str,
) -> None:
    _value, _lease, valid = execution_result(tmp_path)
    arguments = {name: getattr(valid, name) for name in valid.__dataclass_fields__}
    arguments[field] = invalid
    with pytest.raises((TypeError, ValueError), match=message):
        MemoryJobCommit(**arguments)  # type: ignore[arg-type]


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


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("store", object(), "store must be"),
        ("conversations", object(), "conversations must be"),
        ("summary_service", object(), "summary_service must be"),
        ("summary_vector_index", object(), "summary_vector_index must be"),
        ("change_receipts", object(), "change_receipts must be"),
        ("semantic_refresher", object(), "semantic_refresher must be"),
        ("vector_index", object(), "vector_index must be"),
        ("transaction_recovery", object(), "transaction_recovery must be"),
    ],
)
def test_finalizer_constructor_rejects_invalid_dependencies(
    tmp_path: Path,
    field: str,
    invalid: object,
    message: str,
) -> None:
    finalizer = runtime(tmp_path).components.workflow.runner.committed_finalizer
    arguments = {
        "store": finalizer.store,
        "conversations": finalizer.conversations,
        "summary_service": finalizer.summary_service,
        "summary_vector_index": finalizer.summary_vector_index,
        "change_receipts": finalizer.change_receipts,
        "semantic_refresher": finalizer.semantic_refresher,
        "vector_index": finalizer.vector_index,
        "transaction_recovery": finalizer.transaction_recovery,
    }
    arguments[field] = invalid
    with pytest.raises(TypeError, match=message):
        MemoryCommittedJobFinalizer(**arguments)  # type: ignore[arg-type]


def test_finalizer_rejects_invalid_finalize_inputs(tmp_path: Path) -> None:
    value, lease, commit = execution_result(tmp_path)
    finalizer = value.components.workflow.runner.committed_finalizer
    with pytest.raises(TypeError, match="lease must be"):
        asyncio.run(finalizer.finalize(object(), commit.journal))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="journal must be"):
        asyncio.run(finalizer.finalize(lease, object()))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="summary_generated must be"):
        asyncio.run(finalizer.finalize(lease, commit.journal, summary_generated=1))  # type: ignore[arg-type]


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
    with pytest.raises(TypeError, match="receipt must be"):
        finalizer._refresh_semantic_layers(object())  # type: ignore[arg-type]


def test_full_completion_result_is_strict_and_all_durable_steps_are_true(tmp_path: Path) -> None:
    _value, result = completion_result(tmp_path)
    assert isinstance(result, MemoryJobCompletion)
    assert result.job.status is MemoryJobStatus.COMMITTED
    assert result.change_receipt.state is MemoryChangeReceiptState.COMMITTED
    assert result.summary_indexed and result.vector_indexed and result.journal_cleaned


@pytest.mark.parametrize("field", ["summary_generated", "summary_indexed", "vector_indexed", "journal_cleaned"])
def test_completion_result_rejects_non_boolean_flags(tmp_path: Path, field: str) -> None:
    _value, valid = completion_result(tmp_path)
    arguments = {name: getattr(valid, name) for name in valid.__dataclass_fields__}
    arguments[field] = 1
    with pytest.raises(TypeError, match="must be boolean"):
        MemoryJobCompletion(**arguments)
