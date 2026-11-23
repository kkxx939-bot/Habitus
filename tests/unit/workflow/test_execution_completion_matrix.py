"""MemoryJob 规划、执行（写卡片、记下"卡片已写完"）与完成三阶段的编排和错误边界矩阵。"""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path

import pytest

from habitus.infrastructure.store.contracts import PathLock
from habitus.memory.conversation import ConversationAddress
from habitus.memory.model import MemoryKind
from habitus.memory.recall import (
    MemoryRecallTarget,
)
from habitus.memory.uri import MemoryURI
from habitus.memory.workflow import MemoryJobError, MemoryJobStatus
from habitus.memory.workflow.execution import MemoryJobWrite
from habitus.pre.conversation import ConversationBatch
from habitus.runtime import build_runtime
from tests.helpers import BASE_TIME, closed_turn, document, segment, segment_summary
from tests.integration.test_memory_job_full_chain import dependencies, runtime_config
from tests.lock_store import ProcessLocalLockStore


def runtime(tmp_path: Path):
    providers, _chats = dependencies()
    value = build_runtime(
        runtime_config(tmp_path),
        providers=providers,
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
    value, lease, _written = execution_result(tmp_path)
    return value, asyncio.run(value.components.workflow.runner.committed_finalizer.finalize(lease))


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


def test_executor_writes_cards_and_records_the_stage_but_leaves_job_running_for_completion(tmp_path: Path) -> None:
    value, lease, result = execution_result(tmp_path)
    assert isinstance(result, MemoryJobWrite)
    assert result.job.cards_written
    assert result.job.changed_uris == tuple(str(uri) for uri in result.commit.written_uris)
    current = value.components.workflow.jobs.assert_current(lease)
    assert current.status is MemoryJobStatus.RUNNING and current.cards_written
    assert result.summary_generated


def test_job_write_requires_a_job_whose_cards_are_written(tmp_path: Path) -> None:
    _value, lease, valid = execution_result(tmp_path)
    with pytest.raises(ValueError, match="cards are written"):
        MemoryJobWrite(valid.commit, lease.job, valid.summary_generated)


def test_finalizer_rejects_a_job_whose_cards_are_not_written(tmp_path: Path) -> None:
    value = runtime(tmp_path)
    _address, job = queued_job(value)
    lease = value.components.workflow.jobs.claim(job, "matrix-worker")
    with pytest.raises(MemoryJobError, match="cards are written"):
        asyncio.run(value.components.workflow.runner.committed_finalizer.finalize(lease))


def test_full_completion_commits_the_job_and_keeps_the_written_addresses(tmp_path: Path) -> None:
    _value, result = completion_result(tmp_path)
    assert result.status is MemoryJobStatus.COMMITTED
    assert result.cards_written


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
