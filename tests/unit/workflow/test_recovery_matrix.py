"""STAGED 发布恢复的状态组合矩阵（多文件事务的恢复已随阶段 3 去掉）。"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from habitus.infrastructure.store.contracts import PathLock
from habitus.memory.conversation import ConversationAddress
from habitus.memory.workflow import (
    MemoryJobExecutionError,
    MemoryJobNotReadyError,
    MemoryJobStore,
)
from habitus.memory.workflow.recovery import MemoryStagedJobRecovery
from tests.helpers import BASE_TIME, segment
from tests.lock_store import ProcessLocalLockStore


def staged_job(tmp_path: Path):
    store = MemoryJobStore(
        tmp_path / "workflow",
        PathLock(ProcessLocalLockStore()),
        memory_root=tmp_path / "memory",
    )
    source = segment(segment_id="000000000000-000000000001")
    address = ConversationAddress(source.conversation_id, date(2026, 7, 1))
    return store, store.stage(address, source), source


class FakeJobStore:
    """记录 STAGED 恢复分支，不模拟真实持久化实现。"""

    def __init__(self, *, ready_error: BaseException | None = None) -> None:
        self.ready_error = ready_error
        self.failed_with: BaseException | None = None
        self.activated = False

    def require_staged_ready(self, job):
        if self.ready_error is not None:
            raise self.ready_error
        return job

    def record_staged_failure(self, job, error):
        self.failed_with = error
        return job

    def activate(self, job):
        self.activated = True
        return job


class FakeConversationJournal:
    def __init__(self, source, *, sealed=None, error: BaseException | None = None) -> None:
        self.source = source
        self.sealed = sealed or source
        self.error = error
        self.seal_calls: list[tuple[object, int]] = []
        self.layout = SimpleNamespace(segment_range=lambda _segment_id: (source.start_sequence, source.end_sequence))

    def read_state(self, _address):
        return SimpleNamespace(archived_through_sequence=None)

    def seal(self, address, *, through_sequence, before_history_publish=None):
        self.seal_calls.append((address, through_sequence))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(segment=self.sealed)


def staged_recovery(source, jobs: FakeJobStore, conversations: FakeConversationJournal):
    current = object.__new__(MemoryStagedJobRecovery)
    current.jobs = jobs
    current.conversations = conversations
    return current


def test_non_staged_job_is_returned_without_touching_conversation(tmp_path: Path) -> None:
    store, staged, source = staged_job(tmp_path)
    queued = store.activate(staged)
    jobs = FakeJobStore()
    conversations = FakeConversationJournal(source, error=AssertionError("不应发布"))

    assert staged_recovery(source, jobs, conversations).recover(queued) is queued
    assert conversations.seal_calls == []


def test_not_ready_staged_job_is_propagated_without_recording_failure(tmp_path: Path) -> None:
    _store, job, source = staged_job(tmp_path)
    error = MemoryJobNotReadyError(BASE_TIME + timedelta(minutes=1))
    jobs = FakeJobStore(ready_error=error)
    recovery = staged_recovery(source, jobs, FakeConversationJournal(source))

    with pytest.raises(MemoryJobNotReadyError) as raised:
        recovery.recover(job)
    assert raised.value is error
    assert jobs.failed_with is None


@pytest.mark.parametrize(
    "sealed",
    [
        segment(conversation_id="other", segment_id="000000000000-000000000001"),
        segment(segment_id="000000000010-000000000011"),
    ],
)
def test_recovered_segment_identity_or_digest_mismatch_records_failure(
    tmp_path: Path,
    sealed,
) -> None:
    _store, job, source = staged_job(tmp_path)
    jobs = FakeJobStore()
    recovery = staged_recovery(source, jobs, FakeConversationJournal(source, sealed=sealed))

    with pytest.raises(MemoryJobExecutionError, match="could not publish") as raised:
        recovery.recover(job)
    assert raised.value.job is job
    assert jobs.failed_with is not None
    assert not jobs.activated


def test_conversation_publish_error_is_wrapped_and_recorded(tmp_path: Path) -> None:
    _store, job, source = staged_job(tmp_path)
    jobs = FakeJobStore()
    failure = OSError("disk full")
    recovery = staged_recovery(
        source,
        jobs,
        FakeConversationJournal(source, error=failure),
    )

    with pytest.raises(MemoryJobExecutionError) as raised:
        recovery.recover(job)
    assert raised.value.__cause__ is failure
    assert jobs.failed_with is failure


def test_matching_recovered_segment_activates_staged_job(tmp_path: Path) -> None:
    _store, job, source = staged_job(tmp_path)
    jobs = FakeJobStore()
    conversations = FakeConversationJournal(source)

    assert staged_recovery(source, jobs, conversations).recover(job) is job
    assert jobs.activated
    assert conversations.seal_calls[0][1] == source.end_sequence
