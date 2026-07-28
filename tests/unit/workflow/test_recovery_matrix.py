"""STAGED 发布恢复与已提交事务续跑的状态组合矩阵。"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from infrastructure.store.contracts import PathLock
from infrastructure.store.locks import ProcessLocalLockStore
from memory.conversation import ConversationAddress, ConversationMessageJournal
from memory.editor import (
    MemoryCommitTransaction,
    MemoryTransactionJournal,
    MemoryTransactionJournalError,
    MemoryTransactionJournalState,
)
from memory.snapshot import MemorySnapshotReader
from memory.tree import MemoryTree
from memory.workflow import (
    MemoryJobExecutionError,
    MemoryJobNotReadyError,
    MemoryJobStore,
)
from memory.workflow.receipt import MemoryChangeReceiptState, MemoryChangeReceiptStore, MemoryChangeSource
from memory.workflow.recovery import MemoryJobTransactionRecovery, MemoryStagedJobRecovery
from tests.helpers import BASE_TIME, codec, segment


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

    def seal(self, address, *, through_sequence):
        self.seal_calls.append((address, through_sequence))
        if self.error is not None:
            raise self.error
        return SimpleNamespace(segment=self.sealed)


class FakeTransactionJournal:
    def __init__(self, record=None, *, discard_error: BaseException | None = None) -> None:
        self.record = record
        self.discard_error = discard_error
        self.discarded: list[str] = []

    def try_read(self, _transaction_id):
        return self.record

    def discard_terminal(self, transaction_id):
        if self.discard_error is not None:
            raise self.discard_error
        self.discarded.append(transaction_id)


class FakeTransaction:
    def __init__(self, record=None, *, discard_error: BaseException | None = None) -> None:
        self.journal = FakeTransactionJournal(record, discard_error=discard_error)
        self.recovered: list[bool] = []

    def recover_pending(self, *, discard_terminal):
        self.recovered.append(discard_terminal)
        return ("a" * 32,)


class FakeReceiptStore:
    def __init__(self, receipt=None) -> None:
        self.receipt = receipt
        self.discarded: list[MemoryChangeSource] = []

    def try_read(self, _source):
        return self.receipt

    def discard_prepared(self, source):
        self.discarded.append(source)
        return True


def staged_recovery(source, jobs: FakeJobStore, conversations: FakeConversationJournal):
    current = object.__new__(MemoryStagedJobRecovery)
    current.jobs = jobs
    current.conversations = conversations
    return current


def transaction_recovery(transaction: FakeTransaction, receipts: FakeReceiptStore):
    current = object.__new__(MemoryJobTransactionRecovery)
    current.transaction = transaction
    current.change_receipts = receipts
    return current


def test_staged_recovery_constructor_rejects_invalid_collaborator(
    tmp_path: Path,
) -> None:
    path_lock = PathLock(ProcessLocalLockStore())
    conversations = ConversationMessageJournal(tmp_path / "conversation", path_lock)
    jobs = MemoryJobStore(
        tmp_path / "workflow",
        path_lock,
        memory_root=tmp_path / "memory",
    )
    with pytest.raises(TypeError, match="conversations"):
        MemoryStagedJobRecovery(object(), jobs)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="jobs"):
        MemoryStagedJobRecovery(conversations, object())  # type: ignore[arg-type]


def test_non_staged_job_is_returned_without_touching_conversation(tmp_path: Path) -> None:
    store, staged, source = staged_job(tmp_path)
    queued = store.activate(staged)
    jobs = FakeJobStore()
    conversations = FakeConversationJournal(source, error=AssertionError("不应发布"))

    assert staged_recovery(source, jobs, conversations).recover(queued) is queued
    assert conversations.seal_calls == []


@pytest.mark.parametrize("invalid", [None, object(), "job", 1, True])
def test_staged_recovery_requires_memory_job(tmp_path: Path, invalid: object) -> None:
    _store, _job, source = staged_job(tmp_path)
    recovery = staged_recovery(source, FakeJobStore(), FakeConversationJournal(source))
    with pytest.raises(TypeError, match="job"):
        recovery.recover(invalid)  # type: ignore[arg-type]


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


def test_transaction_recovery_constructor_rejects_invalid_collaborator(
    tmp_path: Path,
) -> None:
    document_codec = codec()
    tree = MemoryTree(tmp_path / "memory", document_codec=document_codec)
    transaction = MemoryCommitTransaction(
        tree,
        MemorySnapshotReader(tree),
        PathLock(ProcessLocalLockStore()),
        MemoryTransactionJournal(tmp_path / "workflow" / "transactions", document_codec),
    )
    receipts = MemoryChangeReceiptStore(tmp_path / "workflow", document_codec)
    with pytest.raises(TypeError, match="transaction"):
        MemoryJobTransactionRecovery(object(), receipts)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="change_receipts"):
        MemoryJobTransactionRecovery(transaction, object())  # type: ignore[arg-type]


def test_transaction_recovery_delegates_without_discarding_terminal_journals() -> None:
    transaction = FakeTransaction()
    recovery = transaction_recovery(transaction, FakeReceiptStore())

    assert recovery.recover_pending() == ("a" * 32,)
    assert transaction.recovered == [False]


@pytest.mark.parametrize("invalid", [None, object(), "job", 1, True])
def test_transaction_inspection_requires_memory_job(tmp_path: Path, invalid: object) -> None:
    _store, _job, _source = staged_job(tmp_path)
    recovery = transaction_recovery(FakeTransaction(), FakeReceiptStore())
    with pytest.raises(TypeError, match="job"):
        recovery.inspect(invalid)  # type: ignore[arg-type]


def test_committed_journal_is_returned_for_finalization(tmp_path: Path) -> None:
    _store, job, _source = staged_job(tmp_path)
    record = SimpleNamespace(state=MemoryTransactionJournalState.COMMITTED)
    recovery = transaction_recovery(FakeTransaction(record), FakeReceiptStore())

    assert recovery.inspect(job) is record


def test_rolled_back_journal_discards_prepared_receipt_and_terminal_log(tmp_path: Path) -> None:
    _store, job, _source = staged_job(tmp_path)
    record = SimpleNamespace(state=MemoryTransactionJournalState.ROLLED_BACK)
    transaction = FakeTransaction(record)
    receipts = FakeReceiptStore()
    recovery = transaction_recovery(transaction, receipts)

    assert recovery.inspect(job) is None
    assert receipts.discarded == [MemoryChangeSource.from_job(job)]
    assert transaction.journal.discarded == [job.transaction_id]


def test_missing_journal_with_no_receipt_is_a_clean_restart(tmp_path: Path) -> None:
    _store, job, _source = staged_job(tmp_path)
    recovery = transaction_recovery(FakeTransaction(), FakeReceiptStore())
    assert recovery.inspect(job) is None


def test_missing_journal_discards_orphan_prepared_receipt(tmp_path: Path) -> None:
    _store, job, _source = staged_job(tmp_path)
    receipt = SimpleNamespace(state=MemoryChangeReceiptState.PREPARED)
    receipts = FakeReceiptStore(receipt)
    recovery = transaction_recovery(FakeTransaction(), receipts)

    assert recovery.inspect(job) is None
    assert receipts.discarded == [MemoryChangeSource.from_job(job)]


def test_missing_journal_with_committed_receipt_is_integrity_error(tmp_path: Path) -> None:
    _store, job, _source = staged_job(tmp_path)
    receipt = SimpleNamespace(state=MemoryChangeReceiptState.COMMITTED)
    recovery = transaction_recovery(FakeTransaction(), FakeReceiptStore(receipt))

    with pytest.raises(RuntimeError, match="committed change receipt"):
        recovery.inspect(job)


@pytest.mark.parametrize("invalid", [None, object(), "source", 1, True])
def test_discard_uncommitted_requires_change_source(invalid: object) -> None:
    recovery = transaction_recovery(FakeTransaction(), FakeReceiptStore())
    with pytest.raises(TypeError, match="source"):
        recovery.discard_uncommitted(invalid)  # type: ignore[arg-type]


def test_discard_uncommitted_preserves_committed_transaction_and_receipt(tmp_path: Path) -> None:
    _store, job, _source = staged_job(tmp_path)
    source = MemoryChangeSource.from_job(job)
    record = SimpleNamespace(state=MemoryTransactionJournalState.COMMITTED)
    receipts = FakeReceiptStore(SimpleNamespace(state=MemoryChangeReceiptState.PREPARED))
    recovery = transaction_recovery(FakeTransaction(record), receipts)

    recovery.discard_uncommitted(source)
    assert receipts.discarded == []


@pytest.mark.parametrize(
    "journal_state",
    [None, MemoryTransactionJournalState.PREPARED, MemoryTransactionJournalState.ROLLED_BACK],
)
def test_discard_uncommitted_cleans_only_prepared_receipt_and_rolled_back_log(
    tmp_path: Path,
    journal_state,
) -> None:
    _store, job, _source = staged_job(tmp_path)
    source = MemoryChangeSource.from_job(job)
    record = None if journal_state is None else SimpleNamespace(state=journal_state)
    transaction = FakeTransaction(record)
    receipts = FakeReceiptStore(SimpleNamespace(state=MemoryChangeReceiptState.PREPARED))
    recovery = transaction_recovery(transaction, receipts)

    recovery.discard_uncommitted(source)
    assert receipts.discarded == [source]
    assert transaction.journal.discarded == (
        [job.transaction_id]
        if journal_state is MemoryTransactionJournalState.ROLLED_BACK
        else []
    )


def test_discard_terminal_reports_success_or_journal_error() -> None:
    successful = transaction_recovery(FakeTransaction(), FakeReceiptStore())
    assert successful.discard_terminal("a" * 32) is True

    failed = transaction_recovery(
        FakeTransaction(discard_error=MemoryTransactionJournalError("busy")),
        FakeReceiptStore(),
    )
    assert failed.discard_terminal("b" * 32) is False
