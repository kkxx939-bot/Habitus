"""Conversation、Job、Receipt 联合生命周期的状态组合与安全门矩阵。"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import date, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from memory.conversation import ConversationAddress
from memory.editor import MemoryTransactionJournalState
from memory.workflow import (
    ConversationLifecycleError,
    ConversationLifecycleMaintenanceResult,
    ConversationLifecycleManager,
    MemoryChangeReceipt,
    MemoryChangeReceiptState,
    MemoryChangeSource,
)
from pre.conversation import (
    ConversationSummarySourceKind,
    ConversationSummarySourceRef,
)
from tests.helpers import BASE_TIME, closed_turn, segment
from tests.unit.workflow.test_lifecycle_manager import lifecycle_manager


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("compactor", object(), "compactor must be"),
        ("journal", object(), "journal must be"),
        ("segment_store", object(), "segment_store must be"),
        ("range_store", object(), "range_store must be"),
        ("summary_vector_index", object(), "summary_vector_index must be"),
        ("jobs", object(), "jobs must be"),
        ("receipts", object(), "receipts must be"),
        ("transaction_journal", object(), "transaction_journal must be"),
        ("summary_config", object(), "summary_config must be"),
        ("workflow_config", object(), "workflow_config must be"),
    ],
)
def test_constructor_rejects_invalid_collaborator(
    tmp_path: Path,
    field: str,
    invalid: object,
    message: str,
) -> None:
    manager, _calls = lifecycle_manager(tmp_path, compaction_enabled=False)
    arguments = {
        "compactor": manager.compactor,
        "journal": manager.journal,
        "segment_store": manager.segment_store,
        "range_store": manager.range_store,
        "summary_vector_index": manager.summary_vector_index,
        "jobs": manager.jobs,
        "receipts": manager.receipts,
        "transaction_journal": manager.transaction_journal,
        "summary_config": manager.summary_config,
        "workflow_config": manager.workflow_config,
    }
    arguments[field] = invalid
    with pytest.raises(TypeError, match=message):
        ConversationLifecycleManager(**arguments)  # type: ignore[arg-type]


def test_constructor_rejects_summary_config_different_from_compactor(tmp_path: Path) -> None:
    manager, _calls = lifecycle_manager(tmp_path, compaction_enabled=False)
    with pytest.raises(ValueError, match="share one lifecycle config"):
        ConversationLifecycleManager(
            manager.compactor,
            manager.journal,
            manager.segment_store,
            manager.range_store,
            manager.summary_vector_index,
            manager.jobs,
            manager.receipts,
            manager.transaction_journal,
            summary_config=replace(manager.summary_config, cleanup_batch_size=1),
        )


@pytest.mark.parametrize("now", ["2026-07-01", BASE_TIME.replace(tzinfo=None)])
def test_maintain_once_rejects_invalid_explicit_time(tmp_path: Path, now: object) -> None:
    manager, _calls = lifecycle_manager(tmp_path, compaction_enabled=False)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    with pytest.raises((TypeError, ValueError), match="lifecycle now"):
        asyncio.run(manager.maintain_once(address, now=now))  # type: ignore[arg-type]


def test_maintain_once_rejects_invalid_conversation_address(tmp_path: Path) -> None:
    manager, _calls = lifecycle_manager(tmp_path, compaction_enabled=False)
    with pytest.raises(TypeError, match="address must be"):
        asyncio.run(manager.maintain_once(None, now=BASE_TIME))  # type: ignore[arg-type]


def test_maintain_once_normalizes_offset_time_to_utc(tmp_path: Path) -> None:
    manager, _calls = lifecycle_manager(tmp_path, compaction_enabled=False)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    offset = BASE_TIME.astimezone(timezone(timedelta(hours=8)))
    result = asyncio.run(manager.maintain_once(address, now=offset))
    assert result.summary_indexed


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("compaction", object(), "compaction must be"),
        ("summary_indexed", 1, "summary_indexed must be"),
        ("purged_history_segment_ids", ["id"], "must contain"),
        ("released_history_segment_ids", ("",), "must contain"),
        ("deleted_segment_summary_ids", (1,), "must contain"),
        ("deleted_range_summary_ids", None, "must contain"),
        ("deleted_memory_receipt_ids", ("",), "must contain"),
        ("deleted_memory_job_sequences", (0,), "positive integers"),
        ("deleted_memory_job_sequences", (True,), "positive integers"),
    ],
)
def test_maintenance_result_rejects_invalid_fields(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    manager, _calls = lifecycle_manager(tmp_path, compaction_enabled=False)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    valid = asyncio.run(manager.maintain_once(address, now=BASE_TIME))
    arguments = {name: getattr(valid, name) for name in valid.__dataclass_fields__}
    arguments[field] = value
    with pytest.raises(TypeError, match=message):
        ConversationLifecycleMaintenanceResult(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("segment_id", "released_through", "expected"),
    [
        ("000000000000-000000000001", -1, False),
        ("000000000000-000000000001", 0, False),
        ("000000000000-000000000001", 1, True),
        ("000000000002-000000000003", 2, False),
        ("000000000002-000000000003", 3, True),
    ],
)
def test_history_release_check_uses_segment_end_sequence(
    segment_id: str,
    released_through: int,
    expected: bool,
) -> None:
    assert ConversationLifecycleManager._source_history_is_released(segment_id, released_through) is expected


def _source_ref(
    start: int,
    end: int,
    *,
    kind: ConversationSummarySourceKind = ConversationSummarySourceKind.SEGMENT,
) -> ConversationSummarySourceRef:
    return ConversationSummarySourceRef(
        kind,
        f"{start:012d}-{end:012d}",
        f"{start + 1:064x}",
        start,
        end,
    )


def test_parent_map_indexes_each_source_to_its_parent() -> None:
    first = _source_ref(0, 1)
    second = _source_ref(2, 3)
    parent = SimpleNamespace(source_refs=(first, second))
    result = ConversationLifecycleManager._parent_map(
        (parent,),  # type: ignore[arg-type]
        expected_kind=ConversationSummarySourceKind.SEGMENT,
    )
    assert result == {first.summary_id: (parent, first), second.summary_id: (parent, second)}


def test_parent_map_rejects_wrong_source_kind() -> None:
    parent = SimpleNamespace(source_refs=(_source_ref(0, 1, kind=ConversationSummarySourceKind.RANGE),))
    with pytest.raises(ConversationLifecycleError, match="invalid source kind"):
        ConversationLifecycleManager._parent_map(
            (parent,),  # type: ignore[arg-type]
            expected_kind=ConversationSummarySourceKind.SEGMENT,
        )


def test_parent_map_rejects_source_covered_by_multiple_parents() -> None:
    reference = _source_ref(0, 1)
    first = SimpleNamespace(source_refs=(reference,))
    second = SimpleNamespace(source_refs=(reference,))
    with pytest.raises(ConversationLifecycleError, match="multiple parents"):
        ConversationLifecycleManager._parent_map(
            (first, second),  # type: ignore[arg-type]
            expected_kind=ConversationSummarySourceKind.SEGMENT,
        )


def _committed_workflow(manager: ConversationLifecycleManager, address: ConversationAddress):
    source_segment = segment(
        conversation_id=address.conversation_id,
        segment_id="000000000000-000000000001",
        messages=closed_turn(),
    )
    queued = manager.jobs.activate(manager.jobs.stage(address, source_segment))
    job = manager.jobs.complete(manager.jobs.claim(queued, "worker"))
    change_source = MemoryChangeSource.from_job(job)
    receipt = MemoryChangeReceipt(
        source=change_source,
        state=MemoryChangeReceiptState.COMMITTED,
        prepared_at=BASE_TIME,
        committed_at=BASE_TIME,
        expected_created_uris=(),
        expected_updated_uris=(),
        expected_deleted_uris=(),
        unchanged_uris=(),
        prepared_node_changes=(),
        identity_changes=(),
        added_relations=(),
        removed_relations=(),
        node_changes=(),
    )
    manager.receipts._create(receipt)
    return source_segment, job, receipt


def test_workflow_commit_gate_accepts_committed_job_and_receipt_without_recovery_log(tmp_path: Path) -> None:
    manager, _calls = lifecycle_manager(tmp_path, compaction_enabled=False)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    source_segment, _job, _receipt = _committed_workflow(manager, address)
    assert manager._workflow_is_committed(address, source_segment)


def test_workflow_commit_gate_rejects_missing_job(tmp_path: Path) -> None:
    manager, _calls = lifecycle_manager(tmp_path, compaction_enabled=False)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    source_segment = segment(segment_id="000000000000-000000000001")
    with pytest.raises(ConversationLifecycleError, match="no durable MemoryJob"):
        manager._workflow_is_committed(address, source_segment)


def test_workflow_commit_gate_returns_false_for_uncommitted_job(tmp_path: Path) -> None:
    manager, _calls = lifecycle_manager(tmp_path, compaction_enabled=False)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    source_segment = segment(segment_id="000000000000-000000000001")
    manager.jobs.activate(manager.jobs.stage(address, source_segment))
    assert not manager._workflow_is_committed(address, source_segment)


def test_workflow_commit_gate_rejects_missing_receipt(tmp_path: Path) -> None:
    manager, _calls = lifecycle_manager(tmp_path, compaction_enabled=False)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    source_segment = segment(segment_id="000000000000-000000000001")
    queued = manager.jobs.activate(manager.jobs.stage(address, source_segment))
    manager.jobs.complete(manager.jobs.claim(queued, "worker"))
    with pytest.raises(ConversationLifecycleError, match="no durable change receipt"):
        manager._workflow_is_committed(address, source_segment)


@pytest.mark.parametrize(
    ("state", "expected", "error"),
    [
        (MemoryTransactionJournalState.PREPARED, False, None),
        (MemoryTransactionJournalState.COMMITTED, True, None),
        (MemoryTransactionJournalState.ROLLED_BACK, None, "rolled-back"),
    ],
)
def test_workflow_commit_gate_interprets_transaction_terminal_state(
    tmp_path: Path,
    state: MemoryTransactionJournalState,
    expected: bool | None,
    error: str | None,
) -> None:
    manager, _calls = lifecycle_manager(tmp_path, compaction_enabled=False)
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    source_segment, job, _receipt = _committed_workflow(manager, address)
    record = manager.transaction_journal._parse(
        {
            "schema": "memory_commit_transaction_v1",
            "transaction_id": job.transaction_id,
            "state": state.value,
            "created_at": "2026-07-01T08:00:00.000000Z",
            "updated_at": "2026-07-01T08:00:00.000000Z",
            "lock_identities": [],
            "entries": [],
        }
    )
    manager.transaction_journal.prepare(replace(record, state=MemoryTransactionJournalState.PREPARED))
    if state is not MemoryTransactionJournalState.PREPARED:
        manager.transaction_journal.mark(job.transaction_id, state, timestamp=BASE_TIME)
    if error is not None:
        with pytest.raises(ConversationLifecycleError, match=error):
            manager._workflow_is_committed(address, source_segment)
    else:
        assert manager._workflow_is_committed(address, source_segment) is expected


@pytest.mark.parametrize(
    ("state", "error"),
    [
        (MemoryTransactionJournalState.PREPARED, "PREPARED"),
        (MemoryTransactionJournalState.ROLLED_BACK, "rolled-back"),
    ],
)
def test_discard_terminal_journal_rejects_non_committed_states(
    tmp_path: Path,
    state: MemoryTransactionJournalState,
    error: str,
) -> None:
    manager, _calls = lifecycle_manager(tmp_path, compaction_enabled=False)
    transaction_id = "a" * 32
    record = manager.transaction_journal._parse(
        {
            "schema": "memory_commit_transaction_v1",
            "transaction_id": transaction_id,
            "state": "prepared",
            "created_at": "2026-07-01T08:00:00.000000Z",
            "updated_at": "2026-07-01T08:00:00.000000Z",
            "lock_identities": [],
            "entries": [],
        }
    )
    manager.transaction_journal.prepare(record)
    if state is MemoryTransactionJournalState.ROLLED_BACK:
        manager.transaction_journal.mark(transaction_id, state, timestamp=BASE_TIME)
    with pytest.raises(ConversationLifecycleError, match=error):
        manager._discard_terminal_journal(transaction_id)


def test_discard_terminal_journal_is_noop_when_absent_and_deletes_committed(tmp_path: Path) -> None:
    manager, _calls = lifecycle_manager(tmp_path, compaction_enabled=False)
    manager._discard_terminal_journal("b" * 32)
    transaction_id = "a" * 32
    record = manager.transaction_journal._parse(
        {
            "schema": "memory_commit_transaction_v1",
            "transaction_id": transaction_id,
            "state": "prepared",
            "created_at": "2026-07-01T08:00:00.000000Z",
            "updated_at": "2026-07-01T08:00:00.000000Z",
            "lock_identities": [],
            "entries": [],
        }
    )
    manager.transaction_journal.prepare(record)
    manager.transaction_journal.mark(transaction_id, MemoryTransactionJournalState.COMMITTED, timestamp=BASE_TIME)
    manager._discard_terminal_journal(transaction_id)
    assert manager.transaction_journal.try_read(transaction_id) is None
