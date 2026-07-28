"""变更回执的严格审计 Schema、耐久存储与 Job 失败分类测试。"""

from datetime import date, timedelta
from pathlib import Path

import pytest

from infrastructure.vector import (
    VectorStoreBusyError,
    VectorStoreConflictError,
    VectorStoreIntegrityError,
)
from memory.conversation import ConversationAddress
from memory.editor.transaction import MemoryCommitConflictError
from memory.model import MemoryKind
from memory.uri import MemoryURI
from memory.workflow.failure import memory_job_failure_is_retryable
from memory.workflow.jobs import MemoryJobError, MemoryJobLeaseLostError
from memory.workflow.receipt import (
    MemoryChangeReceipt,
    MemoryChangeReceiptError,
    MemoryChangeReceiptState,
    MemoryChangeReceiptStore,
    MemoryChangeSource,
    MemoryNodeChange,
    MemoryNodeChangeAction,
    MemoryPreparedNodeChange,
)
from ModelClient import ModelResponseError, ModelTransportError
from tests.helpers import BASE_TIME, codec, document


def source(*, sequence: int = 1) -> MemoryChangeSource:
    return MemoryChangeSource(
        memory_sequence=sequence,
        transaction_id=f"{sequence:032x}",
        conversation_id="conversation-1",
        started_on=date(2026, 7, 1),
        segment_id=f"segment-{sequence}",
        source_segment_digest=f"{sequence:064x}",
    )


def prepared_receipt(*, sequence: int = 1) -> MemoryChangeReceipt:
    uri = MemoryURI.from_address(document(MemoryKind.PROFILE).address)
    return MemoryChangeReceipt(
        source=source(sequence=sequence),
        state=MemoryChangeReceiptState.PREPARED,
        prepared_at=BASE_TIME,
        committed_at=None,
        expected_created_uris=(uri,),
        expected_updated_uris=(),
        expected_deleted_uris=(),
        unchanged_uris=(),
        prepared_node_changes=(
            MemoryPreparedNodeChange(
                action=MemoryNodeChangeAction.CREATE,
                uri=uri,
                before_digest=None,
                expected_after_digest="b" * 64,
            ),
        ),
        identity_changes=(),
        added_relations=(),
        removed_relations=(),
    )


def committed_receipt(*, sequence: int = 1) -> MemoryChangeReceipt:
    prepared = prepared_receipt(sequence=sequence)
    uri = prepared.expected_created_uris[0]
    return MemoryChangeReceipt(
        source=prepared.source,
        state=MemoryChangeReceiptState.COMMITTED,
        prepared_at=prepared.prepared_at,
        committed_at=BASE_TIME + timedelta(seconds=1),
        expected_created_uris=prepared.expected_created_uris,
        expected_updated_uris=(),
        expected_deleted_uris=(),
        unchanged_uris=(),
        prepared_node_changes=prepared.prepared_node_changes,
        identity_changes=(),
        added_relations=(),
        removed_relations=(),
        node_changes=(
            MemoryNodeChange(
                MemoryNodeChangeAction.CREATE,
                uri,
                None,
                1,
                None,
                "a" * 64,
            ),
        ),
    )


def test_receipt_round_trip_preserves_source_and_actual_node_changes() -> None:
    receipt = committed_receipt()
    restored = MemoryChangeReceipt.from_dict(receipt.to_dict())
    assert restored == receipt
    assert restored.changed_uris == receipt.expected_created_uris
    assert restored.source.receipt_id == receipt.source.receipt_id
    assert restored.SCHEMA_VERSION == "memory_change_receipt_v2"


def test_committed_receipt_rejects_outputs_that_differ_from_prepared_intent() -> None:
    prepared = prepared_receipt()
    with pytest.raises(ValueError, match="do not match"):
        MemoryChangeReceipt(
            source=prepared.source,
            state=MemoryChangeReceiptState.COMMITTED,
            prepared_at=prepared.prepared_at,
            committed_at=BASE_TIME + timedelta(seconds=1),
            expected_created_uris=prepared.expected_created_uris,
            expected_updated_uris=(),
            expected_deleted_uris=(),
            unchanged_uris=(),
            prepared_node_changes=prepared.prepared_node_changes,
            identity_changes=(),
            added_relations=(),
            removed_relations=(),
            node_changes=(),
        )


def test_receipt_store_is_canonical_listed_by_sequence_and_cleanup_checks_state(tmp_path: Path) -> None:
    store = MemoryChangeReceiptStore(tmp_path / "workflow", codec())
    second = prepared_receipt(sequence=2)
    first = prepared_receipt(sequence=1)
    store._create(second)
    store._create(first)

    address_receipts = store.list_for_conversation(
        ConversationAddress("conversation-1", date(2026, 7, 1))
    )
    assert address_receipts == (first, second)
    assert store.read(first.source) == first
    with pytest.raises(MemoryChangeReceiptError, match="only a COMMITTED"):
        store.discard_committed(first)
    assert store.discard_prepared(first.source)
    assert not store.discard_prepared(first.source)


@pytest.mark.parametrize(
    ("error", "retryable"),
    [
        (TimeoutError("slow"), True),
        (MemoryCommitConflictError("changed"), True),
        (VectorStoreBusyError("busy"), True),
        (VectorStoreConflictError("generation"), True),
        (ModelTransportError("network"), True),
        (MemoryJobLeaseLostError("lost"), False),
        (VectorStoreIntegrityError("corrupt"), False),
        (ModelResponseError("invalid"), False),
        (MemoryJobError("invalid state"), False),
        (ValueError("bad input"), False),
        (RuntimeError("unknown runtime fault"), True),
    ],
)
def test_failure_policy_retries_only_transient_or_unknown_runtime_faults(
    error: BaseException,
    retryable: bool,
) -> None:
    assert memory_job_failure_is_retryable(error) is retryable
