"""变更回执的严格审计 Schema、耐久存储与 Job 失败分类测试。"""

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pytest

from habitus.infrastructure.store.contracts import PathLock
from habitus.infrastructure.vector import (
    VectorStoreBusyError,
    VectorStoreConflictError,
    VectorStoreError,
    VectorStoreIntegrityError,
)
from habitus.memory.conversation import ConversationAddress
from habitus.memory.editor.extraction import (
    MemoryExtractionCapacityError,
    MemoryExtractionError,
    MemoryExtractionPermanentError,
)
from habitus.memory.editor.retrieval import MemoryRetrievalError
from habitus.memory.editor.transaction import MemoryCommitConflictError
from habitus.memory.model import MemoryKind
from habitus.memory.uri import MemoryURI
from habitus.memory.workflow.failure import MemoryJobFailureKind, memory_job_failure_kind
from habitus.memory.workflow.jobs import MemoryJobConfig, MemoryJobError, MemoryJobLeaseLostError, MemoryJobStore
from habitus.memory.workflow.receipt import (
    MemoryChangeReceipt,
    MemoryChangeReceiptError,
    MemoryChangeReceiptState,
    MemoryChangeReceiptStore,
    MemoryChangeSource,
    MemoryNodeChange,
    MemoryNodeChangeAction,
    MemoryPreparedNodeChange,
)
from habitus.model_client import (
    ModelAuthenticationError,
    ModelConfigurationError,
    ModelContentSafetyError,
    ModelInputTooLargeError,
    ModelPermissionError,
    ModelQuotaError,
    ModelRateLimitError,
    ModelResponseError,
    ModelStructuredOutputError,
    ModelTransportError,
)
from habitus.runtime.consistency import MemoryConsistencyService, MemoryConsistencyState
from tests.helpers import BASE_TIME, closed_turn, codec, document, segment
from tests.lock_store import ProcessLocalLockStore


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

    address_receipts = store.list_for_conversation(ConversationAddress("conversation-1", date(2026, 7, 1)))
    assert address_receipts == (first, second)
    assert store.read(first.source) == first
    with pytest.raises(MemoryChangeReceiptError, match="only a COMMITTED"):
        store.discard_committed(first)
    assert store.discard_prepared(first.source)
    assert not store.discard_prepared(first.source)


WAITS = MemoryJobFailureKind.WAITS_FOR_MODEL
RETRY = MemoryJobFailureKind.RETRYABLE
PERMANENT = MemoryJobFailureKind.PERMANENT


def wrapped(outer: BaseException, cause: BaseException) -> BaseException:
    """照检索、抽取的写法把依赖的失败包一层（``raise outer from cause``）。"""

    outer.__cause__ = cause
    return outer


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (TimeoutError("slow"), RETRY),
        (MemoryCommitConflictError("changed"), RETRY),
        (VectorStoreConflictError("generation"), RETRY),
        # 向量库连不上（断网、超时、5xx、索引重建中）：与等模型同一档（裁定 82 第 3 条）
        (VectorStoreBusyError("busy"), WAITS),
        # 检索、抽取把依赖失败包成自己的 RuntimeError 之后按根因归类
        (wrapped(MemoryRetrievalError("memory semantic search failed"), VectorStoreBusyError("down")), WAITS),
        (wrapped(MemoryExtractionError("additional search failed"), VectorStoreBusyError("down")), WAITS),
        (wrapped(MemoryRetrievalError("memory semantic search failed"), ModelTransportError("embedding")), WAITS),
        (
            wrapped(MemoryRetrievalError("memory semantic search failed"), VectorStoreError("401 Unauthorized")),
            PERMANENT,
        ),
        (wrapped(MemoryRetrievalError("memory semantic search failed"), RuntimeError("unknown")), RETRY),
        (wrapped(MemoryRetrievalError("memory semantic search failed"), ValueError("bad hit")), RETRY),
        (wrapped(MemoryExtractionPermanentError("invalid"), ModelTransportError("network")), PERMANENT),
        # 模型此刻用不了：断网 / 5xx、限流、额度或余额、鉴权、权限、服务端答坏 → 等模型恢复（裁定 80 问题 2）
        (ModelTransportError("network"), WAITS),
        (ModelRateLimitError("rate limit"), WAITS),
        (ModelQuotaError("Insufficient Balance"), WAITS),
        (ModelAuthenticationError("invalid key"), WAITS),
        (ModelPermissionError("denied"), WAITS),
        (ModelResponseError("invalid"), WAITS),
        # 答复用完结构纠错仍不合格：Job 层有界重试（M14）
        (ModelStructuredOutputError("model failed json_schema validation after 2 attempt(s)"), RETRY),
        # 同一份输入再试也一样：第一次就 FAILED
        (ModelInputTooLargeError("too large"), PERMANENT),
        (ModelContentSafetyError("blocked"), PERMANENT),
        (ModelConfigurationError("missing dependency"), PERMANENT),
        (MemoryExtractionCapacityError("too large"), PERMANENT),
        (MemoryExtractionPermanentError("invalid candidates"), PERMANENT),
        (MemoryJobLeaseLostError("lost"), PERMANENT),
        (VectorStoreIntegrityError("corrupt"), PERMANENT),
        (MemoryJobError("invalid state"), PERMANENT),
        (ValueError("bad input"), PERMANENT),
        (RuntimeError("unknown runtime fault"), RETRY),
    ],
)
def test_failure_policy_waits_for_the_model_retries_transient_faults_and_stops_on_deterministic_ones(
    error: BaseException,
    kind: MemoryJobFailureKind,
) -> None:
    assert memory_job_failure_kind(error) is kind


def test_committed_receipt_does_not_finish_consistency_before_job_terminal_state(
    tmp_path: Path,
) -> None:
    jobs = MemoryJobStore(
        tmp_path / "workflow",
        PathLock(ProcessLocalLockStore()),
        memory_root=tmp_path / "memory",
        config=MemoryJobConfig(),
    )
    address = ConversationAddress("conversation-1", date(2026, 7, 1))
    source_segment = segment(
        conversation_id=address.conversation_id,
        segment_id="000000000000-000000000001",
        messages=closed_turn(),
    )
    queued = jobs.activate(jobs.stage(address, source_segment))
    lease = jobs.claim(queued, "worker-1")
    receipts = MemoryChangeReceiptStore(tmp_path / "workflow", codec())
    receipt = replace(
        committed_receipt(),
        source=MemoryChangeSource.from_job(lease.job),
    )
    receipts._create(receipt)
    consistency = MemoryConsistencyService(jobs, receipts)

    assert consistency.inspect(queued).state is MemoryConsistencyState.PENDING
    jobs.complete(lease)
    assert consistency.inspect(queued).state is MemoryConsistencyState.COMMITTED
