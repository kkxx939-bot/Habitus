"""耐久 Job、lease、变更来源、节点变更与 Receipt 状态机契约矩阵。"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta

import pytest

from habitus.memory.document import MemoryLinkType, MemoryStoredLink
from habitus.memory.editor import MemoryIdentityProposalBasis, MemoryNodeDisposition
from habitus.memory.model import MemoryAddress, MemoryKind
from habitus.memory.uri import MemoryURI
from habitus.memory.workflow.jobs import (
    MemoryJob,
    MemoryJobConfig,
    MemoryJobExecutionError,
    MemoryJobLease,
    MemoryJobNotReadyError,
    MemoryJobStatus,
)
from habitus.memory.workflow.jobs.model import MEMORY_JOB_ERROR_MAX_CHARS
from habitus.memory.workflow.receipt import (
    MemoryChangeReceipt,
    MemoryChangeReceiptError,
    MemoryChangeReceiptState,
    MemoryChangeSource,
    MemoryIdentityChange,
    MemoryNodeChange,
    MemoryNodeChangeAction,
    MemoryPreparedNodeChange,
)
from tests.helpers import BASE_TIME, segment

DIGEST = "a" * 64
TRANSACTION_ID = "b" * 32
CLAIM_ID = "c" * 32


def _job(status: MemoryJobStatus, **overrides: object) -> MemoryJob:
    running = status is MemoryJobStatus.RUNNING
    values: dict[str, object] = {
        "memory_sequence": 1,
        "conversation_id": "conversation-1",
        "started_on": date(2026, 7, 1),
        "segment_id": "000000000000-000000000001",
        "source_segment_digest": DIGEST,
        "transaction_id": TRANSACTION_ID,
        "status": status,
        "attempts": 1 if status is MemoryJobStatus.FAILED else 0,
        "claim_id": CLAIM_ID if running else None,
        "claim_generation": 1 if running else 0,
        "worker_id": "worker-1" if running else None,
        "lease_expires_at": BASE_TIME + timedelta(minutes=2) if running else None,
        "next_attempt_at": None,
        "last_error": "failed" if status is MemoryJobStatus.FAILED else None,
        "created_at": BASE_TIME,
        "updated_at": BASE_TIME,
    }
    values.update(overrides)
    return MemoryJob(**values)  # type: ignore[arg-type]


def _source(**overrides: object) -> MemoryChangeSource:
    values: dict[str, object] = {
        "memory_sequence": 1,
        "transaction_id": TRANSACTION_ID,
        "conversation_id": "conversation-1",
        "started_on": date(2026, 7, 1),
        "segment_id": "000000000000-000000000001",
        "source_segment_digest": segment(segment_id="000000000000-000000000001").digest,
    }
    values.update(overrides)
    return MemoryChangeSource(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("attempts", range(1, 13))
@pytest.mark.parametrize(("base", "maximum"), [(0.1, 1.0), (1.0, 60.0), (2, 5), (10, 10)])
def test_job_retry_backoff_is_exponential_and_capped(attempts: int, base: float, maximum: float) -> None:
    config = MemoryJobConfig(retry_base_delay_seconds=base, retry_max_delay_seconds=maximum)
    assert config.retry_delay_seconds(attempts) == min(maximum, base * (2 ** (attempts - 1)))


@pytest.mark.parametrize("attempts", [0, -1, True, False, 1.0, "1", None, [], {}])
def test_job_retry_backoff_rejects_invalid_attempt_count(attempts: object) -> None:
    with pytest.raises(ValueError):
        MemoryJobConfig().retry_delay_seconds(attempts)  # type: ignore[arg-type]


@pytest.mark.parametrize("status", tuple(MemoryJobStatus))
def test_job_accepts_each_coherent_durable_status(status: MemoryJobStatus) -> None:
    value = _job(status)
    assert value.status is status
    assert value.source_identity == (
        value.conversation_id,
        value.started_on,
        value.segment_id,
        value.source_segment_digest,
    )


@pytest.mark.parametrize("status", tuple(MemoryJobStatus))
@pytest.mark.parametrize("memory_sequence", [0, -1, True, False, 1.0, "1", None])
def test_job_rejects_invalid_global_sequence_for_every_status(status: MemoryJobStatus, memory_sequence: object) -> None:
    with pytest.raises(ValueError):
        _job(status, memory_sequence=memory_sequence)


@pytest.mark.parametrize("field", ["conversation_id", "segment_id"])
@pytest.mark.parametrize("value", ["", " ", " leading", "trailing ", None, 1, True])
def test_job_rejects_non_normalized_source_identity_text(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        _job(MemoryJobStatus.STAGED, **{field: value})


@pytest.mark.parametrize("value", [datetime(2026, 7, 1), "2026-07-01", None, 1, True])
def test_job_started_on_requires_calendar_date_not_datetime(value: object) -> None:
    with pytest.raises(ValueError):
        _job(MemoryJobStatus.STAGED, started_on=value)


@pytest.mark.parametrize("field", ["source_segment_digest", "transaction_id"])
@pytest.mark.parametrize("value", ["", "0", "G" * 64, "A" * 64, None, 1, True])
def test_job_rejects_invalid_lowercase_hex_identities(field: str, value: object) -> None:
    if field == "transaction_id" and isinstance(value, str) and len(value) == 64:
        value = value[:32]
    with pytest.raises(ValueError):
        _job(MemoryJobStatus.STAGED, **{field: value})


@pytest.mark.parametrize("attempts", [-1, True, False, 1.0, "1", None])
def test_job_rejects_invalid_attempt_state(attempts: object) -> None:
    with pytest.raises(ValueError):
        _job(MemoryJobStatus.QUEUED, attempts=attempts)


@pytest.mark.parametrize("generation", [-1, True, False, 1.0, "1", None])
def test_job_rejects_invalid_claim_generation(generation: object) -> None:
    with pytest.raises(ValueError):
        _job(MemoryJobStatus.QUEUED, claim_generation=generation)


@pytest.mark.parametrize("status", [status for status in MemoryJobStatus if status is not MemoryJobStatus.RUNNING])
@pytest.mark.parametrize(
    ("field", "value"),
    [("claim_id", CLAIM_ID), ("worker_id", "worker-1"), ("lease_expires_at", BASE_TIME + timedelta(minutes=1))],
)
def test_non_running_job_rejects_any_lease_ownership(status: MemoryJobStatus, field: str, value: object) -> None:
    with pytest.raises(ValueError, match="non-running"):
        _job(status, **{field: value})


@pytest.mark.parametrize("field", ["claim_id", "worker_id", "lease_expires_at"])
@pytest.mark.parametrize("value", [None, "", "invalid worker id!", 0, True])
def test_running_job_requires_complete_normalized_lease_identity(field: str, value: object) -> None:
    kwargs = {field: value}
    if field == "lease_expires_at" and value == "invalid worker id!":
        kwargs[field] = "not-time"
    with pytest.raises(ValueError):
        _job(MemoryJobStatus.RUNNING, **kwargs)


@pytest.mark.parametrize("worker_id", ["w", "worker-1", "worker.one", "worker:one", "A_1", "x" * 128])
def test_worker_id_accepts_exact_operational_character_set(worker_id: str) -> None:
    assert MemoryJob.valid_worker_id(worker_id)
    assert _job(MemoryJobStatus.RUNNING, worker_id=worker_id).worker_id == worker_id


@pytest.mark.parametrize("worker_id", ["", " worker", "worker ", "worker/1", "worker@1", "x" * 129, None, 1, True])
def test_worker_id_rejects_unsafe_or_oversized_values(worker_id: object) -> None:
    assert MemoryJob.valid_worker_id(worker_id) is False
    with pytest.raises(ValueError):
        _job(MemoryJobStatus.RUNNING, worker_id=worker_id)


@pytest.mark.parametrize("status", [MemoryJobStatus.STAGED, MemoryJobStatus.QUEUED])
def test_only_staged_or_queued_jobs_may_carry_retry_schedule(status: MemoryJobStatus) -> None:
    values = {"next_attempt_at": BASE_TIME + timedelta(seconds=1), "attempts": 1, "last_error": "retry"}
    assert _job(status, **values).next_attempt_at is not None
    for other in (MemoryJobStatus.RUNNING, MemoryJobStatus.FAILED, MemoryJobStatus.COMMITTED):
        with pytest.raises(ValueError):
            _job(other, next_attempt_at=BASE_TIME + timedelta(seconds=1))


def test_new_staged_job_cannot_contain_retry_state_but_retried_stage_requires_error() -> None:
    with pytest.raises(ValueError, match="new staged"):
        _job(MemoryJobStatus.STAGED, next_attempt_at=BASE_TIME + timedelta(seconds=1))
    with pytest.raises(ValueError, match="requires its last error"):
        _job(MemoryJobStatus.STAGED, attempts=1)
    assert _job(MemoryJobStatus.STAGED, attempts=1, last_error="retry").last_error == "retry"


@pytest.mark.parametrize("last_error", ["", None, 1, True, "x" * (MEMORY_JOB_ERROR_MAX_CHARS + 1)])
def test_job_rejects_invalid_error_when_failure_state_requires_it(last_error: object) -> None:
    with pytest.raises(ValueError):
        _job(MemoryJobStatus.FAILED, last_error=last_error)


@pytest.mark.parametrize("field", ["created_at", "updated_at"])
@pytest.mark.parametrize("value", [None, "time", 1, date(2026, 7, 1), datetime(2026, 7, 1)])
def test_job_rejects_missing_non_datetime_or_naive_system_timestamps(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        _job(MemoryJobStatus.STAGED, **{field: value})


def test_job_rejects_updated_time_before_creation() -> None:
    with pytest.raises(ValueError, match="precede"):
        _job(MemoryJobStatus.STAGED, updated_at=BASE_TIME - timedelta(seconds=1))


def test_job_lease_exposes_only_running_fencing_identity() -> None:
    running = _job(MemoryJobStatus.RUNNING)
    lease = MemoryJobLease(running)
    assert lease.claim_id == CLAIM_ID
    assert lease.claim_generation == 1
    assert lease.worker_id == "worker-1"
    assert lease.lease_expires_at == running.lease_expires_at
    assert lease.source_identity == running.source_identity
    for status in MemoryJobStatus:
        if status is MemoryJobStatus.RUNNING:
            continue
        with pytest.raises(ValueError, match="RUNNING"):
            MemoryJobLease(_job(status))


@pytest.mark.parametrize("value", [None, {}, [], "job", 1, True])
def test_job_lease_rejects_non_job_values(value: object) -> None:
    with pytest.raises(TypeError):
        MemoryJobLease(value)  # type: ignore[arg-type]


@pytest.mark.parametrize("available_at", [BASE_TIME, BASE_TIME + timedelta(days=1)])
def test_not_ready_error_normalizes_and_exposes_available_time(available_at: datetime) -> None:
    error = MemoryJobNotReadyError(available_at)
    assert error.available_at == available_at
    assert available_at.isoformat() in str(error)


@pytest.mark.parametrize("available_at", [None, "time", date(2026, 7, 1), datetime(2026, 7, 1)])
def test_not_ready_error_requires_timezone_aware_datetime(available_at: object) -> None:
    with pytest.raises(TypeError):
        MemoryJobNotReadyError(available_at)  # type: ignore[arg-type]


@pytest.mark.parametrize("job", [None, _job(MemoryJobStatus.FAILED)])
def test_job_execution_error_optionally_carries_failed_job(job: MemoryJob | None) -> None:
    error = MemoryJobExecutionError("failed", job=job)
    assert error.job is job


@pytest.mark.parametrize("sequence", [1, 2, 2**31])
def test_change_source_round_trip_and_receipt_identity_are_deterministic(sequence: int) -> None:
    source = _source(memory_sequence=sequence, transaction_id=f"{sequence:032x}")
    restored = MemoryChangeSource.from_dict(source.to_dict())
    assert restored == source
    assert len(source.receipt_id) == 64
    assert source.receipt_id == restored.receipt_id


def test_change_source_binds_and_round_trips_the_complete_editor_segment() -> None:
    editor_segment = segment(segment_id="000000000000-000000000001")
    job = _job(
        MemoryJobStatus.STAGED,
        source_segment_digest=editor_segment.digest,
    )
    source = MemoryChangeSource.from_job(job, editor_segment=editor_segment)

    assert source.editor_segment_id == editor_segment.segment_id
    assert source.editor_segment_digest == editor_segment.digest
    restored = MemoryChangeSource.from_dict(source.to_dict())
    assert restored == source
    assert restored.to_dict() == source.to_dict()

    mismatched = segment(
        conversation_id="another-conversation",
        segment_id=editor_segment.segment_id,
    )
    with pytest.raises(ValueError, match="another conversation"):
        MemoryChangeSource.from_job(job, editor_segment=mismatched)


@pytest.mark.parametrize("field", ["editor_segment_id", "editor_segment_digest"])
def test_change_source_rejects_partial_editor_segment_provenance(field: str) -> None:
    value = _source().to_dict()
    value[field] = "a" * 64
    with pytest.raises(ValueError, match="invalid shape|present together"):
        MemoryChangeSource.from_dict(value)


@pytest.mark.parametrize(
    "field",
    ["memory_sequence", "transaction_id", "conversation_id", "started_on", "segment_id", "source_segment_digest"],
)
def test_change_source_parser_rejects_each_missing_field(field: str) -> None:
    value = _source().to_dict()
    value.pop(field)
    with pytest.raises(ValueError):
        MemoryChangeSource.from_dict(value)


@pytest.mark.parametrize("unknown", ["uri", "owner", "revision", "job_id", "evidence"])
def test_change_source_parser_rejects_unknown_fields(unknown: str) -> None:
    value = _source().to_dict()
    value[unknown] = "forbidden"
    with pytest.raises(ValueError):
        MemoryChangeSource.from_dict(value)


def test_change_source_requires_exact_segment_identity_and_digest() -> None:
    source_segment = segment(segment_id="000000000000-000000000001")
    source = _source(source_segment_digest=source_segment.digest)
    source.require_segment(source_segment)
    for wrong in (
        segment(conversation_id="other", segment_id=source_segment.segment_id),
        segment(segment_id="000000000002-000000000003"),
    ):
        with pytest.raises(MemoryChangeReceiptError, match="does not match"):
            source.require_segment(wrong)


@pytest.mark.parametrize("action", [MemoryNodeDisposition.MERGE, MemoryNodeDisposition.DELETE])
def test_identity_change_round_trips_merge_and_delete(action: MemoryNodeDisposition) -> None:
    source_uri = MemoryURI.from_address(MemoryAddress.preference("旧主题"))
    if action is MemoryNodeDisposition.MERGE:
        value = MemoryIdentityChange(
            action,
            source_uri,
            MemoryURI.from_address(MemoryAddress.preference("新主题")),
            MemoryIdentityProposalBasis.DUPLICATE_IDENTITY,
        )
    else:
        value = MemoryIdentityChange(
            action,
            source_uri,
            None,
            MemoryIdentityProposalBasis.EXPLICIT_FORGET,
        )
    assert MemoryIdentityChange.from_dict(value.to_dict()) == value


@pytest.mark.parametrize(
    "action", [MemoryNodeDisposition.CREATE, MemoryNodeDisposition.UPDATE, MemoryNodeDisposition.NOOP]
)
def test_identity_change_rejects_non_retirement_dispositions(action: MemoryNodeDisposition) -> None:
    with pytest.raises(ValueError, match="merge or delete"):
        MemoryIdentityChange(
            action,
            MemoryURI.from_address(MemoryAddress.preference("旧主题")),
            None,
            MemoryIdentityProposalBasis.EXPLICIT_FORGET,
        )


@pytest.mark.parametrize("action", tuple(MemoryNodeChangeAction))
@pytest.mark.parametrize("kind", tuple(MemoryKind))
def test_prepared_node_change_accepts_coherent_action_for_every_memory_kind(
    action: MemoryNodeChangeAction,
    kind: MemoryKind,
) -> None:
    uri = MemoryURI.from_address(
        {
            MemoryKind.PROFILE: MemoryAddress.profile(),
            MemoryKind.PREFERENCE: MemoryAddress.preference("主题"),
            MemoryKind.ENTITY: MemoryAddress.entity("分类", "实体"),
            MemoryKind.TOOL: MemoryAddress.tool("tool.name"),
            MemoryKind.EVENT: MemoryAddress.event(date(2026, 7, 1), "事件"),
            MemoryKind.INTENTION: MemoryAddress.intention("事项"),
        }[kind]
    )
    confirms = kind is MemoryKind.INTENTION and action is MemoryNodeChangeAction.CREATE
    values = {
        MemoryNodeChangeAction.CREATE: (None, "a" * 64),
        MemoryNodeChangeAction.UPDATE: ("a" * 64, "b" * 64),
        MemoryNodeChangeAction.DELETE: ("a" * 64, None),
    }[action]
    prepared = MemoryPreparedNodeChange(action, uri, *values, confirms_intention=confirms)
    assert MemoryPreparedNodeChange.from_dict(prepared.to_dict()) == prepared


@pytest.mark.parametrize("action", tuple(MemoryNodeChangeAction))
@pytest.mark.parametrize(
    ("before", "after"),
    [(None, None), ("a" * 64, None), (None, "b" * 64), ("a" * 64, "b" * 64)],
)
def test_prepared_node_change_enforces_action_specific_before_after_shape(
    action: MemoryNodeChangeAction,
    before: str | None,
    after: str | None,
) -> None:
    valid = {
        MemoryNodeChangeAction.CREATE: before is None and after is not None,
        MemoryNodeChangeAction.UPDATE: before is not None and after is not None,
        MemoryNodeChangeAction.DELETE: before is not None and after is None,
    }[action]
    uri = MemoryURI.from_address(MemoryAddress.profile())
    if valid:
        MemoryPreparedNodeChange(action, uri, before, after)
    else:
        with pytest.raises(ValueError):
            MemoryPreparedNodeChange(action, uri, before, after)


@pytest.mark.parametrize("action", tuple(MemoryNodeChangeAction))
@pytest.mark.parametrize(
    ("before_revision", "after_revision", "before_digest", "after_digest"),
    [
        (None, 1, None, "b" * 64),
        (1, 2, "a" * 64, "b" * 64),
        (1, None, "a" * 64, None),
    ],
)
def test_committed_node_change_accepts_only_action_matching_physical_state(
    action: MemoryNodeChangeAction,
    before_revision: int | None,
    after_revision: int | None,
    before_digest: str | None,
    after_digest: str | None,
) -> None:
    expected_action = {
        (None, 1, None, "b" * 64): MemoryNodeChangeAction.CREATE,
        (1, 2, "a" * 64, "b" * 64): MemoryNodeChangeAction.UPDATE,
        (1, None, "a" * 64, None): MemoryNodeChangeAction.DELETE,
    }[(before_revision, after_revision, before_digest, after_digest)]
    uri = MemoryURI.from_address(MemoryAddress.profile())
    if action is expected_action:
        value = MemoryNodeChange(action, uri, before_revision, after_revision, before_digest, after_digest)
        assert MemoryNodeChange.from_dict(value.to_dict()) == value
    else:
        with pytest.raises(ValueError):
            MemoryNodeChange(action, uri, before_revision, after_revision, before_digest, after_digest)


@pytest.mark.parametrize("field", ["before_revision", "after_revision"])
@pytest.mark.parametrize("value", [0, -1, True, False, 1.0, "1"])
def test_node_change_rejects_invalid_non_null_revision(field: str, value: object) -> None:
    kwargs = {
        "action": MemoryNodeChangeAction.UPDATE,
        "uri": MemoryURI.from_address(MemoryAddress.profile()),
        "before_revision": 1,
        "after_revision": 2,
        "before_digest": "a" * 64,
        "after_digest": "b" * 64,
    }
    kwargs[field] = value
    with pytest.raises(ValueError):
        MemoryNodeChange(**kwargs)  # type: ignore[arg-type]


def _prepared_receipt(**overrides: object) -> MemoryChangeReceipt:
    uri = MemoryURI.from_address(MemoryAddress.profile())
    prepared = MemoryPreparedNodeChange(MemoryNodeChangeAction.CREATE, uri, None, "b" * 64)
    values: dict[str, object] = {
        "source": _source(),
        "state": MemoryChangeReceiptState.PREPARED,
        "prepared_at": BASE_TIME,
        "committed_at": None,
        "expected_created_uris": (uri,),
        "expected_updated_uris": (),
        "expected_deleted_uris": (),
        "unchanged_uris": (),
        "prepared_node_changes": (prepared,),
        "identity_changes": (),
        "added_relations": (),
        "removed_relations": (),
        "node_changes": (),
    }
    values.update(overrides)
    return MemoryChangeReceipt(**values)  # type: ignore[arg-type]


def test_prepared_and_committed_receipt_round_trip_exactly() -> None:
    prepared = _prepared_receipt()
    assert MemoryChangeReceipt.from_dict(prepared.to_dict()) == prepared
    uri = prepared.expected_created_uris[0]
    committed = replace(
        prepared,
        state=MemoryChangeReceiptState.COMMITTED,
        committed_at=BASE_TIME + timedelta(seconds=1),
        node_changes=(MemoryNodeChange(MemoryNodeChangeAction.CREATE, uri, None, 1, None, "c" * 64),),
    )
    assert MemoryChangeReceipt.from_dict(committed.to_dict()) == committed
    assert committed.changed_uris == (uri,)


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "source",
        "state",
        "prepared_at",
        "committed_at",
        "expected_created_uris",
        "expected_updated_uris",
        "expected_deleted_uris",
        "unchanged_uris",
        "prepared_node_changes",
        "identity_changes",
        "added_relations",
        "removed_relations",
        "node_changes",
    ],
)
def test_receipt_parser_rejects_each_missing_field(field: str) -> None:
    value = _prepared_receipt().to_dict()
    value.pop(field)
    with pytest.raises(ValueError):
        MemoryChangeReceipt.from_dict(value)


@pytest.mark.parametrize(
    "field",
    [
        "expected_created_uris",
        "expected_updated_uris",
        "expected_deleted_uris",
        "unchanged_uris",
        "prepared_node_changes",
        "identity_changes",
        "added_relations",
        "removed_relations",
        "node_changes",
    ],
)
@pytest.mark.parametrize("value", [None, (), {}, "items", 1, True])
def test_receipt_parser_requires_json_arrays_for_every_collection(field: str, value: object) -> None:
    raw = _prepared_receipt().to_dict()
    raw[field] = value
    with pytest.raises(ValueError):
        MemoryChangeReceipt.from_dict(raw)


@pytest.mark.parametrize(
    "field", ["expected_created_uris", "expected_updated_uris", "expected_deleted_uris", "unchanged_uris"]
)
def test_receipt_uri_collections_must_be_unique_and_sorted(field: str) -> None:
    profile = MemoryURI.from_address(MemoryAddress.profile())
    preference = MemoryURI.from_address(MemoryAddress.preference("主题"))
    values = tuple(reversed(sorted((profile, preference), key=str)))
    with pytest.raises(ValueError, match="unique and sorted"):
        _prepared_receipt(**{field: values})


def test_receipt_expected_change_sets_are_disjoint() -> None:
    uri = MemoryURI.from_address(MemoryAddress.profile())
    with pytest.raises(ValueError, match="disjoint"):
        _prepared_receipt(expected_updated_uris=(uri,))


def test_prepared_receipt_rejects_commit_time_or_node_changes() -> None:
    uri = MemoryURI.from_address(MemoryAddress.profile())
    change = MemoryNodeChange(MemoryNodeChangeAction.CREATE, uri, None, 1, None, "a" * 64)
    with pytest.raises(ValueError, match="prepared"):
        _prepared_receipt(committed_at=BASE_TIME)
    with pytest.raises(ValueError, match="prepared"):
        _prepared_receipt(node_changes=(change,))


@pytest.mark.parametrize("committed_at", [None, BASE_TIME - timedelta(seconds=1)])
def test_committed_receipt_requires_non_backward_commit_time(committed_at: datetime | None) -> None:
    prepared = _prepared_receipt()
    uri = prepared.expected_created_uris[0]
    change = MemoryNodeChange(MemoryNodeChangeAction.CREATE, uri, None, 1, None, "a" * 64)
    with pytest.raises(ValueError, match="committed_at"):
        replace(prepared, state=MemoryChangeReceiptState.COMMITTED, committed_at=committed_at, node_changes=(change,))


@pytest.mark.parametrize("state", tuple(MemoryChangeReceiptState))
def test_receipt_rejects_same_relation_in_added_and_removed_sets(state: MemoryChangeReceiptState) -> None:
    left = MemoryURI.from_address(MemoryAddress.profile())
    right = MemoryURI.from_address(MemoryAddress.preference("主题"))
    relation = MemoryStoredLink(left, right, MemoryLinkType.RELATED_TO)
    kwargs: dict[str, object] = {"added_relations": (relation,), "removed_relations": (relation,)}
    if state is MemoryChangeReceiptState.COMMITTED:
        prepared = _prepared_receipt()
        uri = prepared.expected_created_uris[0]
        kwargs.update(
            state=state,
            committed_at=BASE_TIME,
            node_changes=(MemoryNodeChange(MemoryNodeChangeAction.CREATE, uri, None, 1, None, "a" * 64),),
        )
    with pytest.raises(ValueError, match="same relation"):
        _prepared_receipt(**kwargs)
