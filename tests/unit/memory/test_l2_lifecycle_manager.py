from __future__ import annotations

import asyncio
import threading
from datetime import UTC, date, datetime, timedelta

import pytest

from habitus.infrastructure.store.contracts import PathLock
from habitus.memory.compaction import (
    MemoryFieldCompactor,
    MemoryLifecycleCommitter,
    MemoryLifecycleMaintenanceConfig,
    MemoryLifecycleManager,
    MemoryLifecycleOperationPhase,
    MemoryLifecycleOperationStore,
    MemoryLifecycleScanStore,
    MemoryRecoveryStore,
)
from habitus.memory.compaction.scan import MemoryCompactionKeep, MemoryLifecycleScanStateError
from habitus.memory.editor import MemoryCommitTransaction, MemoryTransactionJournal
from habitus.memory.model import MemoryKind
from habitus.memory.recall import (
    MemoryRecallLifecycle,
    MemoryRecallLifecycleConfig,
    SQLiteMemoryRecallLifecycleStore,
)
from habitus.memory.snapshot import MemorySnapshotReader
from habitus.memory.tree import MemoryTree
from habitus.memory.uri import MemoryURI
from tests.helpers import codec, document
from tests.lock_store import ProcessLocalLockStore
from tests.unit.conversation.test_summary_generation import RecordingProvider, structured

NOW = datetime(2026, 8, 3, 8, 0, tzinfo=UTC)


def manager(tmp_path, responses, *, maintenance_config=None, recall_config=None):
    current = [NOW]
    tree = MemoryTree(tmp_path / "memory", document_codec=codec())
    reader = MemorySnapshotReader(tree)
    path_lock = PathLock(ProcessLocalLockStore())
    workflow_root = tmp_path / "workflow"
    transaction = MemoryCommitTransaction(
        tree,
        reader,
        path_lock,
        MemoryTransactionJournal(workflow_root / "transactions", tree.document_codec),
        clock=lambda: current[0],
    )
    config = recall_config or MemoryRecallLifecycleConfig(
        preference_half_life_days=1,
        preference_retire_days=30,
        cold2_probe_interval_days=1,
        cold2_probe_limit=2,
        retire_candidate_grace_days=1,
    )
    recall = MemoryRecallLifecycle(
        SQLiteMemoryRecallLifecycleStore(tmp_path / "workflow" / "recall.sqlite3", config=config),
        config=config,
    )
    provider = RecordingProvider(responses)
    refreshed = []

    async def refresh(uris):
        refreshed.append(uris)

    value = MemoryLifecycleManager(
        tree,
        reader,
        recall,
        MemoryFieldCompactor(structured(provider)),
        MemoryRecoveryStore(tree, workflow_root),
        MemoryLifecycleCommitter(transaction, reader),
        MemoryLifecycleOperationStore(tree, path_lock, workflow_root),
        MemoryLifecycleScanStore(tree, path_lock, workflow_root),
        config=maintenance_config,
        clock=lambda: current[0],
        derived_refresh=refresh,
    )
    return value, recall, current, refreshed


def test_cold2_compaction_keeps_recovery_and_actual_use_restores_detailed_l2(tmp_path) -> None:
    value, recall, current, refreshed = manager(
        tmp_path,
        [
            {
                "operations": [
                    {
                        "field": "content",
                        "operation": "update",
                        "content": "- 偏好简洁回答",
                    }
                ]
            }
        ],
    )
    detailed = document(
        MemoryKind.PREFERENCE,
        fields={
            "topic": "回答风格",
            "content": "- 偏好简洁回答\n- 回答开头先给出明确结论\n- 涉及代码时保留精确文件路径",
        },
        timestamp=NOW - timedelta(days=10),
    )
    value.tree.write(detailed)
    uri = MemoryURI.from_address(detailed.address)

    compacted = asyncio.run(value.maintain())
    compact_document = value.tree.read(detailed.address)
    assert compacted.compacted == (uri,)
    assert compact_document.metadata.revision == 2
    assert compact_document.fields["content"] == "- 偏好简洁回答"
    assert value.expand_for_probe(compact_document).fields == detailed.fields
    assert recall.store.read_many((uri,))[0].compacted_at == NOW
    assert refreshed == [(uri,)]

    current[0] = NOW + timedelta(days=1)
    context_use = asyncio.run(
        value.record_context_use(
            (value._target(compact_document),),
            used_at=current[0],
        )
    )
    assert context_use.rejected_uris == ()
    assert context_use.documents[0].fields == detailed.fields
    restored = value.tree.read(detailed.address)
    state = recall.store.read_many((uri,))[0]
    assert restored.metadata.revision == 3
    assert restored.fields == detailed.fields
    assert state.useful_recall_count == 1
    assert state.compacted_at is None
    assert refreshed == [(uri,)]
    assert value.operation_store.pending()[0].phase.value == "derived_pending"
    asyncio.run(value.maintain())
    assert refreshed == [(uri,), (uri,)]


def test_cold2_retires_after_retire_window_from_last_activity_and_grace(tmp_path) -> None:
    value, recall, current, refreshed = manager(
        tmp_path,
        [
            {
                "operations": [
                    {
                        "field": "content",
                        "operation": "update",
                        "content": "- 偏好简洁回答",
                    }
                ]
            }
        ],
    )
    detailed = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "回答风格", "content": "- 偏好简洁回答\n- 回答开头先给结论"},
        timestamp=NOW - timedelta(days=20),
    )
    value.tree.write(detailed)
    uri = MemoryURI.from_address(detailed.address)
    asyncio.run(value.maintain())
    current[0] = NOW + timedelta(days=10)
    candidate = asyncio.run(value.maintain())
    assert candidate.retired == ()
    current[0] = NOW + timedelta(days=12)
    retired = asyncio.run(value.maintain())

    assert retired.retired == (uri,)
    assert value.tree.list_addresses() == ()
    assert recall.store.read_many((uri,)) == ()
    assert value.recovery_store.latest(uri, created_at=detailed.metadata.created_at) is None
    assert refreshed == [(uri,), (uri,)]


def test_actual_use_does_not_restore_stale_recovery_over_newer_semantic_update(tmp_path) -> None:
    value, recall, current, refreshed = manager(
        tmp_path,
        [
            {
                "operations": [
                    {
                        "field": "content",
                        "operation": "update",
                        "content": "- 偏好简洁回答",
                    }
                ]
            }
        ],
    )
    detailed = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "回答风格", "content": "- 偏好简洁回答\n- 开头先给结论"},
        timestamp=NOW - timedelta(days=10),
    )
    value.tree.write(detailed)
    uri = MemoryURI.from_address(detailed.address)
    asyncio.run(value.maintain())
    updated, _commit = value.committer.replace_fields(
        value.snapshot_reader.read(uri),
        {"topic": "回答风格", "content": "- 用户最新要求改为详细解释"},
    )

    assert value.expand_for_probe(updated) == updated
    current[0] = NOW + timedelta(days=1)
    asyncio.run(value.record_context_use((value._target(updated),), used_at=current[0]))

    retained = value.tree.read(detailed.address)
    state = recall.store.read_many((uri,))[0]
    assert retained.fields["content"] == "- 用户最新要求改为详细解释"
    assert state.document_revision == retained.metadata.revision
    assert state.compacted_at is None
    assert refreshed == [(uri,)]


def test_retired_marker_retries_transactional_delete_after_interruption(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, recall, current, refreshed = manager(
        tmp_path,
        [
            {
                "operations": [
                    {
                        "field": "content",
                        "operation": "update",
                        "content": "- 偏好简洁回答",
                    }
                ]
            }
        ],
    )
    detailed = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "回答风格", "content": "- 偏好简洁回答\n- 开头先给结论"},
        timestamp=NOW - timedelta(days=20),
    )
    value.tree.write(detailed)
    uri = MemoryURI.from_address(detailed.address)
    asyncio.run(value.maintain())
    current[0] = NOW + timedelta(days=10)
    first = asyncio.run(value.maintain())
    assert first.retired == ()
    current[0] = NOW + timedelta(days=12)
    delete = value.committer.delete

    def interrupted(_uri):
        raise RuntimeError("interrupted before delete commit")

    monkeypatch.setattr(value.committer, "delete", interrupted)
    failed = asyncio.run(value.maintain())
    assert failed.retired == ()
    assert tuple(item.error_type for item in failed.failures) == ("RuntimeError",)
    assert recall.store.read_many((uri,))[0].retired_at == current[0]
    assert value.tree.read(detailed.address)

    monkeypatch.setattr(value.committer, "delete", delete)
    current[0] = NOW + timedelta(days=13)
    retired = asyncio.run(value.maintain())
    assert retired.retired == (uri,)
    assert value.tree.list_addresses() == ()
    assert recall.store.read_many((uri,)) == ()
    assert refreshed == [(uri,), (uri,)]


def test_lifecycle_scan_cursor_rotates_beyond_first_bounded_page(tmp_path) -> None:
    response = {
        "operations": [
            {
                "field": "content",
                "operation": "update",
                "content": "- 压缩后的偏好",
            }
        ]
    }
    value, _recall, _current, refreshed = manager(
        tmp_path,
        [response],
        maintenance_config=MemoryLifecycleMaintenanceConfig(max_scan_items=1),
    )
    detailed = tuple(
        document(
            MemoryKind.PREFERENCE,
            fields={"topic": topic, "content": f"- {topic} 的详细偏好\n- 保留额外约束"},
            timestamp=NOW - timedelta(days=10),
        )
        for topic in ("a", "b")
    )
    for item in detailed:
        value.tree.write(item)

    first = asyncio.run(value.maintain())
    restarted, _recall2, _current2, restarted_refreshed = manager(
        tmp_path,
        [response],
        maintenance_config=MemoryLifecycleMaintenanceConfig(max_scan_items=1),
    )
    second = asyncio.run(restarted.maintain())

    expected = {MemoryURI.from_address(item.address) for item in detailed}
    assert set((*first.compacted, *second.compacted)) == expected
    assert first.scanned == second.scanned == 1
    assert {uris[0] for uris in (*refreshed, *restarted_refreshed)} == expected


def test_compaction_discards_model_plan_when_l2_changes_during_model_call(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, recall, _current, refreshed = manager(
        tmp_path,
        [
            {
                "operations": [
                    {
                        "field": "content",
                        "operation": "update",
                        "content": "- 旧计划压缩结果",
                    }
                ]
            }
        ],
    )
    detailed = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "回答风格", "content": "- 原始详细偏好\n- 必须保留约束"},
        timestamp=NOW - timedelta(days=10),
    )
    value.tree.write(detailed)
    uri = MemoryURI.from_address(detailed.address)
    compact = value.field_compactor.compact

    async def compact_after_concurrent_update(source):
        result = await compact(source)
        value.committer.replace_fields(
            value.snapshot_reader.read(uri),
            {"topic": "回答风格", "content": "- 并发写入的新偏好"},
        )
        return result

    monkeypatch.setattr(value.field_compactor, "compact", compact_after_concurrent_update)
    result = asyncio.run(value.maintain())

    retained = value.tree.read(detailed.address)
    assert result.compacted == ()
    assert retained.metadata.revision == 2
    assert retained.fields["content"] == "- 并发写入的新偏好"
    assert value.recovery_store.latest(uri, created_at=detailed.metadata.created_at) is None
    assert value.operation_store.pending() == ()
    assert recall.store.read_many((uri,)) == ()
    assert refreshed == []


def test_derived_failure_leaves_durable_operation_and_retries_without_reapplying_l2(
    tmp_path,
) -> None:
    value, recall, current, _refreshed = manager(
        tmp_path,
        [
            {
                "operations": [
                    {
                        "field": "content",
                        "operation": "update",
                        "content": "- 压缩后的偏好",
                    }
                ]
            }
        ],
    )
    detailed = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "回答风格", "content": "- 压缩后的偏好\n- 额外详细约束"},
        timestamp=NOW - timedelta(days=10),
    )
    value.tree.write(detailed)
    uri = MemoryURI.from_address(detailed.address)
    attempts = []

    async def flaky_refresh(uris):
        attempts.append(uris)
        if len(attempts) == 1:
            raise RuntimeError("derived refresh unavailable")

    value.derived_refresh = flaky_refresh
    failed = asyncio.run(value.maintain())
    compacted = value.tree.read(detailed.address)
    assert failed.compacted == ()
    assert tuple(item.error_type for item in failed.failures) == ("RuntimeError",)
    assert compacted.metadata.revision == 2
    assert recall.store.read_many((uri,))[0].compacted_at == NOW
    assert value.operation_store.pending()[0].phase.value == "derived_pending"

    deferred = asyncio.run(value.maintain())
    assert deferred.compacted == ()
    assert attempts == [(uri,)]
    assert value.operation_store.pending()

    current[0] += timedelta(seconds=61)
    resumed = asyncio.run(value.maintain())
    assert resumed.compacted == (uri,)
    assert value.tree.read(detailed.address).metadata.revision == 2
    assert value.operation_store.pending() == ()
    assert attempts == [(uri,), (uri,)]


def test_compaction_activation_failure_resumes_from_prepared_without_losing_baseline(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, recall, current, _refreshed = manager(
        tmp_path,
        [
            {
                "operations": [
                    {
                        "field": "content",
                        "operation": "update",
                        "content": "- 压缩后的偏好",
                    }
                ]
            }
        ],
    )
    detailed = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "回答风格", "content": "- 压缩后的偏好\n- 额外详细约束"},
        timestamp=NOW - timedelta(days=10),
    )
    value.tree.write(detailed)
    uri = MemoryURI.from_address(detailed.address)
    activate = value.recovery_store.activate
    attempts = []

    def interrupted_activation(record, compacted):
        attempts.append(compacted.revision)
        raise RuntimeError("interrupted before recovery activation")

    monkeypatch.setattr(value.recovery_store, "activate", interrupted_activation)
    failed = asyncio.run(value.maintain())

    assert failed.compacted == ()
    assert value.tree.read(detailed.address).metadata.revision == 2
    assert value.operation_store.pending()[0].phase.value == "prepared"
    baseline = value.recovery_store.latest(uri, created_at=detailed.metadata.created_at)
    assert baseline is not None and not baseline.active

    monkeypatch.setattr(value.recovery_store, "activate", activate)
    current[0] += timedelta(seconds=61)
    resumed = asyncio.run(value.maintain())
    assert resumed.compacted == (uri,)
    assert value.operation_store.pending() == ()
    assert value.recovery_store.for_compacted(value.snapshot_reader.read(uri)) is not None
    assert recall.store.read_many((uri,))[0].compacted_at == NOW
    assert attempts == [2]


def test_context_use_uri_lock_prevents_compaction_from_committing_across_use_state(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, recall, _current, _refreshed = manager(
        tmp_path,
        [
            {
                "operations": [
                    {
                        "field": "content",
                        "operation": "update",
                        "content": "- 压缩后的偏好",
                    }
                ]
            }
        ],
    )
    detailed = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "回答风格", "content": "- 详细偏好\n- 额外约束"},
        timestamp=NOW - timedelta(days=10),
    )
    value.tree.write(detailed)
    uri = MemoryURI.from_address(detailed.address)
    entered = threading.Event()
    release = threading.Event()
    record_use = recall.store.record_use

    def blocked_record_use(targets, *, used_at):
        entered.set()
        assert release.wait(timeout=5)
        return record_use(targets, used_at=used_at)

    monkeypatch.setattr(recall.store, "record_use", blocked_record_use)

    async def race():
        context = asyncio.create_task(value.record_context_use((value._target(detailed),), used_at=NOW))
        assert await asyncio.to_thread(entered.wait, 5)
        maintenance = asyncio.create_task(value.maintain())
        await asyncio.sleep(0.05)
        release.set()
        return await context, await maintenance

    context_result, maintenance_result = asyncio.run(race())
    state = recall.store.read_many((uri,))[0]
    assert context_result.rejected_uris == ()
    assert maintenance_result.compacted == ()
    assert value.tree.read(detailed.address).metadata.revision == 1
    assert state.document_revision == 1
    assert state.useful_recall_count == 1
    assert value.operation_store.pending() == ()
    assert value.recovery_store.latest(uri, created_at=detailed.metadata.created_at) is None


def test_new_semantic_revision_cancels_stale_retirement_and_clears_old_state(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, recall, current, _refreshed = manager(
        tmp_path,
        [
            {
                "operations": [
                    {
                        "field": "content",
                        "operation": "update",
                        "content": "- 压缩后的偏好",
                    }
                ]
            }
        ],
    )
    detailed = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "回答风格", "content": "- 详细偏好\n- 额外约束"},
        timestamp=NOW - timedelta(days=20),
    )
    value.tree.write(detailed)
    uri = MemoryURI.from_address(detailed.address)
    asyncio.run(value.maintain())
    current[0] = NOW + timedelta(days=10)
    asyncio.run(value.maintain())
    current[0] = NOW + timedelta(days=12)
    delete = value.committer.delete

    def interrupted_delete(_snapshot):
        raise RuntimeError("interrupted before retirement delete")

    monkeypatch.setattr(value.committer, "delete", interrupted_delete)
    failed = asyncio.run(value.maintain())
    assert failed.retired == ()
    assert recall.store.read_many((uri,))[0].retired_at == current[0]
    assert value.operation_store.pending()[0].kind.value == "retire"

    value.committer.replace_fields(
        value.snapshot_reader.read(uri),
        {"topic": "回答风格", "content": "- 并发写入的新偏好"},
    )
    current_document = value.tree.read(detailed.address)
    context_use = asyncio.run(value.record_context_use((value._target(current_document),), used_at=current[0]))
    assert context_use.rejected_uris == ()
    assert value.operation_store.pending() == ()
    revised_state = recall.store.read_many((uri,))[0]
    assert revised_state.document_revision == current_document.metadata.revision
    assert revised_state.retired_at is None

    monkeypatch.setattr(value.committer, "delete", delete)
    current[0] = NOW + timedelta(days=13)
    resumed = asyncio.run(value.maintain())

    assert resumed.retired == ()
    assert current_document.fields["content"] == "- 并发写入的新偏好"
    assert value.operation_store.pending() == ()
    assert value.recovery_store.latest(uri, created_at=detailed.metadata.created_at) is None


def test_stale_retirement_worker_and_new_revision_context_use_share_operation_lock(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, recall, current, _refreshed = manager(
        tmp_path,
        [
            {
                "operations": [
                    {
                        "field": "content",
                        "operation": "update",
                        "content": "- 压缩后的偏好",
                    }
                ]
            }
        ],
    )
    detailed = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "回答风格", "content": "- 详细偏好\n- 额外约束"},
        timestamp=NOW - timedelta(days=20),
    )
    value.tree.write(detailed)
    uri = MemoryURI.from_address(detailed.address)
    asyncio.run(value.maintain())
    current[0] = NOW + timedelta(days=10)
    asyncio.run(value.maintain())
    current[0] = NOW + timedelta(days=12)
    delete = value.committer.delete

    def interrupted_delete(_snapshot):
        raise RuntimeError("interrupted before retirement delete")

    monkeypatch.setattr(value.committer, "delete", interrupted_delete)
    failed = asyncio.run(value.maintain())
    assert failed.failures
    assert value.operation_store.pending()[0].kind.value == "retire"

    value.committer.replace_fields(
        value.snapshot_reader.read(uri),
        {"topic": "回答风格", "content": "- 并发写入的新偏好"},
    )
    current_document = value.tree.read(detailed.address)
    monkeypatch.setattr(value.committer, "delete", delete)
    current[0] += timedelta(seconds=61)
    read = value.snapshot_reader.read
    entered = threading.Event()
    release = threading.Event()
    call_guard = threading.Lock()

    def blocked_first_read(identity):
        snapshot = read(identity)
        with call_guard:
            should_block = not entered.is_set()
            if should_block:
                entered.set()
        if should_block:
            assert release.wait(timeout=5)
        return snapshot

    monkeypatch.setattr(value.snapshot_reader, "read", blocked_first_read)

    async def race():
        maintenance = asyncio.create_task(value.maintain())
        assert await asyncio.to_thread(entered.wait, 5)
        context = asyncio.create_task(value.record_context_use((value._target(current_document),), used_at=current[0]))
        await asyncio.sleep(0.05)
        release.set()
        return await maintenance, await context

    maintenance_result, context_result = asyncio.run(race())

    assert maintenance_result.failures == ()
    assert context_result.rejected_uris == ()
    assert value.operation_store.pending() == ()
    revised_state = recall.store.read_many((uri,))[0]
    assert revised_state.document_revision == current_document.metadata.revision
    assert revised_state.retired_at is None
    assert value.scan_store.eligible(uri, now=current[0])


def test_l2_committed_stale_retirement_advance_does_not_recreate_scan_failure(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value, recall, current, _refreshed = manager(
        tmp_path,
        [
            {
                "operations": [
                    {
                        "field": "content",
                        "operation": "update",
                        "content": "- 压缩后的偏好",
                    }
                ]
            }
        ],
    )
    detailed = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "回答风格", "content": "- 详细偏好\n- 额外约束"},
        timestamp=NOW - timedelta(days=20),
    )
    value.tree.write(detailed)
    uri = MemoryURI.from_address(detailed.address)
    asyncio.run(value.maintain())
    current[0] = NOW + timedelta(days=10)
    asyncio.run(value.maintain())
    current[0] = NOW + timedelta(days=12)
    advance_owned = value.operation_store.advance_owned

    def interrupted_after_l2(operation, phase, **kwargs):
        advanced = advance_owned(operation, phase, **kwargs)
        if phase is MemoryLifecycleOperationPhase.L2_COMMITTED:
            raise RuntimeError("interrupted after L2 retirement commit")
        return advanced

    monkeypatch.setattr(value.operation_store, "advance_owned", interrupted_after_l2)
    failed = asyncio.run(value.maintain())
    assert failed.failures
    assert not value.tree.exists(detailed.address)
    assert value.operation_store.pending()[0].phase is MemoryLifecycleOperationPhase.L2_COMMITTED

    replacement = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "回答风格", "content": "- 重新创建的新偏好"},
        timestamp=current[0] + timedelta(seconds=1),
    )
    value.tree.write(replacement)
    monkeypatch.setattr(value.operation_store, "advance_owned", advance_owned)
    current[0] += timedelta(seconds=61)
    advance_retire = value._advance_retire_if_current
    entered = threading.Event()
    release = threading.Event()

    def blocked_advance(operation, phase, now):
        entered.set()
        assert release.wait(timeout=5)
        return advance_retire(operation, phase, now)

    monkeypatch.setattr(value, "_advance_retire_if_current", blocked_advance)

    async def race():
        maintenance = asyncio.create_task(value.maintain())
        assert await asyncio.to_thread(entered.wait, 5)
        context = await value.record_context_use(
            (value._target(replacement),),
            used_at=current[0],
        )
        release.set()
        return await maintenance, context

    maintenance_result, context_result = asyncio.run(race())

    assert maintenance_result.failures == ()
    assert context_result.rejected_uris == ()
    assert value.operation_store.pending() == ()
    revised_state = recall.store.read_many((uri,))[0]
    assert revised_state.document_revision == replacement.metadata.revision
    assert revised_state.document_created_at == replacement.metadata.created_at
    assert revised_state.retired_at is None
    assert value.scan_store.eligible(uri, now=current[0])


def test_one_poison_l2_node_does_not_starve_later_scan_candidates(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = {
        "operations": [
            {
                "field": "content",
                "operation": "update",
                "content": "- 有效压缩结果",
            }
        ]
    }
    value, _recall, _current, _refreshed = manager(
        tmp_path,
        [response],
        maintenance_config=MemoryLifecycleMaintenanceConfig(
            max_scan_items=2,
            max_compactions_per_cycle=2,
        ),
    )
    poisoned = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "a", "content": "- a 的详细偏好\n- 额外约束"},
        timestamp=NOW - timedelta(days=10),
    )
    healthy = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "b", "content": "- b 的详细偏好\n- 额外约束"},
        timestamp=NOW - timedelta(days=10),
    )
    value.tree.write(poisoned)
    value.tree.write(healthy)
    compact = value.field_compactor.compact

    async def fail_one(source):
        if source.address == poisoned.address:
            raise RuntimeError("poisoned memory")
        return await compact(source)

    monkeypatch.setattr(value.field_compactor, "compact", fail_one)
    result = asyncio.run(value.maintain())

    assert tuple(item.uri for item in result.failures) == (MemoryURI.from_address(poisoned.address),)
    assert result.compacted == (MemoryURI.from_address(healthy.address),)
    assert value.tree.read(poisoned.address).metadata.revision == 1
    assert value.tree.read(healthy.address).metadata.revision == 2


@pytest.mark.parametrize(
    ("kind", "fields", "field", "compact_text"),
    [
        (
            MemoryKind.ENTITY,
            {
                "category": "项目",
                "name": "Habitus",
                "summary": "Habitus 是用户正在重构的记忆系统，用 Python 写，分行为层与记忆层，计划插件交付。",
            },
            "summary",
            "Habitus 是用户在重构的记忆系统。",
        ),
        (
            MemoryKind.INTENTION,
            {
                "intent_name": "完成记忆系统重构",
                "status": "open",
                "next_step": "补齐测试体系，先写生命周期测试，再写检索测试，最后跑 ci",
            },
            "next_step",
            "补齐测试体系",
        ),
    ],
)
def test_compaction_of_l2_without_optional_fields_commits_and_stays_usable(
    tmp_path,
    kind,
    fields,
    field,
    compact_text,
) -> None:
    value, _recall, current, _refreshed = manager(
        tmp_path,
        [{"operations": [{"field": field, "operation": "update", "content": compact_text}]}],
    )
    detailed = document(kind, fields=fields, timestamp=NOW - timedelta(days=200))
    value.tree.write(detailed)
    uri = MemoryURI.from_address(detailed.address)

    result = asyncio.run(value.maintain())

    compact_document = value.tree.read(detailed.address)
    assert result.failures == ()
    assert result.compacted == (uri,)
    assert value.operation_store.pending() == ()
    assert set(compact_document.fields) == set(fields)
    assert compact_document.fields[field] == compact_text
    current[0] = NOW + timedelta(days=1)
    context_use = asyncio.run(
        value.record_context_use((value._target(compact_document),), used_at=current[0]),
    )
    assert context_use.rejected_uris == ()
    assert tuple(item.fields for item in context_use.documents) == (detailed.fields,)


def test_compacted_l2_rewritten_by_editor_recompacts_when_cold_again_and_retires(tmp_path) -> None:
    responses = [
        {"operations": [{"field": "content", "operation": "update", "content": "- 简洁"}]},
        {"operations": [{"field": "content", "operation": "update", "content": "- 详细解释"}]},
    ]
    value, _recall, current, _refreshed = manager(tmp_path, responses)
    detailed = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "回答风格", "content": "- 偏好简洁回答\n- 开头先给结论\n- 保留路径"},
        timestamp=NOW - timedelta(days=10),
    )
    value.tree.write(detailed)
    uri = MemoryURI.from_address(detailed.address)
    assert asyncio.run(value.maintain()).compacted == (uri,)
    rewritten, _commit = value.committer.replace_fields(
        value.snapshot_reader.read(uri),
        {"topic": "回答风格", "content": "- 用户改口：要详细解释\n- 每步给出理由\n- 附上命令"},
    )
    assert value.recovery_store.for_compacted(value.snapshot_reader.read(uri)) is None

    current[0] = NOW + timedelta(days=3)
    recompacted = asyncio.run(value.maintain())
    assert recompacted.failures == ()
    assert recompacted.compacted == (uri,)
    assert responses == []
    compact_document = value.tree.read(detailed.address)
    assert compact_document.fields["content"] == "- 详细解释"
    assert value.expand_for_probe(compact_document).fields == rewritten.fields

    current[0] = NOW + timedelta(days=4)
    assert asyncio.run(value.maintain()) == type(recompacted)(1, (), ())
    retired = None
    for day in range(5, 40):
        current[0] = NOW + timedelta(days=day)
        cycle = asyncio.run(value.maintain())
        assert cycle.failures == ()
        if cycle.retired:
            retired = cycle.retired
            break
    assert retired == (uri,)
    assert value.tree.list_addresses() == ()


@pytest.mark.parametrize(
    ("recall_config", "step_days"),
    [
        (
            MemoryRecallLifecycleConfig(
                preference_half_life_days=1,
                preference_retire_days=5,
                retire_candidate_grace_days=1,
            ),
            0.5,
        ),
        (MemoryRecallLifecycleConfig(), 5.0),
    ],
    ids=["short-window", "default-config"],
)
def test_compaction_write_back_does_not_reset_retirement_clock(
    tmp_path,
    recall_config,
    step_days,
) -> None:
    value, recall, current, _refreshed = manager(
        tmp_path,
        [{"operations": [{"field": "content", "operation": "update", "content": "- 简洁"}]}],
        recall_config=recall_config,
    )
    detailed = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "回答风格", "content": "- 偏好简洁回答\n- 开头先给结论\n- 保留路径"},
        timestamp=NOW,
    )
    value.tree.write(detailed)
    uri = MemoryURI.from_address(detailed.address)
    compacted_at = None
    for step in range(1, 1_000):
        current[0] = NOW + timedelta(days=step * step_days)
        if asyncio.run(value.maintain()).compacted == (uri,):
            compacted_at = current[0]
            break
    assert compacted_at is not None
    compacted_document = value.tree.read(detailed.address)
    assert compacted_document.metadata.revision == 2
    assert compacted_document.metadata.updated_at == compacted_at
    last_activity = detailed.metadata.updated_at
    retire = timedelta(days=recall_config.preference_retire_days)
    grace = timedelta(days=recall_config.retire_candidate_grace_days)

    def run_at(moment):
        current[0] = moment
        result = asyncio.run(value.maintain())
        assert result.failures == ()
        state = recall.store.read_many((uri,))
        return result, state[0] if state else None

    _result, before = run_at(last_activity + retire - timedelta(hours=1))
    assert before is not None and before.retire_candidate_at is None
    _result, marked = run_at(last_activity + retire)
    assert marked is not None and marked.retire_candidate_at == last_activity + retire
    within_grace, _state = run_at(last_activity + retire + grace / 2)
    assert within_grace.retired == ()
    assert value.tree.list_addresses() == (detailed.address,)
    after_grace, _state = run_at(last_activity + retire + grace)
    assert after_grace.retired == (uri,)
    assert value.tree.list_addresses() == ()


_KEEP_EVENT = {"operations": [{"field": "summary", "operation": "keep"}]}


def _run_until_deleted(value, recall, current, uri, *, start, limit_days, hourly_near):
    """推进真实维护循环直到删除，记录标候选与删除的时刻。

    关键时刻前后一天逐小时推进，其余每 12 小时一轮，控制单测耗时。
    """

    seen: dict[str, datetime] = {}
    moment = start
    while moment <= start + timedelta(days=limit_days):
        current[0] = moment
        result = asyncio.run(value.maintain())
        state = recall.store.read_many((uri,))
        if state and state[0].retire_candidate_at is not None:
            seen.setdefault("candidate", moment)
        if result.retired == (uri,):
            seen["deleted"] = moment
            return seen
        near = any(abs(moment - point) <= timedelta(days=1) for point in hourly_near)
        moment += timedelta(hours=1 if near else 12)
    return seen


def test_cold_event_answered_keep_is_asked_once_and_still_retires_on_schedule(tmp_path) -> None:
    responses = [dict(_KEEP_EVENT) for _ in range(10)]
    config = MemoryRecallLifecycleConfig()
    value, recall, current, _refreshed = manager(tmp_path, responses, recall_config=config)
    event = document(MemoryKind.EVENT, timestamp=NOW)
    value.tree.write(event)
    uri = MemoryURI.from_address(event.address)

    retire = timedelta(days=config.event_retire_days)
    grace = timedelta(days=config.retire_candidate_grace_days)
    seen = _run_until_deleted(
        value,
        recall,
        current,
        uri,
        start=NOW,
        limit_days=240,
        hourly_near=(NOW + timedelta(days=37), NOW + retire, NOW + retire + grace),
    )

    assert len(responses) == 9
    assert value.tree.list_addresses() == ()
    assert seen == {"candidate": NOW + retire, "deleted": NOW + retire + grace}


def test_keep_verdict_is_not_reasked_until_the_memory_is_used_and_cools_again(tmp_path) -> None:
    responses = [dict(_KEEP_EVENT) for _ in range(10)]
    value, _recall, current, _refreshed = manager(tmp_path, responses, recall_config=MemoryRecallLifecycleConfig())
    event = document(MemoryKind.EVENT, timestamp=NOW - timedelta(days=40))
    value.tree.write(event)
    for hour in range(5):
        current[0] = NOW + timedelta(hours=hour)
        result = asyncio.run(value.maintain())
        assert result.failures == ()
        assert result.compacted == ()
    assert len(responses) == 9

    current[0] = NOW + timedelta(days=1)
    use = asyncio.run(value.record_context_use((value._target(event),), used_at=current[0]))
    assert use.rejected_uris == ()
    current[0] = NOW + timedelta(days=30)
    asyncio.run(value.maintain())
    assert len(responses) == 9
    current[0] = NOW + timedelta(days=60)
    asyncio.run(value.maintain())
    asyncio.run(value.maintain())
    assert len(responses) == 8


def test_retirement_does_not_wait_for_a_compaction_model_that_keeps_failing(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MemoryRecallLifecycleConfig()
    value, recall, current, _refreshed = manager(tmp_path, [], recall_config=config)
    event = document(MemoryKind.EVENT, timestamp=NOW)
    value.tree.write(event)
    uri = MemoryURI.from_address(event.address)
    calls = []

    async def unavailable(source):
        calls.append(source.address)
        raise RuntimeError("compaction model unavailable")

    monkeypatch.setattr(value.field_compactor, "compact", unavailable)

    retire = timedelta(days=config.event_retire_days)
    grace = timedelta(days=config.retire_candidate_grace_days)
    seen = _run_until_deleted(
        value,
        recall,
        current,
        uri,
        start=NOW,
        limit_days=240,
        hourly_near=(NOW + timedelta(days=37), NOW + retire, NOW + retire + grace),
    )

    assert calls
    assert value.tree.list_addresses() == ()
    backoff_cap = timedelta(seconds=value.scan_store.max_retry_seconds)
    assert NOW + retire <= seen["candidate"] <= NOW + retire + backoff_cap
    assert seen["deleted"] == seen["candidate"] + grace


def test_compaction_quota_counts_model_attempts_not_successes(tmp_path) -> None:
    responses = [dict(_KEEP_EVENT) for _ in range(20)]
    value, _recall, current, _refreshed = manager(tmp_path, responses, recall_config=MemoryRecallLifecycleConfig())
    for index in range(10):
        value.tree.write(
            document(
                MemoryKind.EVENT,
                fields={
                    "event_name": f"事件{index}",
                    "event_date": date(2026, 6, 1),
                    "summary": f"第 {index} 件事的经过",
                },
                timestamp=NOW - timedelta(days=40),
            )
        )
    assert len(value.tree.list_addresses()) == 10

    asked = []
    for hour in range(4):
        current[0] = NOW + timedelta(hours=hour)
        before = len(responses)
        result = asyncio.run(value.maintain())
        assert result.failures == ()
        asked.append(before - len(responses))

    assert asked == [4, 4, 2, 0]


def test_compaction_keep_ledger_matches_exact_version_and_stays_bounded(tmp_path) -> None:
    tree = MemoryTree(tmp_path / "memory", document_codec=codec())
    store = MemoryLifecycleScanStore(
        tree,
        PathLock(ProcessLocalLockStore()),
        tmp_path / "workflow",
        max_compaction_keeps=2,
    )
    store.initialize()
    uris = [MemoryURI.parse(f"memory://preferences/p{index}.md") for index in range(3)]
    versions = [
        MemoryCompactionKeep(uri, 1, "a" * 64, None, NOW + timedelta(hours=index)) for index, uri in enumerate(uris)
    ]
    store.record_compaction_keep(versions[0])
    assert store.compaction_kept(versions[0])
    assert not store.compaction_kept(MemoryCompactionKeep(uris[0], 2, "a" * 64, None, NOW))
    assert not store.compaction_kept(MemoryCompactionKeep(uris[0], 1, "b" * 64, None, NOW))
    assert not store.compaction_kept(MemoryCompactionKeep(uris[0], 1, "a" * 64, NOW, NOW))

    store.record_compaction_keep(versions[1])
    store.record_compaction_keep(versions[2])
    assert [store.compaction_kept(item) for item in versions] == [False, True, True]


def test_stale_v1_scan_state_fails_with_the_file_to_delete_and_is_left_untouched(tmp_path) -> None:
    tree = MemoryTree(tmp_path / "memory", document_codec=codec())
    state_path = tmp_path / "workflow" / "lifecycle" / "l2_scan_state.json"
    state_path.parent.mkdir(parents=True)
    stale = b'{"after_uri":null,"failures":[],"schema":"memory_lifecycle_scan_v1"}\n'
    state_path.write_bytes(stale)
    store = MemoryLifecycleScanStore(tree, PathLock(ProcessLocalLockStore()), tmp_path / "workflow")

    with pytest.raises(MemoryLifecycleScanStateError) as raised:
        store.initialize()

    assert str(raised.value) == (
        f"这是旧格式扫描账（memory_lifecycle_scan_v1），删除 {state_path.resolve()} 后重启即可（只存游标与退避）"
    )
    assert state_path.read_bytes() == stale
