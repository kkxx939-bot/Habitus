from __future__ import annotations

import asyncio
import threading
from datetime import datetime, timedelta, timezone

import pytest

from infrastructure.store.contracts import PathLock
from infrastructure.store.locks import ProcessLocalLockStore
from memory.compaction import (
    MemoryFieldCompactor,
    MemoryLifecycleCommitter,
    MemoryLifecycleMaintenanceConfig,
    MemoryLifecycleManager,
    MemoryLifecycleOperationPhase,
    MemoryRecoveryStore,
)
from memory.editor import MemoryCommitTransaction, MemoryTransactionJournal
from memory.model import MemoryKind
from memory.retrieval import (
    MemoryRecallLifecycle,
    MemoryRecallLifecycleConfig,
    SQLiteMemoryRecallLifecycleStore,
)
from memory.snapshot import MemorySnapshotReader
from memory.tree import MemoryTree
from memory.uri import MemoryURI
from tests.helpers import codec, document
from tests.unit.conversation.test_summary_generation import RecordingProvider, structured

NOW = datetime(2026, 8, 3, 8, 0, tzinfo=timezone.utc)


def manager(tmp_path, responses, *, maintenance_config=None):
    current = [NOW]
    tree = MemoryTree(tmp_path / "memory", document_codec=codec())
    reader = MemorySnapshotReader(tree)
    transaction = MemoryCommitTransaction(
        tree,
        reader,
        PathLock(ProcessLocalLockStore()),
        MemoryTransactionJournal(tmp_path / "workflow" / "transactions", tree.document_codec),
        clock=lambda: current[0],
    )
    config = MemoryRecallLifecycleConfig(
        preference_half_life_days=1,
        preference_retire_days=5,
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
        MemoryRecoveryStore(tree),
        MemoryLifecycleCommitter(transaction, reader),
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


def test_cold2_retires_only_after_compaction_retention_window_and_grace(tmp_path) -> None:
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
    asyncio.run(value.record_use((uri,), used_at=current[0]))

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
        context = asyncio.create_task(
            value.record_context_use((value._target(detailed),), used_at=NOW)
        )
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
    context_use = asyncio.run(
        value.record_context_use((value._target(current_document),), used_at=current[0])
    )
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
        context = asyncio.create_task(
            value.record_context_use((value._target(current_document),), used_at=current[0])
        )
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

    assert tuple(item.uri for item in result.failures) == (
        MemoryURI.from_address(poisoned.address),
    )
    assert result.compacted == (MemoryURI.from_address(healthy.address),)
    assert value.tree.read(poisoned.address).metadata.revision == 1
    assert value.tree.read(healthy.address).metadata.revision == 2
