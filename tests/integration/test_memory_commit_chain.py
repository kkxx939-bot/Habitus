"""候选节点、最终身份、双向关系与逐个原子写的主链测试（阶段 3：没有多文件事务）。"""

import logging
from pathlib import Path
from threading import Event, Thread

import pytest

from habitus.infrastructure.editor.snapshot import SnapshotBatch, VersionedSnapshot
from habitus.infrastructure.store.contracts import PathLock
from habitus.memory.compaction import (
    MemoryLifecycleCommitter,
    MemoryLifecycleOperationStore,
    MemoryLifecycleScanStore,
    MemoryRecoveryStore,
)
from habitus.memory.document import MemoryDocumentMetadata, MemoryLinkType, MemoryStoredLink
from habitus.memory.editor import (
    MemoryCandidate,
    MemoryCandidateBatch,
    MemoryCommitConflictError,
    MemoryCommitPlan,
    MemoryCommitStatus,
    MemoryCommitWriter,
    MemoryFinalIdentity,
    MemoryFinalIdentityMap,
    MemoryMutation,
    MemoryMutationAction,
    MemoryMutationPlan,
    MemoryMutationPlanner,
    MemoryMutationReadSet,
    MemoryNodeDisposition,
    MemoryNodeMatch,
    MemoryNodeMatchStatus,
    MemoryPageIdMap,
    MemoryRelationAction,
    MemoryRelationCandidate,
    MemoryRelationPlanner,
    MemoryRelationReadSet,
    MemoryRelationReadSetLoader,
    MemoryRelationResolver,
)
from habitus.memory.model import MemoryKind
from habitus.memory.snapshot import MemorySnapshotReader
from habitus.memory.tree import MemoryTree
from habitus.memory.uri import MemoryURI
from tests.helpers import BASE_TIME, codec, memory_fields
from tests.lock_store import ProcessLocalLockStore


def missing_batch(*uris: MemoryURI) -> SnapshotBatch:
    snapshots = tuple(VersionedSnapshot.missing(str(uri)) for uri in sorted(uris, key=str))
    return SnapshotBatch(snapshots, 0)


def _writer(tree: MemoryTree, reader: MemorySnapshotReader | None = None) -> MemoryCommitWriter:
    return MemoryCommitWriter(
        tree,
        reader or MemorySnapshotReader(tree),
        PathLock(ProcessLocalLockStore()),
        clock=lambda: BASE_TIME,
    )


def _create_plan(profile: MemoryCandidate, preference: MemoryCandidate) -> MemoryCommitPlan:
    candidates = MemoryCandidateBatch(
        profile=(profile,),
        preferences=(preference,),
        relations=(
            MemoryRelationCandidate(
                MemoryRelationAction.ADD,
                profile.page_id,
                preference.page_id,
                MemoryLinkType.BELONGS_TO,
            ),
        ),
    )
    targets = missing_batch(MemoryURI.from_address(profile.address), MemoryURI.from_address(preference.address))
    mutation = MemoryMutationPlanner().plan(
        candidates,
        MemoryMutationReadSet(SnapshotBatch((), 0), targets),
        MemoryPageIdMap(),
    )
    identities = MemoryFinalIdentityMap.from_mutation_plan(mutation, MemoryPageIdMap())
    relations = MemoryRelationResolver().resolve(candidates, identities)
    relation_read_set = MemoryRelationReadSet.build(targets, identities, relations)
    relation_plan = MemoryRelationPlanner().plan(identities, relations, relation_read_set)
    return MemoryCommitPlan.build(mutation, identities, relation_plan)


def test_workflow_state_follows_the_injected_root_not_the_memory_root_parent(tmp_path: Path) -> None:
    """记忆根挪到别处、工作目录不动：L2 生命周期状态落在注入的工作目录，不再从记忆根推。"""

    tree = MemoryTree(tmp_path / "elsewhere" / "memory", document_codec=codec())
    workflow_root = tmp_path / "data" / "workflow"
    path_lock = PathLock(ProcessLocalLockStore())
    recovery = MemoryRecoveryStore(tree, workflow_root)
    operations = MemoryLifecycleOperationStore(tree, path_lock, workflow_root)
    scan = MemoryLifecycleScanStore(tree, path_lock, workflow_root)
    for store in (recovery, operations, scan):
        store.initialize()

    resolved = workflow_root.resolve()
    assert recovery.root == resolved / "lifecycle" / "l2_recovery"
    assert operations.root == resolved / "lifecycle" / "l2_operations"
    assert scan.path == resolved / "lifecycle" / "l2_scan_state.json"
    assert not (tree.root.parent / "workflow").exists()


def test_workflow_state_is_refused_inside_the_memory_tree(tmp_path: Path) -> None:
    tree = MemoryTree(tmp_path / "memory", document_codec=codec())
    inside = tree.root / "workflow"
    path_lock = PathLock(ProcessLocalLockStore())
    with pytest.raises(ValueError, match="outside the L2 memory tree"):
        MemoryRecoveryStore(tree, inside)
    with pytest.raises(ValueError, match="outside the L2 memory tree"):
        MemoryLifecycleOperationStore(tree, path_lock, inside)
    with pytest.raises(ValueError, match="outside the L2 memory tree"):
        MemoryLifecycleScanStore(tree, path_lock, inside)


def test_create_two_nodes_with_link_and_backlink_then_reject_a_stale_plan(tmp_path: Path) -> None:
    tree = MemoryTree(tmp_path / "memory", document_codec=codec())
    profile = MemoryCandidate(100, MemoryKind.PROFILE, memory_fields(MemoryKind.PROFILE))
    preference = MemoryCandidate(101, MemoryKind.PREFERENCE, memory_fields(MemoryKind.PREFERENCE))
    plan = _create_plan(profile, preference)
    writer = _writer(tree)

    result = writer.commit(plan)

    profile_uri = MemoryURI.from_address(profile.address)
    preference_uri = MemoryURI.from_address(preference.address)
    stored_profile = tree.read(profile.address)
    stored_preference = tree.read(preference.address)
    assert result.status is MemoryCommitStatus.UPDATED
    assert result.created_uris == tuple(sorted((profile_uri, preference_uri), key=str))
    assert stored_profile.links == result.added_relations
    assert stored_preference.backlinks == result.added_relations
    assert stored_profile.metadata.revision == stored_preference.metadata.revision == 1
    assert not (tmp_path / "workflow").exists()

    # 同一份计划读到的是"文件还不存在"，写之前发现已经存在：不盖，交回调用方重做
    with pytest.raises(MemoryCommitConflictError):
        writer.commit(plan)


def test_empty_candidate_batch_is_a_noop(tmp_path: Path) -> None:
    tree = MemoryTree(tmp_path / "memory", document_codec=codec())
    empty = SnapshotBatch((), 0)
    mutation = MemoryMutationPlanner().plan(MemoryCandidateBatch(), MemoryMutationReadSet(empty, empty), MemoryPageIdMap())
    identities = MemoryFinalIdentityMap.from_mutation_plan(mutation, MemoryPageIdMap())
    relation_plan = MemoryRelationPlanner().plan(identities, (), MemoryRelationReadSet.build(empty, identities, ()))

    result = _writer(tree).commit(MemoryCommitPlan.build(mutation, identities, relation_plan))

    assert result.status is MemoryCommitStatus.UNCHANGED
    assert result.created_uris == result.updated_uris == result.deleted_uris == ()


def test_reads_are_not_refused_while_a_batch_is_being_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """阶段 3 去掉"有未完成事务就拒读"：写到一半时读树照常返回，看到的是每个文件各自的新旧状态。"""

    tree = MemoryTree(tmp_path / "memory", document_codec=codec())
    profile = MemoryCandidate(100, MemoryKind.PROFILE, memory_fields(MemoryKind.PROFILE))
    preference = MemoryCandidate(101, MemoryKind.PREFERENCE, memory_fields(MemoryKind.PREFERENCE))
    plan = _create_plan(profile, preference)
    reader = MemorySnapshotReader(tree)
    writer = _writer(tree, reader)
    first_written = Event()
    release_writer = Event()
    original_write = tree.write
    writes = 0

    def blocking_write(value):
        nonlocal writes
        result = original_write(value)
        writes += 1
        if writes == 1:
            first_written.set()
            if not release_writer.wait(timeout=5):
                raise TimeoutError("reader did not release the writer")
        return result

    monkeypatch.setattr(tree, "write", blocking_write)
    failures: list[BaseException] = []

    def commit() -> None:
        try:
            writer.commit(plan)
        except BaseException as exc:
            failures.append(exc)

    worker = Thread(target=commit)
    worker.start()
    assert first_written.wait(timeout=5)
    during = tuple(
        snapshot.exists
        for snapshot in reader.read_many(
            (MemoryURI.from_address(profile.address), MemoryURI.from_address(preference.address))
        ).snapshots
    )
    listed = tree.list_addresses()
    release_writer.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert failures == []
    assert sum(during) == 1  # 只写成了一个文件，读照样返回
    assert len(listed) == 1


def _stored(tree: MemoryTree, kind: MemoryKind, fields: dict[str, object], *, links=(), backlinks=()):
    document = codec().build(
        kind,
        fields,
        metadata=MemoryDocumentMetadata(revision=1, created_at=BASE_TIME, updated_at=BASE_TIME),
        links=tuple(sorted(links, key=lambda item: item.identity)),
        backlinks=tuple(sorted(backlinks, key=lambda item: item.identity)),
    )
    tree.write(document)
    return document


def _messy_graph(tree: MemoryTree):
    """A 记着三条坏关联：指向不存在的实体、对 B 只有正向（B 没记反向）、来自 C 的反向（C 没记正向）。"""

    a_fields = {"topic": "回答风格", "content": "- 偏好简洁直接的回答"}
    b_fields = {"category": "项目", "name": "Habitus", "summary": "Habitus 是用户正在重构的记忆系统。"}
    c_fields = {"category": "项目", "name": "看板", "summary": "看板显示记忆来自哪一段对话。"}
    a_uri = MemoryURI.parse("memory://preferences/回答风格.md")
    b_uri = MemoryURI.parse("memory://entities/项目/Habitus.md")
    c_uri = MemoryURI.parse("memory://entities/项目/看板.md")
    ghost = MemoryURI.parse("memory://entities/项目/已删.md")
    to_ghost = MemoryStoredLink(from_uri=a_uri, to_uri=ghost, link_type=MemoryLinkType.BELONGS_TO)
    to_b = MemoryStoredLink(from_uri=a_uri, to_uri=b_uri, link_type=MemoryLinkType.BELONGS_TO)
    from_c = MemoryStoredLink(from_uri=c_uri, to_uri=a_uri, link_type=MemoryLinkType.DERIVED_FROM)
    a = _stored(tree, MemoryKind.PREFERENCE, a_fields, links=(to_ghost, to_b), backlinks=(from_c,))
    _stored(tree, MemoryKind.ENTITY, b_fields)
    _stored(tree, MemoryKind.ENTITY, c_fields)
    return a, a_uri, b_uri, c_uri, to_ghost, to_b, from_c


def test_writing_a_card_repairs_its_dangling_and_one_sided_relations(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """关联规划这条读路径：坏关联不报错、记日志；写到这张卡时修掉，正向为准。"""

    tree = MemoryTree(tmp_path / "memory", document_codec=codec())
    a, a_uri, b_uri, c_uri, to_ghost, to_b, from_c = _messy_graph(tree)
    reader = MemorySnapshotReader(tree)
    before = reader.read(a_uri)
    known = SnapshotBatch((before,), before.size_bytes)
    new_fields = {"topic": "回答风格", "content": "- 偏好简洁直接、先给结论的回答"}
    candidate = MemoryCandidate(1, MemoryKind.PREFERENCE, new_fields)
    mutation = MemoryMutationPlan(
        MemoryMutationReadSet(known, known),
        (
            MemoryMutation(
                MemoryNodeMatch(candidate, a_uri, MemoryNodeMatchStatus.EXISTING, before),
                MemoryMutationAction.UPDATE,
                candidate.fields,
                ("content",),
            ),
        ),
    )
    identities = MemoryFinalIdentityMap((MemoryFinalIdentity(1, MemoryNodeDisposition.UPDATE, a_uri, a_uri),))

    with caplog.at_level(logging.WARNING):
        relation_read_set = MemoryRelationReadSetLoader(reader).load(known, identities, ())
        relation_plan = MemoryRelationPlanner().plan(identities, (), relation_read_set)
    _writer(tree, reader).commit(MemoryCommitPlan.build(mutation, identities, relation_plan))

    stored_a = tree.read(a.address)
    assert stored_a.fields["content"] == new_fields["content"]
    assert to_ghost not in stored_a.links
    assert to_b in stored_a.links
    assert to_b in tree.read(b_uri.to_address()).backlinks  # 正向有、反向缺：给对方补上
    assert from_c not in stored_a.backlinks  # 反向有、对方没有正向：丢掉
    assert from_c not in tree.read(c_uri.to_address()).links
    assert "memory relation skipped" in caplog.text


def test_cold_retirement_of_a_card_with_bad_relations_does_not_fail(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """冷热退休这条读路径：要删的卡带着悬空与单边关联也照常删，邻居身上指向它的关联清掉。"""

    tree = MemoryTree(tmp_path / "memory", document_codec=codec())
    a, a_uri, b_uri, c_uri, *_ = _messy_graph(tree)
    # B 补上对 A 的反向，退休 A 时它要被清掉
    b = tree.read(b_uri.to_address())
    _stored(
        tree,
        MemoryKind.ENTITY,
        dict(b.fields),
        backlinks=(MemoryStoredLink(from_uri=a_uri, to_uri=b_uri, link_type=MemoryLinkType.BELONGS_TO),),
    )
    reader = MemorySnapshotReader(tree)
    committer = MemoryLifecycleCommitter(_writer(tree, reader), reader)

    with caplog.at_level(logging.WARNING):
        result = committer.delete(reader.read(a_uri))

    assert result.deleted_uris == (a_uri,)
    assert not tree.exists(a.address)
    assert tree.read(b_uri.to_address()).backlinks == ()
    assert tree.read(c_uri.to_address()).links == ()
