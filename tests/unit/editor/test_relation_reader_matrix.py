"""最终身份确定后关系闭包读取与快照冲突场景矩阵。"""

from __future__ import annotations

from pathlib import Path

import pytest

from infrastructure.editor.snapshot import SnapshotBatch, VersionedSnapshot
from memory.document import MemoryLinkType, MemoryStoredLink
from memory.editor import (
    MemoryFinalIdentity,
    MemoryFinalIdentityMap,
    MemoryNodeDisposition,
    MemoryRelationAction,
    MemoryRelationIntegrityError,
    MemoryRelationReadConflictError,
    MemoryRelationReadSetLoader,
    MemoryResolvedRelation,
)
from memory.model import MemoryAddress, MemoryKind
from memory.snapshot import MemorySnapshotReader
from memory.tree import MemoryTree
from memory.uri import MemoryURI
from tests.helpers import codec, document


def reader(tmp_path: Path, *documents):
    """构造真实关系树和快照读取器。"""

    tree = MemoryTree(tmp_path / "memory")
    for current in documents:
        tree.write(current)
    return tree, MemorySnapshotReader(tree)


def linked_documents(*, backlink: bool = True):
    """构造偏好到 Profile 的稳定双向关系。"""

    source_base = document(MemoryKind.PREFERENCE)
    target_base = document(MemoryKind.PROFILE)
    source_uri = MemoryURI.from_address(source_base.address)
    target_uri = MemoryURI.from_address(target_base.address)
    relation = MemoryStoredLink(source_uri, target_uri, MemoryLinkType.BELONGS_TO)
    source = codec().build(
        source_base.kind,
        source_base.fields,
        metadata=source_base.metadata,
        links=(relation,),
    )
    target = codec().build(
        target_base.kind,
        target_base.fields,
        metadata=target_base.metadata,
        backlinks=(relation,) if backlink else (),
    )
    return source, target, relation


def noop_identities(*documents) -> MemoryFinalIdentityMap:
    entries = []
    for page_id, current in enumerate(documents, start=1):
        uri = MemoryURI.from_address(current.address)
        entries.append(
            MemoryFinalIdentity(page_id, MemoryNodeDisposition.NOOP, uri, uri)
        )
    return MemoryFinalIdentityMap(tuple(entries))


def test_relation_reader_requires_real_snapshot_reader(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="reader"):
        MemoryRelationReadSetLoader(object())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "invalid", "message"),
    [
        ("known_snapshots", object(), "known_snapshots"),
        ("identities", object(), "identities"),
        ("operations", [], "operations"),
        ("operations", (object(),), "operations"),
    ],
)
def test_relation_reader_rejects_each_invalid_input(
    tmp_path: Path,
    field: str,
    invalid: object,
    message: str,
) -> None:
    _tree, snapshot_reader = reader(tmp_path)
    values = {
        "known_snapshots": SnapshotBatch((), 0),
        "identities": MemoryFinalIdentityMap(()),
        "operations": (),
        field: invalid,
    }
    with pytest.raises(TypeError, match=message):
        MemoryRelationReadSetLoader(snapshot_reader).load(**values)


def test_no_structural_or_explicit_relation_returns_empty_read_set(tmp_path: Path) -> None:
    current = document(MemoryKind.PREFERENCE)
    _tree, snapshot_reader = reader(tmp_path, current)
    known = snapshot_reader.read_many((MemoryURI.from_address(current.address),))

    result = MemoryRelationReadSetLoader(snapshot_reader).load(
        known,
        noop_identities(current),
        (),
    )

    assert result.snapshots == SnapshotBatch((), 0)
    assert result.structural_source_uris == ()


def test_explicit_relation_reads_both_complete_endpoints(tmp_path: Path) -> None:
    source = document(MemoryKind.PREFERENCE)
    target = document(MemoryKind.PROFILE)
    source_uri = MemoryURI.from_address(source.address)
    target_uri = MemoryURI.from_address(target.address)
    operation = MemoryResolvedRelation(
        MemoryRelationAction.ADD,
        source_uri,
        target_uri,
        MemoryLinkType.BELONGS_TO,
    )
    _tree, snapshot_reader = reader(tmp_path, source, target)

    result = MemoryRelationReadSetLoader(snapshot_reader).load(
        SnapshotBatch((), 0),
        noop_identities(source, target),
        (operation,),
    )

    assert tuple(item.identity for item in result.snapshots.snapshots) == tuple(
        sorted((str(source_uri), str(target_uri)))
    )
    assert all(item.exists for item in result.snapshots.snapshots)


def test_explicit_relation_accepts_missing_snapshots_for_planned_creates(tmp_path: Path) -> None:
    source_uri = MemoryURI.from_address(MemoryAddress.preference("新偏好"))
    target_uri = MemoryURI.from_address(MemoryAddress.profile())
    identities = MemoryFinalIdentityMap(
        (
            MemoryFinalIdentity(100, MemoryNodeDisposition.CREATE, None, source_uri),
            MemoryFinalIdentity(101, MemoryNodeDisposition.CREATE, None, target_uri),
        )
    )
    operation = MemoryResolvedRelation(
        MemoryRelationAction.ADD,
        source_uri,
        target_uri,
        MemoryLinkType.BELONGS_TO,
    )
    _tree, snapshot_reader = reader(tmp_path)

    result = MemoryRelationReadSetLoader(snapshot_reader).load(
        SnapshotBatch((), 0),
        identities,
        (operation,),
    )

    assert len(result.snapshots.snapshots) == 2
    assert not any(item.exists for item in result.snapshots.snapshots)


@pytest.mark.parametrize("disposition", [MemoryNodeDisposition.MERGE, MemoryNodeDisposition.DELETE])
def test_structural_change_reads_every_one_hop_neighbor(
    tmp_path: Path,
    disposition: MemoryNodeDisposition,
) -> None:
    source, target, relation = linked_documents()
    survivor = document(
        MemoryKind.PREFERENCE,
        fields={"topic": "新主题", "content": "- 新内容"},
    )
    source_uri = MemoryURI.from_address(source.address)
    target_uri = MemoryURI.from_address(target.address)
    survivor_uri = MemoryURI.from_address(survivor.address)
    entries = [
        MemoryFinalIdentity(
            1,
            disposition,
            source_uri,
            survivor_uri if disposition is MemoryNodeDisposition.MERGE else None,
        ),
        MemoryFinalIdentity(2, MemoryNodeDisposition.NOOP, target_uri, target_uri),
    ]
    documents = [source, target]
    if disposition is MemoryNodeDisposition.MERGE:
        entries.append(MemoryFinalIdentity(3, MemoryNodeDisposition.NOOP, survivor_uri, survivor_uri))
        documents.append(survivor)
    identities = MemoryFinalIdentityMap(tuple(entries))
    _tree, snapshot_reader = reader(tmp_path, *documents)
    known = snapshot_reader.read_many((source_uri,))

    result = MemoryRelationReadSetLoader(snapshot_reader).load(known, identities, ())

    assert result.structural_source_uris == (source_uri,)
    assert result.document(source_uri).links == (relation,)
    assert result.document(target_uri).backlinks == (relation,)
    if disposition is MemoryNodeDisposition.MERGE:
        assert result.snapshot(survivor_uri).exists


def test_structural_source_disappearance_is_reported_before_planning(tmp_path: Path) -> None:
    source_uri = MemoryURI.from_address(MemoryAddress.preference("已删除"))
    identities = MemoryFinalIdentityMap(
        (MemoryFinalIdentity(1, MemoryNodeDisposition.DELETE, source_uri, None),)
    )
    _tree, snapshot_reader = reader(tmp_path)
    known = snapshot_reader.read_many((source_uri,))

    with pytest.raises(MemoryRelationReadConflictError, match="source disappeared"):
        MemoryRelationReadSetLoader(snapshot_reader).load(known, identities, ())


def test_known_revision_change_is_reported_before_relation_planning(tmp_path: Path) -> None:
    source = document(MemoryKind.PREFERENCE)
    source_uri = MemoryURI.from_address(source.address)
    tree, snapshot_reader = reader(tmp_path, source)
    known = snapshot_reader.read_many((source_uri,))
    updated = codec().build(
        source.kind,
        source.fields,
        metadata=source.metadata.next_revision(source.metadata.updated_at),
    )
    tree.write(updated)
    identities = MemoryFinalIdentityMap(
        (MemoryFinalIdentity(1, MemoryNodeDisposition.DELETE, source_uri, None),)
    )

    with pytest.raises(MemoryRelationReadConflictError, match="changed before relation planning"):
        MemoryRelationReadSetLoader(snapshot_reader).load(known, identities, ())


def test_inconsistent_one_hop_pair_is_rejected_after_complete_closure_read(tmp_path: Path) -> None:
    source, target, _relation = linked_documents(backlink=False)
    source_uri = MemoryURI.from_address(source.address)
    target_uri = MemoryURI.from_address(target.address)
    identities = MemoryFinalIdentityMap(
        (
            MemoryFinalIdentity(1, MemoryNodeDisposition.DELETE, source_uri, None),
            MemoryFinalIdentity(2, MemoryNodeDisposition.NOOP, target_uri, target_uri),
        )
    )
    _tree, snapshot_reader = reader(tmp_path, source, target)

    with pytest.raises(MemoryRelationIntegrityError, match="backlink"):
        MemoryRelationReadSetLoader(snapshot_reader).load(
            snapshot_reader.read_many((source_uri,)),
            identities,
            (),
        )


@pytest.mark.parametrize(
    ("changes", "same"),
    [
        ({}, True),
        ({"source_digest": "f" * 64}, False),
        ({"revision": 2}, False),
        ({"state": "missing", "value": None, "revision": None, "source_digest": None, "size_bytes": 0}, False),
    ],
)
def test_snapshot_equality_uses_state_revision_and_digest_only(
    changes: dict[str, object],
    same: bool,
) -> None:
    current = document(MemoryKind.PREFERENCE)
    uri = str(MemoryURI.from_address(current.address))
    left = VersionedSnapshot(
        identity=uri,
        state="found",
        value=current,
        revision=1,
        source_digest="0" * 64,
        size_bytes=10,
    )
    values = {
        "identity": uri,
        "state": "found",
        "value": current,
        "revision": 1,
        "source_digest": "0" * 64,
        "size_bytes": 999,
        **changes,
    }
    right = VersionedSnapshot(**values)

    assert MemoryRelationReadSetLoader._same_snapshot(left, right) is same
