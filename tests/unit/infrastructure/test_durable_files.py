"""可信根目录内不可变、可变和删除文件操作测试。"""

from pathlib import Path

import pytest

from infrastructure.store.filesystem.durable_io.atomic_file import (
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    atomic_replace_bytes,
    atomic_temporary_destination,
    durable_unlink,
    read_regular_bytes,
)
from infrastructure.store.filesystem.path_safety import (
    DurablePathIntegrityError,
    require_safe_artifact_path,
    validate_authoritative_tree,
)


def test_immutable_create_is_idempotent_for_same_bytes_and_conflicts_for_different_bytes(tmp_path: Path) -> None:
    target = tmp_path / "history" / "segment.json"
    assert atomic_create_bytes(target, b"first", artifact_root=tmp_path)
    assert not atomic_create_bytes(target, b"first", artifact_root=tmp_path)
    with pytest.raises(ImmutableArtifactConflictError, match="different content"):
        atomic_create_bytes(target, b"second", artifact_root=tmp_path)
    assert target.read_bytes() == b"first"


def test_atomic_replace_and_durable_unlink_are_safe_and_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "state" / "cursor.json"
    atomic_replace_bytes(target, b"one", artifact_root=tmp_path)
    atomic_replace_bytes(target, b"two", artifact_root=tmp_path)
    assert read_regular_bytes(target, artifact_root=tmp_path, max_bytes=3) == b"two"
    assert durable_unlink(target, artifact_root=tmp_path)
    assert not durable_unlink(target, artifact_root=tmp_path)


def test_read_is_bounded_and_rejects_non_regular_or_outside_paths(tmp_path: Path) -> None:
    target = tmp_path / "payload"
    target.write_bytes(b"1234")
    with pytest.raises(DurablePathIntegrityError, match="read bound"):
        read_regular_bytes(target, artifact_root=tmp_path, max_bytes=3)
    with pytest.raises(DurablePathIntegrityError, match="outside"):
        require_safe_artifact_path(tmp_path, tmp_path.parent / "escape", label="test")


def test_symlink_cannot_be_used_as_root_intermediate_or_leaf(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root_alias = tmp_path / "root-alias"
    root_alias.symlink_to(outside, target_is_directory=True)
    with pytest.raises(DurablePathIntegrityError, match="root"):
        require_safe_artifact_path(root_alias, root_alias / "file", label="test")

    root = tmp_path / "root"
    root.mkdir()
    (root / "alias").symlink_to(outside, target_is_directory=True)
    with pytest.raises(DurablePathIntegrityError, match="symbolic link"):
        atomic_replace_bytes(root / "alias" / "file", b"x", artifact_root=root)
    with pytest.raises(DurablePathIntegrityError, match="symbolic link directory"):
        validate_authoritative_tree(root, label="tree")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (".result.json." + "a" * 32 + ".tmp", "result.json"),
        ("result.json.tmp", None),
        (".result.json.short.tmp", None),
    ],
)
def test_atomic_temporary_name_parser_is_strict(name: str, expected: str | None) -> None:
    assert atomic_temporary_destination(name) == expected

