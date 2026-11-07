"""可信根目录内不可变、可变和删除文件操作测试。

经 ``durable_io`` 包导入：POSIX 上测 ``atomic_file``，Windows 上测 ``windows_file``，两份实现过同一组用例。
"""

import sys
from pathlib import Path

import pytest

from habitus.infrastructure.store.filesystem.durable_io import (
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    atomic_replace_bytes,
    atomic_temporary_destination,
    durable_unlink,
    read_regular_bytes,
)
from habitus.infrastructure.store.filesystem.path_safety import (
    DurablePathIntegrityError,
    require_safe_artifact_path,
    validate_authoritative_tree,
)

# 当前平台实际生效的那份实现模块（POSIX: atomic_file；Windows: windows_file）。
atomic_file_module = sys.modules[atomic_replace_bytes.__module__]


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


def test_read_rejects_parent_directory_replaced_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "root"
    inside = root / "sub"
    inside.mkdir(parents=True)
    target = inside / "payload"
    target.write_bytes(b"inside")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "payload").write_bytes(b"outside")
    parked = root / "parked"
    original = atomic_file_module.require_safe_artifact_path
    swapped = False

    def validate_then_swap(root_value, path_value, *, label):
        nonlocal swapped
        candidate = original(root_value, path_value, label=label)
        if not swapped:
            inside.rename(parked)
            inside.symlink_to(outside, target_is_directory=True)
            swapped = True
        return candidate

    monkeypatch.setattr(atomic_file_module, "require_safe_artifact_path", validate_then_swap)

    with pytest.raises(DurablePathIntegrityError):
        read_regular_bytes(target, artifact_root=root, max_bytes=64)


def test_read_rejects_artifact_root_ancestor_replaced_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted = tmp_path / "trusted"
    root = trusted / "root"
    target = root / "sub" / "payload"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"inside")
    outside = tmp_path / "outside"
    outside_target = outside / "root" / "sub" / "payload"
    outside_target.parent.mkdir(parents=True)
    outside_target.write_bytes(b"outside")
    parked = tmp_path / "trusted-parked"
    original = atomic_file_module.require_safe_artifact_path
    swapped = False

    def validate_then_swap(root_value, path_value, *, label):
        nonlocal swapped
        candidate = original(root_value, path_value, label=label)
        if not swapped:
            trusted.rename(parked)
            trusted.symlink_to(outside, target_is_directory=True)
            swapped = True
        return candidate

    monkeypatch.setattr(atomic_file_module, "require_safe_artifact_path", validate_then_swap)

    with pytest.raises(DurablePathIntegrityError):
        read_regular_bytes(target, artifact_root=root, max_bytes=64)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX 侧用 fchmod 收紧；Windows 侧按 ACL 尽力收紧，见 windows_file")
def test_a_storage_directory_that_cannot_be_restricted_refuses_the_write(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """存储根与途经目录收不紧到 0700（目录不归本用户等）就拒写，不带着宽权限继续（W16 ``except: pass`` 逐条判）。"""

    def refuse(descriptor: int, mode: int) -> None:
        raise PermissionError("operation not permitted")

    monkeypatch.setattr(atomic_file_module.os, "fchmod", refuse)
    target = tmp_path / "root" / "nested" / "file"
    with pytest.raises(DurablePathIntegrityError, match="cannot be restricted to its owner"):
        atomic_replace_bytes(target, b"x", artifact_root=tmp_path / "root")
    assert not target.exists()
    atomic_replace_bytes(target, b"x", artifact_root=tmp_path / "root", restrict_existing_directories=False)
    assert target.read_bytes() == b"x"  # 明说不收紧已有目录的调用照常写（新建目录仍以 0700 创建）
