"""耐久写入层与"仅本人"权限工具的跨平台用例（阶段 3 Windows 原生）。

两边都跑的用例守"普通路径没被弄坏"和极限路径（长路径、中文与空格用户名）；只在 Windows 上跑的用例守
目录联接、``\\\\?\\`` 换算和 DACL；纯函数（SDDL 判定、别名判定）在每个平台都跑。
"""

from __future__ import annotations

import os
import stat
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from habitus.infrastructure.store.filesystem import (
    DurablePathIntegrityError,
    ImmutableArtifactConflictError,
    atomic_create_bytes,
    atomic_replace_bytes,
    durable_rmdir,
    durable_unlink,
    ensure_real_directory,
    grants_group_or_other_access,
    is_path_alias,
    list_real_directory,
    read_regular_bytes,
    real_directory_exists,
    regular_file_exists,
    restrict_directory_to_owner,
)
from habitus.infrastructure.store.filesystem.path_safety import metadata_is_alias
from tests.platform_helpers import (
    dacl_is_protected,
    deny_all_access,
    grant_everyone_read,
    is_private_directory,
    is_private_file,
    make_directory_alias,
)

windows_only = pytest.mark.skipif(sys.platform != "win32", reason="Windows 专属语义（目录联接、\\\\?\\、DACL）")


@windows_only
@pytest.mark.parametrize(
    ("dacl", "private"),
    [
        ("D:P(A;;FA;;;{sid})", True),
        ("D:P", True),
        ("D:P(A;;FA;;;{sid})(D;;FA;;;WD)", True),
        ("D:P(A;;FA;;;{sid})(A;;FA;;;SY)", False),
        ("D:P(A;;FA;;;{sid})(A;;FR;;;WD)", False),
        ("D:P(A;;FA;;;{sid})(A;;0x1200a9;;;BU)", False),
        ('D:P(A;;FA;;;{sid})(XA;;FR;;;WD;(@User.Department == "x(y)"))', False),
        ("D:NO_ACCESS_CONTROL", False),
        ("D:PNO_ACCESS_CONTROL", False),
    ],
)
def test_windows_dacl_judgement_reads_the_real_acl(tmp_path: Path, dacl: str, private: bool) -> None:
    """Windows：按二进制 ACE 与 SID 判定，空指针 DACL、条件 ACE、别的主体放权都判"放给了别人"，且从不抛错。"""

    assert sys.platform == "win32"
    from habitus.infrastructure.store.filesystem import windows_api

    target = tmp_path / "secret.yaml"
    target.write_bytes(b"x")
    windows_api.set_dacl_from_sddl(windows_api.extended_path(target), dacl.format(sid=windows_api.current_user_sid()))
    assert grants_group_or_other_access(target) is (not private)


@pytest.mark.parametrize(
    ("mode", "attributes", "tag", "expected"),
    [
        (stat.S_IFLNK | 0o777, 0, 0, True),
        (stat.S_IFDIR | 0o755, stat.FILE_ATTRIBUTE_REPARSE_POINT, 0xA0000003, True),  # 目录联接
        (stat.S_IFDIR | 0o755, stat.FILE_ATTRIBUTE_REPARSE_POINT, 0xA000000C, True),  # 目录符号链接
        (stat.S_IFREG | 0o644, stat.FILE_ATTRIBUTE_REPARSE_POINT, 0x9000001A, False),  # OneDrive 占位
        (stat.S_IFREG | 0o644, stat.FILE_ATTRIBUTE_REPARSE_POINT, 0x80000013, False),  # 去重
        (stat.S_IFDIR | 0o755, 0, 0, False),
    ],
)
def test_alias_judgement_covers_junctions_but_not_cloud_placeholders(
    mode: int, attributes: int, tag: int, expected: bool
) -> None:
    metadata = SimpleNamespace(st_mode=mode, st_file_attributes=attributes, st_reparse_tag=tag)
    assert metadata_is_alias(metadata) is expected  # type: ignore[arg-type]


def test_durable_writes_are_private_to_the_current_user(tmp_path: Path) -> None:
    root = tmp_path / "root"
    target = root / "nested" / "state.json"
    atomic_replace_bytes(target, b"secret", artifact_root=root)
    created = root / "nested" / "created.json"
    assert atomic_create_bytes(created, b"once", artifact_root=root)

    assert is_private_file(target)
    assert is_private_file(created)
    assert is_private_directory(root)
    assert is_private_directory(root / "nested")
    assert not grants_group_or_other_access(target)


def test_replace_into_a_shared_user_directory_keeps_the_file_private(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir()
    grant_everyone_read(shared)
    target = shared / "config.yaml"

    atomic_replace_bytes(target, b"credentials: {}", artifact_root=shared, restrict_existing_directories=False)

    assert is_private_file(target)
    assert not is_private_directory(shared)
    assert grants_group_or_other_access(shared)


def test_loosened_existing_file_is_tightened_again_by_replace(tmp_path: Path) -> None:
    target = tmp_path / "root" / "state.json"
    atomic_replace_bytes(target, b"one", artifact_root=tmp_path / "root")
    grant_everyone_read(target)
    assert grants_group_or_other_access(target)

    atomic_replace_bytes(target, b"two", artifact_root=tmp_path / "root")

    assert is_private_file(target)
    assert read_regular_bytes(target, artifact_root=tmp_path / "root", max_bytes=8) == b"two"


def _round_trip(root: Path, target: Path) -> None:
    atomic_replace_bytes(target, b"first", artifact_root=root)
    atomic_replace_bytes(target, b"second", artifact_root=root)
    assert read_regular_bytes(target, artifact_root=root, max_bytes=16) == b"second"
    assert regular_file_exists(target, artifact_root=root)
    immutable = target.with_name("immutable.bin")
    assert atomic_create_bytes(immutable, b"x", artifact_root=root)
    assert not atomic_create_bytes(immutable, b"x", artifact_root=root)
    with pytest.raises(ImmutableArtifactConflictError):
        atomic_create_bytes(immutable, b"y", artifact_root=root)
    names = [entry.name for entry in list_real_directory(target.parent, artifact_root=root, max_entries=8)]
    assert names == sorted([immutable.name, target.name])
    assert durable_unlink(target, artifact_root=root)
    assert durable_unlink(immutable, artifact_root=root)
    assert not durable_unlink(target, artifact_root=root)
    assert durable_rmdir(target.parent, artifact_root=root)
    assert not real_directory_exists(target.parent, artifact_root=root)


def test_ordinary_short_path_round_trip(tmp_path: Path) -> None:
    root = tmp_path / "habitus"
    _round_trip(root, root / "memory" / "profile.md")


def test_user_name_with_chinese_and_spaces_round_trip(tmp_path: Path) -> None:
    root = tmp_path / "Users" / "张三 李四" / ".habitus" / "存储 根"
    ensure_real_directory(root / "记忆 树", artifact_root=root)
    _round_trip(root, root / "记忆 树" / "偏好 设置.md")


def _deep_root(tmp_path: Path, minimum_length: int) -> Path:
    root = tmp_path
    while len(str(root)) < minimum_length:
        root = root / ("深层目录-" + "d" * 40)
    return root


def test_path_longer_than_max_path_round_trip(tmp_path: Path) -> None:
    root = _deep_root(tmp_path, 300)
    target = root / "events" / "2026" / "11" / "07" / ("事件-" + "e" * 60 + ".md")
    assert len(str(target)) > 260
    _round_trip(root, target)


@pytest.mark.parametrize("total_length", [259, 260, 261])
def test_max_path_boundary_lengths(tmp_path: Path, total_length: int) -> None:
    root = _deep_root(tmp_path, 200)
    parent = root / "boundary"
    leaf_length = total_length - len(str(parent)) - 1
    assert 1 <= leaf_length <= 200
    target = parent / ("b" * leaf_length)
    assert len(str(target)) == total_length
    atomic_replace_bytes(target, b"edge", artifact_root=root)
    assert read_regular_bytes(target, artifact_root=root, max_bytes=4) == b"edge"
    assert durable_unlink(target, artifact_root=root)


def test_directory_alias_is_rejected_as_root_intermediate_and_entry(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "payload").write_bytes(b"outside")

    alias_root = tmp_path / "alias-root"
    make_directory_alias(alias_root, outside)
    assert is_path_alias(alias_root)
    with pytest.raises(DurablePathIntegrityError):
        atomic_replace_bytes(alias_root / "file", b"x", artifact_root=alias_root)

    root = tmp_path / "root"
    root.mkdir()
    make_directory_alias(root / "alias", outside)
    with pytest.raises(DurablePathIntegrityError):
        atomic_replace_bytes(root / "alias" / "file", b"x", artifact_root=root)
    with pytest.raises(DurablePathIntegrityError):
        read_regular_bytes(root / "alias" / "payload", artifact_root=root, max_bytes=64)
    with pytest.raises(DurablePathIntegrityError):
        list_real_directory(root, artifact_root=root, max_entries=8)
    with pytest.raises(DurablePathIntegrityError):
        durable_rmdir(root / "alias", artifact_root=root)
    assert (outside / "payload").read_bytes() == b"outside"
    assert not (outside / "file").exists()


@windows_only
def test_extended_path_conversion() -> None:
    assert sys.platform == "win32"
    from habitus.infrastructure.store.filesystem import windows_api

    assert windows_api.extended_path("C:\\Users\\张三 李四\\x") == "\\\\?\\C:\\Users\\张三 李四\\x"
    assert windows_api.extended_path("C:/a/b") == "\\\\?\\C:\\a\\b"
    assert windows_api.extended_path("\\\\server\\share\\x") == "\\\\?\\UNC\\server\\share\\x"
    assert windows_api.extended_path("\\\\?\\C:\\already") == "\\\\?\\C:\\already"
    with pytest.raises(ValueError):
        windows_api.extended_path("relative\\x")
    with pytest.raises(ValueError):
        windows_api.extended_path("C:relative")


@windows_only
@pytest.mark.parametrize("segment", ["trailing.", "trailing ", "..", "a:stream", "con", "nul.txt", "x<y", "tab\there"])
def test_segments_that_extended_paths_would_not_normalize_are_rejected(tmp_path: Path, segment: str) -> None:
    root = tmp_path / "root"
    with pytest.raises(DurablePathIntegrityError):
        atomic_replace_bytes(Path(f"{root}\\{segment}\\file"), b"x", artifact_root=root)


@windows_only
def test_replace_waits_out_a_briefly_open_reader_without_share_delete(tmp_path: Path) -> None:
    root = tmp_path / "root"
    target = root / "state.json"
    atomic_replace_bytes(target, b"old", artifact_root=root)
    handle = open(target, "rb")  # noqa: SIM115 - 普通 open() 不带 FILE_SHARE_DELETE，会挡住改名
    releaser = threading.Timer(0.1, handle.close)
    releaser.start()
    started = time.monotonic()
    try:
        atomic_replace_bytes(target, b"new", artifact_root=root)
    finally:
        releaser.join()
        handle.close()
    assert time.monotonic() - started >= 0.05
    assert read_regular_bytes(target, artifact_root=root, max_bytes=8) == b"new"


@windows_only
def test_durable_reader_does_not_block_unlink(tmp_path: Path) -> None:
    """耐久层自己的读句柄带 FILE_SHARE_DELETE：开着时写者照样能删。（覆盖式改名仍会被开着的句柄挡住，靠有界重试。）"""

    assert sys.platform == "win32"
    import os

    from habitus.infrastructure.store.filesystem import windows_api

    root = tmp_path / "root"
    target = root / "state.json"
    atomic_replace_bytes(target, b"old", artifact_root=root)
    descriptor = windows_api.open_regular_file_for_read(windows_api.extended_path(target))
    try:
        assert durable_unlink(target, artifact_root=root)
    finally:
        os.close(descriptor)
    assert not regular_file_exists(target, artifact_root=root)


@windows_only
def test_explicitly_restricted_files_and_directories_are_protected_from_inheritance(tmp_path: Path) -> None:
    root = tmp_path / "root"
    target = root / "nested" / "state.json"
    atomic_replace_bytes(target, b"x", artifact_root=root)
    assert dacl_is_protected(target)
    assert dacl_is_protected(root)
    assert dacl_is_protected(root / "nested")


@windows_only
def test_a_volume_mount_ancestor_is_allowed_but_a_plain_junction_is_not(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """卷挂载到文件夹（企业 VDI）与目录联接同一标签：可信根的上级若是卷根就放行，普通联接照旧拒绝。"""

    import os

    real = tmp_path / "real"
    real.mkdir()
    mounted = tmp_path / "mounted"
    make_directory_alias(mounted, real)
    root = mounted / "habitus"
    with pytest.raises(DurablePathIntegrityError):
        atomic_replace_bytes(root / "x.json", b"x", artifact_root=root)
    original = os.path.ismount
    monkeypatch.setattr(os.path, "ismount", lambda path: Path(path) == mounted or original(path))
    atomic_replace_bytes(root / "x.json", b"x", artifact_root=root)
    assert (real / "habitus" / "x.json").read_bytes() == b"x"


def test_a_leaf_file_symlink_is_never_read_through(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"outside")
    (root / "leaf").symlink_to(outside)
    with pytest.raises(DurablePathIntegrityError):
        read_regular_bytes(root / "leaf", artifact_root=root, max_bytes=64)
    with pytest.raises(DurablePathIntegrityError):
        atomic_replace_bytes(root / "leaf", b"x", artifact_root=root)
    assert outside.read_bytes() == b"outside"


def _metadata_is_unreadable(path: Path) -> bool:
    try:
        os.lstat(path)
    except PermissionError:
        return True
    return False


@pytest.mark.skipif(sys.platform != "win32" and os.geteuid() == 0, reason="POSIX 上 root 不受权限位限制")
def test_alias_check_fails_closed_when_the_parent_cannot_be_read(tmp_path: Path) -> None:
    """与 3.12/3.13 的 ``Path.is_symlink()`` 一样：路径不存在才算"不是别名"，看不了（权限不足）要上抛，不能当成放行
    （3.14 起 pathlib 吞掉全部 OSError，这里有意不跟）。"""

    locked = tmp_path / "locked"
    locked.mkdir()
    child = locked / "child"
    child.mkdir()
    if sys.platform == "win32":
        # Windows 人人有"跳过遍历检查"特权，只锁父目录时子项仍可直接查属性；子项自己也锁上才是"看不了"。
        deny_all_access(child)
    deny_all_access(locked)
    try:
        if not _metadata_is_unreadable(child):
            # 前提造不出来：这台机器上当前账号对拒绝 ACE 视而不见（管理员带备份特权、或系统按名查元数据不走这条检查），
            # 元数据照样读得到，is_path_alias 拿到的就是真实结果，没有"看不了还放行"可防
            pytest.skip("当前账号仍读得到被拒目录的元数据，造不出'看不了'的前提")
        with pytest.raises(PermissionError):
            is_path_alias(child)
    finally:
        restrict_directory_to_owner(locked)
        restrict_directory_to_owner(child)
    assert not is_path_alias(tmp_path / "missing" / "child")
