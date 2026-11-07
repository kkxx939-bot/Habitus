"""可信根目录内的原子耐久字节文件操作（Windows 实现，与 POSIX 的 ``atomic_file`` 同一接口）。

Windows 没有 ``dir_fd`` / ``*at()``、不能打开或 fsync 目录，所以这里按路径逐级操作：

- 每一级都用 ``lstat`` 确认是真实目录，拦住符号链接、目录联接和挂载点（名字代理重解析点）；
  读文件的最终句柄带 ``FILE_FLAG_OPEN_REPARSE_POINT`` 打开再校验。检查与使用之间残留一个极小竞争窗口
  （用户 2026-11-06 定的务实档，差异写在阶段 3 报告"Windows 耐久/安全语义差异"）。
- 原子替换与仅创建一次都用 ``MoveFileExW`` 加 ``MOVEFILE_WRITE_THROUGH``：改名落盘后才返回，
  替代 POSIX 上"fsync 父目录"的耐久保证。
- "仅本人可访问"用只含当前用户 SID、关闭继承的 DACL 表达，替代 0600 / 0700。
- 所有系统调用都走 ``\\\\?\\`` 扩展长度路径，越过 260 字符的 MAX_PATH，不需要改注册表。``\\\\?\\`` 会关掉
  路径规范化，所以每一段都先拒绝 ``.``、``..`` 以及以空格或点结尾的名字。
"""

from __future__ import annotations

import os
import stat
import sys
import uuid
from pathlib import Path

from habitus.foundation.ids import require_safe_path_segment
from habitus.infrastructure.store.filesystem import windows_api
from habitus.infrastructure.store.filesystem.durable_io.common import (
    DurableDirectoryEntry,
    ImmutableArtifactConflictError,
    _write_all,
    atomic_temporary_destination,
)
from habitus.infrastructure.store.filesystem.owner_access import (
    restrict_directory_to_owner,
)
from habitus.infrastructure.store.filesystem.path_safety import (
    DurablePathIntegrityError,
    is_path_alias,
    metadata_is_alias,
    require_safe_artifact_path,
)

assert sys.platform == "win32", "windows_file 是 Windows 实现；POSIX 用 atomic_file"


def _safe_segment(part: str) -> bool:
    """``\\\\?\\`` 关掉了规范化与设备名解析：每段都要是 Windows 上的普通名字——拒 ``.``/``..``、结尾空格或点、
    ``<>:"|?*``（含 ``:`` 备用数据流）、控制字符和 CON/NUL/COM1 这类保留设备名，与 ``ids`` 的名级检查同一套。"""

    try:
        require_safe_path_segment(part, "artifact path segment")
    except ValueError:
        return False
    return True


def _long(path: Path) -> str:
    return windows_api.extended_path(path)


def _lstat(path: Path) -> os.stat_result:
    return os.lstat(_long(path))


def _is_real_directory(path: Path) -> bool:
    metadata = _lstat(path)
    return stat.S_ISDIR(metadata.st_mode) and not metadata_is_alias(metadata)


def _make_directory(path: Path) -> bool:
    """新建一级目录并收成仅本人；已存在返回假。"""

    try:
        windows_api.create_owner_only_directory(_long(path))
    except FileExistsError:
        return False
    return True


def _try_restrict_directory(path: Path) -> None:
    try:
        restrict_directory_to_owner(path)
    except OSError:
        pass


def _walk_boundary(boundary: Path, *, create: bool) -> None:
    """从盘符或共享根起逐级确认（必要时新建）可信根的每一级都是真实目录。"""

    normalized = boundary.expanduser().absolute()
    if not normalized.anchor:
        raise DurablePathIntegrityError("artifact root must be an absolute path")
    current = Path(normalized.anchor)
    try:
        relative = normalized.relative_to(current)
        anchor_is_directory = stat.S_ISDIR(_lstat(current).st_mode)
    except (OSError, ValueError) as exc:
        raise DurablePathIntegrityError("artifact root anchor is not a safe directory") from exc
    if not anchor_is_directory:
        raise DurablePathIntegrityError("artifact root anchor is not a safe directory")
    for part in relative.parts:
        if not _safe_segment(part):
            raise DurablePathIntegrityError("artifact root contains an unsafe directory segment")
        current = current / part
        if create:
            try:
                _make_directory(current)
            except OSError as exc:
                raise DurablePathIntegrityError("artifact root cannot be created through an unsafe ancestor") from exc
        try:
            # 卷挂载到文件夹（企业 VDI 常见）与目录联接同一个标签；可信根的上级若是卷根就放行。
            is_directory = _is_real_directory(current) or os.path.ismount(current)
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise DurablePathIntegrityError("artifact root cannot traverse a symbolic link or non-directory") from exc
        if not is_directory:
            raise DurablePathIntegrityError("artifact root cannot traverse a symbolic link or non-directory")


def _control_parent(
    path: Path,
    artifact_root: str | Path,
    *,
    restrict_existing_directories: bool = True,
) -> Path:
    """逐级确认（必要时新建）目标父目录；语义同 POSIX 的 ``_open_control_parent``。"""

    candidate = Path(path).expanduser().absolute()
    boundary = Path(artifact_root).expanduser().absolute()
    if is_path_alias(boundary):
        raise DurablePathIntegrityError("artifact root cannot be a symbolic link")
    resolved_boundary = boundary.resolve()
    try:
        relative_parent = candidate.parent.relative_to(boundary)
    except ValueError:
        try:
            relative_parent = candidate.parent.relative_to(resolved_boundary)
            boundary = resolved_boundary
        except ValueError as exc:
            raise DurablePathIntegrityError("artifact path is outside its artifact root") from exc
    if not all(_safe_segment(part) for part in (*relative_parent.parts, candidate.name)):
        raise DurablePathIntegrityError("artifact path contains an unsafe directory segment")
    try:
        _walk_boundary(boundary, create=True)
    except (OSError, DurablePathIntegrityError) as exc:
        raise DurablePathIntegrityError("artifact root is not a safe directory") from exc
    if restrict_existing_directories:
        _try_restrict_directory(boundary)
    current = boundary
    for part in relative_parent.parts:
        current = current / part
        try:
            created = _make_directory(current)
            is_directory = _is_real_directory(current)
        except OSError as exc:
            raise DurablePathIntegrityError("artifact path cannot traverse a symbolic link or non-directory") from exc
        if not is_directory:
            raise DurablePathIntegrityError("artifact path cannot traverse a symbolic link or non-directory")
        if restrict_existing_directories and not created:
            _try_restrict_directory(current)
    return current


def _existing_parent(path: Path, artifact_root: str | Path) -> Path:
    candidate = Path(path).expanduser().absolute()
    boundary = Path(artifact_root).expanduser().absolute()
    try:
        relative_parent = candidate.parent.relative_to(boundary)
    except ValueError as exc:
        raise DurablePathIntegrityError("artifact path is outside its artifact root") from exc
    if not all(_safe_segment(part) for part in (*relative_parent.parts, candidate.name)):
        raise DurablePathIntegrityError("artifact path contains an unsafe directory segment")
    try:
        _walk_boundary(boundary, create=False)
    except FileNotFoundError:
        raise
    except (OSError, DurablePathIntegrityError) as exc:
        raise DurablePathIntegrityError("artifact root is not a safe directory") from exc
    current = boundary
    for part in relative_parent.parts:
        current = current / part
        try:
            is_directory = _is_real_directory(current)
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise DurablePathIntegrityError("artifact path cannot traverse a symbolic link or non-directory") from exc
        if not is_directory:
            raise DurablePathIntegrityError("artifact path cannot traverse a symbolic link or non-directory")
    return current


def _write_owner_only_temporary(temporary: Path, encoded: bytes) -> None:
    """排他新建临时文件（出生即带仅本人 DACL），写字节并落盘；凭据字节从不出现在宽权限文件里。"""

    descriptor = windows_api.create_owner_only_file(_long(temporary))
    try:
        _write_all(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _discard_temporary(temporary: Path) -> None:
    try:
        windows_api.unlink_with_retry(_long(temporary))
    except FileNotFoundError:
        pass


def _read_all(descriptor: int, maximum: int | None) -> bytes:
    chunks: list[bytes] = []
    total = 0
    while True:
        size = 1024 * 1024 if maximum is None else min(1024 * 1024, maximum - total + 1)
        chunk = os.read(descriptor, size)
        if not chunk:
            return b"".join(chunks)
        total += len(chunk)
        if maximum is not None and total > maximum:
            raise DurablePathIntegrityError("durable byte file exceeds its read bound")
        chunks.append(chunk)


def _read_collision(destination: Path) -> bytes:
    try:
        descriptor = windows_api.open_regular_file_for_read(_long(destination))
    except OSError as exc:
        raise ImmutableArtifactConflictError(
            "immutable artifact collision is unreadable or not a regular file"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if metadata_is_alias(metadata) or not stat.S_ISREG(metadata.st_mode):
            raise ImmutableArtifactConflictError("immutable artifact collision is not a regular file")
        return _read_all(descriptor, None)
    finally:
        os.close(descriptor)


def _lstat_or_none(path: Path) -> os.stat_result | None:
    try:
        return _lstat(path)
    except FileNotFoundError:
        return None


def atomic_create_bytes(path: Path, encoded: bytes, *, artifact_root: str | Path) -> bool:
    """仅创建一次不可变字节；相同内容的重放不产生变更。"""

    try:
        parent = _control_parent(path, artifact_root)
    except DurablePathIntegrityError as exc:
        raise ImmutableArtifactConflictError(str(exc)) from exc
    destination = parent / path.name
    temporary = parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        if _lstat_or_none(destination) is None:
            _write_owner_only_temporary(temporary, encoded)
            try:
                windows_api.move_file_write_through(_long(temporary), _long(destination), replace=False)
            except FileExistsError:
                pass
            else:
                return True
        if _read_collision(destination) != encoded:
            raise ImmutableArtifactConflictError("immutable artifact identity conflicts with different content")
        return False
    finally:
        _discard_temporary(temporary)


def atomic_replace_bytes(
    path: Path,
    encoded: bytes,
    *,
    artifact_root: str | Path,
    restrict_existing_directories: bool = True,
) -> None:
    """原子创建或替换一个可变普通文件（仅本人可访问）。

    写进用户自己的目录（如配置所在目录）时传 ``restrict_existing_directories=False``：只有新建的目录收成
    仅本人，已有目录的 DACL 不动。
    """

    parent = _control_parent(path, artifact_root, restrict_existing_directories=restrict_existing_directories)
    destination = parent / path.name
    temporary = parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        existing = _lstat_or_none(destination)
        if existing is not None and (metadata_is_alias(existing) or not stat.S_ISREG(existing.st_mode)):
            raise DurablePathIntegrityError("mutable artifact destination is not a regular file")
        _write_owner_only_temporary(temporary, encoded)
        windows_api.move_file_write_through(_long(temporary), _long(destination), replace=True)
    finally:
        _discard_temporary(temporary)


def durable_unlink(path: Path, *, artifact_root: str | Path) -> bool:
    """耐久删除可信根目录内的普通文件；目标不存在时幂等返回 ``False``。"""

    candidate = require_safe_artifact_path(artifact_root, path, label="durable deleted file")
    try:
        target = _existing_parent(candidate, artifact_root) / candidate.name
    except FileNotFoundError:
        return False
    metadata = _lstat_or_none(target)
    if metadata is None:
        return False
    if metadata_is_alias(metadata) or not stat.S_ISREG(metadata.st_mode):
        raise DurablePathIntegrityError("durable deleted path is not a regular file")
    windows_api.unlink_with_retry(_long(target))
    return True


def ensure_real_directory(path: Path, *, artifact_root: str | Path) -> None:
    """逐级创建并确认一个可信根内目录。"""

    candidate = require_safe_artifact_path(artifact_root, path, label="durable directory")
    _control_parent(candidate / ".habitus-directory-anchor", artifact_root)


def durable_rmdir(path: Path, *, artifact_root: str | Path) -> bool:
    """删除一个已验证父目录下的空真实目录。"""

    candidate = require_safe_artifact_path(artifact_root, path, label="durable deleted directory")
    if candidate == Path(artifact_root).expanduser().absolute():
        raise DurablePathIntegrityError("artifact root cannot be removed")
    try:
        target = _existing_parent(candidate, artifact_root) / candidate.name
    except FileNotFoundError:
        return False
    metadata = _lstat_or_none(target)
    if metadata is None:
        return False
    if metadata_is_alias(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise DurablePathIntegrityError("durable deleted path is not a directory")
    windows_api.rmdir_with_retry(_long(target))
    return True


def regular_file_exists(path: Path, *, artifact_root: str | Path) -> bool:
    """不跟随任一目录或叶子别名地判断普通文件是否存在。"""

    candidate = require_safe_artifact_path(artifact_root, path, label="durable regular file")
    try:
        target = _existing_parent(candidate, artifact_root) / candidate.name
    except FileNotFoundError:
        return False
    metadata = _lstat_or_none(target)
    if metadata is None:
        return False
    if metadata_is_alias(metadata) or not stat.S_ISREG(metadata.st_mode):
        raise DurablePathIntegrityError("durable path is not a regular file")
    return True


def _require_listable_directory(path: Path, message: str) -> bool:
    """目录存在且真实时返回真，不存在返回假，其余一律以 ``message`` 拒绝。"""

    try:
        is_directory = _is_real_directory(path)
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise DurablePathIntegrityError(message) from exc
    if not is_directory:
        raise DurablePathIntegrityError(message)
    return True


def real_directory_exists(path: Path, *, artifact_root: str | Path) -> bool:
    """不跟随别名地判断根内真实目录是否存在。"""

    candidate = require_safe_artifact_path(artifact_root, path, label="durable directory")
    boundary = Path(artifact_root).expanduser().absolute()
    if candidate == boundary:
        try:
            _walk_boundary(boundary, create=False)
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise DurablePathIntegrityError("durable directory is not safe") from exc
        return True
    try:
        parent = _existing_parent(candidate, artifact_root)
    except FileNotFoundError:
        return False
    return _require_listable_directory(parent / candidate.name, "durable directory is not safe")


def list_real_directory(
    path: Path,
    *,
    artifact_root: str | Path,
    max_entries: int,
) -> tuple[DurableDirectoryEntry, ...]:
    """有界列出不含别名的直接条目。"""

    if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries <= 0:
        raise ValueError("max_entries must be a positive integer")
    candidate = require_safe_artifact_path(artifact_root, path, label="durable listed directory")
    boundary = Path(artifact_root).expanduser().absolute()
    try:
        if candidate == boundary:
            _walk_boundary(boundary, create=False)
        else:
            directory = _existing_parent(candidate, artifact_root) / candidate.name
            if not _require_listable_directory(directory, "durable listed directory is not safe"):
                return ()
    except FileNotFoundError:
        return ()
    except OSError as exc:
        raise DurablePathIntegrityError("durable listed directory is not safe") from exc
    names = os.listdir(_long(candidate))
    if len(names) > max_entries:
        raise DurablePathIntegrityError("durable directory exceeds its enumeration bound")
    entries: list[DurableDirectoryEntry] = []
    for name in names:
        metadata = _lstat_or_none(candidate / name)
        if metadata is None:
            raise DurablePathIntegrityError("durable directory changed during enumeration")
        if metadata_is_alias(metadata):
            raise DurablePathIntegrityError("durable directory contains a symbolic link")
        entries.append(DurableDirectoryEntry(name=name, mode=metadata.st_mode))
    return tuple(sorted(entries, key=lambda entry: entry.name))


def read_regular_bytes(
    path: Path,
    *,
    artifact_root: str | Path,
    max_bytes: int,
) -> bytes:
    """有界读取一个普通文件，不跟随末级别名。"""

    maximum = int(max_bytes)
    if maximum <= 0:
        raise ValueError("max_bytes must be positive")
    candidate = require_safe_artifact_path(artifact_root, path, label="durable byte file")
    target = _existing_parent(candidate, artifact_root) / candidate.name
    try:
        descriptor = windows_api.open_regular_file_for_read(_long(target))
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise DurablePathIntegrityError("durable byte file cannot be opened safely") from exc
    try:
        metadata = os.fstat(descriptor)
        if metadata_is_alias(metadata):
            raise DurablePathIntegrityError("durable byte file cannot be opened safely")
        if not stat.S_ISREG(metadata.st_mode):
            raise DurablePathIntegrityError("durable byte path is not a regular file")
        if metadata.st_size > maximum:
            raise DurablePathIntegrityError("durable byte file exceeds its read bound")
        return _read_all(descriptor, maximum)
    finally:
        os.close(descriptor)


__all__ = [
    "DurableDirectoryEntry",
    "ImmutableArtifactConflictError",
    "atomic_create_bytes",
    "atomic_replace_bytes",
    "atomic_temporary_destination",
    "durable_rmdir",
    "durable_unlink",
    "ensure_real_directory",
    "list_real_directory",
    "real_directory_exists",
    "read_regular_bytes",
    "regular_file_exists",
]
