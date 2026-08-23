"""可信根目录内的原子耐久字节文件操作。"""

from __future__ import annotations

import os
import re
import stat
import uuid
from dataclasses import dataclass
from pathlib import Path

from infrastructure.store.filesystem.path_safety import (
    DurablePathIntegrityError,
    require_safe_artifact_path,
)


class ImmutableArtifactConflictError(ValueError):
    """仅允许创建的产物身份已绑定到不同字节内容。"""


@dataclass(frozen=True)
class DurableDirectoryEntry:
    name: str
    mode: int

    def is_file(self) -> bool:
        return stat.S_ISREG(self.mode)

    def is_dir(self) -> bool:
        return stat.S_ISDIR(self.mode)


_ATOMIC_TEMPORARY_NAME = re.compile(
    r"^\.(?P<destination>[^/]+)\.(?P<nonce>[0-9a-f]{32})\.tmp$"
)

_DIRECTORY_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
)


def atomic_temporary_destination(name: str) -> str | None:
    """识别本模块可能遗留的临时文件名，并返回原目标文件名。"""

    if not isinstance(name, str):
        raise TypeError("atomic temporary name must be a string")
    match = _ATOMIC_TEMPORARY_NAME.fullmatch(name)
    if match is None:
        return None
    return match.group("destination")


def _write_all(descriptor: int, encoded: bytes) -> None:
    view = memoryview(encoded)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:  # pragma: no cover - 对操作系统契约的防御性检查。
            raise OSError("durable artifact write made no progress")
        view = view[written:]


def _open_boundary_directory(boundary: Path, *, create: bool) -> int:
    """Open every absolute path component relative to its pinned parent descriptor."""

    normalized = boundary.expanduser().absolute()
    anchor_text = normalized.anchor
    if not anchor_text:
        raise DurablePathIntegrityError("artifact root must be an absolute path")
    anchor = Path(anchor_text)
    try:
        relative = normalized.relative_to(anchor)
        descriptor = os.open(anchor, _DIRECTORY_FLAGS)
    except (OSError, ValueError) as exc:
        raise DurablePathIntegrityError("artifact root anchor is not a safe directory") from exc
    try:
        for part in relative.parts:
            if part in {"", ".", ".."}:
                raise DurablePathIntegrityError("artifact root contains an unsafe directory segment")
            if create:
                try:
                    os.mkdir(part, 0o700, dir_fd=descriptor)
                except FileExistsError:
                    pass
                except OSError as exc:
                    raise DurablePathIntegrityError(
                        "artifact root cannot be created through an unsafe ancestor"
                    ) from exc
            try:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except FileNotFoundError:
                raise
            except OSError as exc:
                raise DurablePathIntegrityError(
                    "artifact root cannot traverse a symbolic link or non-directory"
                ) from exc
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_control_parent(path: Path, artifact_root: str | Path) -> int:
    candidate = Path(path).expanduser().absolute()
    boundary = Path(artifact_root).expanduser().absolute()
    if boundary.is_symlink():
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
    if any(part in {"", ".", ".."} for part in relative_parent.parts):
        raise DurablePathIntegrityError("artifact path contains an unsafe directory segment")
    try:
        directory_descriptor = _open_boundary_directory(boundary, create=True)
    except (OSError, DurablePathIntegrityError) as exc:
        raise DurablePathIntegrityError("artifact root is not a safe directory") from exc
    try:
        try:
            os.fchmod(directory_descriptor, 0o700)
        except OSError:
            pass
        for part in relative_parent.parts:
            try:
                os.mkdir(part, 0o700, dir_fd=directory_descriptor)
            except FileExistsError:
                pass
            os.fsync(directory_descriptor)
            try:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=directory_descriptor)
            except OSError as exc:
                raise DurablePathIntegrityError(
                    "artifact path cannot traverse a symbolic link or non-directory"
                ) from exc
            os.close(directory_descriptor)
            directory_descriptor = child
            try:
                os.fchmod(directory_descriptor, 0o700)
            except OSError:
                pass
        return directory_descriptor
    except BaseException:
        os.close(directory_descriptor)
        raise


def _read_regular_file_at(directory_descriptor: int, name: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(name, flags, dir_fd=directory_descriptor)
    except OSError as exc:
        raise ImmutableArtifactConflictError(
            "immutable artifact collision is unreadable or not a regular file"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ImmutableArtifactConflictError("immutable artifact collision is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


def _open_existing_parent(path: Path, artifact_root: str | Path) -> int:
    candidate = Path(path).expanduser().absolute()
    boundary = Path(artifact_root).expanduser().absolute()
    try:
        relative_parent = candidate.parent.relative_to(boundary)
    except ValueError as exc:
        raise DurablePathIntegrityError("artifact path is outside its artifact root") from exc
    if any(part in {"", ".", ".."} for part in relative_parent.parts):
        raise DurablePathIntegrityError("artifact path contains an unsafe directory segment")
    try:
        directory_descriptor = _open_boundary_directory(boundary, create=False)
    except FileNotFoundError:
        raise
    except (OSError, DurablePathIntegrityError) as exc:
        raise DurablePathIntegrityError("artifact root is not a safe directory") from exc
    try:
        for part in relative_parent.parts:
            try:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=directory_descriptor)
            except FileNotFoundError:
                raise
            except OSError as exc:
                raise DurablePathIntegrityError(
                    "artifact path cannot traverse a symbolic link or non-directory"
                ) from exc
            os.close(directory_descriptor)
            directory_descriptor = child
        return directory_descriptor
    except BaseException:
        os.close(directory_descriptor)
        raise


def atomic_create_bytes(path: Path, encoded: bytes, *, artifact_root: str | Path) -> bool:
    """仅创建一次不可变字节；相同内容的重放不产生变更。"""

    try:
        parent_descriptor = _open_control_parent(path, artifact_root)
    except DurablePathIntegrityError as exc:
        raise ImmutableArtifactConflictError(str(exc)) from exc
    temporary_name = f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary_name, flags, 0o600, dir_fd=parent_descriptor)
        try:
            _write_all(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError:
            if _read_regular_file_at(parent_descriptor, path.name) != encoded:
                raise ImmutableArtifactConflictError(
                    "immutable artifact identity conflicts with different content"
                ) from None
            return False
        os.fsync(parent_descriptor)
        return True
    finally:
        try:
            os.unlink(temporary_name, dir_fd=parent_descriptor)
        except FileNotFoundError:
            pass
        os.close(parent_descriptor)


def atomic_replace_bytes(path: Path, encoded: bytes, *, artifact_root: str | Path) -> None:
    """原子创建或替换一个可变普通文件。"""

    parent_descriptor = _open_control_parent(path, artifact_root)
    temporary_name = f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        try:
            existing = os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISREG(existing.st_mode):
                raise DurablePathIntegrityError("mutable artifact destination is not a regular file")

        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(temporary_name, flags, 0o600, dir_fd=parent_descriptor)
        try:
            _write_all(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=parent_descriptor,
            dst_dir_fd=parent_descriptor,
        )
        os.chmod(path.name, 0o600, dir_fd=parent_descriptor, follow_symlinks=False)
        os.fsync(parent_descriptor)
    finally:
        try:
            os.unlink(temporary_name, dir_fd=parent_descriptor)
        except FileNotFoundError:
            pass
        os.close(parent_descriptor)


def durable_unlink(path: Path, *, artifact_root: str | Path) -> bool:
    """耐久删除可信根目录内的普通文件；目标不存在时幂等返回 ``False``。"""

    candidate = require_safe_artifact_path(artifact_root, path, label="durable deleted file")
    try:
        parent_descriptor = _open_existing_parent(candidate, artifact_root)
    except FileNotFoundError:
        return False
    try:
        try:
            metadata = os.stat(candidate.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return False
        if not stat.S_ISREG(metadata.st_mode):
            raise DurablePathIntegrityError("durable deleted path is not a regular file")
        os.unlink(candidate.name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
        return True
    finally:
        os.close(parent_descriptor)


def ensure_real_directory(path: Path, *, artifact_root: str | Path) -> None:
    """以逐级目录描述符创建并确认一个可信根内目录。"""

    candidate = require_safe_artifact_path(artifact_root, path, label="durable directory")
    descriptor = _open_control_parent(candidate / ".habitus-directory-anchor", artifact_root)
    os.close(descriptor)


def durable_rmdir(path: Path, *, artifact_root: str | Path) -> bool:
    """相对已验证父目录耐久删除一个空的真实目录。"""

    candidate = require_safe_artifact_path(artifact_root, path, label="durable deleted directory")
    if candidate == Path(artifact_root).expanduser().absolute():
        raise DurablePathIntegrityError("artifact root cannot be removed")
    try:
        parent_descriptor = _open_existing_parent(candidate, artifact_root)
    except FileNotFoundError:
        return False
    try:
        try:
            metadata = os.stat(candidate.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return False
        if not stat.S_ISDIR(metadata.st_mode):
            raise DurablePathIntegrityError("durable deleted path is not a directory")
        os.rmdir(candidate.name, dir_fd=parent_descriptor)
        os.fsync(parent_descriptor)
        return True
    finally:
        os.close(parent_descriptor)


def regular_file_exists(path: Path, *, artifact_root: str | Path) -> bool:
    """不跟随任一目录或叶子符号链接地判断普通文件是否存在。"""

    candidate = require_safe_artifact_path(artifact_root, path, label="durable regular file")
    try:
        parent_descriptor = _open_existing_parent(candidate, artifact_root)
    except FileNotFoundError:
        return False
    try:
        try:
            metadata = os.stat(candidate.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            return False
        if not stat.S_ISREG(metadata.st_mode):
            raise DurablePathIntegrityError("durable path is not a regular file")
        return True
    finally:
        os.close(parent_descriptor)


def real_directory_exists(path: Path, *, artifact_root: str | Path) -> bool:
    """不跟随符号链接地判断根内真实目录是否存在。"""

    candidate = require_safe_artifact_path(artifact_root, path, label="durable directory")
    boundary = Path(artifact_root).expanduser().absolute()
    if candidate == boundary:
        try:
            descriptor = _open_boundary_directory(boundary, create=False)
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise DurablePathIntegrityError("durable directory is not safe") from exc
        os.close(descriptor)
        return True
    try:
        parent_descriptor = _open_existing_parent(candidate, artifact_root)
    except FileNotFoundError:
        return False
    try:
        try:
            descriptor = os.open(candidate.name, _DIRECTORY_FLAGS, dir_fd=parent_descriptor)
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise DurablePathIntegrityError("durable directory is not safe") from exc
        os.close(descriptor)
        return True
    finally:
        os.close(parent_descriptor)


def list_real_directory(
    path: Path,
    *,
    artifact_root: str | Path,
    max_entries: int,
) -> tuple[DurableDirectoryEntry, ...]:
    """通过已验证目录描述符有界列出不含符号链接的直接条目。"""

    if isinstance(max_entries, bool) or not isinstance(max_entries, int) or max_entries <= 0:
        raise ValueError("max_entries must be a positive integer")
    candidate = require_safe_artifact_path(artifact_root, path, label="durable listed directory")
    boundary = Path(artifact_root).expanduser().absolute()
    parent_descriptor: int | None = None
    if candidate == boundary:
        try:
            descriptor = _open_boundary_directory(boundary, create=False)
        except FileNotFoundError:
            return ()
        except OSError as exc:
            raise DurablePathIntegrityError("durable listed directory is not safe") from exc
    else:
        try:
            parent_descriptor = _open_existing_parent(candidate, artifact_root)
        except FileNotFoundError:
            return ()
        try:
            descriptor = os.open(candidate.name, _DIRECTORY_FLAGS, dir_fd=parent_descriptor)
        except FileNotFoundError:
            os.close(parent_descriptor)
            return ()
        except OSError as exc:
            os.close(parent_descriptor)
            raise DurablePathIntegrityError("durable listed directory is not safe") from exc
    try:
        names = os.listdir(descriptor)
        if len(names) > max_entries:
            raise DurablePathIntegrityError("durable directory exceeds its enumeration bound")
        entries: list[DurableDirectoryEntry] = []
        for name in names:
            try:
                metadata = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            except FileNotFoundError:
                raise DurablePathIntegrityError("durable directory changed during enumeration") from None
            if stat.S_ISLNK(metadata.st_mode):
                raise DurablePathIntegrityError("durable directory contains a symbolic link")
            entries.append(DurableDirectoryEntry(name=name, mode=metadata.st_mode))
        return tuple(sorted(entries, key=lambda entry: entry.name))
    finally:
        os.close(descriptor)
        if parent_descriptor is not None:
            os.close(parent_descriptor)


def read_regular_bytes(
    path: Path,
    *,
    artifact_root: str | Path,
    max_bytes: int,
) -> bytes:
    """有界读取一个普通文件，不跟随末级符号链接。"""

    maximum = int(max_bytes)
    if maximum <= 0:
        raise ValueError("max_bytes must be positive")
    candidate = require_safe_artifact_path(artifact_root, path, label="durable byte file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    parent_descriptor = _open_existing_parent(candidate, artifact_root)
    try:
        try:
            descriptor = os.open(candidate.name, flags, dir_fd=parent_descriptor)
        except FileNotFoundError:
            raise
        except OSError as exc:
            raise DurablePathIntegrityError("durable byte file cannot be opened safely") from exc
        try:
            metadata = os.fstat(descriptor)
            if not stat.S_ISREG(metadata.st_mode):
                raise DurablePathIntegrityError("durable byte path is not a regular file")
            if metadata.st_size > maximum:
                raise DurablePathIntegrityError("durable byte file exceeds its read bound")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(descriptor, min(1024 * 1024, maximum - total + 1))
                if not chunk:
                    return b"".join(chunks)
                total += len(chunk)
                if total > maximum:
                    raise DurablePathIntegrityError("durable byte file exceeds its read bound")
                chunks.append(chunk)
        finally:
            os.close(descriptor)
    finally:
        os.close(parent_descriptor)


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
