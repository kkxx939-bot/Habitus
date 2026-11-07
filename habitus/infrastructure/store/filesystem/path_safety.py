"""耐久本地产物的词法路径完整性检查。"""

from __future__ import annotations

import errno
import os
import stat
import sys
from pathlib import Path

# 重解析点标签的名字代理（name surrogate）位：符号链接、目录联接（junction）、挂载点都带它，它们把路径
# 指向别处；OneDrive 占位、去重这类重解析点不带，它们仍是真实文件或目录，不能当别名拦掉。
_REPARSE_TAG_NAME_SURROGATE = 0x20000000
# 与 CPython ``pathlib`` 的 ``_ignore_error`` 同一张表：这些错误等同"路径不存在"。
_IGNORED_ERRNOS = frozenset({errno.ENOENT, errno.ENOTDIR, errno.EBADF, errno.ELOOP})
_IGNORED_WINERRORS = frozenset({21, 123, 1921})


class DurablePathIntegrityError(RuntimeError):
    """耐久产物路径越出边界，或穿越边界内的符号链接。"""


def metadata_is_alias(metadata: os.stat_result) -> bool:
    """``lstat`` 结果是否为路径别名：符号链接，或 Windows 上的目录联接、挂载点等名字代理重解析点。"""

    if stat.S_ISLNK(metadata.st_mode):
        return True
    attributes = getattr(metadata, "st_file_attributes", 0)
    if not attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        return False
    return bool(getattr(metadata, "st_reparse_tag", 0) & _REPARSE_TAG_NAME_SURROGATE)


def is_path_alias(path: str | os.PathLike[str]) -> bool:
    """路径本身是否为别名（不跟随末级）；与 3.12/3.13 的 ``Path.is_symlink()`` 一样，路径不存在时返回假、其他错误
    照常上抛（3.14 起 pathlib 吞掉全部 OSError，这里有意不跟：看不了的路径不能当成"不是别名"放行）。

    POSIX 上除上述错误处理外等价于 ``Path.is_symlink()``；Windows 上另外拦住 ``is_symlink()`` 看不见的目录联接，并且对
    绝对路径用 ``\\\\?\\`` 扩展长度形式查，超过 260 字符的路径也查得到。
    """

    target: str | os.PathLike[str] = path
    if sys.platform == "win32" and os.path.isabs(path):
        from habitus.infrastructure.store.filesystem import windows_api

        target = windows_api.extended_path(os.path.abspath(path))
    try:
        metadata = os.lstat(target)
    except OSError as exc:
        if isinstance(exc, FileNotFoundError | NotADirectoryError) or exc.errno in _IGNORED_ERRNOS:
            return False
        if getattr(exc, "winerror", None) in _IGNORED_WINERRORS:
            return False
        raise
    except ValueError:
        return False
    return metadata_is_alias(metadata)


def require_safe_artifact_path(
    root: str | Path,
    path: str | Path,
    *,
    label: str,
) -> Path:
    """校验一个精确的词法路径，不跟随边界别名。"""

    boundary = Path(root).expanduser().absolute()
    candidate = Path(path).expanduser().absolute()
    try:
        relative = candidate.relative_to(boundary)
    except ValueError as exc:
        raise DurablePathIntegrityError(f"{label} is outside its artifact root") from exc
    if is_path_alias(boundary):
        raise DurablePathIntegrityError(f"{label} artifact root cannot be a symbolic link")
    current = boundary
    for part in relative.parts:
        if part in {"", ".", ".."}:
            raise DurablePathIntegrityError(f"{label} contains an unsafe path segment")
        current = current / part
        if is_path_alias(current):
            raise DurablePathIntegrityError(f"{label} cannot traverse a symbolic link")
    return candidate


def validate_authoritative_tree(root: str | Path, *, label: str) -> int:
    """拒绝权威目录树中任何位置的目录别名。

    被隔离的载荷属于事实记录，可能是未跟随目标而直接移动的符号链接。
    隔离目录本身仍是权威目录，因此必须是真实目录。叶子产物仍由各自的
    类型校验器负责；类型校验器可以区分权威证明和可丢弃的投影状态，
    并修复后者。
    """

    boundary = Path(root).expanduser().absolute()
    if not boundary.exists():
        return 0
    require_safe_artifact_path(boundary.parent, boundary, label=label)
    checked = 0
    for directory, names, filenames in os.walk(boundary, followlinks=False):
        current = Path(directory)
        relative = current.relative_to(boundary)
        if relative.parts and relative.parts[0] == "quarantine":
            names[:] = []
            continue
        checked += len(names) + len(filenames)
        for name in names:
            candidate = current / name
            if is_path_alias(candidate):
                raise DurablePathIntegrityError(f"{label} contains a symbolic link directory: {candidate}")
        if current == boundary:
            names[:] = [name for name in names if name != "quarantine"]
    return checked


__all__ = [
    "DurablePathIntegrityError",
    "is_path_alias",
    "metadata_is_alias",
    "require_safe_artifact_path",
    "validate_authoritative_tree",
]
