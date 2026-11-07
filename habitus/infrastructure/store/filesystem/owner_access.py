"""“仅本人可访问”的公共权限工具：POSIX 用权限位，Windows 用 DACL。

全仓库凡是要把文件收成 0600、把目录收成 0700、或检查"组与其他人没有权限"的地方都走这里，
不再各自 ``os.chmod`` / 读 ``st_mode``。Windows 上权限位不起作用，"仅本人"改由只含当前用户 SID、
关闭继承的 DACL 表达（``windows_api``，标准库 ctypes 实现，零新依赖）。设权限时 DACL 里不放 SYSTEM 与
Administrators（用户 2026-11-07 定：锁死机器上本人始终完全控制，不依赖管理员）。
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

if sys.platform == "win32":
    from habitus.infrastructure.store.filesystem import windows_api

    def _windows_path(path: str | os.PathLike[str]) -> str:
        """调用方给的任意路径（可能相对、带 ``..``）先按 Win32 规则规范成绝对路径，再换成扩展长度形式。"""

        return windows_api.extended_path(os.path.abspath(os.fspath(path)))


_OWNER_ONLY_FILE_MODE = 0o600
_OWNER_ONLY_DIRECTORY_MODE = 0o700
_GROUP_OR_OTHER_BITS = 0o077


def restrict_file_to_owner(path: str | os.PathLike[str]) -> None:
    """把一个已存在的文件收成只有当前用户可读写（POSIX 0600）。"""

    if sys.platform == "win32":
        windows_api.set_dacl_from_sddl(_windows_path(path), windows_api.owner_only_sddl(directory=False))
    else:
        os.chmod(path, _OWNER_ONLY_FILE_MODE)


def restrict_directory_to_owner(path: str | os.PathLike[str]) -> None:
    """把一个已存在的目录收成只有当前用户可进入（POSIX 0700；Windows 上其下新建项继承同一 DACL）。"""

    if sys.platform == "win32":
        windows_api.set_dacl_from_sddl(_windows_path(path), windows_api.owner_only_sddl(directory=True))
    else:
        os.chmod(path, _OWNER_ONLY_DIRECTORY_MODE)


def ensure_owner_directory(path: str | os.PathLike[str]) -> None:
    """必要时连同上级一起创建目录，再把末级目录收成仅本人可访问。"""

    Path(path).mkdir(parents=True, exist_ok=True, mode=_OWNER_ONLY_DIRECTORY_MODE)
    restrict_directory_to_owner(path)


def make_owner_directories(path: str | os.PathLike[str]) -> None:
    """建出目录及缺失的上级（已存在不报错）。POSIX 与 ``mkdir(mode=0o700, parents=True)`` 完全一致（末级 0700、
    上级按 umask）；Windows 上本次新建的每一级都出生即仅本人（不改已存在目录的 DACL）。"""

    if sys.platform == "win32":
        target = Path(os.path.abspath(os.fspath(path)))
        for directory in reversed((target, *target.parents)):
            try:
                windows_api.create_owner_only_directory(windows_api.extended_path(directory))
            except FileExistsError:
                # 已存在的目录不改 DACL，只给本次新建的目录收权限。
                continue
            except PermissionError:
                if not directory.is_dir():  # 盘根、无权新建但已存在的上级：只要已存在就继续往下
                    raise
    else:
        Path(path).mkdir(mode=_OWNER_ONLY_DIRECTORY_MODE, parents=True, exist_ok=True)


def grants_group_or_other_access(path: str | os.PathLike[str]) -> bool:
    """文件是否把访问权限放给了当前用户以外的人（POSIX 看组与其他人的权限位；Windows 逐条读 DACL）。"""

    if sys.platform == "win32":
        return not windows_api.dacl_grants_only_current_user(_windows_path(path))
    return bool(stat.S_IMODE(os.stat(path).st_mode) & _GROUP_OR_OTHER_BITS)


def describe_access(path: str | os.PathLike[str]) -> str:
    """给人看的权限描述：POSIX 是三位八进制权限，Windows 是 DACL 的 SDDL。"""

    if sys.platform == "win32":
        return windows_api.dacl_sddl(_windows_path(path))
    return f"mode={stat.S_IMODE(os.stat(path).st_mode):03o}"


__all__ = [
    "describe_access",
    "ensure_owner_directory",
    "grants_group_or_other_access",
    "make_owner_directories",
    "restrict_directory_to_owner",
    "restrict_file_to_owner",
]
