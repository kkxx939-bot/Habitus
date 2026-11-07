"""跨平台测试工具："仅本人可访问"的断言与目录别名的构造。

POSIX 上"仅本人"就是 0600 / 0700 权限位；Windows 上是只含当前用户 SID、关闭继承（``D:P``）的 DACL。
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

from habitus.infrastructure.store.filesystem.owner_access import grants_group_or_other_access

IS_WINDOWS = sys.platform == "win32"


def _windows_dacl(path: Path) -> tuple[str, str]:
    assert sys.platform == "win32"
    from habitus.infrastructure.store.filesystem import windows_api

    return windows_api.dacl_sddl(windows_api.extended_path(path.absolute())), windows_api.current_user_sid()


def is_private_file(path: Path) -> bool:
    """POSIX 恰为 0600；Windows 的 DACL 只放给当前用户（显式设的或从仅本人目录继承的都算）。"""

    if sys.platform == "win32":
        return not grants_group_or_other_access(path)
    return stat.S_IMODE(os.stat(path).st_mode) == 0o600


def is_private_directory(path: Path) -> bool:
    """POSIX 恰为 0700；Windows 的 DACL 只放给当前用户，且这条 ACE 继承给其下的新文件与子目录。"""

    if sys.platform == "win32":
        sddl, _sid = _windows_dacl(path)
        return not grants_group_or_other_access(path) and "OICI" in sddl
    return stat.S_IMODE(os.stat(path).st_mode) == 0o700


def dacl_is_protected(path: Path) -> bool:
    """Windows：DACL 关闭了继承（``D:P``），父目录以后再放宽也传不下来。"""

    sddl, _sid = _windows_dacl(path)
    return sddl.startswith("D:P")


def deny_all_access(path: Path) -> None:
    """让文件对当前用户也不可读写（POSIX 0000；Windows 空 DACL），用于证明某次读没有碰盘。"""

    if sys.platform == "win32":
        from habitus.infrastructure.store.filesystem import windows_api

        windows_api.set_dacl_from_sddl(windows_api.extended_path(path.absolute()), "D:P")
    else:
        os.chmod(path, 0)


def make_directory_alias(link: Path, target: Path) -> None:
    """建一个指向 ``target`` 目录的别名：POSIX 用符号链接，Windows 用不需要特权的目录联接（junction）。"""

    if sys.platform == "win32":
        import _winapi

        _winapi.CreateJunction(str(target), str(link))
    else:
        link.symlink_to(target, target_is_directory=True)


def grant_everyone_read(path: Path) -> None:
    """把一个文件或目录放宽给所有人可读（POSIX 0644/0755；Windows 加一条可继承的 Everyone 读 ACE）。"""

    if sys.platform == "win32":
        from habitus.infrastructure.store.filesystem import windows_api

        sid = windows_api.current_user_sid()
        windows_api.set_dacl_from_sddl(
            windows_api.extended_path(path.absolute()),
            f"D:AI(A;OICI;FA;;;{sid})(A;OICI;GR;;;WD)",
        )
    else:
        os.chmod(path, 0o755 if path.is_dir() else 0o644)


def grant_group_read(path: Path) -> None:
    """只放给"组"读（POSIX 0640；Windows 加一条 Builtin Users 读 ACE）——组和其他人任何一位放开都算不私有。"""

    if sys.platform == "win32":
        from habitus.infrastructure.store.filesystem import windows_api

        sid = windows_api.current_user_sid()
        windows_api.set_dacl_from_sddl(windows_api.extended_path(path.absolute()), f"D:P(A;;FA;;;{sid})(A;;FR;;;BU)")
    else:
        os.chmod(path, 0o640)


def call_within_seconds(seconds: float, call):  # type: ignore[no-untyped-def]
    """在限定秒数内调用并返回结果；不返回就抛 TimeoutError（防住死循环挂住整个测试进程）。

    用守护线程而不是 ``signal.alarm``：后者 Windows 上没有。超时后那条线程留在后台，不影响判定。
    """

    import threading

    outcome: dict[str, object] = {}

    def run() -> None:
        try:
            outcome["value"] = call()
        except BaseException as exc:  # 原样交回调用线程
            outcome["error"] = exc

    worker = threading.Thread(target=run, daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise TimeoutError(f"did not return within {seconds} s")
    if "error" in outcome:
        raise outcome["error"]  # type: ignore[misc]
    return outcome["value"]
