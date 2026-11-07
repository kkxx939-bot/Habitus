"""耐久写入层 POSIX 与 Windows 两份实现共用的类型与小工具。"""

from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass


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


_ATOMIC_TEMPORARY_NAME = re.compile(r"^\.(?P<destination>[^/]+)\.(?P<nonce>[0-9a-f]{32})\.tmp$")


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


__all__ = [
    "DurableDirectoryEntry",
    "ImmutableArtifactConflictError",
    "atomic_temporary_destination",
]
