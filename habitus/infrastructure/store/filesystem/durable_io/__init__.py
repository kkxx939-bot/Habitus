"""崩溃安全且独立于领域的本地持久化原语。

同一接口两份实现：POSIX 用逐级目录描述符 + ``*at()`` 的 ``atomic_file``，Windows 用按路径逐级校验重解析点、
``MoveFileExW`` 写穿改名和 DACL 的 ``windows_file``。在这里按平台二选一导入，调用方只认本包。
"""

import sys

from habitus.infrastructure.store.filesystem.durable_io.common import (
    DurableDirectoryEntry,
    ImmutableArtifactConflictError,
    atomic_temporary_destination,
)

if sys.platform == "win32":
    from habitus.infrastructure.store.filesystem.durable_io.windows_file import (
        atomic_create_bytes,
        atomic_replace_bytes,
        durable_rmdir,
        durable_unlink,
        ensure_real_directory,
        list_real_directory,
        read_regular_bytes,
        real_directory_exists,
        regular_file_exists,
    )
else:
    from habitus.infrastructure.store.filesystem.durable_io.atomic_file import (
        atomic_create_bytes,
        atomic_replace_bytes,
        durable_rmdir,
        durable_unlink,
        ensure_real_directory,
        list_real_directory,
        read_regular_bytes,
        real_directory_exists,
        regular_file_exists,
    )

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
