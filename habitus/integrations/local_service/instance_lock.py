"""HTTP 服务实例锁；不参与 Conversation 或 Memory 领域并发。"""

from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from types import TracebackType

from habitus.infrastructure.store.filesystem import is_path_alias, make_owner_directories, restrict_file_to_owner

if sys.platform == "win32":
    import msvcrt

    # 锁住远在载荷之后的一个字节：载荷所在的区间不受锁影响，别的进程仍可读出锁文件里的 pid。
    _WINDOWS_LOCK_OFFSET = 0x7FFFFFFE

    def _lock_exclusive_nonblocking(descriptor: int) -> None:
        os.lseek(descriptor, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
        try:
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        except PermissionError as exc:
            raise BlockingIOError("lock region is held by another process") from exc
        finally:
            os.lseek(descriptor, 0, os.SEEK_SET)

    def _unlock(descriptor: int) -> None:
        os.lseek(descriptor, _WINDOWS_LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock_exclusive_nonblocking(descriptor: int) -> None:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(descriptor: int) -> None:
        fcntl.flock(descriptor, fcntl.LOCK_UN)


class ServiceInstanceLockError(RuntimeError):
    """同一 storage root 已有本地 HTTP 服务持有实例锁。"""


class ServiceInstanceLock:
    """使用进程持有的文件锁（POSIX ``flock``，Windows ``msvcrt.locking``）保证一套本地服务只有一个实例。"""

    def __init__(self, path: str | Path) -> None:
        resolved = Path(path).expanduser().absolute()
        if resolved.name in {"", ".", ".."}:
            raise ValueError("service lock path must name a file")
        self.path = resolved
        self._descriptor: int | None = None

    @property
    def acquired(self) -> bool:
        return self._descriptor is not None

    def acquire(self) -> None:
        if self._descriptor is not None:
            return
        self._ensure_parent()
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise ServiceInstanceLockError("failed to open the local service instance lock") from exc
        try:
            _lock_exclusive_nonblocking(descriptor)
        except BlockingIOError as exc:
            os.close(descriptor)
            raise ServiceInstanceLockError("another Habitus local service already owns this storage root") from exc
        try:
            payload = {
                "schema_version": "habitus_local_service_lock_v1",
                "pid": os.getpid(),
                "hostname": socket.gethostname(),
            }
            encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
            restrict_file_to_owner(self.path)
            os.ftruncate(descriptor, 0)
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        except BaseException:
            _unlock(descriptor)
            os.close(descriptor)
            raise
        self._descriptor = descriptor

    def release(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        try:
            _unlock(descriptor)
        finally:
            os.close(descriptor)

    def _ensure_parent(self) -> None:
        parent = self.path.parent
        make_owner_directories(parent)
        if is_path_alias(parent) or not parent.is_dir():
            raise ServiceInstanceLockError("service lock parent must be a real directory")
        if is_path_alias(self.path):
            raise ServiceInstanceLockError("service lock cannot be a symbolic link")

    def __enter__(self) -> ServiceInstanceLock:
        self.acquire()
        return self

    def __exit__(
        self,
        _error_type: type[BaseException] | None,
        _error: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.release()


__all__ = ["ServiceInstanceLock", "ServiceInstanceLockError"]
