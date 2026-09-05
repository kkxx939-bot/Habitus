"""HTTP 服务实例锁；不参与 Conversation 或 Memory 领域并发。"""

from __future__ import annotations

import json
import os
import socket
from pathlib import Path
from types import TracebackType

try:
    import fcntl
except ImportError as exc:  # pragma: no cover - 当前正式目标为 macOS/Linux
    raise RuntimeError("Habitus local service requires POSIX file locking") from exc


class ServiceInstanceLockError(RuntimeError):
    """同一 storage root 已有本地 HTTP 服务持有实例锁。"""


class ServiceInstanceLock:
    """使用进程持有的 flock 保证一套本地服务只有一个实例。"""

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
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise ServiceInstanceLockError("failed to open the local service instance lock") from exc
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
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
            os.ftruncate(descriptor, 0)
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        except BaseException:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
            raise
        self._descriptor = descriptor

    def release(self) -> None:
        descriptor = self._descriptor
        if descriptor is None:
            return
        self._descriptor = None
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    def _ensure_parent(self) -> None:
        parent = self.path.parent
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if parent.is_symlink() or not parent.is_dir():
            raise ServiceInstanceLockError("service lock parent must be a real directory")
        if self.path.is_symlink():
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
