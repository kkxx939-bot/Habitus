"""Behavior SQLite 的路径、连接、PRAGMA 与显式事务边界。"""

from __future__ import annotations

import os
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from behavior.config import BehaviorStoreConfig
from behavior.errors import BehaviorStoreError, LegacyBehaviorStoreError


class BehaviorSQLiteConnection:
    def __init__(
        self,
        root: Path,
        *,
        config: BehaviorStoreConfig,
    ) -> None:
        if not isinstance(root, Path) or not root.is_absolute():
            raise ValueError("Behavior root must be an absolute Path")
        if not isinstance(config, BehaviorStoreConfig):
            raise TypeError("config must be BehaviorStoreConfig")
        self.root = root
        self.path = root / "behavior.sqlite3"
        self.config = config

    @property
    def initialized(self) -> bool:
        return self.path.is_file() and not self.path.is_symlink()

    def initialize_path(self) -> Path:
        if any(path.is_symlink() for path in (self.root, *self.root.parents)):
            raise BehaviorStoreError("Behavior root and its ancestors cannot be symbolic links")
        if self.root.exists() and not self.root.is_dir():
            raise BehaviorStoreError("Behavior root must be a directory")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        legacy_path = self.root / ("evidence_" + "claims.sqlite3")
        if legacy_path.exists() or legacy_path.is_symlink():
            raise LegacyBehaviorStoreError(
                "Legacy Behavior data is incompatible and requires explicit Behavior-only cleanup; "
                "Memory and Conversation data are unaffected"
            )
        if self.path.is_symlink():
            raise BehaviorStoreError("Behavior database cannot be a symbolic link")
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise BehaviorStoreError("Behavior database path cannot be opened safely") from exc
        else:
            os.close(descriptor)
        os.chmod(self.path, 0o600)
        return self.path

    def connect(self) -> sqlite3.Connection:
        if not self.initialized:
            raise BehaviorStoreError("Behavior database is not initialized")
        if any(
            path.is_symlink()
            for path in (Path(str(self.path) + "-wal"), Path(str(self.path) + "-shm"))
        ):
            raise BehaviorStoreError("Behavior database sidecars cannot be symbolic links")
        connection = sqlite3.connect(
            self.path,
            timeout=self.config.sqlite_timeout_seconds,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(f"PRAGMA busy_timeout={int(self.config.sqlite_timeout_seconds * 1000)}")
        mode = connection.execute("PRAGMA journal_mode").fetchone()[0]
        if str(mode).casefold() != "wal":
            mode = self._enable_wal(connection)
        if str(mode).casefold() != "wal":
            connection.close()
            raise BehaviorStoreError("Behavior database did not enter WAL mode")
        if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
            connection.close()
            raise BehaviorStoreError("Behavior database foreign keys are disabled")
        return connection

    def _enable_wal(self, connection: sqlite3.Connection) -> str:
        deadline = time.monotonic() + self.config.sqlite_timeout_seconds
        while True:
            try:
                return str(connection.execute("PRAGMA journal_mode=WAL").fetchone()[0])
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).casefold() and "busy" not in str(exc).casefold():
                    connection.close()
                    raise BehaviorStoreError("Behavior database WAL initialization failed") from exc
                if time.monotonic() >= deadline:
                    connection.close()
                    raise BehaviorStoreError("Behavior database WAL initialization timed out") from exc
                time.sleep(0.02)

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        connection = self.connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def database_size(self, connection: sqlite3.Connection | None = None) -> int:
        logical_size = 0
        if connection is not None:
            page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
            page_count = int(connection.execute("PRAGMA page_count").fetchone()[0])
            logical_size = page_size * page_count
        physical_size = sum(
            path.stat().st_size
            for path in (self.path, Path(str(self.path) + "-wal"), Path(str(self.path) + "-shm"))
            if path.exists() and not path.is_symlink()
        )
        return max(logical_size, physical_size)

    def projected_write_size(
        self,
        connection: sqlite3.Connection,
        *,
        encoded_bytes: int,
        btree_writes: int,
    ) -> int:
        if encoded_bytes < 0 or btree_writes <= 0:
            raise ValueError("write estimate inputs must be positive")
        page_size = int(connection.execute("PRAGMA page_size").fetchone()[0])
        wal_frame_bytes = page_size + 24
        conservative_journal_growth = wal_frame_bytes * (btree_writes * 2 + 1) + 32
        return self.database_size(connection) + encoded_bytes + conservative_journal_growth

    def close(self) -> None:
        """连接按操作关闭；该方法保留显式 Runtime 生命周期对称性。"""
