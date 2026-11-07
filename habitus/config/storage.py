"""Habitus 耐久数据共同根目录配置。"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

from habitus.config.loader import ConfigError, construct_config, group_fields, required_field
from habitus.infrastructure.store.filesystem import is_path_alias
from habitus.infrastructure.store.sqlite import SQLiteLockStoreConfig


@dataclass(frozen=True)
class StorageConfig:
    """Conversation、Memory 和 Workflow 的共同物理根目录。"""

    root: str | Path
    sqlite_lock: SQLiteLockStoreConfig = field(default_factory=SQLiteLockStoreConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.root, str | Path):
            raise TypeError("storage.root must be a filesystem path")
        if isinstance(self.root, str) and not self.root.strip():
            raise ConfigError("storage.root cannot be empty")
        if not isinstance(self.sqlite_lock, SQLiteLockStoreConfig):
            raise TypeError("storage.sqlite_lock must be SQLiteLockStoreConfig")
        requested = Path(self.root).expanduser().absolute()
        if is_path_alias(requested):
            raise ConfigError("storage.root cannot be a symbolic link")
        object.__setattr__(self, "root", _platform_root(requested.resolve(strict=False)))

    @classmethod
    def from_mapping(cls, value: object) -> StorageConfig:
        data = group_fields(cls, value, "config.storage")
        root = required_field(data, "root", path="config.storage")
        if not isinstance(root, str | Path):
            raise ConfigError("'config.storage.root' must be a filesystem path")
        return cls(
            root=root,
            sqlite_lock=construct_config(
                SQLiteLockStoreConfig,
                data.get("sqlite_lock", {}),
                "config.storage.sqlite_lock",
            ),
        )


def _platform_root(resolved: Path) -> Path:
    """Windows 上把存储根换成 ``\\\\?\\`` 扩展长度形式（用户 2026-11-07 定）。

    默认 Windows 没开长路径支持（LongPathsEnabled=0），而且公司机器常改不了注册表；存储根下的会话、记忆、SQLite 文件
    加上长会话 ID 很容易超过 260 字符。耐久写入层自己会加这个前缀，但列目录、实例锁、SQLite 等直接用路径的地方不经它；
    在根上换一次，其下拼出来的所有路径都带上前缀，统一越过 MAX_PATH。POSIX 原样返回。
    """

    if sys.platform != "win32":
        return resolved
    from habitus.infrastructure.store.filesystem import windows_api

    return Path(windows_api.extended_path(resolved))


__all__ = ["StorageConfig"]
