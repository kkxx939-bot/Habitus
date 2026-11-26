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
        if isinstance(self.root, str) and not self.root.strip():
            raise ConfigError("storage.root cannot be empty")
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
    """Windows 上把存储根换成 ``\\\\?\\`` 扩展长度形式；POSIX 原样返回。

    Windows 默认没开长路径支持且常改不了注册表，存储根下的路径加上长会话 ID 很容易超过 260 字符；列目录、实例锁、SQLite
    等直接用路径的地方不经耐久写入层加前缀，所以在根上换一次，其下拼出来的路径都越过 MAX_PATH。
    """

    if sys.platform != "win32":
        return resolved
    from habitus.infrastructure.store.filesystem import windows_api

    return Path(windows_api.extended_path(resolved))


__all__ = ["StorageConfig"]
