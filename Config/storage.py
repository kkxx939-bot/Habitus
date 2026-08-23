"""Habitus 耐久数据共同根目录配置。"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from Config.loader import ConfigError, construct_config, group_fields, required_field
from infrastructure.store.sqlite import SQLiteLockStoreConfig


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
        if requested.is_symlink():
            raise ConfigError("storage.root cannot be a symbolic link")
        object.__setattr__(self, "root", requested.resolve(strict=False))

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


__all__ = ["StorageConfig"]
