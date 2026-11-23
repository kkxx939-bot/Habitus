"""目录 L0/L1 刷新结果。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from habitus.memory.model import MemoryDirectory


class MemorySemanticRefreshStatus(str, Enum):
    WRITTEN = "written"
    DELETED = "deleted"
    UNCHANGED = "unchanged"
    MISSING = "missing"


@dataclass(frozen=True)
class MemorySemanticRefreshResult:
    """一个目录的 L0/L1 刷新结果。"""

    directory: MemoryDirectory
    status: MemorySemanticRefreshStatus
    abstract_path: Path | None = None
    overview_path: Path | None = None


__all__ = [
    "MemorySemanticRefreshResult",
    "MemorySemanticRefreshStatus",
]
