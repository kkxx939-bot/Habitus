"""行为目录 L0/L1 生成使用的受控数据模型。

与 memory 语义层同构：快照是"生成一次概览所需的全部直接子项"，digest 是内容身份。行为侧的
特有点：**日快照并入同日的观测空白**（用户裁定：空白如实写进当日叙述）——所以条目类型除了
文档与子目录，还区分 occurrence 与 gap。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from habitus.behavior.model import BehaviorDirectory
from habitus.foundation.integrity import canonical_digest


class BehaviorSemanticEntryKind(str, Enum):
    OCCURRENCE = "occurrence"
    GAP = "gap"
    DIRECTORY = "directory"


@dataclass(frozen=True)
class BehaviorSemanticEntry:
    """快照中的一个直接子项：行为文档、同日空白段或子目录摘要。"""

    name: str
    kind: BehaviorSemanticEntryKind
    content: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("behavior semantic entry name must be non-empty")
        if self.name != self.name.strip() or any(ord(char) < 32 for char in self.name):
            raise ValueError("behavior semantic entry name contains unsafe characters")
        object.__setattr__(self, "kind", BehaviorSemanticEntryKind(self.kind))
        if not isinstance(self.content, str):
            raise TypeError("behavior semantic entry content must be a string")


@dataclass(frozen=True)
class BehaviorDirectorySnapshot:
    """生成一次目录概览所需的完整快照；``directory`` 是层文件的落点。"""

    directory: BehaviorDirectory
    entries: tuple[BehaviorSemanticEntry, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.directory, BehaviorDirectory):
            raise TypeError("snapshot directory must be a BehaviorDirectory")
        normalized = tuple(self.entries)
        if any(not isinstance(entry, BehaviorSemanticEntry) for entry in normalized):
            raise TypeError("snapshot entries must contain BehaviorSemanticEntry values")
        identities = [(entry.kind.value, entry.name) for entry in normalized]
        if len(identities) != len(set(identities)):
            raise ValueError("snapshot contains duplicate direct entries")
        object.__setattr__(self, "entries", normalized)

    @property
    def digest(self) -> str:
        return canonical_digest(
            {
                "directory": list(self.directory.identity_parts),
                "entries": [
                    {"name": entry.name, "kind": entry.kind.value, "content": entry.content}
                    for entry in self.entries
                ],
            }
        )


class BehaviorSemanticRefreshStatus(str, Enum):
    # 与 memory 的刻意分叉：无 DELETED——add-only 树的目录不会被清空回收，空目录只是"还没有
    # 可摘要的内容"（EMPTY），层文件一旦写过就随下次内容出现被覆盖，不存在删除路径。
    WRITTEN = "written"
    UNCHANGED = "unchanged"
    MISSING = "missing"
    EMPTY = "empty"


@dataclass(frozen=True)
class BehaviorSemanticRefreshResult:
    """一个目录的 L0/L1 刷新结果；``source_digest`` 字段名与 memory 侧对齐。"""

    directory: BehaviorDirectory
    status: BehaviorSemanticRefreshStatus
    source_digest: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.directory, BehaviorDirectory):
            raise TypeError("result directory must be a BehaviorDirectory")
        object.__setattr__(self, "status", BehaviorSemanticRefreshStatus(self.status))


__all__ = [
    "BehaviorDirectorySnapshot",
    "BehaviorSemanticEntry",
    "BehaviorSemanticEntryKind",
    "BehaviorSemanticRefreshResult",
    "BehaviorSemanticRefreshStatus",
]
