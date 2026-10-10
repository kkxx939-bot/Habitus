"""对外只读（设计 六）：编号 → 条目；编号 → 现在对应哪些编号；当前版本号；某版之后的变更。

（原来还有"原话 → 编号"提示索引给当日实况用；新口径下原话几乎不逐字重复，命中 1.6%，已删——未封口的链改为
直接走白天归类，见 ``runtime/unsealed.py``。）
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from habitus.behavior.kinds.changes import VersionRecord
from habitus.behavior.kinds.ids import ClassId, is_marker
from habitus.behavior.kinds.model import BehaviorClass, Vocabulary
from habitus.behavior.kinds.store import BehaviorKindStore


class VocabularyReader:
    def __init__(self, store: BehaviorKindStore) -> None:
        if not isinstance(store, BehaviorKindStore):
            raise TypeError("store must be BehaviorKindStore")
        self.store = store

    def snapshot(self) -> Vocabulary:
        return self.store.read()

    def version(self) -> int:
        return self.snapshot().version

    def describe(self, token: str) -> BehaviorClass | None:
        """编号 → 条目（含停用的）；占位标记给 ``None``；既不是编号也不是占位标记的 token 硬失败。"""

        class_id = _class_id(token)
        if class_id is None:
            return None
        return self.snapshot().classes.get(class_id)

    def current_ids(self, token: str) -> tuple[str, ...]:
        """编号 → 现在对应哪些在用编号（顺着合并、拆分往下找）；占位标记给空。"""

        class_id = _class_id(token)
        if class_id is None:
            return ()
        return tuple(str(item) for item in self.snapshot().descendants(class_id))

    def class_names(self) -> Mapping[str, str]:
        """编号 → 类名（含停用的）：给下游渲染"给人和模型看的名字"用，编号仍是身份。"""

        return MappingProxyType({str(item.id): item.name for item in self.snapshot().classes.values()})

    def changes_since(self, version: int) -> tuple[VersionRecord, ...]:
        """``version`` 之后的每一版（新增、拆分、合并、改判据与迁移），按版本号从小到大；派生树据此同步。"""

        if isinstance(version, bool) or not isinstance(version, int) or version < 0:
            raise ValueError("version must be a non-negative integer")
        return tuple(record for record in self.store.records() if record.version > version)


def _class_id(token: str) -> ClassId | None:
    return None if is_marker(token) else ClassId.parse(token)


__all__ = ["VocabularyReader"]
