"""词表口的实现：用基础词表的只读口装出语义树要的 ``ClassCatalog``（裁定 20）。

scene 不 import 词表包，所以"变更日志怎么读成拆分 / 合并"住在组合根（occurrence 上的编号怎么读归事件序列 ``series.reader``）。
变更日志一版是一组原子操作，这里按语义读成语义树要的几样：

- ``Branch(a, b)``、a 没停用 → b 从 a 拆出来（接收了 a 的一部分条目）；
- ``RetireClass(a)`` + ``Branch(a, b)`` → a 并进 b；
- 迁移（``moves``）→ 哪些 occurrence 的编号从哪改到哪。
"""

from __future__ import annotations

from habitus.behavior.kinds.changes import Branch, RetireClass, VersionRecord
from habitus.behavior.kinds.reader import VocabularyReader
from habitus.scene.concepts.catalog import CatalogChanges, CatalogClass, MovedOccurrence


class VocabularyCatalog:
    """按 ``ClassCatalog`` 协议实现。每次调用现读词表（夜批一夜读几次，词表文件很小）。"""

    def __init__(self, reader: VocabularyReader) -> None:
        if not isinstance(reader, VocabularyReader):
            raise TypeError("reader must be VocabularyReader")
        self.reader = reader

    def classes(self) -> tuple[CatalogClass, ...]:
        return tuple(
            CatalogClass(
                id=str(item.id), name=item.name, criterion=item.criterion, lane=item.lane.value, active=item.active
            )
            for item in self.reader.snapshot().classes.values()
        )

    def version(self) -> int:
        return self.reader.version()

    def changes_since(self, version: int) -> CatalogChanges:
        records = self.reader.changes_since(version)
        split_from: dict[str, str] = {}
        merged_into: dict[str, str] = {}
        moved: list[MovedOccurrence] = []
        for record in records:
            _read(record, split_from, merged_into)
            moved.extend(
                MovedOccurrence(uri=move.occurrence, source=move.source, target=move.target) for move in record.moves
            )
        return CatalogChanges(
            since=version,
            version=records[-1].version if records else max(version, self.reader.version()),
            split_from=split_from,
            merged_into=merged_into,
            moved=tuple(moved),
        )


def _read(
    record: VersionRecord,
    split_from: dict[str, str],
    merged_into: dict[str, str],
) -> None:
    """一版的操作读成拆分 / 合并（新增、改名、改判据不进变更流，同步时按全部类核对）。"""

    retired = {str(item.class_id) for item in record.operations if isinstance(item, RetireClass)}
    for operation in record.operations:
        if isinstance(operation, Branch):
            source, target = str(operation.source), str(operation.target)
            if source in retired:
                merged_into[source] = target
            else:
                split_from[target] = source


__all__ = ["VocabularyCatalog"]
