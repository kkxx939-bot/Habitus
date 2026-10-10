"""预测层读语义树关系的那一侧（语义树新方案 ``13`` 第 9 步；裁定 24：先接现在的预测层，反馈后做）。

每条 lane 取盘上最近一晚的关系表，连同概念集、是第几晚，一起交给装配器（``foresight.RelationTable``）。按"各 lane 最近一晚
+ 那晚文件的修改指纹"缓存：没变就不重读（预测层每个槽一拍，一天近百拍，关系表一天只变一次）；重跑最晚那一晚重写了文件也读得到。
概念集在同一晚的关系之前写（B2 → B4），重读时一起重读，两者对得上。
"""

from __future__ import annotations

from datetime import date

from habitus.foresight import RelationTable
from habitus.scene.concepts import ConceptStore
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.relations.store import RelationStore


class SceneRelations:
    """``() → RelationTable``：每条 lane 盘上最近一晚的关系表 + 概念集，按（各 lane 最近一晚, 那晚文件的指纹）缓存。"""

    def __init__(self, relations: RelationStore, concepts: ConceptStore) -> None:
        if not isinstance(relations, RelationStore):
            raise TypeError("relations must be a RelationStore")
        if not isinstance(concepts, ConceptStore):
            raise TypeError("concepts must be a ConceptStore")
        self.relations = relations
        self.concepts = concepts
        self._key: tuple[tuple[str, date | None, tuple[int, int] | None], ...] | None = None
        self._current = RelationTable((), ConceptSet(()), None)

    def __call__(self) -> RelationTable:
        nights = [(lane, self.relations.latest_night(lane)) for lane in self.relations.lanes()]
        # 指纹里带上文件的修改时间与大小：存储允许重写最晚那一晚，只看"是哪一晚"会一直读着重写前的（第四轮评审 E14）
        key = tuple(
            (lane, night, None if night is None else self.relations.stamp(lane, night)) for lane, night in nights
        )
        if key != self._key:
            loaded = tuple(
                relation
                for lane, night in nights
                if night is not None
                for relation in self.relations.read(lane, night).relations
            )
            latest = max((night for _lane, night in nights if night is not None), default=None)
            self._current = RelationTable(loaded, self.concepts.read_all() if loaded else ConceptSet(()), latest)
            self._key = key
        return self._current


__all__ = ["SceneRelations"]
