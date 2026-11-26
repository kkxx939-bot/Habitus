"""一代要量哪些关系：短跨度两两 × 四段、长跨度、够样本的前因 × 条件，以及冻住的行依赖的东西还在不在。

链、块、集中期这些材料由 ``LaneTests`` 一代只算一次，这里只读它们；怎么量在 ``evaluation``。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import TYPE_CHECKING

from habitus.scene.concepts.model import ConceptSet
from habitus.scene.relations.conditions import (
    MIN_PER_SIDE,
    Condition,
    Template,
    join_conditions,
    mechanical,
)
from habitus.scene.relations.engine import RelationKey, comparable
from habitus.scene.relations.spans import MARKER_SEGMENTS, SEGMENTS, Segment

if TYPE_CHECKING:  # 只给类型检查看：``lane_tests`` 在运行时导入本模块
    from habitus.scene.relations.lane_tests import LaneTests


class RelationCandidates:
    def __init__(self, tests: LaneTests) -> None:
        self.tests = tests

    def keys(self) -> tuple[RelationKey, ...]:
        """全部要量的关系：短跨度两两 × 四段；长跨度（行为概念的四种标志、情境概念成立的日子）；够样本的前因 × 条件。"""

        line, concepts = self.tests.line, self.tests.concepts
        lane = line.lane
        # 停用的概念不量：它不会再有新记录，量出来只会是"它没了"（裁定 27）
        names = [name for name in line.concepts() if concepts.lane_of(name) == lane and not concepts[name].retired]
        found = [
            RelationKey(lane, antecedent, consequent, segment)
            for antecedent in names
            if line.marks.get(antecedent)
            for consequent in names
            if comparable(concepts, antecedent, consequent)
            for segment in SEGMENTS
        ]
        for antecedent in names:
            for segment in MARKER_SEGMENTS:
                if self.tests.episodes(antecedent, segment):
                    found.extend(
                        RelationKey(lane, antecedent, consequent, segment)
                        for consequent in names
                        if comparable(concepts, antecedent, consequent)
                    )
        for situation in sorted(line.situations):
            if self.tests.episodes(situation, Segment.SITUATION):
                found.extend(RelationKey(lane, situation, consequent, Segment.SITUATION) for consequent in names)
        found.extend(self._moderated(names))
        return tuple(found)

    def _moderated(self, names: list[str]) -> list[RelationKey]:
        """带条件的关系：单个条件对每个够样本的前因都列（裁定 62；列出来进不进检验族按层级定，裁定 95、
        ``evaluation._families``）；第二个条件起只在已成立的带条件关系上
        加一个（{C1} 成立才列 {C1, C2}），条件池同样是机械列举的，最多 ``max_combined_conditions`` 个。
        同一个集合从几条父关系长出来只有一条键。"""

        found: list[RelationKey] = []
        line, concepts = self.tests.line, self.tests.concepts
        window = self.tests.config.transition_window_slots
        pools: dict[str, tuple[Condition, ...]] = {}
        for antecedent in names:
            chains = self.tests.chains(antecedent)
            if len(chains) < 2 * MIN_PER_SIDE:
                continue  # 两边各 ``MIN_PER_SIDE`` 次的最低要求；够不够开检验由每条关系自己的最小可达 p 定（``ready_to_test``）
            conditions = mechanical(line, concepts, antecedent, chains, window=window)
            pools[antecedent] = conditions
            for condition in sorted(conditions, key=lambda item: item.text):
                for consequent in names:
                    if not comparable(concepts, antecedent, consequent) or _circular(concepts, condition, consequent):
                        continue
                    for segment in SEGMENTS:
                        found.append(RelationKey(line.lane, antecedent, consequent, segment, condition.text))
        found.extend(self._grown(names, pools))
        return found

    def _grown(self, names: list[str], pools: Mapping[str, tuple[Condition, ...]]) -> list[RelationKey]:
        """已成立的带条件关系上再加一个条件（第二个及以上）。同一个集合从几条父关系长出来只有一条键。"""

        limit = self.tests.config.thresholds.max_combined_conditions
        concepts = self.tests.concepts
        grown: set[RelationKey] = set()
        for parent in sorted(self.tests.established):
            held = parent.conditions
            pool = pools.get(parent.antecedent)
            if not held or len(held) >= limit or pool is None:
                continue
            if parent.consequent not in names or parent.segment not in SEGMENTS:
                continue
            for condition in pool:
                if condition in held or _circular(concepts, condition, parent.consequent):
                    continue
                grown.add(replace(parent, condition=join_conditions((*held, condition))))
        return sorted(grown)

    def still_defined(self, key: RelationKey) -> bool:
        """这条关系依赖的东西还在不在（冻住的行的寿命）：前因与后果成了上下级；集中期 / 停了一阵随变点重算整段没了、带的条件有一个
        不在机械列举里了（消失），就是不在了。只是前因这阵子没有新链的照样算在——它会回来。
        几个条件合起来的还要求父关系仍然成立，那一条由状态折叠按这一代的结果判（``state._drop_orphaned_combinations``）。"""

        line = self.tests.line
        if self.tests.concepts.related(key.antecedent, key.consequent):
            # 上下级之间不量关系（裁定 105）：之前一代留下的这一对的行不再带着，由状态折叠清掉
            return False
        if key.segment in (Segment.HIGH_STATE, Segment.LOW_STATE) and key.antecedent in line.concepts():
            if not self.tests.episodes(key.antecedent, key.segment):
                return False
        if key.condition:
            chains = self.tests.chains(key.antecedent)
            if not chains:
                return True
            window = self.tests.config.transition_window_slots
            listed = {
                condition.text
                for condition in mechanical(line, self.tests.concepts, key.antecedent, chains, window=window)
            }
            return all(item.text in listed for item in key.conditions)
        return True


def _circular(concepts: ConceptSet, condition: Condition, consequent: str) -> bool:
    """ "B 刚出现过"不是调节：条件里的那一类就是后果、或是后果的上级 / 下级（上下级之间不量关系，裁定 105）时，
    它说的是 B 自己扎堆，不是 A 被什么左右。"""

    return condition.template is Template.TIMELINE and (
        condition.value == consequent or concepts.related(condition.value, consequent)
    )


__all__ = ["RelationCandidates"]
