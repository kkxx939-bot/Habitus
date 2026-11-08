"""一晚要量哪些关系（「量哪些」）：短跨度两两 × 四段、长跨度、够样本的前因 × 条件，以及冻住的行依赖的东西还在不在。

链、块、集中期这些材料由 ``LaneTests`` 一晚只算一次，这里只读它们；怎么量在 ``evaluation``。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import date
from typing import TYPE_CHECKING

from habitus.scene.concepts.model import ConceptSet, members_retired
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
    """``proposed``：模型提的条件长出来的键 → 只在哪天之后的数据上检验（枚举时记下，量的时候用）。"""

    def __init__(self, tests: LaneTests) -> None:
        self.tests = tests
        self.proposed: dict[RelationKey, date] = {}

    def keys(self) -> tuple[RelationKey, ...]:
        """全部要量的关系：短跨度两两 × 四段；长跨度（行为概念的四种标志、情境概念成立的日子）；够样本的前因 × 条件。"""

        line, concepts = self.tests.line, self.tests.concepts
        lane = line.lane
        # 停用的概念（类被拆改；细分 / 汇总概念成员类全停了）不再量：它不会再有新记录，量出来只会是"它没了"（裁定 27 第 9 条）
        names = [
            name for name in line.concepts() if concepts.lane_of(name) == lane and not members_retired(concepts, name)
        ]
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
            if self._derived_from_history(situation):
                continue
            if self.tests.episodes(situation, Segment.SITUATION):
                found.extend(RelationKey(lane, situation, consequent, Segment.SITUATION) for consequent in names)
        found.extend(self._moderated(names))
        return tuple(found)

    def _derived_from_history(self, situation: str) -> bool:
        """由行为历史派生的情境（「昨日有 X」「连续 X 中」）不当前因，只当调节条件（2026-10-12 用户裁定）。

        它当前因问的问题别处已经在问：「昨日有 X 的日子 Y 多吗」= 次日段的 X → Y，「连续 X 中的日子 Y 多吗」= X 的集中期标志；
        Y 换成 X 自己就成了同义反复（真实模型盘：昨日有交材料 → 交材料 58.8% 对 0%，候选）。日历来的情境（周末、调休、
        月初）不从行为算，照旧进情境段。调节条件那条路（``conditions.mechanical`` 的概念模板）不受影响。
        """

        concepts = self.tests.concepts
        if situation not in concepts:
            return False
        rule = concepts[situation].situation
        return rule is not None and rule.basis.reads_history

    def _moderated(self, names: list[str]) -> list[RelationKey]:
        """带条件的关系：单个条件对每个够样本的前因都列（现状，裁定 62 第 1 条）；第二个条件起只在已成立的带条件关系上
        加一个（{C1} 成立才列 {C1, C2}），条件池同样是机械列举 + 模型提的，最多 ``max_combined_conditions`` 个。
        同一个集合从几条父关系长出来只有一条键。"""

        found: list[RelationKey] = []
        line, concepts = self.tests.line, self.tests.concepts
        window = self.tests.config.transition_window_slots
        pools: dict[str, dict[Condition, date | None]] = {}
        for antecedent in names:
            chains = self.tests.chains(antecedent)
            if len(chains) < 2 * MIN_PER_SIDE:
                continue  # 两边各 ``MIN_PER_SIDE`` 次的最低要求；够不够开检验由每条关系自己的最小可达 p 定（``ready_to_test``）
            conditions: dict[Condition, date | None] = {
                condition: None for condition in mechanical(line, concepts, antecedent, chains, window=window)
            }
            # 模型提的条件只在提出之后的数据上检验（防"看过数据再提"）；与机械列举撞上的不算新条件，照旧在全部历史上量——
            # 不然它会把机械条件攒下的证据清掉（第四轮评审人工智能 1：第六批 21 条提议里 20 条与机械列举重复）
            conditions.update(
                {
                    condition: since
                    for condition, since in self.tests.proposals.get(antecedent, ())
                    if condition not in conditions
                }
            )
            pools[antecedent] = conditions
            for condition, since in sorted(conditions.items(), key=lambda item: item[0].text):
                for consequent in names:
                    if not comparable(concepts, antecedent, consequent) or _circular(concepts, condition, consequent):
                        continue
                    for segment in SEGMENTS:
                        key = RelationKey(line.lane, antecedent, consequent, segment, condition.text)
                        found.append(key)
                        if since is not None:
                            self.proposed[key] = since
        found.extend(self._grown(names, pools))
        return found

    def _grown(self, names: list[str], pools: Mapping[str, Mapping[Condition, date | None]]) -> list[RelationKey]:
        """已成立的带条件关系上再加一个条件（第二个及以上）。同一个集合从几条父关系长出来只有一条键：新加的那个条件
        是机械列举的就在全部历史上量，几条路都是模型提的才只看最早提出之后的数据。"""

        limit = self.tests.config.thresholds.max_combined_conditions
        grown: dict[RelationKey, list[date | None]] = {}
        for parent in sorted(self.tests.established):
            held = parent.conditions
            pool = pools.get(parent.antecedent)
            if not held or len(held) >= limit or pool is None:
                continue
            if parent.consequent not in names or parent.segment not in SEGMENTS:
                continue
            for condition, since in pool.items():
                if condition in held or _circular(self.tests.concepts, condition, parent.consequent):
                    continue
                key = replace(parent, condition=join_conditions((*held, condition)))
                grown.setdefault(key, []).append(since)
        for key, sinces in grown.items():
            if all(since is not None for since in sinces):
                self.proposed[key] = min(since for since in sinces if since is not None)
        return sorted(grown)

    def still_defined(self, key: RelationKey) -> bool:
        """这条关系依赖的东西还在不在（冻住的行的寿命，26 号清单 5.6）：集中期 / 停了一阵随变点重算整段没了、带的条件有一个
        既不在机械列举里也不在模型提议里了（被拒、消失），就是不在了。只是前因这阵子没有新链的照样算在——它会回来。
        几个条件合起来的还要求父关系仍然成立，那一条由状态折叠按当晚的结果判（``state._drop_orphaned_combinations``）。"""

        line = self.tests.line
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
            proposed = {condition.text for condition, _since in self.tests.proposals.get(key.antecedent, ())}
            return all(item.text in listed or item.text in proposed for item in key.conditions)
        return True


def _circular(concepts: ConceptSet, condition: Condition, consequent: str) -> bool:
    """ "B 刚出现过"不是调节：条件里的那一类就是（或读同一批记录于）后果时，它说的是 B 自己扎堆，不是 A 被什么左右。"""

    if condition.template is not Template.TIMELINE:
        return False
    return condition.value == consequent or not comparable(concepts, condition.value, consequent)


__all__ = ["RelationCandidates"]
