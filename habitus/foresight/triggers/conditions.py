"""白天判带条件的关系"此刻条件成不成立"，与同一族里取最具体的那一条（裁定 61、62）。

判法借语义树的 ``evaluator``（同一份规则，``scene/relations/conditions.py``）在 ``triggers.marks`` 排出的时间线上判。
条件只有两种形状：时间线条件指向一个词表节点（``ConditionSpec.classes`` 第一个是它自己，其余是它的下级，对它是"不知道"）、概念条件指向一个日历情境；机会表里出现别的形状（时间线条件没有类、概念条件
不是日历）是表本身错了，明确报错，不当成"判不了"。

一族：同一（前因, 后果, 段）的不带条件的那条与它的各个带条件版本。这一刻只取条件全成立、挂的条件最多的那一条
当输入；一样多的取键文本最小的（同一次结果只记一次，不重复记功记过）；带条件的都不成立就是不带条件的那条。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace

from habitus.foresight.opportunities.model import ConditionSpec, RelationOpportunities
from habitus.scene.relations import Chain, LaneTimeline, Template, all_hold, evaluator
from habitus.scene.relations import parse_condition as parse


def judge_for(
    line: LaneTimeline, conditions: Sequence[ConditionSpec], *, window: int
) -> Callable[[Chain], bool | None]:
    """几个条件合起来判，判法与语义树同一个（``all_hold``）：任一不成立 → 不成立；都成立 → 成立；有判不了的 → 判不了。"""

    return all_hold([_judge(line, spec, window) for spec in conditions])


def _judge(line: LaneTimeline, spec: ConditionSpec, window: int) -> Callable[[Chain], bool | None]:
    condition = parse(spec.text)
    if condition.template is Template.TIMELINE and not spec.classes:
        raise ValueError(f"timeline condition {spec.text!r} names no vocabulary class")
    if condition.template is Template.CONCEPT and spec.calendar_key is None:
        raise ValueError(f"concept condition {spec.text!r} is not a calendar situation")
    if condition.template is Template.TIMELINE:
        # 白天的时间线按每一行自己的编号挂记号：条件的节点自己的记号就在它名下；它下级的记号对它是"不知道"（裁定 105），
        # 合成一份挂成条件的下级记号另给判法（窗里只有下级的判不了）——不改时间线本身
        below = sorted(
            {mark.uri: mark for token in spec.classes[1:] for mark in line.marks.get(token, [])}.values(),
            key=lambda item: (item.slot, item.started_at, item.uri),
        )
        line = replace(line, shadows={**line.shadows, condition.value: below})
    return evaluator(line, condition, window=window)


def families(relations: Iterable[RelationOpportunities]) -> dict[str, tuple[RelationOpportunities, ...]]:
    """按不带条件的关系身份分族（条件少的在前）。"""

    found: dict[str, list[RelationOpportunities]] = {}
    for relation in relations:
        found.setdefault(relation.base_key, []).append(relation)
    return {
        base: tuple(sorted(members, key=lambda item: (len(item.conditions), item.key)))
        for base, members in found.items()
    }


def most_specific(
    members: Sequence[RelationOpportunities], chain: Chain, line: LaneTimeline, *, window: int
) -> RelationOpportunities | None:
    """这一族里此刻该取的那一条：条件全成立、条件最多的；都不成立取不带条件的；连不带条件的都不在族里就没有。"""

    holding = [
        relation
        for relation in members
        if not relation.conditions or judge_for(line, relation.conditions, window=window)(chain) is True
    ]
    if not holding:
        return None
    most = max(len(relation.conditions) for relation in holding)
    return min((relation for relation in holding if len(relation.conditions) == most), key=lambda item: item.key)


__all__ = ["families", "judge_for", "most_specific"]
