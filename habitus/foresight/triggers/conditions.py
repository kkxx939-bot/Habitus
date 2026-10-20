"""白天判带条件的关系"此刻条件成不成立"，与同一族里取最具体的那一条（裁定 61 第 6 条、62 第 4 条）。

判法借语义树的 ``evaluator``（同一份规则，``scene/relations/conditions.py``）在 ``triggers.marks`` 排出的时间线上判；
白天判不了的（时间线条件指向细分概念、映射器打的情境标签）一律"判不了"，不当成不成立。

**一族**：同一（前因, 后果, 段）的不带条件的那条与它的各个带条件版本。这一刻只取**条件全成立、挂的条件最多**的那一条
当输入；一样多的取键文本最小的（同一次结果只记一次，不重复记功记过）；带条件的都不成立就是不带条件的那条。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

from habitus.foresight.opportunities.model import ConditionSpec, RelationOpportunities
from habitus.scene.relations import Chain, LaneTimeline, Template, evaluator
from habitus.scene.relations import parse_condition as parse


def judge_for(
    line: LaneTimeline, conditions: Sequence[ConditionSpec], *, window: int
) -> Callable[[Chain], bool | None]:
    """几个条件合起来判：任一不成立 → 不成立；都成立 → 成立；有判不了的 → 判不了。空集合恒成立。"""

    judges = [_judge(line, spec, window) for spec in conditions]

    def judge(chain: Chain) -> bool | None:
        flags = [item(chain) for item in judges]
        if any(flag is False for flag in flags):
            return False
        if any(flag is None for flag in flags):
            return None
        return True

    return judge


def _judge(line: LaneTimeline, spec: ConditionSpec, window: int) -> Callable[[Chain], bool | None]:
    condition = parse(spec.text)
    if condition.template is Template.TIMELINE and not spec.classes:
        return _unknown  # 细分概念：白天没有映射器的命中
    if condition.template is Template.CONCEPT and spec.calendar_key is None:
        return _unknown  # 映射器打的情境标签：白天没有
    return evaluator(line, condition, window=window)


def _unknown(_chain: Chain) -> bool | None:
    return None


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
