"""日历情境（裁定 59 第 7 条，2026-10-18 定）：周几、几号、月末、日型，按外部条件的日历键机械生成，不由模型写。

外部条件只是给每一天打的约束标签，量的是行为。

- **维度与取值从他自己的数据里攒出来**（``calendar_concepts`` 的 ``observed``）：维度是日历给的键，取值看这条 lane 的日子里实际出现过
  哪些才长出哪个日历情境，不一上来给齐——新加坡、美国永远不会出现"调休上班"，没碰到过节假日就没有"节假日"。每条 lane 都看得见日历，
  所以不挂 lane。
- **取值归组也攒出来**（``value_sets_to_add``）：同一件事在一个维度的几个取值上各自抬高地成了候选或成立（晨跑的周一、三、六；叫车的
  工作日、调休上班），合成一个"周一、周三、周六"的情境再检验一次，样本是单个取值的几倍；和别的关系一样要过前向验证才成立。
- **粗的能解释的不报细的**（``most_specific``）：同状态、同方向时，集合包住的单个取值 / 小集合不留，月末包住的 31 号不留。周几与日型
  不当嵌套：检验周几时日型是固定住的（``relations.calendar``），两条都成立说的是两件事。调用方按状态与方向分好组再交进来。
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime

from habitus.conditions import (
    DAY_OF_MONTH_KEY,
    DAY_TYPE_KEY,
    MONTH_END_KEY,
    WEEKDAY_KEY,
    WEEKDAY_NAMES,
    YES,
    DayType,
)
from habitus.scene.concepts.model import ConceptDefinition, ConceptOrigin, ConceptRole, ConceptSet, ConceptSource
from habitus.scene.concepts.situation import SituationBasis, SituationRule

_SOURCE = ConceptSource(ConceptOrigin.CALENDAR)
#: 每个维度取值的先后（名字按这个顺序排，集合的名字才唯一）。月末只有"是"一个取值，不合集合。
_ORDER: Mapping[str, tuple[str, ...]] = {
    WEEKDAY_KEY: WEEKDAY_NAMES,
    DAY_OF_MONTH_KEY: tuple(str(day) for day in range(1, 32)),
    MONTH_END_KEY: (YES,),
    DAY_TYPE_KEY: tuple(kind.value for kind in DayType),
}
GROUPABLE = (WEEKDAY_KEY, DAY_OF_MONTH_KEY, DAY_TYPE_KEY)


def value_name(key: str, value: str) -> str:
    """一个取值的显示名：周几、日型就是取值本身，几号写成"20号"，月末写成"月末"。"""

    if key == DAY_OF_MONTH_KEY:
        return f"{value}号"
    if key == MONTH_END_KEY:
        return "月末"
    return value


def calendar_concepts(
    *, now: datetime, observed: Iterable[tuple[str, str]] | None = None
) -> tuple[ConceptDefinition, ...]:
    """单个取值的日历情境。``observed`` 是数据里出现过的（键, 取值）——只给这些；不给就给全部取值（测试与工具用）。"""

    seen = None if observed is None else set(observed)
    found = []
    for key, values in _ORDER.items():
        for value in values:
            if seen is None or (key, value) in seen:
                found.append(_concept(value_name(key, value), key, (value,), now=now))
    return tuple(found)


def value_set(key: str, values: Iterable[str], *, now: datetime) -> ConceptDefinition:
    """一个维度上几个取值合成的一个日历情境，名字按取值顺序排（"周一、周三、周六""工作日、调休上班"）。"""

    if key not in GROUPABLE:
        raise ValueError(f"values of {key!r} are not grouped")
    wanted = set(values)
    ordered = tuple(value for value in _ORDER[key] if value in wanted)
    if len(ordered) < 2:
        raise ValueError("a value set names at least two values")
    return _concept("、".join(value_name(key, value) for value in ordered), key, ordered, now=now)


def weekday_set(names: Iterable[str], *, now: datetime) -> ConceptDefinition:
    return value_set(WEEKDAY_KEY, names, now=now)


def is_calendar(concepts: ConceptSet, identity: str) -> bool:
    return identity in concepts and concepts[identity].source.origin is ConceptOrigin.CALENDAR


def calendar_key(concepts: ConceptSet, identity: str) -> tuple[str, tuple[str, ...]] | None:
    """日历情境盯的键与取值；不是日历情境就是 None。"""

    if not is_calendar(concepts, identity):
        return None
    rule = concepts[identity].situation
    if rule is None or rule.value is None:
        return None
    return rule.value, rule.values


def most_specific(concepts: ConceptSet, antecedents: Sequence[str]) -> tuple[str, ...]:
    """同一个后果、同状态同方向的几个前因里，被更粗的日历前因解释了的不留（集合包住的取值、月末包住的 31 号）；
    非日历前因原样保留。"""

    keyed = {item: key for item in antecedents if (key := calendar_key(concepts, item)) is not None}
    dropped = {
        item
        for item, key in keyed.items()
        if any(_covered(key, other) for name, other in keyed.items() if name != item)
    }
    return tuple(item for item in antecedents if item not in dropped)


def _covered(mine: tuple[str, tuple[str, ...]], other: tuple[str, tuple[str, ...]]) -> bool:
    key, values = mine
    other_key, other_values = other
    if key == other_key:
        return set(values) < set(other_values)
    return key == DAY_OF_MONTH_KEY and values == ("31",) and other_key == MONTH_END_KEY


def value_sets_to_add(
    concepts: ConceptSet, held: Mapping[str, Sequence[str]], *, now: datetime
) -> tuple[ConceptDefinition, ...]:
    """后果 → 它此刻**抬高地**处在候选或成立的日历前因（调用方只交抬高的）。同一维度的单个取值两个及以上的，合成一个集合
    （概念集里还没有的才给）。"""

    found: dict[str, ConceptDefinition] = {}
    for antecedents in held.values():
        by_key: dict[str, list[str]] = {}
        for item in antecedents:
            rule = calendar_key(concepts, item)
            if rule is not None and rule[0] in GROUPABLE and len(rule[1]) == 1:
                by_key.setdefault(rule[0], []).append(rule[1][0])
        for dimension, values in by_key.items():
            if len(values) < 2:
                continue
            concept = value_set(dimension, values, now=now)
            if concept.name not in concepts:
                found[concept.name] = concept
    return tuple(found[name] for name in sorted(found))


def _concept(name: str, key: str, values: tuple[str, ...], *, now: datetime) -> ConceptDefinition:
    return ConceptDefinition(
        name=name,
        definition=f"当地日历上的{name}",
        role=ConceptRole.DAY_TYPE,
        source=_SOURCE,
        created_at=now,
        situation=SituationRule(SituationBasis.CALENDAR, value=key, values=values),
    )


__all__ = [
    "GROUPABLE",
    "calendar_concepts",
    "calendar_key",
    "is_calendar",
    "most_specific",
    "value_name",
    "value_set",
    "value_sets_to_add",
    "weekday_set",
]
