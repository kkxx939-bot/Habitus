"""profile.md 与 entities/：从关系读数里投影出"这个人"与"和谁 · 在哪"。

- **profile**：未被推翻的关系 top-N（过了累积度门槛、区间不含零、没被测过的情境调节、账没混口径；无节律型另一套
  判据见 ``FulfilmentReading.unrefuted``），按区间**离零较近的那一端**排——保守端说"至少有这么大"；"效应 / 区间宽"会被零宽区间劫持（六次全 +90 分算出 9e+10）。不同单位按各自
  的尺度折算（概率 1、分钟 60、次数 1）。作息骨架与四个约束量要读预测树的曝光与率分布，由组合根注入的口给——那一半随刀 5 接线。
- **entities**：按对象类情境概念（"和 A 一起""在办公室"）聚合：哪些关系在这个对象在场时强度不同——就是 relations 里
  按那个概念分账的那一切片。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from habitus.scene.concepts.model import ConceptRole, ConceptSet
from habitus.scene.views.relations import Moderation, RelationReading


@dataclass(frozen=True)
class ProfileView:
    stable_relations: tuple[RelationReading, ...]
    total_relations: int
    unsettled_open_claims: int


def profile_view(readings: Iterable[RelationReading], *, top_n: int = 10) -> ProfileView:
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n <= 0:
        raise ValueError("top_n is a positive integer")
    items = tuple(readings)
    stable = sorted((r for r in items if r.unrefuted), key=_prominence, reverse=True)
    return ProfileView(
        stable_relations=tuple(stable[:top_n]),
        total_relations=len(items),
        unsettled_open_claims=sum(r.open_claims for r in items),
    )


_UNIT_SCALE = {"probability": 1.0, "minutes": 60.0, "times": 1.0}


def _prominence(reading: RelationReading) -> float:
    """区间离零较近的那一端的绝对值（保守效应量），按单位尺度折算。

    无节律型没有"比平时多几成"这个效应量，排的是**兑现率区间的下界**——"至少这么常兑现"，同样是保守端。
    """

    if reading.fulfilment is not None:
        interval = reading.fulfilment.interval
        return 0.0 if interval is None else max(0.0, interval.low)
    interval = reading.strength.interval
    if interval is None or interval.contains_zero:
        return 0.0
    conservative = min(abs(interval.low), abs(interval.high))
    return conservative / _UNIT_SCALE.get(reading.strength.unit, 1.0)


@dataclass(frozen=True)
class EntitySlice:
    concept: str
    moderations: tuple[tuple[str, Moderation], ...]


def entity_slices(readings: Iterable[RelationReading], concepts: ConceptSet) -> tuple[EntitySlice, ...]:
    """对象类情境概念 → (假设身份, 那条关系按它分账的结果)。"""

    by_concept: dict[str, list[tuple[str, Moderation]]] = {}
    for reading in readings:
        for moderation in reading.stability.moderations:
            if moderation.situation in concepts and concepts[moderation.situation].role is ConceptRole.OBJECT:
                by_concept.setdefault(moderation.situation, []).append((reading.hypothesis_identity, moderation))
    return tuple(EntitySlice(concept, tuple(sorted(items, key=lambda pair: pair[0]))) for concept, items in sorted(by_concept.items()))


__all__ = ["EntitySlice", "ProfileView", "entity_slices", "profile_view"]
