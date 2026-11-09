"""一晚的关系检验：每条 lane 全部两两都量，数据说了算（语义树新方案 ``13`` ①③）。

```
每条 lane：
  前因 A × 后果 B × 四段跨度  →  每个都按层配好对照（spans；同一条链配两道，两道都过才算）
  → 最小可达 p 过不了线的：样本不够（不进检验族、不给区间）
  → 其余做加权 BH（FDR 10%）            第 1 道
  → 过了的：按块看稳不稳                  第 2 道
           区间下限够不够大               第 3 道
```

两条 lane 各成一个检验族（裁定 21：两条 lane 相互独立、不干扰）。不比的对（机械排除）：自己对自己（那是复现节律，
归 ``scene.recurrence``）；读同一批记录的（汇总与成员、基础与它的汇总、细分与它的源类）。

本模块只算"今晚的数据说什么"。候选要在之后的新数据上复现才成立（前向验证）、成立后按段看新数据（维持检验，从上一次判过起不重叠地攒满一段）——
这两步要跨夜的状态，由状态折叠（第 5 步）调 ``evaluate`` 在对应的子集上做。先验权重由第 7 步给，这里只收。

本模块是关系的身份、读数与第 3 道的判法；一条 lane 一晚的检验在 ``lane_tests``（量哪些在 ``candidates``、怎么量在 ``evaluation``）。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import date
from enum import Enum

from habitus.scene.concepts.model import ConceptKind, ConceptSet
from habitus.scene.relations import stats
from habitus.scene.relations.conditions import (
    Condition,
    join_conditions,
    parse_conditions,
)
from habitus.scene.relations.config import RelationConfig
from habitus.scene.relations.spans import (
    Chain,
    Segment,
)


@dataclass(frozen=True, order=True)
class RelationKey:
    """关系身份 = （lane, 前因, 后果, 跨度段, 条件）。条件为空 = 不带条件；带条件的是"在这些条件下 A → B 比少一个条件时
    更强 / 更弱"（调节；组合前因"A 与 C 一起"也是一种条件），条件是一个集合，写成 ``conditions.join_conditions`` 的键
    （几个 ``Condition.text`` 排序后用 ``+`` 拼起来；裁定 61 第 6 条）。"""

    lane: str
    antecedent: str
    consequent: str
    segment: Segment
    condition: str = ""

    def __post_init__(self) -> None:
        # 键会写进关系表与迁移日志的一行里：换行、行分隔符这类不可打印的字符一个都不收
        for label in ("lane", "antecedent", "consequent", "condition"):
            value = getattr(self, label)
            if not value.isprintable():
                raise ValueError(f"relation key {label} must be printable text")

    @property
    def conditions(self) -> tuple[Condition, ...]:
        """这条关系挂的条件（没有是空元组）。"""

        return parse_conditions(self.condition)

    def parents(self) -> tuple[RelationKey, ...]:
        """少任意一个条件的那几条键（组合长在它们上面；单个条件的父是不带条件的那条；不带条件的没有父）。"""

        items = self.conditions
        return tuple(
            replace(self, condition=join_conditions(items[:index] + items[index + 1 :])) for index in range(len(items))
        )

    @property
    def text(self) -> str:
        base = f"{self.lane}|{self.antecedent}|{self.consequent}|{self.segment.value}"
        return f"{base}|{self.condition}" if self.condition else base


class Verdict(str, Enum):
    #: 最小可达 p 也过不了线：不进检验族、不给区间。
    SPARSE = "sparse"
    #: 进了检验族，没过第 1 道。
    TESTED = "tested"
    #: 过了第 1 道（今晚的数据上显著）；前向验证由状态折叠接着做。
    SIGNIFICANT = "significant"


@dataclass(frozen=True)
class Blocks:
    """一个前因的块：块长（天）= max(1 天, 合并扎堆之后相邻两次间隔的中位数)，从它第一次出现那天起算。"""

    days: float
    origin: date

    def of(self, chain: Chain) -> int:
        return math.floor((chain.day - self.origin).days / self.days)


@dataclass(frozen=True)
class RelationTest:
    key: RelationKey
    verdict: Verdict
    antecedents: int
    blocks: int
    block_days: float
    effect: stats.Effect
    p_value: float
    minimum_p: float
    upward: bool
    weight: float = 1.0
    #: 样本不够时：照现在的层结构，大约还要几次全朝一个方向的 A 才可能过线（粗估，给人看）。
    more_needed: int | None = None
    #: 第 2 道：各块效应异质性的 p、去掉任何一块方向都不变，合起来就是"时间上稳"。
    heterogeneity_p: float | None = None
    leave_one_out: bool | None = None
    stable: bool | None = None
    #: 第 3 道：差值区间、相对差区间，区间离 0 最近的一端够不够大。
    interval: tuple[float, float] | None = None
    relative_interval: tuple[float, float] | None = None
    large_enough: bool | None = None
    #: 从后果看：B 里约多少是 A 带出来的（超额比例，不另算一份证据）。
    excess_share: float | None = None
    #: 支撑例子里 A、B 目标相同的比例（只用结构字段；算法不看，高的关系对融合口径敏感）。
    same_event_share: float | None = None


@dataclass(frozen=True)
class LaneNight:
    lane: str
    cutoff: date
    tests: tuple[RelationTest, ...]
    family: int
    #: 开测门第 1 道整组到不了、没逐条量的带条件关系，按段计数（它们必是稀疏，不建检验对象）。
    skipped: Mapping[str, int] = field(default_factory=dict)
    #: 不加权时过第 1 道的关系（给了先验权重时状态折叠据它打"靠先验"；没给权重时就是过了第 1 道的那些）。
    plain_significant: frozenset[RelationKey] = frozenset()
    #: 只给观测：四族（不带条件短跨度 ``plain`` / 长跨度 ``long`` / 带条件 ``conditioned`` / 按组 ``groups``）各自的族大小
    #: （``*_family``）与过第 1 道数（``*_passed``）、带条件那一族各条进族路（``entry_parent`` / ``entry_carried`` /
    #: ``entry_group``）的条数、逐条量了几条（``measured``）。不进比较。
    breakdown: Mapping[str, int] = field(default_factory=dict, compare=False)

    def significant(self) -> tuple[RelationTest, ...]:
        return tuple(test for test in self.tests if test.verdict is Verdict.SIGNIFICANT)


@dataclass(frozen=True)
class Subset:
    """在一个子集（前向验证的新数据、维持检验那一段的新数据）上的读数。"""

    antecedents: int
    blocks: int
    effect: stats.Effect
    p_value: float
    interval: tuple[float, float] | None
    relative_interval: tuple[float, float] | None
    #: 这几层朝这条关系的方向全中（朝上全阳、朝下全阴）能到的单侧 p：不小于 α 时判了必输。
    minimum_p: float


def large_enough(
    spread: tuple[float, float] | None,
    relative: tuple[float, float] | None,
    *,
    upward: bool,
    config: RelationConfig,
) -> bool:
    """第 3 道：区间离 0 最近的那一端，同时够得上绝对差与相对差。对照率常为 0、说不清相对差的，相对差那条按满足算
    （从无到有本身就是最大的相对变化）。"""

    if spread is None:
        return False
    near = spread[0] if upward else -spread[1]
    if near < config.thresholds.min_absolute_lift:
        return False
    if relative is None:
        return True
    near_relative = relative[0] if upward else -relative[1]
    return near_relative >= config.thresholds.min_relative_lift


def gone(
    spread: tuple[float, float] | None, relative: tuple[float, float] | None, *, upward: bool, config: RelationConfig
) -> bool:
    """失效：区间离 0 最远的那一端都已够不上第 3 道（有把握说它没了）。只是不显著（证据弱、区间宽）不算。"""

    if spread is None:
        return False
    far = spread[1] if upward else -spread[0]
    if far < config.thresholds.min_absolute_lift:
        return True
    if relative is None:
        return False
    far_relative = relative[1] if upward else -relative[0]
    return far_relative < config.thresholds.min_relative_lift


def comparable(concepts: ConceptSet, antecedent: str, consequent: str) -> bool:
    """这一对要不要比：不比自己；不比读同一批记录的（汇总 / 基础类有交集；细分与它的源类或含源类的汇总）。"""

    if antecedent == consequent:
        return False
    if concepts.overlaps(antecedent, consequent):
        return False
    first, second = concepts[antecedent], concepts[consequent]
    for refinement, other in ((first, second), (second, first)):
        if refinement.kind is ConceptKind.REFINEMENT and other.kind in (ConceptKind.BASE, ConceptKind.GROUP):
            if refinement.classes[0] in other.classes:
                return False
    return True


__all__ = [
    "Blocks",
    "LaneNight",
    "RelationKey",
    "RelationTest",
    "Subset",
    "Verdict",
    "comparable",
    "gone",
    "large_enough",
]
