"""外部条件：那一刻**已知的事实**，一组受控的键值。

与观测门相对：观测门进的是"某人在某时做了什么"（行为，走观测 → 融合 → 归约 → 行为树），事实门进的是
"那一刻外面是什么样"——天气、在不在家、在哪个项目、传感器读数。它不是行为，所以不进行为树、不进预测树；
它住在语义层这一侧，和 ``calendar`` 的日型接缝同一家族，只是粒度从"一天一句话"细到"一刻一组键值"。

用在两处：预测层的承诺落账时记下"说这话那一刻的条件"（结算之后就是 loss 的证据分布，见
``TODO(FORESIGHT-LOSS-001)``），将来关联时记下"这次发生与哪些条件有关"。

契约四条：

1. **只答那一刻已知的**。事后才知道的事不是条件，写进去等于把答案泄给判断。这条靠数据源自觉，接口拦不住。
2. **答不了的键不出现**。缺一个键不等于"条件不合"——所以承诺上同时记下"问了哪些键"（``keys``）与"答了什么"
   （``at``）：只看答案分不清"这个源那天离线"和"从来没有这个键"。
3. **要答得了历史**。关联是夜批，会问"那天那时是多少"；只有实时值的源（传感器）得自己留日志，
   在成为提供者之前先把历史补上。
4. **``when`` 是主体的本地时刻**。日型、时段这类按钟点分的条件换个时区就全错；接口只能拦住不带时区的值。

键与值的形状由**数据源自己声明**（``FactKey``：命名空间化的名字、类别还是数值、数值的单位），不预设一张
从人类行为归纳出来的条件分类表——那种表只增不减、永远缺一类。声明是为了让下游知道"这个键怎么比"
（类别按份额、数值按分位），而键名与声明会随承诺一起被记下来，所以**改名等于作废历史**。

现在**没有任何真实提供者**：日型由 ``calendar`` 那个接缝负责（读时算、不冻结，与条件相反），天气、
在哪个项目、传感器都要各自的源与历史，按 ``TODO(FORESIGHT-LOSS-001)`` 逐个接。在那之前 ``NoFacts`` 是
显式的"一个条件都没有"——不拿一个没有数据源的合成键去充数（那会让第一版 loss 的数字看起来稳定，
实际上什么都没量到）。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

#: 一刻的条件：键值对，按键升序、不重复。值一律是文本；数值型的键由声明里的 ``kind`` 说明怎么解释。
Conditions = tuple[tuple[str, str], ...]

#: 一个键是怎么比的：类别按份额（此刻这个取值在证据里占几成），数值按分位（此刻这个值落在证据分布的哪里）。
FACT_KINDS = ("类别", "数值")


@dataclass(frozen=True)
class FactKey:
    """数据源对一个键的声明：叫什么、怎么比、什么单位。

    ``name`` 用命名空间（``天气.温度``、``手表.心率``）：键会被写进承诺、将来还会被关联记录引用，
    两个源撞同一个裸名字就只能靠改名解决，而改名等于作废历史。
    """

    name: str
    kind: str
    unit: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or self.name != self.name.strip():
            raise ValueError("fact key name must be non-empty text without surrounding whitespace")
        if self.kind not in FACT_KINDS:
            raise ValueError(f"fact key kind must be one of {FACT_KINDS}, not {self.kind!r}")
        if self.unit is not None and (not isinstance(self.unit, str) or not self.unit):
            raise ValueError("fact key unit must be non-empty text or None")
        if self.kind == "类别" and self.unit is not None:
            raise ValueError("a categorical fact key has no unit")


@runtime_checkable
class FactProvider(Protocol):
    """一个外部条件源。

    ``version`` 随"这个源答出来的东西变了口径"一起改（换了数据源、换了取值词表、换了单位）：它会随承诺
    落盘，读历史的人靠它知道那批条件是谁按什么口径答的。
    """

    @property
    def version(self) -> str: ...

    def keys(self) -> tuple[FactKey, ...]: ...

    def at(self, when: datetime) -> Conditions: ...


def require_local(when: object) -> datetime:
    """条件的时刻必须带时区（契约第 4 条）。接口拦不住"传了 UTC"，但拦得住"没有时区"。"""

    if not isinstance(when, datetime) or when.utcoffset() is None:
        raise TypeError("fact time must be a timezone-aware local datetime")
    return when


class NoFacts:
    """显式的"一个条件都没有"。不是占位：没有任何源时，"外面是什么样"我们确实答不出。"""

    version = "none"

    def keys(self) -> tuple[FactKey, ...]:
        return ()

    def at(self, when: datetime) -> Conditions:
        require_local(when)
        return ()


class CompositeFacts:
    """把几个源并成一个。**一个键只能有一个源**：两个源给同一个键，哪个赢都是错的——那是数据归属没定。"""

    def __init__(self, *providers: FactProvider) -> None:
        for provider in providers:
            if not isinstance(provider, FactProvider):
                raise TypeError("every fact provider must implement FactProvider")
        seen: set[str] = set()
        for provider in providers:
            for key in provider.keys():
                if key.name in seen:
                    raise ValueError(f"fact key {key.name!r} is offered by two providers")
                seen.add(key.name)
        self.providers: tuple[FactProvider, ...] = providers

    @property
    def version(self) -> str:
        return "+".join(sorted(provider.version for provider in self.providers)) or "none"

    def keys(self) -> tuple[FactKey, ...]:
        return tuple(sorted((key for provider in self.providers for key in provider.keys()), key=lambda k: k.name))

    def at(self, when: datetime) -> Conditions:
        require_local(when)
        return conditions_of(*(provider.at(when) for provider in self.providers))


def conditions_of(*groups: Sequence[tuple[str, str]]) -> Conditions:
    """合并、去重、按键排序。同一个键给出**两个不同的值**是矛盾，硬拒；给两遍同一个值只是重复，收下。"""

    merged: dict[str, str] = {}
    for group in groups:
        for key, value in group:
            if not isinstance(key, str) or not key or not isinstance(value, str) or not value:
                raise ValueError("conditions must be pairs of non-empty text")
            if key in merged and merged[key] != value:
                raise ValueError(f"condition {key!r} has two values: {merged[key]!r} and {value!r}")
            merged[key] = value
    return tuple(sorted(merged.items()))


__all__ = [
    "FACT_KINDS",
    "CompositeFacts",
    "Conditions",
    "FactKey",
    "FactProvider",
    "NoFacts",
    "conditions_of",
    "require_local",
]
