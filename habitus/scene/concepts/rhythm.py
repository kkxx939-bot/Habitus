"""节律：一个行为概念在钟面上的作息形状。

**它回答四个问题**：几点、一天几个机会、每个机会多大概率、多久一次。一个"机会"就是曲线上的一个峰，
节律给的是"典型的一天"（七个周几平均）。

**读它的两处**：触点①写概念时看这个类的作息（"一天三次"就不该定成一个概念）；常态按峰分份
（``occurrences.baselines``）。两处都够得到 ``concepts/``，所以放在这一支。

**为什么只有协议在这里**：峰来自预测树，而 ``scene`` 不许 import ``prediction``（架构测试按传递闭包
钉死）。实现住组合根（``runtime/scene_rhythms.py``），这里只定形状。

**分型的判据是节律，不是猜**（2026-09-27 裁定）：树上天天有峰的（早餐、就寝）是**节律型**；
只在一两个周几冒头、或整周都没有峰的（打球、就诊）是**无节律型**。``has_rhythm`` 就是这条线。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from habitus.scene.concepts.model import MINUTES_PER_DAY, ConceptError, concept_identity

#: 一天最多认几个机会。一天十几个峰的行为（操作手机）多半是概念定得太宽，不是真有十几次机会——
#: 保护闸，超了就截断并报出来。数值待 54 天重放复核。
MAX_RHYTHM_PEAKS = 12
#: 七个周几里至少几天有峰才算"有节律"。5 = 工作日型的行为（只在工作日写代码）仍算有节律，
#: 而只在周末冒头的（打球）落到无节律型那一套——**待定值**，重放时按真实覆盖复核。
MIN_RHYTHM_DAYS = 5


@dataclass(frozen=True)
class RhythmPeak:
    """典型一天里的一次机会：第几次、几点到几点、这个峰上至少发生一次的概率。"""

    ordinal: int
    start_minute: int
    end_minute: int
    probability: float

    def __post_init__(self) -> None:
        if isinstance(self.ordinal, bool) or not isinstance(self.ordinal, int) or self.ordinal < 1:
            raise ConceptError("a rhythm peak's ordinal counts from 1")
        for label in ("start_minute", "end_minute"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ConceptError(f"rhythm peak {label} must be an integer minute of day")
        # 跨午夜的峰（就寝 23:00–01:00）起点在当天、终点过 24:00：终点最多到次日同一刻（评审 A-7 / B-8 / C-7）。
        if (
            not 0 <= self.start_minute < MINUTES_PER_DAY
            or not self.start_minute < self.end_minute <= self.start_minute + MINUTES_PER_DAY
        ):
            raise ConceptError("a rhythm peak starts inside one day and ends within 24 hours of its start")
        if isinstance(self.probability, bool) or not isinstance(self.probability, int | float):
            raise ConceptError("rhythm peak probability must be a number")
        if not 0.0 < self.probability <= 1.0:
            raise ConceptError("rhythm peak probability lies in (0, 1]")

    @property
    def wraps_midnight(self) -> bool:
        return self.end_minute > MINUTES_PER_DAY

    @property
    def label(self) -> str:
        tail = "（次日）" if self.wraps_midnight else ""
        return f"{_clock(self.start_minute)}–{_clock(self.end_minute)}{tail}"

    def render(self) -> str:
        return f"第 {self.ordinal} 次机会 {self.label}，概率 {self.probability:.0%}"


@dataclass(frozen=True)
class Rhythm:
    """一个行为概念的作息：典型一天的那几个峰 + 七个周几里几天有峰 + 复发间隔。

    ``peaks`` 空表示树上一个峰都取不到（它的类还没有曲线，或太低频）。
    ``recurrence_hours`` 是复发间隔的中位数（小时）；取不到就 None。
    """

    concept: str
    peaks: tuple[RhythmPeak, ...] = ()
    days_with_peaks: int = 0
    recurrence_hours: float | None = None
    #: 给模型看的名字（基础概念的 ``concept`` 是类编号，显示用类名）；不给就用 ``concept``。
    label: str | None = None

    def __post_init__(self) -> None:
        concept_identity(self.concept)
        peaks = tuple(self.peaks)
        if any(not isinstance(peak, RhythmPeak) for peak in peaks):
            raise ConceptError("rhythm peaks must be RhythmPeak values")
        if len(peaks) > MAX_RHYTHM_PEAKS:
            raise ConceptError(f"a rhythm carries at most {MAX_RHYTHM_PEAKS} peaks")
        if [peak.ordinal for peak in peaks] != list(range(1, len(peaks) + 1)):
            raise ConceptError("rhythm peak ordinals are 1..n in clock order")
        for earlier, later in zip(peaks, peaks[1:], strict=False):
            if later.start_minute < earlier.end_minute:
                raise ConceptError("rhythm peaks must not overlap")
        object.__setattr__(self, "peaks", peaks)
        if (
            isinstance(self.days_with_peaks, bool)
            or not isinstance(self.days_with_peaks, int)
            or not 0 <= self.days_with_peaks <= 7
        ):
            raise ConceptError("days_with_peaks counts 0–7 weekdays")
        if bool(peaks) != bool(self.days_with_peaks):
            raise ConceptError("a rhythm has peaks on some weekday, or no peaks at all")
        if self.recurrence_hours is not None:
            if isinstance(self.recurrence_hours, bool) or not isinstance(self.recurrence_hours, int | float):
                raise ConceptError("recurrence_hours must be a number or None")
            if self.recurrence_hours <= 0:
                raise ConceptError("recurrence_hours must be positive")

    @property
    def has_rhythm(self) -> bool:
        """节律型 / 无节律型的分界线，由树上的峰判，不由模型猜。"""

        return bool(self.peaks) and self.days_with_peaks >= MIN_RHYTHM_DAYS

    @property
    def opportunities_per_day(self) -> int:
        return len(self.peaks)

    def render(self) -> str:
        """给提示词的一行。无节律的说清"树上看不出节律"，好让模型知道不必猜第几次机会。"""

        name = self.label or self.concept
        if not self.peaks:
            return f"{name}：树上取不到峰（太低频或还没命中过），按无节律写"
        clock = "、".join(peak.render() for peak in self.peaks)
        every = "" if self.recurrence_hours is None else f"，平均每 {self.recurrence_hours:.0f} 小时一次"
        shape = "有节律" if self.has_rhythm else f"只在 {self.days_with_peaks} 个周几有峰，按无节律写"
        return f"{name}：一天 {self.opportunities_per_day} 个机会（{clock}），7 个周几里 {self.days_with_peaks} 天有它{every}；{shape}"


class RhythmProvider(Protocol):
    """组合根注入的节律口。键是行为概念的名字；取不到的概念不在里面（调用方按"无节律"处理）。"""

    def rhythms(self) -> Mapping[str, Rhythm]: ...


def _clock(minute_of_day: int) -> str:
    hour, minute = divmod(minute_of_day % MINUTES_PER_DAY, 60)
    return f"{hour:02d}:{minute:02d}"


__all__ = [
    "MAX_RHYTHM_PEAKS",
    "MIN_RHYTHM_DAYS",
    "Rhythm",
    "RhythmPeak",
    "RhythmProvider",
]
