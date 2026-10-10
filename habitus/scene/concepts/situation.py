"""情境怎么算：情境概念上那条**机械**说明。

情境命中不经 LLM（"``situation_hits`` 由算法直接得出"，见 ``occurrences/model.py``）。但算法得知道
「赶工中」到底指什么——所以基准（触点①）在定义情境概念时**顺手写下这条说明**，算法照着算。
语义还是模型给的（"连续三天写代码到深夜"这个想法），数是算法的（数到底连了几天）。

**初期只有三族**（2026-09-27 裁定"状态类情境初期只有日型/对象/派生"；「开放承诺」那一族 2026-09-30 取消——
"约了球还没打"把前因（约球）和后果（打球）当成了一个行为的一开一合，而它们是两个行为，只看前者对后者的影响）：

| 族 | 基准 | 例 |
|---|---|---|
| 日型 | ``WEEKDAYS`` 名义周几 · ``CALENDAR_NOTE`` 当地日历那句话里含某个词 | 「周末」=(周六,周日) · 「调休日」含"补班" |
| 对象 | ``SUBJECT`` 同在者里有谁 · ``PLACE`` 地点是哪儿 | 「和家人一起」 · 「在公司」 |
| 派生 | ``STREAK`` 某概念往前连着几个 24 小时都命中 · ``YESTERDAY`` 之前 24 小时内命中过 | 「赶工中」 · 「昨晚晚睡」 |

派生的"一天"以**这条 occurrence 的开始时刻**为锚往前数 24 小时，不按日历日（B14；评审 A-8 / B-11 / C-8：
02:10 的晚睡落在今天的目录，按日历日"昨天"会漏掉刚熬完夜的这个早上）。

**没有这条说明的情境概念永远不会命中**，这不是缺陷而是事实：「出差中」这类状态要等事实门
（``scene/facts.py``）接上真实数据源才有得算，在那之前它进得了概念集，只是分不出层。
编一个"名字里含出差就算"的关键词规则反而会造出一批假的分层。

**为什么不放进 ``MechanicalRule``**：那一条判的是"这一条 occurrence 的数值落在哪个区间"（比常态晚多少），
量在这条记录自己身上；情境问的是"那一刻外面是什么样"，要看日历、看对象、看历史命中。两者的输入完全不同，
合成一个会让每个调用点都得先分辨"这条规则要喂什么材料"。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from habitus.foundation.text import clean_line

MAX_STREAK_DAYS = 30
MAX_VALUE_CHARS = 120


class SituationError(ValueError):
    """情境说明与自己的约束矛盾（基准写了一条算法算不了的）。"""


class SituationBasis(str, Enum):
    """算这个情境要看什么。每一族的参数不同，构造时按族核对，缺一个就拒。"""

    WEEKDAYS = "weekdays"
    CALENDAR_NOTE = "calendar_note"
    SUBJECT = "subject"
    PLACE = "place"
    STREAK = "streak"
    YESTERDAY = "yesterday"

    @property
    def needs_concept(self) -> bool:
        return self in {SituationBasis.STREAK, SituationBasis.YESTERDAY}

    @property
    def needs_value(self) -> bool:
        return self in {SituationBasis.CALENDAR_NOTE, SituationBasis.SUBJECT, SituationBasis.PLACE}

    @property
    def reads_history(self) -> bool:
        """要看别的日子的命中——所以这个情境只在那些天已经映射过之后才算得准。"""

        return self in {SituationBasis.STREAK, SituationBasis.YESTERDAY}


@dataclass(frozen=True)
class SituationRule:
    """一条情境说明。字段按族用：``weekdays`` 只给日型，``value`` 给日历/同在/地点，``concept``（+``grade``、
    ``days``）给派生。"""

    basis: SituationBasis
    weekdays: tuple[int, ...] = ()
    value: str | None = None
    concept: str | None = None
    grade: str | None = None
    days: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "basis", SituationBasis(self.basis))
        self._check_weekdays()
        self._check_value()
        self._check_concept()
        self._check_days()

    def _check_weekdays(self) -> None:
        days = tuple(self.weekdays)
        if self.basis is SituationBasis.WEEKDAYS:
            if not days or any(isinstance(day, bool) or not isinstance(day, int) or not 0 <= day <= 6 for day in days):
                raise SituationError("a weekday situation names one or more weekdays, Monday=0 … Sunday=6")
            if len(set(days)) != len(days):
                raise SituationError("weekdays must be distinct")
            object.__setattr__(self, "weekdays", tuple(sorted(set(days))))
            return
        if days:
            raise SituationError(f"{self.basis.value} takes no weekdays")

    def _check_value(self) -> None:
        if self.basis.needs_value:
            text = clean_line(self.value)
            if not text or len(text) > MAX_VALUE_CHARS:
                raise SituationError(f"{self.basis.value} needs a short non-empty value to match")
            object.__setattr__(self, "value", text)
            return
        if self.value is not None:
            raise SituationError(f"{self.basis.value} takes no value")

    def _check_concept(self) -> None:
        """只核对形状。"这个概念在不在概念集里"由 ``ConceptSet`` 核对——跨对象的检查不放在值对象里，
        否则本模块要 import 概念模型，而概念模型正引用本模块。"""

        if self.basis.needs_concept:
            name = clean_line(self.concept)
            if not name or len(name) > MAX_VALUE_CHARS:
                raise SituationError(f"{self.basis.value} names the concept it watches")
            object.__setattr__(self, "concept", name)
            if self.grade is not None:
                grade = clean_line(self.grade)
                if not grade:
                    raise SituationError("a situation grade filter must be non-empty text")
                object.__setattr__(self, "grade", grade)
            return
        if self.concept is not None or self.grade is not None:
            raise SituationError(f"{self.basis.value} watches no concept")

    def _check_days(self) -> None:
        if isinstance(self.days, bool) or not isinstance(self.days, int):
            raise SituationError("days must be an integer")
        if self.basis is SituationBasis.STREAK:
            if not 2 <= self.days <= MAX_STREAK_DAYS:
                raise SituationError(f"a streak runs 2–{MAX_STREAK_DAYS} days; one day is not a streak")
            return
        if self.days != 1:
            raise SituationError(f"{self.basis.value} looks at one day")

    def payload(self) -> dict[str, object]:
        """进指纹与文档的形状：只含这一族真正用到的字段。"""

        data: dict[str, object] = {"basis": self.basis.value}
        if self.basis is SituationBasis.WEEKDAYS:
            data["weekdays"] = list(self.weekdays)
        if self.basis.needs_value:
            data["value"] = self.value
        if self.basis.needs_concept:
            data["concept"] = self.concept
            data["grade"] = self.grade
        if self.basis is SituationBasis.STREAK:
            data["days"] = self.days
        return data

    def criterion(self) -> str:
        """给人读的一句话（也进概念文档）。"""

        names = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")
        if self.basis is SituationBasis.WEEKDAYS:
            return "名义上是" + "/".join(names[day] for day in self.weekdays)
        if self.basis is SituationBasis.CALENDAR_NOTE:
            return f"当地日历那句话里含「{self.value}」"
        if self.basis is SituationBasis.SUBJECT:
            return f"同在的人里有「{self.value}」"
        if self.basis is SituationBasis.PLACE:
            return f"地点是「{self.value}」"
        graded = "" if self.grade is None else f"（{self.grade}档）"
        if self.basis is SituationBasis.STREAK:
            return f"「{self.concept}」{graded}往前连着 {self.days} 个 24 小时都命中"
        return f"之前 24 小时内命中过「{self.concept}」{graded}"


__all__ = ["MAX_STREAK_DAYS", "SituationBasis", "SituationError", "SituationRule"]
