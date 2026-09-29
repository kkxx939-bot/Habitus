"""映射那两个必填参数的生产者：``situation_for`` 与 ``baseline_for``。

``map_closed_day`` 要的两个 callable 都在这里装好——它们的材料横跨四处（当地日历、账本、命中历史、
这条 occurrence 自己的字段），而那四处只有组合根同时够得到：``occurrences`` 那一支不许触达 ``ledger``
（映射者看不到假设与账），``views`` 不能被底层引用。

**一天一装**：情境与常态都是"这一天的"，所以按天构造一次（``for_day``），一天里的每条 occurrence 复用。
一天要读的东西只有两批：前 N 天的命中签名（``history_days`` 决定 N），和那天起点的开放承诺。

**开放承诺按"这一天的起点"算**，不是按"现在"：
- 重放 54 天历史时按"现在"会答出今天的账，等于把未来泄给过去；
- 当天下午才立起来的承诺，本不该影响当天早上那条行为。
一天之内承诺的开合因此看不见，这是**故意的近似**（也正好挡住泄漏），真要更细就得逐条 occurrence 问一次账。
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from datetime import date, datetime, time, tzinfo

from habitus.behavior.document import BehaviorDocument
from habitus.behavior.model import BehaviorKind
from habitus.scene.calendar import DayTypeCalendar
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.hypotheses.model import Hypothesis
from habitus.scene.ledger.store import LedgerStore
from habitus.scene.occurrences.baselines import RECENT_WINDOW_DAYS, BaselineSnapshot, baseline_table
from habitus.scene.occurrences.model import ConceptHit, ConceptHits
from habitus.scene.occurrences.situations import SituationInputs, day_signature, history_days, situation_hits
from habitus.scene.occurrences.store import ConceptHitStore
from habitus.scene.views.intentions import open_consequents_at


class SceneDayContext:
    """一天的情境与常态。``situation_for`` / ``baseline_for`` 直接交给 ``map_closed_day``。"""

    def __init__(
        self,
        day: date,
        *,
        concepts: ConceptSet,
        hits: ConceptHitStore,
        calendar: DayTypeCalendar,
        timezone: tzinfo,
        ledger: LedgerStore | None = None,
        hypotheses: Mapping[str, Hypothesis] | None = None,
        subject: str | None = None,
        recent_days: int = RECENT_WINDOW_DAYS,
    ) -> None:
        if not isinstance(concepts, ConceptSet):
            raise TypeError("concepts must be a ConceptSet")
        if not isinstance(hits, ConceptHitStore):
            raise TypeError("hits must be a ConceptHitStore")
        if not isinstance(calendar, DayTypeCalendar):
            raise TypeError("calendar must be a DayTypeCalendar")
        self.day = day
        self.concepts = concepts
        self.subject = subject
        self.note = calendar.describe(day)
        start = datetime.combine(day, time.min, tzinfo=timezone)
        self.open_consequents = (
            frozenset()
            if ledger is None or hypotheses is None
            else open_consequents_at(ledger, hypotheses, moment=start)
        )
        self.history = {past: day_signature(hits.read_day(past), concepts) for past in history_days(concepts, day)}
        self.baselines: BaselineSnapshot = baseline_table(
            _before(hits, day), concepts, day=day, recent_days=recent_days
        )

    def situation_for(self, document: BehaviorDocument) -> Sequence[ConceptHit]:
        if document.kind is not BehaviorKind.OCCURRENCE:
            raise TypeError("situation_for reads occurrence documents")
        subjects = tuple(str(item) for item in document.fields.get("subjects", ()))
        place = document.fields.get("place")
        return situation_hits(
            self.concepts,
            SituationInputs(
                day=self.day,
                calendar_note=self.note,
                subjects=tuple(item for item in subjects if item != self.subject) if self.subject else subjects,
                place=None if place is None else str(place),
                open_consequents=self.open_consequents,
                history=self.history,
            ),
        )

    def baseline_for(self, _document: BehaviorDocument) -> Mapping[str, str]:
        """常态是"这一天的"，与是哪条 occurrence 无关——同一天两条记录拿到同一份，重放才对得上。"""

        return self.baselines.values


def _before(hits: ConceptHitStore, day: date) -> Iterator[ConceptHits]:
    """常态要的记录：``ALL`` 窗理论上是全部历史，实际按盘上**已经做完的那些天**给（``day`` 之前的全部）。

    一年 365 天的量级还读得动；再长要另设增量累积——记在待改清单里，不在这里悄悄截断（截断会让
    "历来常态"其实只是"最近一年"，而读侧以为它是全部）。
    """

    for past in sorted(hits.days_done()):
        if past < day:
            yield from hits.read_day(past)


__all__ = ["SceneDayContext"]
