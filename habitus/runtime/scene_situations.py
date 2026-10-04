"""映射那两个必填参数的生产者：``situation_for`` 与 ``baseline_for``。

``map_closed_day`` 要的两个 callable 都在这里装好——它们的材料横跨三处（当地日历、命中历史、
这条 occurrence 自己的字段），而 ``occurrences`` 那一支不许触达别的存储，``views`` 不能被底层引用。

**一天一装**：情境与常态都是"这一天的"，所以按天构造一次，一天里的每条 occurrence 复用。
前几天的命中事件在构造时读一次；**当天的**在每次问的时候现读——派生情境按事件锚定（"这条 occurrence
之前 24 小时内命中过「晚睡」没有"），凌晨 02:10 的晚睡就在今天的目录里，而今天正在被逐条映射、
边映射边落盘，所以只能现读、再按时刻截到这条之前。

**情境不再读账**（2026-09-30）：「约了球还没打」那一族取消了，情境只看日历、对象、历史命中。
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import date, tzinfo

from habitus.behavior.document import BehaviorDocument
from habitus.behavior.model import BehaviorKind
from habitus.scene.calendar import DayTypeCalendar
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.concepts.rhythm import Rhythm
from habitus.scene.occurrences.baselines import RECENT_WINDOW_DAYS, BaselineSnapshot, baseline_table, local_minute
from habitus.scene.occurrences.model import ConceptHits
from habitus.scene.occurrences.situations import (
    HitEvent,
    SituationInputs,
    SituationOutcome,
    history_days,
    hit_events,
    situation_hits,
)
from habitus.scene.occurrences.store import ConceptHitStore


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
        subject: str | None = None,
        recent_days: int = RECENT_WINDOW_DAYS,
        rhythms: Mapping[str, Rhythm] | None = None,
        slack_minutes: int = 0,
    ) -> None:
        if not isinstance(concepts, ConceptSet):
            raise TypeError("concepts must be a ConceptSet")
        if not isinstance(hits, ConceptHitStore):
            raise TypeError("hits must be a ConceptHitStore")
        if not isinstance(calendar, DayTypeCalendar):
            raise TypeError("calendar must be a DayTypeCalendar")
        self.day = day
        self.concepts = concepts
        self.hits = hits
        self.timezone = timezone
        self.subject = subject
        self.note = calendar.describe(day)
        self.reads_today = day in history_days(concepts, day)
        self.history: tuple[HitEvent, ...] = hit_events(
            (record for past in history_days(concepts, day) if past < day for record in hits.read_day(past)), concepts
        )
        self.baselines: BaselineSnapshot = baseline_table(
            _before(hits, day), concepts, day=day, recent_days=recent_days, rhythms=rhythms, slack_minutes=slack_minutes, timezone=timezone
        )

    def situation_for(self, document: BehaviorDocument) -> SituationOutcome:
        if document.kind is not BehaviorKind.OCCURRENCE:
            raise TypeError("situation_for reads occurrence documents")
        subjects = tuple(str(item) for item in document.fields.get("subjects", ()))
        place = document.fields.get("place")
        moment = document.address.started_at
        history = self.history
        if self.reads_today:
            today = hit_events((record for record in self.hits.read_day(self.day) if record.started_at < moment), self.concepts)
            history = tuple(sorted((*history, *today), key=lambda item: (item.at.timestamp(), item.identity)))
        return situation_hits(
            self.concepts,
            SituationInputs(
                day=self.day,
                moment=moment,
                calendar_note=self.note,
                subjects=tuple(item for item in subjects if item != self.subject) if self.subject else subjects,
                place=None if place is None else str(place),
                history=history,
            ),
        )

    def baseline_for(self, document: BehaviorDocument) -> Mapping[str, str]:
        """常态是"这一天的"，只按这条 occurrence 落在哪个钟面峰挑那一份（R3-26）——同一天同一峰的记录拿到同一份，重放才对得上。"""

        if document.kind is not BehaviorKind.OCCURRENCE:
            raise TypeError("baseline_for reads occurrence documents")
        return self.baselines.values_at(local_minute(document.address.started_at, self.timezone))


def _before(hits: ConceptHitStore, day: date) -> Iterator[ConceptHits]:
    """常态要的记录：``ALL`` 窗理论上是全部历史，实际按盘上**已经做完的那些天**给（``day`` 之前的全部）。

    一年 365 天的量级还读得动；再长要另设增量累积——记在待改清单里，不在这里悄悄截断（截断会让
    "历来常态"其实只是"最近一年"，而读侧以为它是全部）。
    """

    for past in sorted(hits.days_done()):
        if past < day:
            yield from hits.read_day(past)


__all__ = ["SceneDayContext"]
