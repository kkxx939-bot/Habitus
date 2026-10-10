"""关系检验与状态折叠测试共用的合成数据：一个人在工作日 9:00–18:00 用电脑，按固定种子随机做几类事，可以埋一条关系。"""

from __future__ import annotations

import random
from datetime import date, datetime, timedelta, timezone

from habitus.scene.concepts import ConceptSet
from habitus.scene.relations import RelationConfig, RelationKey, Segment
from habitus.series import EventSeries, SeriesRecord
from tests.unit.kind_ids import kind_id
from tests.unit.scene.concept_fixtures import base

CST = timezone(timedelta(hours=8))
START = date(2026, 6, 1)  # 周一
CONFIG = RelationConfig(slot_minutes=15, transition_window_slots=3, recurrence_window_days=90)
TITLES = ("讨论方案", "修改代码", "调研", "写文档", "提交推送")
CONCEPTS = ConceptSet([base(title) for title in TITLES])
DISCUSS, CODE, RESEARCH, DOCS, PUSH = (kind_id(title) for title in TITLES)


def at(day: date, hour: int, minute: int) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=CST)


def record(day: date, hour: int, minute: int, title: str, *, number: int = 0) -> SeriesRecord:
    started = at(day, hour, minute)
    return SeriesRecord(
        uri=f"behavior://occurrences/{day:%Y/%m/%d}/{title}{number}--{started:%H%M}.md",
        name=f"{title}{number}",
        day=day,
        started_at=started,
        last_observed_at=started + timedelta(minutes=10),
        kind_token=kind_id(title),
        lane="session",
        classified=True,
        reminded=False,
        summary=title,
    )


def workdays(count: int) -> list[date]:
    days: list[date] = []
    current = START
    while len(days) < count:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)
    return days


def planted(
    days: int = 40,
    *,
    seed: int = 7,
    follow: float = 0.9,
    linked: tuple[int, int] | None = None,
    moderated: bool = False,
) -> EventSeries:
    """每天：9:00 调研开工、17:50 写文档收工（两件固定作息，专门考钟点的影子）；中间随机时刻做调研、提交推送；
    讨论方案一天 1–2 次，``follow`` 的概率在 15–30 分钟后接修改代码；修改代码另外每天随机来 1 次。
    ``linked`` 给了就只在第 [起, 止) 个工作日埋这条关系（其余日子讨论方案之后不接修改代码）。
    ``moderated``：一半的讨论方案之前 15–30 分钟先调研过，**只有这些**之后才接修改代码（调节：先调研再讨论才动手改）。"""

    generator = random.Random(seed)
    records: list[SeriesRecord] = []
    for index, day in enumerate(workdays(days)):
        chance = follow if linked is None or linked[0] <= index < linked[1] else 0.0
        number = 0

        def add(hour: int, minute: int, title: str, day: date = day) -> None:
            nonlocal number
            number += 1
            records.append(record(day, hour, minute, title, number=number))

        def anytime() -> tuple[int, int]:
            minute = generator.randint(9 * 60 + 5, 17 * 60 + 15)
            return divmod(minute, 60)

        add(9, 0, "调研")
        add(17, 50, "写文档")
        for _ in range(generator.randint(1, 2)):
            hour, minute = anytime()
            primed = moderated and generator.random() < 0.5
            if primed:
                add(*divmod(hour * 60 + minute - generator.randint(15, 30), 60), "调研")
            add(hour, minute, "讨论方案")
            if generator.random() < chance and (primed or not moderated):
                add(*divmod(hour * 60 + minute + generator.randint(15, 30), 60), "修改代码")
        add(*anytime(), "修改代码")
        add(*anytime(), "调研")
        if generator.random() < 0.3:
            add(*anytime(), "提交推送")
    records.sort(key=lambda item: (item.started_at, item.uri))
    cutoff = workdays(days)[-1] + timedelta(days=1)
    return EventSeries(cutoff=cutoff, records=tuple(records))


def key(antecedent: str, consequent: str, segment: Segment = Segment.CHAIN) -> RelationKey:
    return RelationKey("session", antecedent, consequent, segment)
