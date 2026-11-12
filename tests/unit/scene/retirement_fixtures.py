"""关系行退休测试用的合成数据（实时化阶段 3）：一条 session lane 上两个类，前因每天 10:00、后果每天 16:00（隔得比转移窗口远，
链上从不相接：前因 → 后果的读数既不显著、也不会在前向验证上被判，带着的候选就停在"接着攒"）。哪一端从哪天起不再来、
覆盖与日型怎么给，由各用例自己定。

只用现有接口：退休的新接口（``Status.RETIRED``、``Watching`` 等）在用例文件里。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta

from habitus.conditions import DAY_TYPE_KEY
from habitus.scene.concepts import ConceptSet
from habitus.scene.concepts.calendar_situations import calendar_concepts, calendar_key
from habitus.scene.relations import (
    LaneState,
    LaneTests,
    LaneTimeline,
    Relation,
    RelationConfig,
    RelationKey,
    Segment,
    Status,
    build_timelines,
    fill_calendar,
)
from habitus.scene.relations.longspan import MEASURE_VERSIONS
from habitus.series.model import EventSeries, SeriesRecord
from habitus.temporal.coverage import Coverage
from tests.unit.kind_ids import kind_id
from tests.unit.scene.concept_fixtures import CREATED, base
from tests.unit.scene.relation_fixtures import CONFIG
from tests.unit.temporal.fixtures import at, record, series

LANE = "session"
SLOT_MINUTES = 15
#: 第 0 天（周一）。
DAY0 = date(2026, 6, 1)
CAUSE_TITLE, EFFECT_TITLE = "退休前因", "退休后果"
CAUSE, EFFECT = kind_id(CAUSE_TITLE), kind_id(EFFECT_TITLE)
#: 日历情境（日型 = 节假日）：只在"日历情境当前因"那一条里用。
HOLIDAY = "节假日"
WORKDAY = "工作日"


def day(index: int) -> date:
    return DAY0 + timedelta(days=index)


def concepts(*, cause_retired: bool = False) -> ConceptSet:
    return ConceptSet(
        (
            base(CAUSE_TITLE, retired=cause_retired),
            base(EFFECT_TITLE),
            *calendar_concepts(now=CREATED, observed=[(DAY_TYPE_KEY, HOLIDAY)]),
        )
    )


def chain_key(antecedent: str = CAUSE, consequent: str = EFFECT) -> RelationKey:
    return RelationKey(LANE, antecedent, consequent, Segment.CHAIN)


def holiday_key() -> RelationKey:
    return RelationKey(LANE, HOLIDAY, EFFECT, Segment.SITUATION)


def daily(kind: str, days: range | list[int], hour: int) -> list[SeriesRecord]:
    return [record(at(day(index), hour), kind) for index in days]


def build(
    *,
    cause_days: range | list[int],
    effect_days: range | list[int],
    through: int,
    concept_set: ConceptSet | None = None,
    holiday_days: range | None = None,
) -> tuple[EventSeries, LaneTimeline, ConceptSet]:
    """前因在 ``cause_days`` 那几天 10:00 来、后果在 ``effect_days`` 那几天 16:00 来；空白算到第 ``through`` 天
    （``series.last_day``，截止日 = 它的次日）。``holiday_days`` 给了，就按天把日历情境「节假日」机械填上（那几天成立、其余人在场的日子不成立）。"""

    found = concept_set if concept_set is not None else concepts()
    events = series(
        [*daily(CAUSE, cause_days, 10), *daily(EFFECT, effect_days, 16)], cutoff=day(through) + timedelta(days=1)
    )
    # 时间线只读槽宽与转移窗口：用关系测试共用的那份配置
    line = build_timelines(events, found, {}, CONFIG)[LANE]
    if holiday_days is not None:
        rule = calendar_key(found, HOLIDAY)
        assert rule is not None
        fill_calendar(
            line,
            {HOLIDAY: rule},
            line.days(),
            lambda moment: {DAY_TYPE_KEY: HOLIDAY if (moment - DAY0).days in holiday_days else WORKDAY},
        )
    return events, line, found


def lane_tests(line: LaneTimeline, concept_set: ConceptSet, config: RelationConfig, events: EventSeries) -> LaneTests:
    return LaneTests(line, concept_set, config, events.cutoff)


def coverage(first: int, last: int, *, unwatched: range = range(0)) -> Coverage:
    """第 ``first``–``last`` 天每槽都在看，``unwatched`` 那几天整天没在看（0.0）。"""

    per_day = 24 * 60 // SLOT_MINUTES
    return Coverage(
        slot_minutes=SLOT_MINUTES,
        days={day(index): (0.0 if index in unwatched else 1.0,) * per_day for index in range(first, last + 1)},
    )


def every_day(_day: date) -> str | None:
    return WORKDAY


def holidays_from(index: int) -> Callable[[date], str | None]:
    """第 ``index`` 天起是节假日，之前是工作日。"""

    def day_type(moment: date) -> str | None:
        return HOLIDAY if (moment - DAY0).days >= index else WORKDAY

    return day_type


def previous_state(night_index: int, *relations: Relation) -> LaneState:
    """接的那一代（截止日 = 第 ``night_index`` 天）：带着给定的行，量法版本是当前的（否则带着的行会因"量法改版"被清掉）。"""

    return LaneState(
        lane=LANE, night=day(night_index), family=0, relations=tuple(relations), measure_versions=MEASURE_VERSIONS
    )


def candidate(key: RelationKey, *, discovered: int, needed: int = 40) -> Relation:
    """带着的候选：前向验证要攒 ``needed`` 次（默认攒不够，停在"接着攒"）。"""

    return Relation(key=key, status=Status.CANDIDATE, upward=True, discovered_on=day(discovered), forward_needed=needed)


def rejected(key: RelationKey, *, on: int) -> Relation:
    return Relation(key=key, status=Status.REJECTED, upward=True, discovered_on=day(on - 5), rejected_on=day(on))


def expired(key: RelationKey, *, on: int) -> Relation:
    return Relation(
        key=key,
        status=Status.EXPIRED,
        upward=True,
        discovered_on=day(on - 10),
        established_on=day(on - 5),
        expired_on=day(on),
        maintained_through=day(on),
        maintenance_needed=40,
        misses=2,
    )


def established(key: RelationKey, *, on: int) -> Relation:
    """成立的关系：维持检验每段要攒 40 次（攒不够，这一代不判）。"""

    return Relation(
        key=key,
        status=Status.ESTABLISHED,
        upward=True,
        discovered_on=day(on - 5),
        forward_passed_on=day(on),
        established_on=day(on),
        maintained_through=day(on),
        maintenance_needed=40,
        forward_needed=4,
    )


__all__ = [
    "CAUSE",
    "DAY0",
    "EFFECT",
    "HOLIDAY",
    "LANE",
    "SLOT_MINUTES",
    "WORKDAY",
    "build",
    "candidate",
    "chain_key",
    "concepts",
    "coverage",
    "day",
    "established",
    "every_day",
    "expired",
    "holiday_key",
    "holidays_from",
    "lane_tests",
    "previous_state",
    "rejected",
]
