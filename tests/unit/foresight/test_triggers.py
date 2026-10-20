"""触发与候选（第 3 步）：事件 / 情境 / 日子三种触发在埋了关系的合成数据上各出什么，反向前因清单标在场与不在场，装配成预期。
每个机会都建（裁定 49）：这回已经做了的照建、不退场，时窗保留历史那几次的实际时刻、不挪到此刻。"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone

import pytest

from habitus.conditions import HolidayTable, HolidayYear, LocalCalendar
from habitus.foresight import ForesightError, Moment, Seen, form_expectations
from habitus.foresight.ledger import expectation_identity
from habitus.foresight.opportunities import (
    KindDays,
    Opportunity,
    OpportunityTable,
    RelationOpportunities,
    build_table,
)
from habitus.foresight.triggers import antecedents_of, day_drafts, event_drafts, situation_drafts, timeline_of
from habitus.foresight.triggers.draft import instance_at
from habitus.scene.concepts import ConceptSet
from habitus.scene.concepts.calendar_situations import calendar_concepts
from habitus.scene.concepts.rhythm import RhythmRulers
from habitus.scene.recurrence import recurrences_of
from habitus.scene.relations import Relation, Segment, Status, build_timelines
from habitus.series import EventSeries, SeriesRecord
from habitus.temporal import DayInterval, OccurrenceDay
from habitus.temporal.slots import absolute_slot
from tests.unit.scene.concept_fixtures import CREATED
from tests.unit.scene.relation_fixtures import CODE, CONCEPTS, CONFIG, DISCUSS, DOCS, RESEARCH, key, planted, record

CST = timezone(timedelta(hours=8))
RULERS = RhythmRulers(15, 2700, 3, 5.0, 5.0, 0.10, 35)
#: 带日历情境的概念集（周三、20 号、工作日……按天机械填）。
CALENDAR_CONCEPTS = ConceptSet(
    (*(CONCEPTS[identity] for identity in CONCEPTS.behaviors()), *calendar_concepts(now=CREATED))
)


def relation(
    antecedent: str, consequent: str, segment: Segment = Segment.CHAIN, *, status: Status = Status.ESTABLISHED
) -> Relation:
    return Relation(key=key(antecedent, consequent, segment), status=status, upward=True)


def table_of(
    series: EventSeries, relations: tuple[Relation, ...], *, calendar: LocalCalendar | None = None
) -> OpportunityTable:
    concepts = CONCEPTS if calendar is None else CALENDAR_CONCEPTS
    line = build_timelines(series, concepts, {}, CONFIG, calendar=calendar)["session"]
    return build_table(
        line,
        relations,
        concepts,
        night=series.cutoff,
        relation_night=series.cutoff,
        slot_minutes=CONFIG.slot_minutes,
        window_slots=CONFIG.transition_window_slots,
        recurrences=recurrences_of(series, RULERS),
    )


def seen_until(series: EventSeries, now: datetime) -> tuple[Seen, ...]:
    return tuple(
        Seen(item.started_at, min(item.last_observed_at, now), item.kind_token, item.uri)
        for item in series.records
        if item.started_at <= now and (now.date() - item.day).days <= 7
    )


def moment_at(now: datetime, calendar: LocalCalendar | None = None) -> Moment:
    found = (calendar or LocalCalendar()).for_expectations(now.date())
    return Moment(at=now, slot=(now.hour * 60 + now.minute) // CONFIG.slot_minutes, calendar=dict(found))


def kind_of(token: str, label: str, days: list[tuple[date, str, datetime, int]]) -> KindDays:
    """手搭一个类的复现：每天（日子、第一条 URI、几点、几次自然）；没有自然的那天第一条就是被提醒才做的，结尾在它的间隔截尾。"""

    found = tuple(
        OccurrenceDay(day, uri, moment, 1, natural, uri if natural else None, moment if natural else None)
        for day, uri, moment, natural in days
    )
    intervals = tuple(
        DayInterval(a, b, (b.day - a.day).days, 1.0, b.natural == 0) for a, b in zip(found, found[1:], strict=False)
    )
    return KindDays(token, label, found, intervals)


def just_after(record: SeriesRecord) -> datetime:
    """这条记录最后所见那一槽过完之后的第一拍再过两分钟：前因已经封口（至少一槽没再见到）。"""

    seen = record.last_observed_at
    slot_start = seen.replace(minute=seen.minute - seen.minute % CONFIG.slot_minutes, second=0, microsecond=0)
    return slot_start + timedelta(minutes=CONFIG.slot_minutes + 2)


def discussion_followed_by_code(series: EventSeries) -> tuple[SeriesRecord, SeriesRecord]:
    """最后几个工作日里一对「讨论方案 → 15–30 分钟后修改代码」：那天在它之前没有修改代码，中间没有别的修改代码，
    且修改代码来在讨论方案封口（``just_after``）之后——不然"刚发生"那一拍它已经发生了。"""

    records = series.records
    for index in range(len(records) - 1, 0, -1):
        item = records[index]
        if item.kind_token != DISCUSS:
            continue
        later = [other for other in records if other.day == item.day and other.started_at > item.started_at]
        if any(
            other.kind_token == CODE and other.started_at < item.started_at
            for other in records
            if other.day == item.day
        ):
            continue
        if (
            later
            and later[0].kind_token == CODE
            and timedelta(minutes=15) <= later[0].started_at - item.started_at <= timedelta(minutes=30)
            and later[0].started_at > just_after(item)
        ):
            return item, later[0]
    raise AssertionError("the planted series has no such pair")


# ── 事件 ────────────────────────────────────────────────────────────────────


def test_a_discussion_just_finished_makes_code_a_candidate_with_the_window_from_history() -> None:
    series = planted()
    table = table_of(series, (relation(DISCUSS, CODE),))
    discussion, code = discussion_followed_by_code(series)
    now = just_after(discussion)
    rows = seen_until(series, now)
    assert all(item.source != code.uri for item in rows)  # 接着的那条修改代码还没发生
    line = timeline_of(rows, lane="session", slot_minutes=CONFIG.slot_minutes, horizon=now.date() + timedelta(days=1))
    moment = moment_at(now)
    now_slot = absolute_slot(now, slot_minutes=CONFIG.slot_minutes)
    (draft,) = event_drafts(table, line, moment, now_slot=now_slot, statuses=("established",))
    (stored,) = table.relations
    assert draft.kind_token == CODE and draft.source == "事件" and draft.trigger == stored.key
    assert draft.trigger_instance == instance_at(discussion.started_at)  # 前因的开始时刻，封口前后不变
    assert draft.opportunities == stored.opportunities and draft.hits == stored.hits > 20
    assert len(draft.basis) == draft.hits and all("修改代码" in uri for uri in draft.basis)
    start, end = draft.window
    assert start <= end <= draft.deadline and now <= draft.deadline
    # 从前因做完那一槽起算：这之后的修改代码都算这条说中的
    assert draft.observed_from <= now and draft.observed_from <= code.started_at
    # 同一条链的边界：链做完那一槽起、转移窗口 + 1 槽；历史上来了的都在这里面
    boundary = datetime.combine(now.date(), time(0), tzinfo=CST) + timedelta(
        minutes=(
            absolute_slot(discussion.last_observed_at, slot_minutes=15)
            - now.date().toordinal() * 96
            + CONFIG.transition_window_slots
            + 1
        )
        * 15
    )
    assert draft.deadline == boundary and end <= boundary
    # 反向：修改代码的前因清单里这条关系在场、出处就是这条讨论方案
    (antecedent,) = antecedents_of(table, CODE, line, moment, now_slot=now_slot, statuses=("established",))
    assert (
        antecedent.relation == stored.key
        and antecedent.present
        and antecedent.upward
        and antecedent.source == discussion.uri
    )


def test_once_the_code_started_the_opportunity_is_still_recorded_as_a_hit() -> None:
    """后果已经来了照建（裁定 49）：那是一次命中，不建账上就只剩没来的那几次；结算从前因做完起算，会结成准时 / 早到。"""

    series = planted()
    table = table_of(series, (relation(DISCUSS, CODE),))
    discussion, code = discussion_followed_by_code(series)
    now = code.started_at + timedelta(minutes=1)
    line = timeline_of(seen_until(series, now), lane="session", slot_minutes=15, horizon=now.date() + timedelta(days=1))
    drafts = event_drafts(
        table, line, moment_at(now), now_slot=absolute_slot(now, slot_minutes=15), statuses=("established",)
    )
    (draft,) = [item for item in drafts if item.trigger_instance == instance_at(discussion.started_at)]
    assert draft.observed_from < code.started_at <= draft.deadline


def test_yesterdays_docs_make_research_a_next_day_candidate_until_it_happens() -> None:
    series = planted()
    table = table_of(series, (relation(DOCS, RESEARCH, Segment.NEXT_DAY),))
    today = date(2026, 7, 21)  # 周二；周一 17:50 写文档
    now = datetime.combine(today, time(8, 30), tzinfo=CST)
    rows = seen_until(series, now)
    line = timeline_of(rows, lane="session", slot_minutes=15, horizon=today + timedelta(days=1))
    (draft,) = event_drafts(
        table, line, moment_at(now), now_slot=absolute_slot(now, slot_minutes=15), statuses=("established",)
    )
    yesterday_docs = next(
        item for item in series.records if item.day == today - timedelta(days=1) and item.kind_token == DOCS
    )
    assert draft.kind_token == RESEARCH and draft.trigger_instance == instance_at(yesterday_docs.started_at)
    # 历史上次日的调研第一次都是 09:00 开工那条：时窗是今天 09:00 起的一槽（单一时刻只是一个点，拍点落不上）
    assert draft.window == (
        datetime.combine(today, time(9, 0), tzinfo=CST),
        datetime.combine(today, time(9, 15), tzinfo=CST),
    )
    assert draft.deadline == datetime.combine(today + timedelta(days=1), time(0), tzinfo=CST)
    assert draft.observed_from == datetime.combine(today, time(0), tzinfo=CST)
    # 09:00 的调研发生之后照建（同一个身份）：从今天零点起算，结算会给准时
    later = datetime.combine(today, time(9, 10), tzinfo=CST)
    line = timeline_of(seen_until(series, later), lane="session", slot_minutes=15, horizon=today + timedelta(days=1))
    (again,) = event_drafts(
        table, line, moment_at(later), now_slot=absolute_slot(later, slot_minutes=15), statuses=("established",)
    )
    assert again.trigger_instance == draft.trigger_instance and again.window == draft.window
    (antecedent,) = antecedents_of(
        table,
        RESEARCH,
        line,
        moment_at(later),
        now_slot=absolute_slot(later, slot_minutes=15),
        statuses=("established",),
    )
    assert antecedent.present and antecedent.source == yesterday_docs.uri


def test_candidate_relations_only_enter_when_the_shadow_flag_is_on_and_absent_antecedents_are_listed() -> None:
    series = planted()
    table = table_of(
        series, (relation(DISCUSS, CODE), relation(RESEARCH, CODE, Segment.REST_OF_DAY, status=Status.CANDIDATE))
    )
    discussion, _code = discussion_followed_by_code(series)
    now = just_after(discussion)
    line = timeline_of(seen_until(series, now), lane="session", slot_minutes=15, horizon=now.date() + timedelta(days=1))
    moment, now_slot = moment_at(now), absolute_slot(now, slot_minutes=15)
    assert len(event_drafts(table, line, moment, now_slot=now_slot, statuses=("established",))) == 1
    both = event_drafts(table, line, moment, now_slot=now_slot, statuses=("established", "candidate"))
    assert {item.trigger for item in both} == {
        item.key for item in table.relations
    }  # 09:00 调研早已过了转移窗口：当天剩下的在场
    # 下班前：讨论方案不在场（不在同一条链里）、调研仍在当天剩下的里
    evening = datetime.combine(now.date(), time(17, 40), tzinfo=CST)
    line = timeline_of(
        seen_until(series, evening), lane="session", slot_minutes=15, horizon=now.date() + timedelta(days=1)
    )
    listed = antecedents_of(
        table,
        CODE,
        line,
        moment_at(evening),
        now_slot=absolute_slot(evening, slot_minutes=15),
        statuses=("established", "candidate"),
    )
    by_key = {item.relation: item for item in listed}
    assert by_key[key(RESEARCH, CODE, Segment.REST_OF_DAY).text].present
    chain_item = by_key[key(DISCUSS, CODE).text]
    assert chain_item.present is (
        absolute_slot(evening, slot_minutes=15) - max(m.end for m in line.marks[DISCUSS] if m.day == now.date()) <= 3
    )
    assert all(item.source == "" for item in listed if not item.present)


def test_an_event_candidate_leaves_once_the_latest_historical_arrival_has_passed_with_one_slot_of_grace() -> None:
    """手搭一条关系：前因 10:00–10:10（做完那一槽 10:00 开头），历史上后果来在做完后 5 分钟与 20 分钟。"""

    day = date(2026, 7, 20)
    ended = datetime.combine(day - timedelta(days=1), time(10, 0), tzinfo=CST)
    stored = RelationOpportunities(
        key=key(DISCUSS, CODE).text,
        antecedent=DISCUSS,
        consequent=CODE,
        antecedent_label="讨论方案",
        consequent_label="修改代码",
        antecedent_classes=(DISCUSS,),
        consequent_classes=(CODE,),
        segment="chain",
        status="established",
        upward=True,
        items=tuple(
            Opportunity(f"a{n}", (f"a{n}",), ended.date(), ended, True, f"c{n}", ended + timedelta(minutes=minutes))
            for n, minutes in ((1, 5), (2, 20))
        ),
    )
    table = OpportunityTable("session", day, day, 15, 3, (stored,), ())
    discussion = Seen(
        datetime.combine(day, time(10, 0), tzinfo=CST),
        datetime.combine(day, time(10, 10), tzinfo=CST),
        DISCUSS,
        "today-a",
    )

    def drafts_at(hour: int, minute: int):  # type: ignore[no-untyped-def]
        now = datetime.combine(day, time(hour, minute), tzinfo=CST)
        line = timeline_of((discussion,), lane="session", slot_minutes=15, horizon=day + timedelta(days=1))
        return event_drafts(
            table, line, moment_at(now), now_slot=absolute_slot(now, slot_minutes=15), statuses=("established",)
        )

    assert drafts_at(10, 12) == ()  # 前因最后所见就在这一槽：还在进行中，不建
    (draft,) = drafts_at(10, 16)
    assert draft.window == (  # 历史那几次的实际时刻：做完后 5 到 20 分钟，不挪到此刻
        datetime.combine(day, time(10, 5), tzinfo=CST),
        datetime.combine(day, time(10, 20), tzinfo=CST),
    )
    assert draft.observed_from == datetime.combine(day, time(10, 0), tzinfo=CST)
    assert draft.trigger_instance == instance_at(discussion.started_at) and draft.basis == ("c1", "c2")
    assert drafts_at(10, 26) == ()  # 过了 10:20、也过了宽限的一槽：退场
    assert drafts_at(10, 50) == ()  # 仍在同一条链的段里（3 槽），但历史上从没这么晚来过
    # 只命中过一次的关系：样本少就没有读数，不建
    single = OpportunityTable("session", day, day, 15, 3, (replace(stored, items=stored.items[:1]),), ())
    now = datetime.combine(day, time(10, 16), tzinfo=CST)
    line = timeline_of((discussion,), lane="session", slot_minutes=15, horizon=day + timedelta(days=1))
    assert (
        event_drafts(
            single, line, moment_at(now), now_slot=absolute_slot(now, slot_minutes=15), statuses=("established",)
        )
        == ()
    )


# ── 情境 ────────────────────────────────────────────────────────────────────


def test_a_wednesday_situation_builds_an_expectation_on_wednesdays_from_the_times_on_past_wednesdays() -> None:
    """语义树上成立的"周三 → 写文档"：今天是周三、到了历史上周三写文档的时刻，就建；别的日子不建（裁定 59）。"""

    series = planted()
    calendar = LocalCalendar()
    table = table_of(series, (relation("周三", DOCS, Segment.SITUATION),), calendar=calendar)
    (stored,) = table.relations
    assert stored.segment == "situation" and stored.calendar_key == "日历.周几" and stored.calendar_values == ("周三",)
    wednesday = date(2026, 7, 22)
    early = datetime.combine(wednesday, time(17, 0), tzinfo=CST)
    assert (
        situation_drafts(table, moment_at(early), done_today=frozenset(), statuses=("established",)) == ()
    )  # 还没到时窗
    now = datetime.combine(wednesday, time(17, 52), tzinfo=CST)
    (draft,) = situation_drafts(table, moment_at(now), done_today=frozenset(), statuses=("established",))
    past = [item for item in stored.items if item.day < wednesday]
    assert draft.kind_token == DOCS and draft.source == "情境" and draft.trigger == stored.key
    assert draft.trigger_instance == "2026-07-22" and draft.hits == len(past) == draft.opportunities
    assert all(item.day.weekday() == 2 for item in stored.items)  # 机会只是周三那几天
    assert draft.window == (now.replace(minute=50), now.replace(hour=18, minute=5))  # 往常 17:50，至少一槽宽
    assert draft.deadline == datetime.combine(wednesday + timedelta(days=1), time(0), tzinfo=CST)
    assert draft.observed_from == datetime.combine(wednesday, time(0), tzinfo=CST)
    # 过了窗：没做的退场，做了的照建（裁定 49）
    late = now.replace(hour=20)
    assert situation_drafts(table, moment_at(late), done_today=frozenset(), statuses=("established",)) == ()
    (kept,) = situation_drafts(table, moment_at(late), done_today=frozenset({DOCS}), statuses=("established",))
    assert kept.trigger_instance == draft.trigger_instance
    # 反向前因清单：周三那条关系今天在场，周四不在场
    line = timeline_of(seen_until(series, now), lane="session", slot_minutes=15, horizon=wednesday + timedelta(days=1))
    for moment, present in (
        (moment_at(now), True),
        (moment_at(datetime.combine(date(2026, 7, 23), time(17, 52), tzinfo=CST)), False),
    ):
        (listed,) = antecedents_of(
            table, DOCS, line, moment, now_slot=absolute_slot(moment.at, slot_minutes=15), statuses=("established",)
        )
        assert listed.present is present and listed.relation == stored.key and listed.source == ""
    # 周四：前因不成立，不建；候选关系只在影子开着时进
    thursday = datetime.combine(date(2026, 7, 23), time(17, 52), tzinfo=CST)
    assert situation_drafts(table, moment_at(thursday), done_today=frozenset(), statuses=("established",)) == ()
    candidate = table_of(
        series, (relation("周三", DOCS, Segment.SITUATION, status=Status.CANDIDATE),), calendar=calendar
    )
    assert situation_drafts(candidate, moment_at(now), done_today=frozenset(), statuses=("established",)) == ()
    assert situation_drafts(candidate, moment_at(now), done_today=frozenset(), statuses=("established", "candidate"))
    # 只命中过一次：样本少就没有读数
    once = OpportunityTable("session", table.night, table.night, 15, 3, (replace(stored, items=tuple(past[:1])),), ())
    assert situation_drafts(once, moment_at(now), done_today=frozenset(), statuses=("established",)) == ()


def test_a_workday_situation_is_not_built_on_a_holiday() -> None:
    """锚在工作日的规律：节假日那天前因不成立，自然不建，不用另写否决。"""

    holiday = date(2026, 7, 22)
    year = HolidayYear("CN", 2026, "测试", "1", frozenset({5, 6}), {holiday: "测试假"}, frozenset())
    calendar = LocalCalendar(HolidayTable({2026: year}))
    table = table_of(planted(), (relation("工作日", RESEARCH, Segment.SITUATION),), calendar=calendar)
    (stored,) = table.relations
    assert holiday not in {item.day for item in stored.items}  # 放假那天不是工作日的机会
    tuesday = datetime.combine(date(2026, 7, 21), time(9, 5), tzinfo=CST)
    assert situation_drafts(table, moment_at(tuesday, calendar), done_today=frozenset(), statuses=("established",))
    on_holiday = datetime.combine(holiday, time(9, 5), tzinfo=CST)
    assert (
        situation_drafts(table, moment_at(on_holiday, calendar), done_today=frozenset(), statuses=("established",))
        == ()
    )
    # 没有放假数据时：建预期按名义周末给日型（统计里是未知，不进机会）
    assert moment_at(on_holiday).calendar["日历.日型"] == "工作日"


# ── 日子 ────────────────────────────────────────────────────────────────────


def test_the_interval_trigger_is_a_discrete_hazard_by_elapsed_days_and_the_window_is_the_cluster() -> None:
    series = planted()
    table = table_of(series, ())
    research = table.kind(RESEARCH)
    assert research is not None
    gaps = [(b.day - a.day).days for a, b in zip(research.days, research.days[1:], strict=False)]
    assert set(gaps) == {1, 3}  # 每个工作日 09:00 调研：隔 1 天，过周末隔 3 天
    today = series.cutoff  # 2026-07-25 周六，没有记录；离上次（周五）1 天
    now = datetime.combine(today, time(8, 0), tzinfo=CST)
    drafts = day_drafts(table, moment_at(now), done_today=frozenset(), recent=35)
    (interval,) = [item for item in drafts if item.kind_token == RESEARCH and item.trigger == "interval"]
    last = research.days[-1]
    assert last.day == date(2026, 7, 24) and interval.trigger_instance == f"{last.first_uri}@2026-07-25"
    # 逝去 1 天：历史上每次间隔都 ≥ 1（全在险），恰好隔 1 天的是命中；1 与 3 不相邻，这一簇只有"第 1 天"
    assert interval.opportunities == len(gaps) and interval.hits == gaps.count(1) == len(interval.basis)
    assert interval.window == (
        datetime.combine(today, time(9, 0), tzinfo=CST),
        datetime.combine(today, time(9, 15), tzinfo=CST),
    )
    assert interval.deadline == datetime.combine(date(2026, 7, 26), time(0), tzinfo=CST)
    # 过了这一簇的时窗：退场（不再是"上次 + 最大间隔"那种伸到下周的窗）
    evening = datetime.combine(today, time(20, 0), tzinfo=CST)
    assert not [
        item
        for item in day_drafts(table, moment_at(evening), done_today=frozenset(), recent=35)
        if item.kind_token == RESEARCH and item.trigger == "interval"
    ]
    # 周一（逝去 3 天）：在险的只剩过周末的那几次，命中全是它们——另一簇、另一条实例
    monday = datetime.combine(date(2026, 7, 27), time(8, 0), tzinfo=CST)
    (later,) = [
        item
        for item in day_drafts(table, moment_at(monday), done_today=frozenset(), recent=35)
        if item.kind_token == RESEARCH and item.trigger == "interval"
    ]
    assert later.trigger_instance == f"{last.first_uri}@2026-07-27"
    assert later.opportunities == later.hits == gaps.count(3) and later.window[0] == monday.replace(hour=9)
    # 逝去 2 天、5 天：历史上没有恰好隔这么久做过的，不是它的日子；今天已做的照建（裁定 49）
    for day in (date(2026, 7, 26), date(2026, 7, 29)):
        moment = moment_at(datetime.combine(day, time(8, 0), tzinfo=CST))
        assert not [
            item
            for item in day_drafts(table, moment, done_today=frozenset(), recent=35)
            if item.kind_token == RESEARCH and item.trigger == "interval"
        ]
    (done,) = [
        item
        for item in day_drafts(table, moment_at(now), done_today=frozenset({RESEARCH}), recent=35)
        if item.kind_token == RESEARCH and item.trigger == "interval"
    ]
    assert done.trigger_instance == interval.trigger_instance
    assert interval.observed_from == datetime.combine(today, time(0), tzinfo=CST)
    # 做了的过了窗也照建；没做的过了窗退场（上面晚上那一拍）
    assert [
        item
        for item in day_drafts(table, moment_at(evening), done_today=frozenset({RESEARCH}), recent=35)
        if item.kind_token == RESEARCH and item.trigger == "interval"
    ]


def test_consecutive_gaps_are_one_cluster_whichever_day_it_is_recomputed_on() -> None:
    """间隔 2、3、4 天连成一簇：第 2 天建的那条与第 3 天重算的是同一个身份（不然第 3 天那次发生被同一本账结成两次命中）。"""

    start = date(2026, 6, 1)
    offsets = [0]
    for gap in (2, 3, 4, 2, 3, 4, 3):
        offsets.append(offsets[-1] + gap)
    days = tuple(start + timedelta(days=offset) for offset in offsets)
    kind = kind_of(
        "s-k0077", "买菜", [(day, f"veg-{day:%m%d}", datetime.combine(day, time(18, 0), tzinfo=CST), 1) for day in days]
    )
    table = OpportunityTable("session", days[-1], days[-1], 15, 3, (), (kind,))
    instances = []
    for elapsed in (2, 3, 4):
        moment = moment_at(datetime.combine(days[-1] + timedelta(days=elapsed), time(8, 0), tzinfo=CST))
        (draft,) = [
            item for item in day_drafts(table, moment, done_today=frozenset(), recent=35) if item.trigger == "interval"
        ]
        instances.append(draft.trigger_instance)
    assert instances == [f"veg-{days[-1]:%m%d}@{days[-1] + timedelta(days=2)}"] * 3


# ── 装配 ────────────────────────────────────────────────────────────────────


def test_assembly_gives_each_draft_an_identity_without_the_night_and_attaches_the_antecedents() -> None:
    series = planted()
    table = table_of(series, (relation(DISCUSS, CODE),))
    discussion, _code = discussion_followed_by_code(series)
    now = just_after(discussion)
    rows = seen_until(series, now)
    found = form_expectations(moment=moment_at(now), table=table, rows=rows, generation="g1", version="3", recent=35)
    assert found and all(
        item.formed_at == now and item.night == table.night and item.generation == "g1" and item.version == "3"
        for item in found
    )
    event = next(item for item in found if item.source == "事件")
    assert event.expectation_id == expectation_identity(CODE, "事件", event.trigger, instance_at(discussion.started_at))
    assert event.antecedents and event.antecedents[0].present and event.antecedents[0].source == discussion.uri
    assert event.level == 0 and event.observed_from <= event.formed_at
    # 日子触发：今天已做的类照建（裁定 49），从今天零点起算
    assert all(
        item.observed_from == datetime.combine(now.date(), time(0), tzinfo=CST)
        for item in found
        if item.source == "日子"
    )
    done_today = {row.kind_token for row in rows if row.day == now.date()}
    assert any(item.kind_token in done_today for item in found if item.source == "日子")
    assert {item.expectation_id for item in found} == {
        expectation_identity(item.kind_token, item.source, item.trigger, item.trigger_instance) for item in found
    }


def test_the_moment_slot_must_use_the_table_slot_width() -> None:
    series = planted()
    table = table_of(series, ())
    now = datetime.combine(series.cutoff, time(8, 0), tzinfo=CST)
    with pytest.raises(ForesightError, match="moment.slot"):
        form_expectations(moment=Moment(at=now, slot=0), table=table, rows=(), recent=35)
    assert form_expectations(moment=moment_at(now), table=None, rows=(), recent=35) == ()


def test_a_window_that_closed_since_the_last_look_is_still_built_when_nothing_was_done() -> None:
    """拍点晚了一分钟：时窗 21:00–21:15，上一拍 21:01、这一拍 21:16。没做的那次也照建，不然账上只丢没做的、把账往上抬。"""

    from habitus.foresight.triggers.draft import clamp

    start = datetime(2026, 9, 3, 21, 0, tzinfo=CST)
    end = start + timedelta(minutes=15)
    boundary = start.replace(hour=23, minute=59)
    now = start + timedelta(minutes=16)
    assert clamp(start, end, boundary, now=now, slot_minutes=15) is None  # 只看此刻：退场
    assert clamp(start, end, boundary, now=now, slot_minutes=15, caught_from=now - timedelta(minutes=15)) is not None
    late = now + timedelta(hours=2)
    assert clamp(start, end, boundary, now=late, slot_minutes=15, caught_from=late - timedelta(minutes=15)) is None
    with pytest.raises(ForesightError):
        Moment(at=now, slot=84, looked_since=now + timedelta(minutes=1))


def test_the_interval_window_follows_the_latest_habit_not_the_whole_history() -> None:
    """26 号清单 4.7：隔一天跑一次，前 40 次 19:00 跑，后来改成 07:00 又跑了 40 次。时窗只看最近 M 次命中（M = 35），
    时窗末不再拖在 19:00；依据只列最近 M 次。在险间隔照旧数全历史。"""

    start = date(2025, 6, 1)
    days = [start + timedelta(days=2 * index) for index in range(80)]
    runs = kind_of(
        "s-k0088",
        "跑步",
        [
            (day, f"run-{day:%y%m%d}", datetime.combine(day, time(19 if index < 40 else 7, 0), tzinfo=CST), 1)
            for index, day in enumerate(days)
        ],
    )
    table = OpportunityTable("session", days[-1], days[-1], 15, 3, (), (runs,))
    early = moment_at(datetime.combine(days[-1] + timedelta(days=2), time(5, 0), tzinfo=CST))
    (recent,) = day_drafts(table, early, done_today=frozenset(), recent=35)
    assert recent.window[1].time() < time(8, 0) and recent.opportunities == 79
    (whole,) = day_drafts(table, early, done_today=frozenset(), recent=80)
    assert whole.window[1].time() >= time(19, 0)  # 全历史时窗末拖到 19:00
    found = form_expectations(moment=early, table=table, rows=(), recent=35)
    assert len(found) == 1 and len(found[0].basis) == 35 and found[0].basis[-1] == f"run-{days[-1]:%y%m%d}"


def test_a_day_done_only_after_a_reminder_is_unknown_to_the_interval_trigger() -> None:
    """裁定 54：被提醒了才做的那天，不提醒做不做看不到——分子分母都不计（以它结尾的间隔截尾）；时窗与出处取当天第一次自然
    发生的，提醒的时刻不许漏回"他往常几点做"。"""

    # 每 7 天买一次菜，第三次是提醒了才买的——那一次落在这一簇里，不是机会
    start = date(2026, 6, 1)
    days = [start + timedelta(days=7 * index) for index in range(5)]
    veg = kind_of(
        "s-k0077",
        "买菜",
        [
            (day, f"veg-{day:%m%d}", datetime.combine(day, time(18, 0), tzinfo=CST), 0 if index == 2 else 1)
            for index, day in enumerate(days)
        ],
    )
    weekly = OpportunityTable("session", days[-1], days[-1], 15, 3, (), (veg,))
    later = moment_at(datetime.combine(days[-1] + timedelta(days=7), time(8, 0), tzinfo=CST))
    (interval,) = [
        item for item in day_drafts(weekly, later, done_today=frozenset(), recent=35) if item.trigger == "interval"
    ]
    assert (interval.hits, interval.opportunities) == (3, 3)
    # 时窗与出处：每周五改代码，第三个周五 07:00 被提醒改了一次、09:30 又自己改了一次；另几周都是 09:30
    fridays = [date(2026, 6, 5) + timedelta(days=7 * index) for index in range(4)]
    records = [record(day, 9, 30, "修改代码", number=2) for day in fridays]
    records.append(replace(record(fridays[2], 7, 0, "修改代码", number=1), reminded=True))
    for offset in range(30):
        records.append(record(date(2026, 6, 1) + timedelta(days=offset), 12, 0, "调研", number=7))
    today = fridays[-1] + timedelta(days=7)
    series = EventSeries(cutoff=today, records=tuple(sorted(records, key=lambda item: (item.started_at, item.uri))))
    built = table_of(series, ())
    third = next(item for item in built.kind(CODE).days if item.day == fridays[2])  # type: ignore[union-attr]
    assert (third.count, third.natural) == (2, 1)
    assert third.first_uri.endswith("修改代码1--0700.md") and third.natural_uri.endswith("修改代码2--0930.md")  # type: ignore[union-attr]
    early = datetime.combine(today, time(6, 0), tzinfo=CST)
    (code,) = [
        item
        for item in day_drafts(built, moment_at(early), done_today=frozenset(), recent=35)
        if item.kind_token == CODE and item.trigger == "interval"
    ]
    assert code.window[0].time() == time(9, 30)
    assert all(uri.endswith("修改代码2--0930.md") for uri in code.basis)


# ── 带条件的关系（裁定 61 第 6 条、62 第 4 条） ──────────────────────────────


def _primed_and_plain_discussions(series: EventSeries, line, window: int):  # type: ignore[no-untyped-def]
    """一条之前 15–30 分钟先调研过的讨论方案，与一条没有的（都在最后几个工作日，封口之后才接修改代码或根本没接）。"""

    from habitus.scene.relations.conditions import Condition, Template, evaluator

    judge = evaluator(line, Condition(Template.TIMELINE, RESEARCH), window=window)
    primed = plain = None
    for chain in reversed(line_chains(line, window)):
        record = next(item for item in series.records if item.uri == chain.anchor.uri)
        flag = judge(chain)
        if flag and primed is None:
            primed = record
        elif flag is False and plain is None:
            plain = record
        if primed is not None and plain is not None:
            return primed, plain
    raise AssertionError("the moderated series has no such pair")


def line_chains(line, window: int):  # type: ignore[no-untyped-def]
    from habitus.scene.relations import chains_of

    return chains_of(line, DISCUSS, window)


def test_a_family_forms_only_its_most_specific_holding_relation_and_the_base_when_no_condition_holds() -> None:
    """不带条件的 + 带条件的是一族：先调研过的那次只建带条件那条（前因清单里也只有它在场）；没调研过的那次只建不带条件的。"""

    from habitus.scene.relations import RelationKey

    series = planted(moderated=True)
    primed_key = RelationKey("session", DISCUSS, CODE, Segment.CHAIN, f"timeline:{RESEARCH}")
    table = table_of(
        series, (relation(DISCUSS, CODE), Relation(key=primed_key, status=Status.ESTABLISHED, upward=True))
    )
    assert {item.condition_text for item in table.relations} == {"", f"timeline:{RESEARCH}"}
    full = build_timelines(series, CONCEPTS, {}, CONFIG)["session"]
    primed, plain = _primed_and_plain_discussions(series, full, CONFIG.transition_window_slots)
    for discussion, expected in ((primed, primed_key.text), (plain, key(DISCUSS, CODE).text)):
        now = just_after(discussion)
        rows = seen_until(series, now)
        line = timeline_of(
            rows,
            lane="session",
            slot_minutes=CONFIG.slot_minutes,
            horizon=now.date() + timedelta(days=1),
            conditions=tuple(spec for item in table.relations for spec in item.conditions),
        )
        moment, now_slot = moment_at(now), absolute_slot(now, slot_minutes=CONFIG.slot_minutes)
        drafts = [
            item
            for item in event_drafts(table, line, moment, now_slot=now_slot, statuses=("established",))
            if item.trigger_instance == instance_at(discussion.started_at)
        ]
        assert [item.trigger for item in drafts] == [expected]
        listed = {
            item.relation: item
            for item in antecedents_of(table, CODE, line, moment, now_slot=now_slot, statuses=("established",))
        }
        assert listed[expected].present and listed[expected].source == discussion.uri
        other = next(text for text in listed if text != expected)
        assert not listed[other].present and listed[other].source == ""
    # 装配出来的预期按同一规则：先调研过的那次记在带条件的那本账上
    now = just_after(primed)
    formed = form_expectations(
        moment=moment_at(now), table=table, rows=seen_until(series, now), shadow_unestablished=False, recent=35
    )
    mine = [item for item in formed if item.trigger_instance == instance_at(primed.started_at)]
    assert [item.trigger for item in mine] == [primed_key.text]


def test_conditions_the_day_cannot_judge_fall_back_to_the_plain_relation() -> None:
    """映射器打的情境标签白天没有：那种条件判不了，不当成不成立——这一次只建不带条件的那条。"""

    from habitus.scene.relations import RelationKey

    series = planted(moderated=True)
    tagged = RelationKey("session", DISCUSS, CODE, Segment.CHAIN, "concept:situation-only-the-mapper-knows")
    table = table_of(series, (relation(DISCUSS, CODE), Relation(key=tagged, status=Status.ESTABLISHED, upward=True)))
    assert any(item.conditions and item.conditions[0].calendar_key is None for item in table.relations)
    discussion, _code = discussion_followed_by_code(series)
    now = just_after(discussion)
    line = timeline_of(
        seen_until(series, now),
        lane="session",
        slot_minutes=15,
        horizon=now.date() + timedelta(days=1),
        conditions=tuple(spec for item in table.relations for spec in item.conditions),
    )
    drafts = event_drafts(
        table, line, moment_at(now), now_slot=absolute_slot(now, slot_minutes=15), statuses=("established",)
    )
    assert [item.trigger for item in drafts if item.trigger_instance == instance_at(discussion.started_at)] == [
        key(DISCUSS, CODE).text
    ]
