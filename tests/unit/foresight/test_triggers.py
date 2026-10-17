"""触发与候选（第 3 步）：事件 / 钟面 / 日子三种触发在埋了关系的合成数据上各出什么，反向前因清单标在场与不在场，装配成预期。
每个机会都建（裁定 49）：这回已经做了的照建、不退场，时窗保留历史那几次的实际时刻、不挪到此刻。"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, time, timedelta, timezone

import pytest

from habitus.foresight import ForesightError, Moment, Seen, form_expectations
from habitus.foresight.ledger import expectation_identity
from habitus.foresight.opportunities import (
    KindDay,
    KindDays,
    Opportunity,
    OpportunityTable,
    RelationOpportunities,
    build_table,
)
from habitus.foresight.opportunities.codec import decode_table, encode_table
from habitus.foresight.triggers import antecedents_of, clock_drafts, day_drafts, event_drafts, timeline_of
from habitus.foresight.triggers.draft import instance_at
from habitus.foresight.triggers.marks import absolute_slot
from habitus.prediction.model import DayCurve, NodeCounts, NodeStatistics, PredictionTree, SlotKey
from habitus.scene.relations import Relation, Segment, Status, build_timelines
from habitus.series import EventSeries, SeriesRecord
from tests.unit.scene.relation_fixtures import CODE, CONCEPTS, CONFIG, DISCUSS, DOCS, RESEARCH, key, planted, record

CST = timezone(timedelta(hours=8))
SLOTS = 96


def relation(
    antecedent: str, consequent: str, segment: Segment = Segment.CHAIN, *, status: Status = Status.ESTABLISHED
) -> Relation:
    return Relation(key=key(antecedent, consequent, segment), status=status, upward=True)


def table_of(series: EventSeries, relations: tuple[Relation, ...]) -> OpportunityTable:
    line = build_timelines(series, CONCEPTS, {}, CONFIG)["session"]
    return build_table(
        line,
        relations,
        CONCEPTS,
        night=series.cutoff,
        relation_night=series.cutoff,
        slot_minutes=CONFIG.slot_minutes,
        window_slots=CONFIG.transition_window_slots,
    )


def seen_until(series: EventSeries, now: datetime) -> tuple[Seen, ...]:
    return tuple(
        Seen(item.started_at, min(item.last_observed_at, now), item.kind_token, item.uri)
        for item in series.records
        if item.started_at <= now and (now.date() - item.day).days <= 7
    )


def moment_at(now: datetime) -> Moment:
    return Moment(at=now, slot=(now.hour * 60 + now.minute) // CONFIG.slot_minutes, weekday=now.weekday())


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


def tree_with(
    curves: dict[tuple[int, str], DayCurve], nodes: dict[tuple[SlotKey, str], NodeStatistics]
) -> PredictionTree:
    return PredictionTree(
        built_at=datetime(2026, 7, 25, 3, 0, tzinfo=CST),
        reference_day=date(2026, 7, 24),
        config_digest="d" * 64,
        slot_minutes=CONFIG.slot_minutes,
        nodes=nodes,
        curves=curves,
        weekday_baselines={},
        edges={},
        parallels={},
        parallel_totals={},
        recurrences={},
        exposure={},
        baselines={},
        actions=(),
        observed_days=40,
        censored_transitions=0.0,
    )


def peak(slots: dict[int, float]) -> DayCurve:
    values = tuple(slots.get(slot, 0.0) for slot in range(SLOTS))
    return DayCurve(
        marginal=values, hazard=values, cumulative=tuple(0.0 for _ in range(SLOTS)), trend=None, trend_n_eff=0.0
    )


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
    table = OpportunityTable("session", day, day, 15, 3, (ended.date(),), (stored,), ())
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
    single = OpportunityTable(
        "session", day, day, 15, 3, (ended.date(),), (replace(stored, items=stored.items[:1]),), ()
    )
    now = datetime.combine(day, time(10, 16), tzinfo=CST)
    line = timeline_of((discussion,), lane="session", slot_minutes=15, horizon=day + timedelta(days=1))
    assert (
        event_drafts(
            single, line, moment_at(now), now_slot=absolute_slot(now, slot_minutes=15), statuses=("established",)
        )
        == ()
    )


# ── 钟面 ────────────────────────────────────────────────────────────────────


def test_the_clock_trigger_takes_its_days_from_the_tree_and_the_window_from_their_actual_times() -> None:
    series = planted()
    table = table_of(series, ())
    tuesdays = tuple(day for day in table.present_days if day.weekday() == 1)[:3]
    tree = tree_with(
        {(1, RESEARCH): peak({36: 0.9})},
        {(SlotKey(1, 36), RESEARCH): NodeStatistics(n_eff=3.0, counts=NodeCounts(occurred_days=3.0), days=tuesdays)},
    )
    today = date(2026, 7, 21)  # 周二
    early = datetime.combine(today, time(8, 50), tzinfo=CST)
    assert clock_drafts(tree, table, moment_at(early), done_today=frozenset()) == ()  # 还没到钟点
    now = datetime.combine(today, time(9, 5), tzinfo=CST)
    (draft,) = clock_drafts(tree, table, moment_at(now), done_today=frozenset())
    assert (
        draft.kind_token == RESEARCH
        and draft.source == "钟面"
        and draft.trigger == "weekday=1"
        and draft.trigger_instance == "2026-07-21"
    )
    assert draft.window == (datetime.combine(today, time(9, 0), tzinfo=CST), datetime.combine(today, time(9, 15), tzinfo=CST))
    assert draft.deadline == datetime.combine(today + timedelta(days=1), time(0), tzinfo=CST)
    assert draft.observed_from == datetime.combine(today, time(0), tzinfo=CST)
    assert draft.hits == 3 and draft.opportunities == sum(
        1 for day in table.present_days if day.weekday() == 1 and day < today
    )
    calendar = table.kind(RESEARCH)
    assert calendar is not None and draft.basis == tuple(
        item.first_uri for item in calendar.days if item.day in tuesdays
    )
    # 今天已做的照建（裁定 49）；过了窗还照建（那是一次命中），没做的过了窗才退场
    (done,) = clock_drafts(tree, table, moment_at(now), done_today=frozenset({RESEARCH}))
    assert done.trigger_instance == draft.trigger_instance
    noon = datetime.combine(today, time(12, 0), tzinfo=CST)
    assert clock_drafts(tree, table, moment_at(noon), done_today=frozenset()) == ()
    (kept,) = clock_drafts(tree, table, moment_at(noon), done_today=frozenset({RESEARCH}))
    assert kept.deadline >= noon
    # 曲线是收缩借来的、这个周几其实没起过头：对不上历史，不建
    borrowed = tree_with({(1, RESEARCH): peak({36: 0.9})}, {})
    assert clock_drafts(borrowed, table, moment_at(now), done_today=frozenset()) == ()
    # 出处日只有两天：样本少就没有读数，不建；没有机会表就没有实际时刻，也不建
    thin = tree_with(
        {(1, RESEARCH): peak({36: 0.9})},
        {
            (SlotKey(1, 36), RESEARCH): NodeStatistics(
                n_eff=2.0, counts=NodeCounts(occurred_days=2.0), days=tuesdays[:2]
            )
        },
    )
    assert clock_drafts(thin, table, moment_at(now), done_today=frozenset()) == ()
    assert clock_drafts(tree, None, moment_at(now), done_today=frozenset()) == ()


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
    drafts = day_drafts(table, moment_at(now), done_today=frozenset())
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
        for item in day_drafts(table, moment_at(evening), done_today=frozenset())
        if item.kind_token == RESEARCH and item.trigger == "interval"
    ]
    # 周一（逝去 3 天）：在险的只剩过周末的那几次，命中全是它们——另一簇、另一条实例
    monday = datetime.combine(date(2026, 7, 27), time(8, 0), tzinfo=CST)
    (later,) = [
        item
        for item in day_drafts(table, moment_at(monday), done_today=frozenset())
        if item.kind_token == RESEARCH and item.trigger == "interval"
    ]
    assert later.trigger_instance == f"{last.first_uri}@2026-07-27"
    assert later.opportunities == later.hits == gaps.count(3) and later.window[0] == monday.replace(hour=9)
    # 逝去 2 天、5 天：历史上没有恰好隔这么久做过的，不是它的日子；今天已做的照建（裁定 49）
    for day in (date(2026, 7, 26), date(2026, 7, 29)):
        moment = moment_at(datetime.combine(day, time(8, 0), tzinfo=CST))
        assert not [
            item
            for item in day_drafts(table, moment, done_today=frozenset())
            if item.kind_token == RESEARCH and item.trigger == "interval"
        ]
    (done,) = [
        item
        for item in day_drafts(table, moment_at(now), done_today=frozenset({RESEARCH}))
        if item.kind_token == RESEARCH and item.trigger == "interval"
    ]
    assert done.trigger_instance == interval.trigger_instance
    assert interval.observed_from == datetime.combine(today, time(0), tzinfo=CST)
    # 做了的过了窗也照建；没做的过了窗退场（上面晚上那一拍）
    assert [
        item
        for item in day_drafts(table, moment_at(evening), done_today=frozenset({RESEARCH}))
        if item.kind_token == RESEARCH and item.trigger == "interval"
    ]


def test_consecutive_gaps_are_one_cluster_whichever_day_it_is_recomputed_on() -> None:
    """间隔 2、3、4 天连成一簇：第 2 天建的那条与第 3 天重算的是同一个身份（不然第 3 天那次发生被同一本账结成两次命中）。"""

    start = date(2026, 6, 1)
    offsets = [0]
    for gap in (2, 3, 4, 2, 3, 4, 3):
        offsets.append(offsets[-1] + gap)
    days = tuple(start + timedelta(days=offset) for offset in offsets)
    kind = KindDays(
        "s-k0077",
        "买菜",
        tuple(KindDay(day, f"veg-{day:%m%d}", datetime.combine(day, time(18, 0), tzinfo=CST), 1, 1) for day in days),
    )
    present = tuple(start + timedelta(days=offset) for offset in range(offsets[-1] + 10))
    table = OpportunityTable("session", days[-1], days[-1], 15, 3, present, (), (kind,))
    instances = []
    for elapsed in (2, 3, 4):
        moment = moment_at(datetime.combine(days[-1] + timedelta(days=elapsed), time(8, 0), tzinfo=CST))
        (draft,) = [item for item in day_drafts(table, moment, done_today=frozenset()) if item.trigger == "interval"]
        instances.append(draft.trigger_instance)
    assert instances == [f"veg-{days[-1]:%m%d}@{days[-1] + timedelta(days=2)}"] * 3


def test_the_day_of_month_trigger_needs_two_months_and_keeps_one_account_per_class() -> None:
    short = table_of(planted(), ())
    today = short.night  # 2026-07-25
    now = datetime.combine(today, time(8, 0), tzinfo=CST)
    research = short.kind(RESEARCH)
    assert research is not None and {d.day.month for d in research.days if d.day.day == 25} == {6}
    # 40 个工作日里 25 号只做过一个月：样本少就没有读数，不建
    assert not [
        item for item in day_drafts(short, moment_at(now), done_today=frozenset()) if item.trigger != "interval"
    ]
    # 天天做的类 10 号也做，但 9 号 11 号也做：10 号对它不是日子，两个月也不建
    longer = table_of(planted(days=60), ())  # 6 月初到 8 月下旬：6 月 10 日与 7 月 10 日都是工作日
    today = date(2026, 8, 10)
    now = datetime.combine(today, time(8, 0), tzinfo=CST)
    assert not [
        item for item in day_drafts(longer, moment_at(now), done_today=frozenset()) if item.trigger == "day_of_month"
    ]
    # 手搭一个每月 10 号还信用卡的类：三个月各一次、都在 09:00 前后
    present = tuple(date(2026, 5, 1) + timedelta(days=offset) for offset in range(102))  # 5-01 … 8-10
    card = KindDays(
        "s-k0099",
        "还信用卡",
        tuple(
            KindDay(day, f"card-{day:%m}", datetime.combine(day, time(9, minute), tzinfo=CST), 1, 1)
            for day, minute in ((date(2026, 5, 10), 0), (date(2026, 6, 10), 20), (date(2026, 7, 10), 5))
        ),
    )
    table = OpportunityTable("session", today, today, 15, 3, present, (), (card,))
    (dom,) = [
        item for item in day_drafts(table, moment_at(now), done_today=frozenset()) if item.trigger == "day_of_month"
    ]
    assert dom.kind_token == "s-k0099" and dom.trigger_instance == "2026-08-10"
    assert dom.hits == 3 and dom.opportunities == 3 and dom.basis == ("card-05", "card-06", "card-07")
    assert dom.window == (
        datetime.combine(today, time(9, 0), tzinfo=CST),
        datetime.combine(today, time(9, 20), tzinfo=CST),
    )
    assert dom.deadline == datetime.combine(date(2026, 8, 11), time(0), tzinfo=CST)
    # 10 号前后一天也做过的那个月不算（那是一对、归离上次够久管）：剩一个月，不建
    paired = KindDays(
        "s-k0099",
        "还信用卡",
        card.days
        + (KindDay(date(2026, 7, 11), "card-07b", datetime.combine(date(2026, 7, 11), time(9, 0), tzinfo=CST), 1, 1),),
    )
    both = OpportunityTable("session", today, today, 15, 3, present, (), (paired,))
    (less,) = [
        item for item in day_drafts(both, moment_at(now), done_today=frozenset()) if item.trigger == "day_of_month"
    ]
    assert less.hits == 2 and less.basis == ("card-05", "card-06")
    # 只在一个月做过：不建；过了那几次的时刻：退场
    thin = OpportunityTable(
        "session", today, today, 15, 3, present, (), (KindDays("s-k0099", "还信用卡", card.days[:1]),)
    )
    assert day_drafts(thin, moment_at(now), done_today=frozenset()) == ()
    evening = datetime.combine(today, time(20, 0), tzinfo=CST)
    assert not [
        item for item in day_drafts(table, moment_at(evening), done_today=frozenset()) if item.trigger == "day_of_month"
    ]


# ── 装配 ────────────────────────────────────────────────────────────────────


def test_assembly_gives_each_draft_an_identity_without_the_night_and_attaches_the_antecedents() -> None:
    series = planted()
    table = table_of(series, (relation(DISCUSS, CODE),))
    discussion, _code = discussion_followed_by_code(series)
    now = just_after(discussion)
    rows = seen_until(series, now)
    found = form_expectations(moment=moment_at(now), tree=None, table=table, rows=rows, generation="g1", version="3")
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
        item.observed_from == datetime.combine(now.date(), time(0), tzinfo=CST) for item in found if item.source == "日子"
    )
    done_today = {row.kind_token for row in rows if row.day == now.date()}
    assert any(item.kind_token in done_today for item in found if item.source == "日子")
    assert {item.expectation_id for item in found} == {
        expectation_identity(item.kind_token, item.source, item.trigger, item.trigger_instance) for item in found
    }


def test_the_tree_and_the_table_must_share_one_slot_width() -> None:
    series = planted()
    table = table_of(series, ())
    now = datetime.combine(series.cutoff, time(8, 0), tzinfo=CST)
    wrong = tree_with({}, {})
    object.__setattr__(wrong, "slot_minutes", 30)
    with pytest.raises(ForesightError, match="slot width"):
        form_expectations(moment=moment_at(now), tree=wrong, table=table, rows=())
    with pytest.raises(ForesightError, match="moment.slot"):
        form_expectations(
            moment=Moment(at=now, slot=0, weekday=now.weekday()), tree=tree_with({}, {}), table=table, rows=()
        )
    assert form_expectations(moment=moment_at(now), tree=None, table=None, rows=()) == ()


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
        Moment(at=now, slot=84, weekday=3, looked_since=now + timedelta(minutes=1))


def test_a_day_done_only_after_a_reminder_is_unknown_to_the_day_and_clock_triggers() -> None:
    """裁定 54：被提醒了才做的那天，不提醒做不做看不到——分子分母都不计；时窗与出处取当天第一次自然发生的，提醒的时刻
    不许漏回"他往常几点做"。只扣分子的写法会把它读成一次没做（还信用卡 3 个月里 7 月是提醒后才还的：2/3 而不是 2/2）。"""

    def kd(day: date, natural: int, hour: int = 9) -> KindDay:
        return KindDay(day, f"card-{day:%m%d}", datetime.combine(day, time(hour, 0), tzinfo=CST), 1, natural)

    today = date(2026, 8, 10)
    now = datetime.combine(today, time(8, 0), tzinfo=CST)
    present = tuple(date(2026, 5, 1) + timedelta(days=offset) for offset in range(102))
    card = KindDays(
        "s-k0099",
        "还信用卡",
        (kd(date(2026, 5, 10), 1), kd(date(2026, 6, 10), 1), kd(date(2026, 7, 10), 0)),
    )
    table = OpportunityTable("session", today, today, 15, 3, present, (), (card,))
    (dom,) = [item for item in day_drafts(table, moment_at(now), done_today=frozenset()) if item.trigger == "day_of_month"]
    assert (dom.hits, dom.opportunities) == (2, 2)
    # 离上次够久：每 7 天买一次菜，第三次是提醒了才买的——那一次落在这一簇里，不是机会
    start = date(2026, 6, 1)
    days = [start + timedelta(days=7 * index) for index in range(5)]
    veg = KindDays("s-k0077", "买菜", tuple(kd(day, 0 if index == 2 else 1, 18) for index, day in enumerate(days)))
    weekly = OpportunityTable(
        "session", days[-1], days[-1], 15, 3, tuple(start + timedelta(days=o) for o in range(60)), (), (veg,)
    )
    later = moment_at(datetime.combine(days[-1] + timedelta(days=7), time(8, 0), tzinfo=CST))
    (interval,) = [item for item in day_drafts(weekly, later, done_today=frozenset()) if item.trigger == "interval"]
    assert (interval.hits, interval.opportunities) == (3, 3)
    # 时窗与出处：7 月 10 号 07:00 被提醒改了一次代码、09:30 又自己改了一次；另两个月都是 09:30
    records = []
    for day in (date(2026, 5, 10), date(2026, 6, 10), date(2026, 7, 10)):
        records += [record(day, 9, 30, "修改代码", number=2), record(day, 9, 0, "调研"), record(day, 20, 0, "写文档")]
    records.append(replace(record(date(2026, 7, 10), 7, 0, "修改代码", number=1), reminded=True))
    for offset in range(102):
        day = date(2026, 5, 1) + timedelta(days=offset)
        if day.day != 10:
            records.append(record(day, 12, 0, "调研", number=7))
    series = EventSeries(cutoff=today, records=tuple(sorted(records, key=lambda item: (item.started_at, item.uri))))
    built = table_of(series, ())
    july = next(item for item in built.kind(CODE).days if item.day == date(2026, 7, 10))  # type: ignore[union-attr]
    assert (july.count, july.natural) == (2, 1)
    assert july.first_uri.endswith("修改代码1--0700.md") and july.natural_uri.endswith("修改代码2--0930.md")  # type: ignore[union-attr]
    early = datetime.combine(today, time(6, 0), tzinfo=july.first_at.tzinfo)
    (code,) = [
        item
        for item in day_drafts(built, moment_at(early), done_today=frozenset())
        if item.kind_token == CODE and item.trigger == "day_of_month"
    ]
    assert code.window[0].time() == time(9, 30)
    assert all(uri.endswith("修改代码2--0930.md") for uri in code.basis)
    # 钟面：周一 4 次自然、1 次只有提醒后才做——那个周一不在分母里
    mondays = [date(2026, 6, 1) + timedelta(days=7 * index) for index in range(6)]
    calendar = KindDays(RESEARCH, "调研", tuple(kd(day, 0 if index == 4 else 1, 10) for index, day in enumerate(mondays[:5])))
    clock_table = OpportunityTable(
        "session", mondays[4], mondays[4], 15, 3, tuple(mondays[:5]), (), (calendar,)
    )
    node = NodeStatistics(n_eff=4.0, counts=NodeCounts(occurred_days=4.0), days=tuple(mondays[:4]))
    tree = tree_with({}, {(SlotKey(weekday=0, slot=40), RESEARCH): node})
    at_ten = datetime.combine(mondays[5], time(10, 5), tzinfo=CST)
    (clock,) = clock_drafts(tree, clock_table, moment_at(at_ten), done_today=frozenset())
    assert (clock.hits, clock.opportunities) == (4, 4)


def test_a_kind_day_names_its_first_natural_occurrence_and_round_trips() -> None:
    at = datetime(2026, 7, 10, 7, 0, tzinfo=CST)
    later = at.replace(hour=9, minute=30)
    assert KindDay(at.date(), "a", at, 1, 1).natural_at == at  # 全是自然的：就是第一次那条
    with pytest.raises(ForesightError, match="first natural"):
        KindDay(at.date(), "a", at, 2, 1)  # 有被提醒的又不说哪条是第一次自然的
    with pytest.raises(ForesightError, match="no natural"):
        KindDay(at.date(), "a", at, 1, 0, "a", at)
    mixed = KindDay(at.date(), "a", at, 2, 1, "b", later)
    kinds = (KindDays("s-k0001", "类", (mixed, KindDay(date(2026, 7, 11), "c", at + timedelta(days=1), 1, 0))),)
    table = OpportunityTable("session", date(2026, 7, 11), date(2026, 7, 11), 15, 3, (), (), kinds)
    assert decode_table(encode_table(table)) == table
