"""单元的历史每晚按当晚的语义树重标（裁定 55 新版第 2 条）：一条压低的关系长出来那一晚，它在场的那几刻换进新组合，
平常那一组不再被拖低；下一次同样的组合出现，单元读低。v0.6 的周二健身、约游泳那几次就是这个形状。"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from habitus.foresight import Evidence, ForesightError, Seen, Tolerance, climb
from habitus.foresight.history import (
    Moment,
    UnitHistory,
    build_history,
    decode_history,
    encode_history,
    present_inputs,
    record_of,
)
from habitus.foresight.ledger import AccountFact, Antecedent, Expectation
from habitus.foresight.opportunities import OpportunityTable, RelationOpportunities
from habitus.foresight.triggers import timeline_of
from habitus.foresight.units import DOWN, Input, Level, PastMoment, unit_reading

CST = timezone(timedelta(hours=8))
GYM, SWIM = "s-k0002", "s-k0077"
RELATION = "session|s-k0077|s-k0002|next_day"
EVIDENCE = Evidence(alpha=0.05, half_life=8.0)
NIGHT = date(2026, 9, 30)
TUESDAYS = [date(2026, 7, 7) + timedelta(weeks=week) for week in range(12)]
SWIM_DAYS = {TUESDAYS[0], TUESDAYS[3], TUESDAYS[5], TUESDAYS[7], TUESDAYS[9]}


def built(expectations, fact_of, **kwargs):  # type: ignore[no-untyped-def]
    """按旧口径（预期 + 它的预测轴事实）造记录再重标：分量不打折。"""

    records = [record_of(item, fact_of(item)) for item in expectations]
    return build_history("session", NIGHT, records, **kwargs)[0]


def at(day: date, hour: int, minute: int = 0) -> datetime:
    return datetime(day.year, day.month, day.day, hour, minute, tzinfo=CST)


def gym(day: date) -> Expectation:
    """周二 19:30 健身的「离上次够久」预期（当天零点起算、23:59 前算数）。"""

    start = at(day, 0)
    return Expectation(
        expectation_id=f"gym-{day.isoformat()}",
        kind_token=GYM,
        day=day,
        formed_at=start,
        source="日子",
        trigger="weekday=1",
        trigger_instance=day.isoformat(),
        window=(at(day, 19, 0), at(day, 20, 0)),
        deadline=at(day, 23, 59),
        basis=(),
        opportunities=20,
        hits=18,
        observed_from=start,
    )


def table(*, with_relation: bool, status: str = "candidate") -> OpportunityTable:
    relations = (
        (
            RelationOpportunities(
                key=RELATION,
                antecedent=SWIM,
                consequent=GYM,
                antecedent_label="约游泳",
                consequent_label="健身",
                antecedent_classes=(SWIM,),
                consequent_classes=(GYM,),
                segment="next_day",
                status=status,
                upward=False,
                items=(),
            ),
        )
        if with_relation
        else ()
    )
    return OpportunityTable("session", NIGHT, NIGHT, 15, 3, relations, ())


def line():  # type: ignore[no-untyped-def]
    """前一天晚上说了"和老王约好明天游泳"的那几个周一。"""

    rows = [
        Seen(at(day - timedelta(days=1), 21, 0), at(day - timedelta(days=1), 21, 5), SWIM, f"behavior://swim-{day}.md")
        for day in sorted(SWIM_DAYS)
    ]
    return timeline_of(rows, lane="session", slot_minutes=15, horizon=NIGHT)


def answers() -> dict[str, AccountFact]:
    """约了游泳的那几个周二没去健身（看得见的落空），其余周二准时去了。"""

    found = {}
    for day in TUESDAYS:
        item = gym(day)
        missed = day in SWIM_DAYS
        found[item.expectation_id] = AccountFact(
            item.account, item.formed_at, not missed, 1.0, "过期待决" if missed else "准时", item.expectation_id
        )
    return found


def test_a_lowering_relation_that_has_grown_moves_its_moments_into_a_new_combination() -> None:
    expectations = [gym(day) for day in TUESDAYS]
    facts = answers()
    before = built(
        expectations,
        lambda item: facts.get(item.expectation_id),
        table=table(with_relation=False),
        line=line(),
        statuses=("candidate",),
    )
    assert {moment.signature for moment in before.of(GYM)} == {
        frozenset({expectations[0].account})
    }  # 关系还没长出来：都是「离上次够久」
    after = built(
        expectations,
        lambda item: facts.get(item.expectation_id),
        table=table(with_relation=True),
        line=line(),
        statuses=("candidate",),
    )
    suppressed = frozenset({expectations[0].account, f"{DOWN}{RELATION}"})
    by_signature: dict[frozenset[str], list[bool]] = {}
    for moment in after.of(GYM):
        by_signature.setdefault(moment.signature, []).append(moment.came)
    assert by_signature[suppressed] == [False] * len(SWIM_DAYS)  # 约了游泳的那几刻：换进新组合，全没来
    assert all(by_signature[frozenset({expectations[0].account})])  # 平常那一组不再被拖低
    # 关系状态不在认的范围里（只认成立的、它还是候选）：不重标
    unrecognised = built(
        expectations,
        lambda item: facts.get(item.expectation_id),
        table=table(with_relation=True),
        line=line(),
        statuses=("established",),
    )
    assert unrecognised == before


def test_after_relabelling_the_unit_reads_low_with_the_suppressor_and_normal_without_it() -> None:
    expectations = [gym(day) for day in TUESDAYS]
    facts = answers()
    history = built(
        expectations,
        lambda item: facts.get(item.expectation_id),
        table=table(with_relation=True),
        line=line(),
        statuses=("candidate",),
    )
    now = at(NIGHT + timedelta(days=7), 18, 0)
    today = gym(NIGHT + timedelta(days=7))
    standing = climb(history.of(GYM), today.account, now=now, tolerance=Tolerance(line=2 / 3), evidence=EVIDENCE)
    plain = Input(today, standing)
    swim_again = Input(
        Expectation(**{**today.__dict__, "antecedents": (Antecedent(RELATION, True, False, "behavior://swim.md"),)}),
        standing,
    )
    low = unit_reading((swim_again,), history.of(GYM), now=now, evidence=EVIDENCE, prior_share=1 / 3)
    normal = unit_reading((plain,), history.of(GYM), now=now, evidence=EVIDENCE, prior_share=1 / 3)
    assert low.level is Level.SAME_SET and normal.level is Level.SAME_SET
    assert low.reading.upper < 2 / 3  # 又约了游泳：不预测
    # 没重标时（关系还不在）平常那一组被约游泳那几次拖着；重标之后它只剩自己的那几刻（都来了）
    unrelabelled = built(
        expectations,
        lambda item: facts.get(item.expectation_id),
        table=table(with_relation=False),
        line=line(),
        statuses=("candidate",),
    )
    mixed = unit_reading((plain,), unrelabelled.of(GYM), now=now, evidence=EVIDENCE, prior_share=1 / 3)
    assert (
        normal.reading.mean > 0.85 and mixed.reading.mean < 0.65 and normal.reading.lower > mixed.reading.lower + 0.25
    )


def test_present_inputs_needs_a_moment_and_the_history_round_trips() -> None:
    with pytest.raises(ForesightError):
        present_inputs((), table=table(with_relation=True), line=line(), statuses=("candidate",))
    history = UnitHistory(
        "session", NIGHT, {GYM: (Moment(PastMoment(at(NIGHT, 0), frozenset({"a", f"{DOWN}b"}), False, 0.4), ("x",)),)}
    )
    assert decode_history(encode_history(history), lane="session", night=NIGHT) == history
    with pytest.raises(ForesightError, match="malformed"):
        decode_history(encode_history(history), lane="session", night=NIGHT + timedelta(days=1))
    with pytest.raises(ForesightError, match="malformed"):
        decode_history("{}", lane="session", night=NIGHT)


def test_a_moment_carries_the_strength_of_the_fact_that_decided_it() -> None:
    """一刻的分量与账同一把尺：窗外很远才做的那次、后补的那次只算打折的那一份（2026-10-17 四方评审：原来一律记 1）。"""

    item = gym(TUESDAYS[0])
    late = AccountFact(item.account, item.formed_at, True, 0.25, "晚了但做了", item.expectation_id)
    (moment,) = built([item], lambda _item: late, table=table(with_relation=False), line=line(), statuses=()).of(GYM)
    assert moment.came and moment.weight == pytest.approx(0.25)
    caught = AccountFact(item.account, item.formed_at, False, 0.4, "后补", item.expectation_id)
    (missed,) = built([item], lambda _item: caught, table=table(with_relation=False), line=line(), statuses=()).of(GYM)
    assert not missed.came and missed.weight == pytest.approx(0.4)


def test_a_calendar_situation_that_has_grown_moves_every_past_day_it_held_into_its_input() -> None:
    """新关系"周二 → 健身"长出来那一晚，过去的周二都换进这个输入（看那天日历条件成不成立，时间线上语义树按天填的那一份）；
    日历答不出的那天不算在场。"""

    key = "session|周二|s-k0002|situation"
    situation = RelationOpportunities(
        key=key,
        antecedent="周二",
        consequent=GYM,
        antecedent_label="周二",
        consequent_label="健身",
        antecedent_classes=("周二",),
        consequent_classes=(GYM,),
        segment="situation",
        status="established",
        upward=True,
        items=(),
        calendar_key="日历.周几",
        calendar_values=("周二",),
    )
    calendar_line = line()
    calendar_line.calendar_conditions.update(
        {day: {"日历.周几": "周二"} for day in TUESDAYS[1:]}
    )  # 第一个周二日历答不出
    expectations = [gym(day) for day in TUESDAYS]
    facts = answers()
    relabelled = built(
        expectations,
        lambda item: facts.get(item.expectation_id),
        table=OpportunityTable("session", NIGHT, NIGHT, 15, 3, (situation,), ()),
        line=calendar_line,
        statuses=("established",),
    )
    with_input = frozenset({expectations[0].account, f"{GYM}\n情境\n{key}"})
    signatures = [moment.signature for moment in relabelled.of(GYM)]
    assert signatures[0] == frozenset({expectations[0].account}) and set(signatures[1:]) == {with_input}


def test_a_moment_retires_only_when_every_layer_it_belongs_to_has_moved_past_it() -> None:
    """26 号清单 4.9：同样多那一层里排在后面的已有 M 刻还不够，同一组里也要有 M 刻——少见的组合（取消出差在场的那几刻）
    不会因为平常那一组天天来就被挤掉；诊断够得着的那一段一律不删。"""

    from habitus.foresight.history import Moment, retired
    from habitus.foresight.units import PastMoment

    start = at(NIGHT - timedelta(days=400), 9, 0)
    plain, rare = frozenset({"a"}), frozenset({"b"})
    moments = [Moment(PastMoment(start + timedelta(days=index), rare, True), (f"rare{index}",)) for index in range(3)]
    moments += [
        Moment(PastMoment(start + timedelta(days=10 + index), plain, True), (f"plain{index}",)) for index in range(10)
    ]
    gone = retired(moments, window=4, before=start + timedelta(days=12))
    assert gone == {"plain0", "plain1"}  # 平常那一组前两刻之后各有 4 刻以上；少见的那 3 刻同一组里后面不到 4 刻，留着
    assert retired(moments, window=4, before=start) == frozenset()  # 都在诊断够得着的那一段里


def test_relabelling_credits_each_moment_to_the_most_specific_established_combination_and_skips_coincidences() -> None:
    """带条件的关系成立那一晚，历史上条件成立的各刻归到组合边、不带条件那条只数条件不成立的几刻（裁定 61 第 6 条）；
    语义树判成巧合（效应不够大）的输入不进签名。"""

    from habitus.foresight.opportunities import ConditionSpec, TreeReading

    plain = RelationOpportunities(
        key="session|s-k0077|s-k0002|next_day",
        antecedent=SWIM,
        consequent=GYM,
        antecedent_label="约游泳",
        consequent_label="健身",
        antecedent_classes=(SWIM,),
        consequent_classes=(GYM,),
        segment="next_day",
        status="established",
        upward=True,
        items=(),
    )
    coach = "s-k0099"
    conditioned = RelationOpportunities(
        key=f"session|s-k0077|s-k0002|next_day|timeline:{coach}",
        antecedent=SWIM,
        consequent=GYM,
        antecedent_label="约游泳",
        consequent_label="健身",
        antecedent_classes=(SWIM,),
        consequent_classes=(GYM,),
        segment="next_day",
        status="established",
        upward=True,
        items=(),
        conditions=(ConditionSpec(f"timeline:{coach}", classes=(coach,)),),
    )
    expectations = [gym(day) for day in TUESDAYS]
    facts = answers()
    # 约游泳之前半小时先约了教练的那几个周一：条件成立
    coached = sorted(SWIM_DAYS)[:2]
    rows = [
        Seen(at(day - timedelta(days=1), 21, 0), at(day - timedelta(days=1), 21, 5), SWIM, f"behavior://swim-{day}.md")
        for day in sorted(SWIM_DAYS)
    ] + [
        Seen(
            at(day - timedelta(days=1), 20, 30),
            at(day - timedelta(days=1), 20, 40),
            coach,
            f"behavior://coach-{day}.md",
        )
        for day in coached
    ]
    line_with_coach = timeline_of(rows, lane="session", slot_minutes=15, horizon=NIGHT)
    table_both = OpportunityTable("session", NIGHT, NIGHT, 15, 3, (plain, conditioned), ())
    history = built(
        expectations,
        lambda item: facts.get(item.expectation_id),
        table=table_both,
        line=line_with_coach,
        statuses=("established",),
    )
    base_input = f"{GYM}\n事件\n{plain.key}"
    combo_input = f"{GYM}\n事件\n{conditioned.key}"
    by_day = {moment.at.date(): moment.signature for moment in history.of(GYM)}
    for day in TUESDAYS:
        if day in coached:
            assert combo_input in by_day[day] and base_input not in by_day[day]
        elif day in SWIM_DAYS:
            assert base_input in by_day[day] and combo_input not in by_day[day]
        else:
            assert base_input not in by_day[day] and combo_input not in by_day[day]
    # 巧合：语义树判效应不够大的输入不进签名（权重照数，在边上）
    coincidental = OpportunityTable(
        "session",
        NIGHT,
        NIGHT,
        15,
        3,
        (RelationOpportunities(**{**plain.__dict__, "reading": TreeReading(0.9, 0.85, large_enough=False)}),),
        (),
    )
    quiet = built(
        expectations,
        lambda item: facts.get(item.expectation_id),
        table=coincidental,
        line=line(),
        statuses=("established",),
    )
    assert all(base_input not in moment.signature for moment in quiet.of(GYM))
