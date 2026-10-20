"""机会表（N1）：在语义树同一条时间线上，对每条成立关系列出历史每次前因之后后果来没来；日历情境关系按天列机会；
每个类的复现（语义树登记的）；编解码往返。"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from habitus.conditions import HolidayTable, HolidayYear, LocalCalendar
from habitus.foresight import ForesightError
from habitus.foresight.opportunities import (
    DayWatch,
    KindDays,
    Opportunity,
    OpportunityTable,
    RelationOpportunities,
    build_table,
    decode_table,
    encode_table,
)
from habitus.scene.concepts import ConceptSet
from habitus.scene.concepts.calendar_situations import calendar_concepts, weekday_set
from habitus.scene.concepts.rhythm import RhythmRulers
from habitus.scene.recurrence import coverage_by_lane, recurrences_of
from habitus.scene.relations import Relation, RelationKey, Segment, Status, build_timelines, chains_of, merge_for
from habitus.temporal import DayInterval, OccurrenceDay
from tests.unit.kind_ids import kind_id
from tests.unit.scene.concept_fixtures import CREATED, base, group
from tests.unit.scene.relation_fixtures import CODE, CONCEPTS, CONFIG, DISCUSS, DOCS, RESEARCH, TITLES, key, planted

CST = timezone(timedelta(hours=8))
NIGHT = date(2026, 7, 27)
RULERS = RhythmRulers(15, 2700, 3, 5.0, 5.0, 0.10, 35)


def established(
    antecedent: str, consequent: str, segment: Segment = Segment.CHAIN, *, status: Status = Status.ESTABLISHED
) -> Relation:
    return Relation(key=key(antecedent, consequent, segment), status=status, upward=True)


def table(
    relations: tuple[Relation, ...], concepts: ConceptSet = CONCEPTS, series=None, calendar=None
) -> OpportunityTable:  # type: ignore[no-untyped-def]
    series = series if series is not None else planted()
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
        coverage=coverage_by_lane(series, RULERS)["session"],
        calendar=calendar,
    )


def test_the_table_copies_how_much_of_each_day_was_watched_and_its_day_type() -> None:
    """账的久别要的两样（26 号清单 3.3）：每天看着的比例照抄 lane 的观测覆盖，日型照抄时间线上的日历条件。"""

    series = planted()
    coverage = coverage_by_lane(series, RULERS)["session"]
    year = series.cutoff.year
    holidays = HolidayTable({year: HolidayYear("CN", year, "test", "1")})
    found = table((established(DISCUSS, CODE),), series=series, calendar=LocalCalendar(holidays))
    assert [item.day for item in found.watch] == list(coverage.days)
    assert all(found.watched(item.day) == coverage.day_share(item.day) for item in found.watch)
    # 每一天都有日型，没记录的日子也有（时间线上的日历条件只有有记录的日子）
    assert all(found.day_type(day) == holidays.day_type(day).value for day in coverage.days)  # type: ignore[union-attr]
    assert {item.day_type for item in found.watch} == {"工作日", "周末"}
    assert (
        found.watched(series.cutoff + timedelta(days=30)) == 0.0
        and found.day_type(series.cutoff + timedelta(days=30)) is None
    )
    assert table((established(DISCUSS, CODE),), series=series).day_type(found.watch[0].day) is None  # 没接日历


def test_every_antecedent_chain_is_an_opportunity_and_hits_name_the_consequent_record() -> None:
    """埋的关系「讨论方案 → 修改代码（同一条链）」：机会 = 讨论方案合并后的每条链；来了的那几次指向那条修改代码、时刻在做完之后。"""

    series = planted()
    line = build_timelines(series, CONCEPTS, {}, CONFIG)["session"]
    found = table((established(DISCUSS, CODE),), series=series)
    (relation,) = found.relations
    assert relation.key == key(DISCUSS, CODE).text and relation.segment == "chain" and relation.status == "established"
    assert relation.antecedent_classes == (DISCUSS,) and relation.consequent_classes == (CODE,)
    assert relation.antecedent_label == "讨论方案" and relation.consequent_label == "修改代码"
    chains = merge_for(Segment.CHAIN, chains_of(line, DISCUSS, CONFIG.transition_window_slots))
    assert len(relation.items) == len(chains) > 30
    assert [item.antecedent_uri for item in relation.items] == [chain.anchor.uri for chain in chains]
    assert relation.hits / relation.opportunities > 0.85  # 造数时 0.9 的概率接修改代码
    for item in relation.items:
        assert item.antecedent_uri in item.antecedent_uris
        if item.came:
            assert item.consequent_uri is not None and "修改代码" in item.consequent_uri
            assert item.consequent_at is not None and item.consequent_at >= item.ended_at
            assert item.consequent_at - item.ended_at <= timedelta(
                minutes=CONFIG.slot_minutes * (CONFIG.transition_window_slots + 1)
            )
        else:
            assert item.consequent_uri is None and item.consequent_at is None
    assert found.by_consequent(CODE) == (relation,) and found.relation(relation.key) is relation
    assert found.by_consequent(DISCUSS) == ()


def test_each_segment_has_its_own_window() -> None:
    """同一对概念四段各一张清单：次日段的来了指向次日那条，第 2–7 天段指向 2–7 天内最早那条；伸过截止日的结果未知。"""

    found = table(
        tuple(
            established(DOCS, RESEARCH, segment)
            for segment in (Segment.REST_OF_DAY, Segment.NEXT_DAY, Segment.DAYS_2_7)
        )
    )
    by_segment = {item.segment: item for item in found.relations}
    assert set(by_segment) == {"rest_of_day", "next_day", "days_2_7"}
    # 17:50 写文档是收工前最后一件：当天剩下的在场槽几乎没有 → 当天剩下的全是未知
    rest = by_segment["rest_of_day"]
    assert rest.opportunities == 0 and rest.unknown == len(rest.items) > 0
    for item in by_segment["next_day"].items:
        if item.came:
            assert item.consequent_at is not None and (item.consequent_at.date() - item.day).days == 1
    week = by_segment["days_2_7"]
    for item in week.items:
        if item.came:
            assert item.consequent_at is not None and 2 <= (item.consequent_at.date() - item.day).days <= 7
    # 最后几天的窗口伸过截止日：结果未知
    assert week.items[-1].came is None and by_segment["next_day"].items[-1].came is None


def test_only_short_relations_between_base_or_group_concepts_enter_and_conditioned_ones_only_when_established() -> None:
    """长跨度、后果是汇总概念的不进；前因是汇总概念的进，类编号列全；候选的进但标状态；失效的不进；
    带条件的只收成立的（裁定 61 第 6 条），候选的带条件关系不进。"""

    concepts = ConceptSet([base(title) for title in TITLES] + [group("沟通", "讨论与调研", "讨论方案", "调研")])
    primed = f"timeline:{RESEARCH}"
    relations = (
        established(DISCUSS, CODE),
        established("沟通", CODE),
        established(CODE, "沟通"),
        Relation(key=RelationKey("session", DISCUSS, CODE, Segment.SITUATION), status=Status.ESTABLISHED, upward=True),
        Relation(
            key=RelationKey("session", DISCUSS, CODE, Segment.CHAIN, primed), status=Status.ESTABLISHED, upward=True
        ),
        Relation(key=RelationKey("session", DOCS, CODE, Segment.CHAIN, primed), status=Status.CANDIDATE, upward=True),
        established(RESEARCH, CODE, status=Status.CANDIDATE),
        established(DOCS, CODE, status=Status.EXPIRED),
        Relation(key=RelationKey("physical", DISCUSS, CODE, Segment.CHAIN), status=Status.ESTABLISHED, upward=True),
    )
    found = table(relations, concepts)
    assert {(item.antecedent, item.consequent, item.status, item.condition_text) for item in found.relations} == {
        (DISCUSS, CODE, "established", ""),
        (DISCUSS, CODE, "established", primed),
        ("沟通", CODE, "established", ""),
        (RESEARCH, CODE, "candidate", ""),
    }
    merged = next(item for item in found.relations if item.antecedent == "沟通")
    assert merged.antecedent_classes == (DISCUSS, RESEARCH) and merged.antecedent_label == "沟通"


def test_a_conditioned_relation_lists_only_the_chains_where_its_conditions_held() -> None:
    """带条件的关系的机会只列条件全成立的链；条件带着白天判它要的材料（时间线条件 → 类编号；日历条件 → 键与取值）。"""

    from habitus.scene.relations.conditions import evaluator_all, parse_conditions

    series = planted(moderated=True)
    concepts = ConceptSet([base(title) for title in TITLES] + list(calendar_concepts(now=CREATED)))
    primed = f"timeline:{RESEARCH}"
    both = f"concept:周三+{primed}"
    found = table(
        (
            established(DISCUSS, CODE),
            Relation(
                key=RelationKey("session", DISCUSS, CODE, Segment.CHAIN, primed), status=Status.ESTABLISHED, upward=True
            ),
            Relation(
                key=RelationKey("session", DISCUSS, CODE, Segment.CHAIN, both), status=Status.ESTABLISHED, upward=True
            ),
        ),
        concepts,
        series=series,
        calendar=LocalCalendar(),
    )
    by_condition = {item.condition_text: item for item in found.relations}
    plain, conditioned, combined = by_condition[""], by_condition[primed], by_condition[both]
    assert conditioned.base_key == combined.base_key == plain.key
    line = build_timelines(series, concepts, {}, CONFIG, calendar=LocalCalendar())["session"]
    judge = evaluator_all(line, parse_conditions(primed), window=CONFIG.transition_window_slots)
    held = {chain.anchor.uri for chain in chains_of(line, DISCUSS, CONFIG.transition_window_slots) if judge(chain)}
    assert 0 < len(conditioned.items) < len(plain.items)
    assert {item.antecedent_uri for item in conditioned.items} == held
    assert {item.antecedent_uri for item in combined.items} <= held
    assert all(item.day.weekday() == 2 for item in combined.items)
    (spec,) = conditioned.conditions
    assert spec.text == primed and spec.classes == (RESEARCH,) and spec.calendar_key is None
    weekday = next(spec for spec in combined.conditions if spec.text.startswith("concept:"))
    assert weekday.calendar_key == "日历.周几" and weekday.calendar_values == ("周三",)
    text = encode_table(found)
    assert decode_table(text, lane="session", night=found.night) == found


def test_calendar_situation_relations_list_each_day_the_condition_held_and_keep_only_the_most_specific() -> None:
    """ "周三 → 写文档"：每个人在场的周三是一次机会，起算是那天零点，来了的指向那天第一次写文档；同一个后果几条日历关系，
    被更粗的解释了的不留（周几集合在就不留被它包住的单个周几）；周几与日型两条都留（检验周几时日型是固定住的）。"""

    def situation(antecedent: str) -> Relation:
        return Relation(key=key(antecedent, DOCS, Segment.SITUATION), status=Status.ESTABLISHED, upward=True)

    both = weekday_set(("周三", "周五"), now=CREATED)
    concepts = ConceptSet([base(title) for title in TITLES] + [*calendar_concepts(now=CREATED), both])
    found = table((situation("周三"), situation("工作日")), concepts, calendar=LocalCalendar())
    assert {item.antecedent for item in found.relations} == {"周三", "工作日"}
    wednesday = found.relation(key("周三", DOCS, Segment.SITUATION).text)
    assert wednesday is not None and wednesday.within_day_types == ()  # 没有放假数据：日型都答不出，不约束
    assert (
        wednesday.antecedent == "周三"
        and wednesday.calendar_key == "日历.周几"
        and wednesday.calendar_values == ("周三",)
    )
    assert wednesday.items and all(item.day.weekday() == 2 for item in wednesday.items)
    for item in wednesday.items:
        assert item.antecedent_uri == f"calendar:{item.day.isoformat()}" and item.ended_at.hour == 0
        assert item.came and item.consequent_at is not None and item.consequent_at.hour == 17  # 17:50 收工写文档
    (merged,) = table((situation("周三"), situation(both.name)), concepts, calendar=LocalCalendar()).relations
    assert merged.antecedent == both.name and merged.calendar_values == ("周三", "周五")
    assert {item.day.weekday() for item in merged.items} == {2, 4}
    # 只在同状态之间比：候选的集合不挤掉成立的周三（2026-10-18 四方评审）
    candidate = Relation(key=key(both.name, DOCS, Segment.SITUATION), status=Status.CANDIDATE, upward=True)
    kept = table((situation("周三"), candidate), concepts, calendar=LocalCalendar()).relations
    assert {item.antecedent for item in kept} == {"周三", both.name}
    # 压低的日历关系不进机会表：一个取值高了，同一维的其余取值就显得低，那只是反面（S8）
    lowering = Relation(key=key(both.name, DOCS, Segment.SITUATION), status=Status.ESTABLISHED, upward=False)
    assert {
        item.antecedent for item in table((situation("周三"), lowering), concepts, calendar=LocalCalendar()).relations
    } == {"周三"}


def test_a_weekday_relation_names_the_day_types_it_was_seen_on_and_lists_only_those_days() -> None:
    """检验周几时日型固定住了，所以周几关系记下"在哪几种日型上见过他做"：只在工作日的周三写过文档，国庆的周三不是机会、
    也不建预期（S8；没见过的日型怎么办待用户定，默认不外推）。"""

    from habitus.conditions import HolidayTable, HolidayYear
    from habitus.series import EventSeries

    holiday = date(2026, 7, 22)  # 周三
    calendar = LocalCalendar(
        HolidayTable({2026: HolidayYear("CN", 2026, "测试", "1", frozenset({5, 6}), {holiday: "测试假"}, frozenset())})
    )
    whole = planted()
    series = EventSeries(
        cutoff=whole.cutoff,
        records=tuple(item for item in whole.records if not (item.day == holiday and item.kind_token == DOCS)),
    )
    concepts = ConceptSet([base(title) for title in TITLES] + list(calendar_concepts(now=CREATED)))
    relation = Relation(key=key("周三", DOCS, Segment.SITUATION), status=Status.ESTABLISHED, upward=True)
    (wednesday,) = table((relation,), concepts, series=series, calendar=calendar).relations
    assert wednesday.within_day_types == ("工作日",)
    assert holiday not in {item.day for item in wednesday.items} and wednesday.items
    holiday_wednesday = {"日历.周几": "周三", "日历.日型": "节假日"}
    workday_wednesday = {"日历.周几": "周三", "日历.日型": "工作日"}
    assert wednesday.calendar_holds(workday_wednesday) and not wednesday.calendar_holds(holiday_wednesday)
    assert not wednesday.calendar_holds(holiday_wednesday, official=True)  # 节假日按官方含义也是休息日，仍不建
    assert wednesday.calendar_holds({"日历.周几": "周三"})  # 日型答不出时不拿它约束


def test_unseen_day_types_follow_the_official_meaning_only_when_asked() -> None:
    """日型关系"工作日 → 叫车"：调休上班那天默认不外推；``official`` 时按放假数据的官方含义归进上班日。"""

    relation = RelationOpportunities(
        key="session|工作日|car|situation",
        antecedent="工作日",
        consequent="car",
        antecedent_label="工作日",
        consequent_label="叫车",
        antecedent_classes=("工作日",),
        consequent_classes=("car",),
        segment="situation",
        status="established",
        upward=True,
        items=(),
        calendar_key="日历.日型",
        calendar_values=("工作日",),
    )
    makeup = {"日历.周几": "周六", "日历.日型": "调休上班"}
    assert not relation.calendar_holds(makeup) and relation.calendar_holds(makeup, official=True)
    assert not relation.calendar_holds({"日历.周几": "周六", "日历.日型": "节假日"}, official=True)
    with pytest.raises(ForesightError, match="day types it was seen on"):
        replace(relation, within_day_types=("工作日",))
    with pytest.raises(ForesightError, match="unseen_day_types"):
        OpportunityTable("session", NIGHT, NIGHT, 15, 3, (), (), unseen_day_types="guess")


def test_the_kind_calendar_lists_each_day_with_its_first_record() -> None:
    """每个类的复现（语义树登记的）：发生过的每一天的第一次那条、几点，相邻两天的间隔；没发生过的类不列。"""

    series = planted(days=5)
    found = table((), series=series)
    research = found.kind(RESEARCH)
    assert research is not None and research.label == "调研" and len(research.days) == 5
    for item in research.days:
        assert item.first_at.hour == 9 and item.first_at.minute == 0 and item.count >= 1  # 9:00 开工那条总是第一次
        assert "调研" in item.first_uri and item.first_at.date() == item.day
    assert research.last is not None and research.last.day == series.last_day
    assert [item.days for item in research.intervals] == [1, 1, 1, 1]
    assert found.kind(kind_id("不存在的类")) is None
    assert {item.kind_token for item in found.kinds} <= {kind_id(title) for title in TITLES}


def test_the_table_round_trips_through_canonical_json() -> None:
    concepts = ConceptSet([base(title) for title in TITLES] + list(calendar_concepts(now=CREATED)))
    found = table(
        (
            established(DISCUSS, CODE),
            established(DOCS, RESEARCH, Segment.NEXT_DAY),
            Relation(key=key("周三", DOCS, Segment.SITUATION), status=Status.ESTABLISHED, upward=True),
        ),
        concepts,
        calendar=LocalCalendar(),
    )
    assert any(item.segment == "situation" for item in found.relations) and found.kinds and found.watch
    text = encode_table(found)
    assert decode_table(text, lane="session", night=found.night) == found
    assert encode_table(decode_table(text)) == text
    with pytest.raises(ForesightError, match="does not match"):
        decode_table(text, lane="physical")
    with pytest.raises(ForesightError, match="canonical"):
        decode_table(text.replace("}", "} "))
    with pytest.raises(ForesightError, match="schema"):
        decode_table(text.replace("opportunities/v8", "opportunities/v7"))


def test_the_shapes_refuse_contradictions() -> None:
    at = datetime(2026, 7, 1, 10, 0, tzinfo=CST)
    chance = Opportunity("a", ("a",), at.date(), at, True, "b", at + timedelta(minutes=20))
    with pytest.raises(ForesightError, match="containing the anchor"):
        Opportunity("a", ("b",), at.date(), at, False)
    with pytest.raises(ForesightError, match="exactly when it came"):
        Opportunity("a", ("a",), at.date(), at, False, "b", at)
    with pytest.raises(ForesightError, match="exactly when it came"):
        Opportunity("a", ("a",), at.date(), at, True)
    with pytest.raises(ForesightError, match="timezone-aware"):
        Opportunity("a", ("a",), at.date(), at.replace(tzinfo=None), None)
    relation = dict(
        key="k",
        antecedent="x",
        consequent="y",
        antecedent_label="X",
        consequent_label="Y",
        antecedent_classes=("x",),
        consequent_classes=("y",),
        segment="chain",
        status="established",
        upward=True,
    )
    with pytest.raises(ForesightError, match="segment"):
        RelationOpportunities(**{**relation, "segment": "weekly"}, items=())  # type: ignore[arg-type]
    with pytest.raises(ForesightError, match="calendar key"):
        RelationOpportunities(**{**relation, "segment": "situation"}, items=())  # type: ignore[arg-type]
    with pytest.raises(ForesightError, match="calendar key"):
        RelationOpportunities(**relation, items=(), calendar_key="日历.周几", calendar_values=("周三",))  # type: ignore[arg-type]
    with pytest.raises(ForesightError, match="status"):
        RelationOpportunities(**{**relation, "status": "expired"}, items=())  # type: ignore[arg-type]
    with pytest.raises(ForesightError, match="time order"):
        RelationOpportunities(
            **relation, items=(chance, Opportunity("c", ("c",), at.date(), at - timedelta(hours=1), None))
        )  # type: ignore[arg-type]
    day = OccurrenceDay(at.date(), "a", at, 1, 1, "a", at)
    later = OccurrenceDay(at.date() + timedelta(days=2), "b", at + timedelta(days=2), 1, 1, "b", at + timedelta(days=2))
    with pytest.raises(ForesightError, match="distinct and ascending"):
        KindDays("s-k0001", "x", (day, day), (DayInterval(day, day, 1, 1.0, False),))
    with pytest.raises(ForesightError, match="one interval"):
        KindDays("s-k0001", "x", (day, later))
    with pytest.raises(ForesightError, match="neighbouring days"):
        KindDays("s-k0001", "x", (day, later), (DayInterval(later, day, 2, 1.0, False),))
    assert KindDays("s-k0001", "x", (day, later), (DayInterval(day, later, 2, 1.0, False),)).last == later
    good = RelationOpportunities(**relation, items=(chance,))  # type: ignore[arg-type]
    assert (good.opportunities, good.hits, good.unknown) == (1, 1, 0)
    with pytest.raises(ForesightError, match="appears twice"):
        OpportunityTable("session", NIGHT, NIGHT, 15, 3, (good, good), ())
    with pytest.raises(ForesightError, match="watch share"):
        DayWatch(NIGHT, 1.5)
    with pytest.raises(ForesightError, match="day_type"):
        DayWatch(NIGHT, 1.0, "")
    with pytest.raises(ForesightError, match="DayWatch records"):
        OpportunityTable("session", NIGHT, NIGHT, 15, 3, (), (), watch=[DayWatch(NIGHT, 1.0)])  # type: ignore[arg-type]
    with pytest.raises(ForesightError, match="distinct and ascending"):
        OpportunityTable("session", NIGHT, NIGHT, 15, 3, (), (), watch=(DayWatch(NIGHT, 1.0), DayWatch(NIGHT, 0.5)))
