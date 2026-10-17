"""机会表（N1）：在语义树同一条时间线上，对每条成立关系列出历史每次前因之后后果来没来；每个类发生过的日子；编解码往返。"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pytest

from habitus.foresight import ForesightError
from habitus.foresight.opportunities import (
    KindDay,
    KindDays,
    Opportunity,
    OpportunityTable,
    RelationOpportunities,
    build_table,
    decode_table,
    encode_table,
)
from habitus.scene.concepts import ConceptSet
from habitus.scene.relations import Relation, RelationKey, Segment, Status, build_timelines, chains_of, merge_for
from tests.unit.kind_ids import kind_id
from tests.unit.scene.concept_fixtures import base, group
from tests.unit.scene.relation_fixtures import CODE, CONCEPTS, CONFIG, DISCUSS, DOCS, RESEARCH, TITLES, key, planted

CST = timezone(timedelta(hours=8))
NIGHT = date(2026, 7, 27)


def established(
    antecedent: str, consequent: str, segment: Segment = Segment.CHAIN, *, status: Status = Status.ESTABLISHED
) -> Relation:
    return Relation(key=key(antecedent, consequent, segment), status=status, upward=True)


def table(relations: tuple[Relation, ...], concepts: ConceptSet = CONCEPTS, series=None) -> OpportunityTable:  # type: ignore[no-untyped-def]
    series = series if series is not None else planted()
    line = build_timelines(series, concepts, {}, CONFIG)["session"]
    return build_table(
        line,
        relations,
        concepts,
        night=series.cutoff,
        relation_night=series.cutoff,
        slot_minutes=CONFIG.slot_minutes,
        window_slots=CONFIG.transition_window_slots,
    )


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
    assert found.by_consequent(CODE) == (relation,) and found.by_antecedent(DISCUSS) == (relation,)
    assert found.by_consequent(DISCUSS) == () and found.present_days == line.days()


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


def test_only_short_unconditioned_relations_between_base_or_group_concepts_enter() -> None:
    """长跨度、带条件、后果是汇总概念的不进；前因是汇总概念的进，类编号列全；候选的进但标状态；失效的不进。"""

    concepts = ConceptSet([base(title) for title in TITLES] + [group("沟通", "讨论与调研", "讨论方案", "调研")])
    relations = (
        established(DISCUSS, CODE),
        established("沟通", CODE),
        established(CODE, "沟通"),
        Relation(key=RelationKey("session", DISCUSS, CODE, Segment.SITUATION), status=Status.ESTABLISHED, upward=True),
        Relation(
            key=RelationKey("session", DISCUSS, CODE, Segment.CHAIN, "weekday=4"),
            status=Status.ESTABLISHED,
            upward=True,
        ),
        established(RESEARCH, CODE, status=Status.CANDIDATE),
        established(DOCS, CODE, status=Status.EXPIRED),
        Relation(key=RelationKey("physical", DISCUSS, CODE, Segment.CHAIN), status=Status.ESTABLISHED, upward=True),
    )
    found = table(relations, concepts)
    assert {(item.antecedent, item.consequent, item.status) for item in found.relations} == {
        (DISCUSS, CODE, "established"),
        ("沟通", CODE, "established"),
        (RESEARCH, CODE, "candidate"),
    }
    merged = next(item for item in found.relations if item.antecedent == "沟通")
    assert merged.antecedent_classes == (DISCUSS, RESEARCH) and merged.antecedent_label == "沟通"
    assert found.by_antecedent(RESEARCH) == tuple(
        item for item in found.relations if item.antecedent in ("沟通", RESEARCH)
    )


def test_the_kind_calendar_lists_each_day_with_its_first_record() -> None:
    """每个类发生过的每一天：第一次那条、几点、当天几次；没发生过的类不列。"""

    series = planted(days=5)
    found = table((), series=series)
    research = found.kind(RESEARCH)
    assert research is not None and research.label == "调研" and len(research.days) == 5
    for item in research.days:
        assert item.first_at.hour == 9 and item.first_at.minute == 0 and item.count >= 2  # 9:00 开工那条总是第一次
        assert "调研" in item.first_uri and item.first_at.date() == item.day
    assert research.last is not None and research.last.day == series.last_day
    assert found.kind(kind_id("不存在的类")) is None
    assert {item.kind_token for item in found.kinds} <= {kind_id(title) for title in TITLES}


def test_the_table_round_trips_through_canonical_json() -> None:
    found = table((established(DISCUSS, CODE), established(DOCS, RESEARCH, Segment.NEXT_DAY)))
    text = encode_table(found)
    assert decode_table(text, lane="session", night=found.night) == found
    assert encode_table(decode_table(text)) == text
    with pytest.raises(ForesightError, match="does not match"):
        decode_table(text, lane="physical")
    with pytest.raises(ForesightError, match="canonical"):
        decode_table(text.replace("}", "} "))
    with pytest.raises(ForesightError, match="schema"):
        decode_table(text.replace("opportunities/v3", "opportunities/v2"))


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
        RelationOpportunities(**{**relation, "segment": "situation"}, items=())  # type: ignore[arg-type]
    with pytest.raises(ForesightError, match="status"):
        RelationOpportunities(**{**relation, "status": "expired"}, items=())  # type: ignore[arg-type]
    with pytest.raises(ForesightError, match="time order"):
        RelationOpportunities(
            **relation, items=(chance, Opportunity("c", ("c",), at.date(), at - timedelta(hours=1), None))
        )  # type: ignore[arg-type]
    day = KindDay(at.date(), "a", at, 1, 1)
    with pytest.raises(ForesightError, match="starts on that day"):
        KindDay(at.date() + timedelta(days=1), "a", at, 1, 1)
    with pytest.raises(ForesightError, match="distinct and ascending"):
        KindDays("s-k0001", "x", (day, day))
    good = RelationOpportunities(**relation, items=(chance,))  # type: ignore[arg-type]
    assert (good.opportunities, good.hits, good.unknown) == (1, 1, 0)
    with pytest.raises(ForesightError, match="appears twice"):
        OpportunityTable("session", NIGHT, NIGHT, 15, 3, (), (good, good), ())
    with pytest.raises(ForesightError, match="present_days"):
        OpportunityTable("session", NIGHT, None, 15, 3, (NIGHT, NIGHT), (), ())
