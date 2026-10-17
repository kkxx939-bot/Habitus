"""对照答案找原因（裁定 55）：一次落空他接下来做了什么、在不在语义关联里；白天那一拍核"上次的原因今天又在不在"。

例子照用户的：订机票之后没订酒店（去订了家人的机票 / 开了会 / 什么起因都看不到），周二没去健身（约了人去游泳 / 节假日）。
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta

from habitus.foresight.causes import CauseMaterials, cause_holds, find_cause
from habitus.foresight.ledger import Antecedent, Cause, Expectation
from habitus.foresight.model import Seen
from habitus.scene.concepts import ConceptSet
from habitus.scene.relations import Relation, RelationKey, Segment, Status, build_timelines
from habitus.series import EventSeries
from tests.unit.kind_ids import kind_id
from tests.unit.scene.concept_fixtures import base, group
from tests.unit.scene.relation_fixtures import CONFIG, at, record

TITLES = ("订机票", "订酒店", "订家人机票", "开会", "约人", "游泳", "健身", "看邮件")
FLIGHT, HOTEL, FAMILY, MEETING, DATE, SWIM, GYM, MAIL = (kind_id(title) for title in TITLES)
TRAVEL = "出行预订"
CONCEPTS = ConceptSet([*(base(title) for title in TITLES), group(TRAVEL, "订行程里的票和住处", "订酒店", "订家人机票")])
MONDAY = date(2026, 6, 1)
TUESDAY = MONDAY + timedelta(days=1)


def line_of(*rows: tuple[date, int, int, str]):  # type: ignore[no-untyped-def]
    records = sorted(
        (record(when, hour, minute, title, number=index) for index, (when, hour, minute, title) in enumerate(rows)),
        key=lambda item: (item.started_at, item.uri),
    )
    series = EventSeries(cutoff=max(row[0] for row in rows) + timedelta(days=1), records=tuple(records))
    return build_timelines(series, CONCEPTS, {}, CONFIG)["session"], {item.name: item for item in records}


def materials(line, relations=(), notes=None) -> CauseMaterials:  # type: ignore[no-untyped-def]
    return CauseMaterials(line, CONCEPTS, tuple(relations), CONFIG.transition_window_slots, notes or (lambda day: None))


def relation(antecedent: str, consequent: str, segment: Segment = Segment.CHAIN, *, status=Status.ESTABLISHED, condition: str = "") -> Relation:  # type: ignore[no-untyped-def]
    return Relation(key=RelationKey("session", antecedent, consequent, segment, condition), status=status, upward=True)


def hotel_after_flight(started: datetime, *, antecedents: tuple[Antecedent, ...] = ()) -> Expectation:
    """订机票（``started``）之后同一条链里该订酒店：时窗 15–45 分钟，最晚时限 2 小时。"""

    return Expectation(
        expectation_id="hotel",
        kind_token=HOTEL,
        day=started.date(),
        formed_at=started + timedelta(minutes=10),
        source="事件",
        trigger=RelationKey("session", FLIGHT, HOTEL, Segment.CHAIN).text,
        trigger_instance=started.isoformat(),
        window=(started + timedelta(minutes=15), started + timedelta(minutes=45)),
        deadline=started + timedelta(hours=2),
        basis=(),
        opportunities=10,
        hits=9,
        observed_from=started + timedelta(minutes=10),
        antecedents=antecedents,
    )


def gym_on_tuesday(*, antecedents: tuple[Antecedent, ...] = ()) -> Expectation:
    formed = at(TUESDAY, 18, 0)
    return Expectation(
        expectation_id="gym",
        kind_token=GYM,
        day=TUESDAY,
        formed_at=formed,
        source="钟面",
        trigger="weekday=1",
        trigger_instance=TUESDAY.isoformat(),
        window=(formed, formed + timedelta(hours=1)),
        deadline=formed + timedelta(hours=3),
        basis=(),
        opportunities=10,
        hits=9,
        observed_from=formed,
        antecedents=antecedents,
    )


def same(expectation: Expectation, line, **kwargs) -> Cause:  # type: ignore[no-untyped-def]
    return find_cause(expectation, materials(line, **kwargs), current=lambda token: (token,))


def test_doing_the_sibling_under_the_same_group_is_the_same_matter() -> None:
    """订完机票去订了家人的机票：和订酒店同在「出行预订」下——粒度，不是关联错了。"""

    line, rows = line_of((MONDAY, 9, 0, "看邮件"), (MONDAY, 10, 0, "订机票"), (MONDAY, 10, 30, "订家人机票"))
    found = same(hotel_after_flight(at(MONDAY, 10, 0)), line)
    assert (found.kind, found.classes, found.uris, found.relation) == ("同一件事", (FAMILY,), (rows["订家人机票2"].uri,), TRAVEL)


def test_a_usual_cue_that_never_came_is_the_missing_cue_unless_it_was_there_or_came_later() -> None:
    """健身惯常在「看邮件」之后的当天剩下时间里（关系表里成立 / 候选的都算，形成时的前因清单不全）：这天没看邮件 → 线索没了；
    中午看过 → 线索在；时窗开始之后才看 → 也算来了；压低健身的、带条件的关系不算惯常前因。"""

    usual = relation(MAIL, GYM, Segment.REST_OF_DAY, status=Status.CANDIDATE)
    line, _rows = line_of((TUESDAY, 9, 0, "开会"), (TUESDAY, 19, 0, "开会"))
    found = same(gym_on_tuesday(), line, relations=(usual,))
    assert (found.kind, found.classes, found.relation, found.segment) == ("线索没了", (MAIL,), usual.key.text, "rest_of_day")
    noon, _rows = line_of((TUESDAY, 9, 0, "开会"), (TUESDAY, 12, 0, "看邮件"), (TUESDAY, 19, 0, "开会"))
    assert same(gym_on_tuesday(), noon, relations=(usual,)).kind == "还没进关联"  # 下一步是 19:00 开会
    late, _rows = line_of((TUESDAY, 9, 0, "开会"), (TUESDAY, 18, 30, "看邮件"))
    assert same(gym_on_tuesday(), late, relations=(usual,)).kind == "还没进关联"
    down = Relation(key=usual.key, status=Status.ESTABLISHED, upward=False)
    assert same(gym_on_tuesday(), line, relations=(down,)).kind == "还没进关联"
    conditioned = relation(MAIL, GYM, Segment.REST_OF_DAY, condition="field:weekend")
    assert same(gym_on_tuesday(), line, relations=(conditioned,)).kind == "还没进关联"
    assert same(gym_on_tuesday(), line, relations=(relation(MAIL, GYM, Segment.REST_OF_DAY, status=Status.TESTED),)).kind == "还没进关联"
    # 两条惯常前因都没来：取它不来健身就掉得最多的那条，不按编号、不按超额比例（甲盘探针 03-24 的真实读数：
    # 「看今天的日程」份额 0.59、有它 0.14 对没它 0.05；「预约保洁」份额 0.56、有它 0.71 对没它 0）
    daily = replace(relation(DATE, GYM, Segment.REST_OF_DAY), tonight=_reading(DATE, GYM, share=0.59, rates=(0.14, 0.05)))
    decisive = replace(relation(MAIL, GYM, Segment.REST_OF_DAY), tonight=_reading(MAIL, GYM, share=0.56, rates=(0.71, 0.0)))
    for order in ((daily, decisive), (decisive, daily)):
        assert same(gym_on_tuesday(), line, relations=order).relation == decisive.key.text
    # 两个角色对调（排序不靠编号碰巧对）
    daily = replace(relation(MAIL, GYM, Segment.REST_OF_DAY), tonight=_reading(MAIL, GYM, share=0.59, rates=(0.14, 0.05)))
    decisive = replace(relation(DATE, GYM, Segment.REST_OF_DAY), tonight=_reading(DATE, GYM, share=0.56, rates=(0.71, 0.0)))
    assert same(gym_on_tuesday(), line, relations=(daily, decisive)).relation == decisive.key.text


def _reading(antecedent: str, consequent: str, *, share: float, rates: tuple[float, float]):  # type: ignore[no-untyped-def]
    from habitus.scene.relations import RelationTest, Verdict
    from habitus.scene.relations.stats import Effect

    key = RelationKey("session", antecedent, consequent, Segment.REST_OF_DAY)
    return RelationTest(key, Verdict.SIGNIFICANT, 20, 6, 1.0, Effect(20, *rates), 0.001, 0.0001, True, excess_share=share)


def test_a_day_type_unlike_every_day_the_habit_was_learnt_on_is_the_missing_cue() -> None:
    line, _rows = line_of((TUESDAY, 9, 0, "开会"))
    holiday = lambda day: "节假日" if day == TUESDAY else None  # noqa: E731
    item = gym_on_tuesday()
    found = find_cause(item, materials(line, notes=holiday), current=lambda token: (token,), basis_days=(MONDAY - timedelta(days=6),))
    assert (found.kind, found.condition) == ("线索没了", "日型=节假日")
    # 依据那几次里也有节假日：日型不是原因
    same_note = lambda day: "节假日"  # noqa: E731
    assert find_cause(item, materials(line, notes=same_note), current=lambda token: (token,), basis_days=(MONDAY,)).kind == "没看到下一步"


def test_the_substitute_brought_on_by_its_own_established_cause_is_another_cause() -> None:
    """周二 17:30 约了人，18:05 去游泳（「约人 → 游泳」同一条链成立）：该健身的时候他接下来去游泳了，因为别的安排。"""

    # 20:00 还在（人在场盖住约人之后的整个转移窗口，那一次的结果才看得清）
    line, rows = line_of((TUESDAY, 9, 0, "开会"), (TUESDAY, 17, 30, "约人"), (TUESDAY, 18, 5, "游泳"), (TUESDAY, 20, 0, "看邮件"))
    found = same(gym_on_tuesday(), line, relations=(relation(DATE, SWIM),))
    assert (found.kind, found.classes, found.segment) == ("别的起因", (DATE,), "chain")
    assert found.uris == (rows["约人1"].uri, rows["游泳2"].uri)
    # 只是已检、不显著的关系不算起因
    tested = same(gym_on_tuesday(), line, relations=(relation(DATE, SWIM, status=Status.TESTED),))
    assert (tested.kind, tested.classes, tested.uris) == ("还没进关联", (SWIM,), (rows["游泳2"].uri,))
    # 早上约的人带出的是早上那一次游泳；晚上这一次前面没有约人：不能借早上那条链说它是起因
    morning, _rows = line_of(
        (TUESDAY, 9, 0, "约人"), (TUESDAY, 9, 20, "游泳"), (TUESDAY, 12, 0, "开会"), (TUESDAY, 18, 30, "游泳"), (TUESDAY, 20, 0, "看邮件")
    )
    assert same(gym_on_tuesday(), morning, relations=(relation(DATE, SWIM),)).kind == "还没进关联"
    # 起因带出的那件事在时窗开始之前就结束了（早上约人、早上游泳）：它不是下一步，下一步是 20:00 看邮件（甲盘探针 03-28 误归的样子）
    early, _rows = line_of((TUESDAY, 8, 0, "约人"), (TUESDAY, 8, 20, "游泳"), (TUESDAY, 20, 0, "看邮件"))
    found = same(gym_on_tuesday(), early, relations=(relation(DATE, SWIM),))
    assert (found.kind, found.classes) == ("还没进关联", (MAIL,))


def test_after_the_antecedent_the_branch_is_named_by_how_much_the_tree_knows_about_it() -> None:
    """订完机票去开会：「订机票 → 开会」成立 = 本来就有分支；带条件的「订机票 → 订酒店｜周末」这天是工作日 = 条件解释了；
    没有条件 = 条件不够；表里没有这条、或已检不显著 = 开会还没进关联（记下来攒次数）。"""

    line, _rows = line_of((MONDAY, 9, 0, "看邮件"), (MONDAY, 10, 0, "订机票"), (MONDAY, 10, 30, "开会"))
    item = hotel_after_flight(at(MONDAY, 10, 0))
    branch = relation(FLIGHT, MEETING)
    weekend = relation(FLIGHT, HOTEL, condition="field:weekend")
    conditioned = same(item, line, relations=(branch, weekend))
    assert (conditioned.kind, conditioned.condition, conditioned.holds_when) == ("分支有条件", "field:weekend", False)
    assert same(item, line, relations=(branch,)).kind == "分支条件不够"
    # 人在场盖住订机票之后的整个窗口（开会那一次看得清是订机票带出来的）：订机票带出开会仍是订机票之后的分支，不是"别的起因"
    seen_through, _rows = line_of((MONDAY, 9, 0, "看邮件"), (MONDAY, 10, 0, "订机票"), (MONDAY, 10, 30, "开会"), (MONDAY, 11, 30, "看邮件"))
    assert same(item, seen_through, relations=(branch,)).kind == "分支条件不够"
    assert same(item, line).kind == "还没进关联"
    # 下一步只看接下来的第一件：先去游泳、后来才开会，游泳就是下一步，它还没进关联
    both, _rows = line_of((MONDAY, 9, 0, "看邮件"), (MONDAY, 10, 0, "订机票"), (MONDAY, 10, 20, "游泳"), (MONDAY, 10, 40, "开会"))
    found = same(item, both, relations=(branch,))
    assert (found.kind, found.classes) == ("还没进关联", (SWIM,))
    assert same(item, line, relations=(relation(FLIGHT, MEETING, status=Status.TESTED),)).kind == "还没进关联"


def test_a_next_action_outside_every_relation_is_kept_to_accumulate_and_no_action_is_the_observation_edge() -> None:
    """该订菜的时候他接下来去缴了水电，缴水电还没进任何关联：记「还没进关联」带上缴水电，攒次数（不是"习惯在变"）。
    到最晚时限什么都没做：没看到下一步。"""

    line, rows = line_of((TUESDAY, 9, 0, "开会"), (TUESDAY, 18, 40, "看邮件"))
    found = same(gym_on_tuesday(), line)
    assert (found.kind, found.classes, found.uris) == ("还没进关联", (MAIL,), (rows["看邮件1"].uri,))
    quiet, _rows = line_of((TUESDAY, 9, 0, "开会"))
    assert same(gym_on_tuesday(), quiet) == Cause("没看到下一步")


def seen(day: date, hour: int, minute: int, kind: str) -> Seen:
    started = at(day, hour, minute)
    return Seen(started, started + timedelta(minutes=10), kind, f"behavior://{kind}-{hour}{minute}.md")


def test_the_daytime_check_finds_last_times_cause_again_only_where_it_can_be_checked() -> None:
    """下一个周二：又约了人（起因那一类此刻正处在它带出游泳的那一段里）→ 又在；惯常的前因又没来 → 又在；工作日的分支条件
    → 又在；习惯在变说的是他自己，此刻核不出来。"""

    next_tuesday = TUESDAY + timedelta(days=7)
    now = at(next_tuesday, 18, 15)
    item = gym_on_tuesday()

    def holds(cause: Cause, rows: tuple[Seen, ...] = (), expectation: Expectation = item) -> bool:
        return cause_holds(
            cause, expectation, rows, now=now, lane="session", slot_minutes=15, window_slots=3, day_note=lambda day: None
        )

    other = Cause("别的起因", classes=(DATE,), segment="chain", relation="r")
    assert holds(other, (seen(next_tuesday, 18, 0, DATE),))
    assert not holds(other, (seen(next_tuesday, 9, 0, DATE),))  # 早上约的，过了那一段
    assert not holds(other)
    cue = RelationKey("session", MAIL, GYM, Segment.REST_OF_DAY).text
    missing = Cause("线索没了", classes=(MAIL,), relation=cue, segment="rest_of_day")
    assert holds(missing)  # 今天到此刻没看邮件
    assert not holds(missing, (seen(next_tuesday, 12, 0, MAIL),))
    assert holds(Cause("线索没了", condition="日型=节假日")) is False
    weekday = Cause("分支有条件", relation="r", condition="field:weekend", holds_when=False)
    assert holds(weekday, expectation=hotel_after_flight(at(next_tuesday, 10, 0)))
    assert not holds(weekday, expectation=hotel_after_flight(at(next_tuesday + timedelta(days=4), 10, 0)))
    assert not holds(Cause("还没进关联", classes=(SWIM,)))
    assert not holds(Cause("没看到下一步"))
