"""日历情境（裁定 59 第 7 条）：由外部条件的日历机械给出，取值从数据里攒出来；按维度取对照；粗的能解释的不报细的；取值合成集合。"""

from __future__ import annotations

from datetime import date, timedelta

from habitus.conditions import LocalCalendar
from habitus.scene.concepts.calendar_situations import (
    calendar_concepts,
    calendar_key,
    most_specific,
    value_set,
    value_sets_to_add,
    weekday_set,
)
from habitus.scene.concepts.model import ConceptSet
from habitus.scene.relations import LaneTests, Segment, build_timelines
from tests.unit.scene.concept_fixtures import CREATED, base, cid
from tests.unit.scene.relation_fixtures import CONFIG
from tests.unit.temporal.fixtures import at, record, series

START = date(2026, 3, 2)  # 周一
MEETING = base("例会材料", "准备例会材料")
CODING = base("写代码", "改动代码文件")


def _concepts() -> ConceptSet:
    return ConceptSet((MEETING, CODING, *calendar_concepts(now=CREATED)))


def _weeks(count: int):  # type: ignore[no-untyped-def]
    found = []
    for offset in range(7 * count):
        day = START + timedelta(days=offset)
        found.append(record(at(day, 10, 0), cid("写代码")))  # 天天在场
        if day.weekday() == 2:
            found.append(record(at(day, 17, 0), cid("例会材料")))
    return found


def test_calendar_situations_are_filled_by_day_without_records_judging_them() -> None:
    concepts = _concepts()
    line = build_timelines(
        series(_weeks(4), cutoff=START + timedelta(days=28)), concepts, {}, CONFIG, calendar=LocalCalendar()
    )["session"]
    wednesday = line.situations["周三"]
    assert wednesday[START + timedelta(days=2)] is True and wednesday[START] is False
    assert "工作日" not in line.situations  # 没有放假数据：日型答不出，一天都不填（不知道 ≠ 否）
    assert line.calendar_days("周三") is wednesday and line.calendar_days(cid("写代码")) is None


def test_a_wednesday_habit_becomes_a_situation_relation() -> None:
    concepts = _concepts()
    weeks = series(_weeks(10), cutoff=START + timedelta(days=70))
    line = build_timelines(weeks, concepts, {}, CONFIG, calendar=LocalCalendar())["session"]
    tests = LaneTests(line, concepts, CONFIG, weeks.cutoff)
    keys = {(key.antecedent, key.consequent) for key in tests.keys() if key.segment is Segment.SITUATION}
    assert ("周三", cid("例会材料")) in keys
    measure = tests.measure(
        next(
            key
            for key in tests.keys()
            if (key.antecedent, key.segment) == ("周三", Segment.SITUATION) and key.consequent == cid("例会材料")
        )
    )
    effect = measure.effect()
    assert effect.treated_rate == 1.0 and effect.control_rate == 0.0  # 周三那几天全做、同一周别的日子都没做


def test_only_values_seen_in_the_data_become_calendar_situations() -> None:
    """取值从数据里攒出来，不一上来给齐：新加坡、美国永远不会出现"调休上班"，没碰到过节假日就没有"节假日"。"""

    seen = calendar_concepts(
        now=CREATED, observed=[("日历.周几", "周三"), ("日历.日型", "工作日"), ("日历.几号", "20"), ("日历.月末", "否")]
    )
    assert [item.name for item in seen] == ["周三", "20号", "工作日"]  # 月末"否"不是一个情境
    assert len(calendar_concepts(now=CREATED)) == 7 + 31 + 1 + 4  # 不给就是全部（工具与测试用）


def test_the_coarser_calendar_antecedent_explains_the_finer() -> None:
    """同状态同方向时：集合包住的取值、小集合不留；月末包住的 31 号不留。周几与日型不当嵌套（检验周几时日型固定住了）；
    互不包含的各自留着。"""

    small = weekday_set(["周一", "周三"], now=CREATED)
    concepts = ConceptSet((*calendar_concepts(now=CREATED), weekday_set(["周一", "周三", "周六"], now=CREATED), small))
    assert most_specific(concepts, ["周一", "周三", "周一、周三、周六", "周五"]) == ("周一、周三、周六", "周五")
    assert most_specific(concepts, ["周一、周三", "周一、周三、周六"]) == ("周一、周三、周六",)
    assert most_specific(concepts, ["31号", "月末"]) == ("月末",)
    assert most_specific(concepts, ["周三", "工作日"]) == ("周三", "工作日")
    assert most_specific(concepts, ["周五", "20号"]) == ("周五", "20号")
    assert most_specific(concepts, ["20号", "月末"]) == ("20号", "月末")
    assert most_specific(concepts, ["出差中", "工作日"]) == ("出差中", "工作日")


def test_value_sets_grow_on_any_dimension_from_two_or_more_held_values() -> None:
    singles = calendar_concepts(now=CREATED)
    concepts = ConceptSet(singles)
    held = {"晨跑": ["周六", "周一", "周三"], "健身": ["周二"], "叫车": ["工作日", "调休上班"], "交房租": ["月末"]}
    added = value_sets_to_add(concepts, held, now=CREATED)
    assert [item.name for item in added] == ["周一、周三、周六", "工作日、调休上班"]
    grown = ConceptSet((*singles, *added))
    assert calendar_key(grown, "周一、周三、周六") == ("日历.周几", ("周一", "周三", "周六"))
    assert calendar_key(grown, "工作日、调休上班") == ("日历.日型", ("工作日", "调休上班"))
    assert value_sets_to_add(grown, {"晨跑": ["周一", "周三", "周六"]}, now=CREATED) == ()
    assert value_set("日历.几号", ["15", "1"], now=CREATED).name == "1号、15号"


def _night(records, calendar, *, weeks: int):  # type: ignore[no-untyped-def]
    observed = {item for day in {record.day for record in records} for item in calendar.of_day(day)}
    concepts = ConceptSet((MEETING, CODING, CAR, *calendar_concepts(now=CREATED, observed=observed)))
    data = series(records, cutoff=START + timedelta(days=7 * weeks))
    line = build_timelines(data, concepts, {}, CONFIG, calendar=calendar)["session"]
    tests = LaneTests(line, concepts, CONFIG, data.cutoff)
    return {
        (test.key.antecedent, test.key.consequent): test
        for test in tests.night().tests
        if test.key.segment is Segment.SITUATION
    }


CAR = base("叫车", "叫车上班")


def _cn_calendar(holidays=(), makeup=()):  # type: ignore[no-untyped-def]
    from habitus.conditions import HolidayTable, HolidayYear

    year = HolidayYear(
        "CN", 2026, "测试", "1", frozenset({5, 6}), {day: "测试假" for day in holidays}, frozenset(makeup)
    )
    return LocalCalendar(HolidayTable({2026: year}))


def test_every_workday_is_anchored_on_the_workday_not_on_monday_and_friday() -> None:
    """工作日天天叫车（20 周）：按维度取对照、比周几时日型固定住，"工作日"成立，单个周几比不出差别（S8）。原来拿前后紧挨着的
    日子当对照，只有周一、周五显著（四方评审复现过）。没有放假数据时日型答不出，只能在周几这一维上看：周一到周五都抬高。"""

    calendar = _cn_calendar()
    records = []
    for offset in range(7 * 20):
        day = START + timedelta(days=offset)
        records.append(record(at(day, 10, 0), cid("写代码")))
        if day.weekday() < 5:
            records.append(record(at(day, 8, 0), cid("叫车")))
    found = _night(records, calendar, weeks=20)
    workday = found[("工作日", cid("叫车"))]
    assert workday.verdict.value == "significant" and workday.upward
    assert all(
        found[(name, cid("叫车"))].verdict.value != "significant" for name in ("周一", "周二", "周三", "周四", "周五")
    )
    blind = _night(records, LocalCalendar(), weeks=20)
    assert all(
        blind[(name, cid("叫车"))].verdict.value == "significant" and blind[(name, cid("叫车"))].upward
        for name in ("周一", "周三", "周五")
    )
    assert ("工作日", cid("叫车")) not in blind  # 日型答不出：没有"工作日"这个情境


def test_a_holiday_wednesday_does_not_dilute_the_wednesday_and_a_late_habit_is_read_from_its_start() -> None:
    """国庆的周三没开例会：比周几时日型固定住，它只和同一周的节假日比（那一层两边都没做，不给检验添证据），"周三"照样显著。
    例会从第 11 周才开始：从第一次做起才算，读数是 9/10（那个国庆的周三没做），不是被前 10 周稀释成 0.45（S8；四方评审算法 M5）。"""

    holiday = START + timedelta(days=7 * 14 + 2)  # 第 15 周的周三
    calendar = _cn_calendar(holidays=(holiday, holiday + timedelta(days=1), holiday - timedelta(days=1)))
    records = []
    for offset in range(7 * 20):
        day = START + timedelta(days=offset)
        records.append(record(at(day, 10, 0), cid("写代码")))
        if day.weekday() == 2 and offset >= 7 * 10 and day != holiday:
            records.append(record(at(day, 17, 0), cid("例会材料")))
    wednesday = _night(records, calendar, weeks=20)[("周三", cid("例会材料"))]
    assert wednesday.verdict.value == "significant" and wednesday.upward
    assert (
        wednesday.effect.antecedents == 10
        and wednesday.effect.treated_rate == 0.9
        and wednesday.effect.control_rate == 0.0
    )


def test_comparisons_stay_within_the_same_week_so_a_fading_habit_is_not_read_as_a_holiday_effect() -> None:
    """写代码前 15 周天天做、之后停了；节假日只落在第 18、19 周的工作日。按同一周分层，那两周节假日与平常日子都没做，比不出东西；
    拿全部历史比的话，节假日会被读成"节假日 → 写代码 ↓"——那是习惯淡了，不是节假日（S8：只在同一段时间里比）。"""

    holidays = tuple(
        day for day in (START + timedelta(days=7 * 17 + offset) for offset in range(14)) if day.weekday() < 5
    )
    calendar = _cn_calendar(holidays=holidays)
    records = []
    for offset in range(7 * 20):
        day = START + timedelta(days=offset)
        records.append(record(at(day, 9, 0), cid("叫车")))  # 天天在场
        if offset < 7 * 15:
            records.append(record(at(day, 10, 0), cid("写代码")))
    holiday = _night(records, calendar, weeks=20)[("节假日", cid("写代码"))]
    assert holiday.verdict.value != "significant"
