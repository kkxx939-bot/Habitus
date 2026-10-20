"""当地日历：周几、几号、月末任何一天都答得出；日型只按放假安排答，那一年没有数据就是未知（不是普通工作日）。"""

from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from habitus.conditions import (
    DAY_OF_MONTH_KEY,
    DAY_TYPE_KEY,
    MONTH_END_KEY,
    WEEKDAY_KEY,
    CalendarDataError,
    DayType,
    FactProvider,
    HolidayTable,
    HolidayYear,
    LocalCalendar,
    nominal_day_type,
)
from habitus.conditions.calendar_data import load_holiday_table, parse_holiday_year

CST = timezone(timedelta(hours=8))
# 测试用的假想年份：不冒充任何一年真实的放假安排。
YEAR = 2099


def _year(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "region": "CN",
        "year": YEAR,
        "source": "测试用的假想通知",
        "version": "t1",
        "weekend": [5, 6],
        "holidays": {f"{YEAR}-10-01": "国庆节", f"{YEAR}-10-02": "国庆节"},
        "makeup_workdays": [f"{YEAR}-10-10"],
    }
    payload.update(overrides)
    return payload


def _calendar() -> LocalCalendar:
    return LocalCalendar(HolidayTable({YEAR: parse_holiday_year(json.dumps(_year()))}))


def test_day_type_follows_the_holiday_schedule() -> None:
    calendar = _calendar()
    assert calendar.day_type(date(YEAR, 10, 1)) is DayType.HOLIDAY
    assert calendar.day_type(date(YEAR, 10, 10)) is DayType.MAKEUP_WORKDAY  # 调休上班
    assert calendar.day_type(date(YEAR, 10, 12)) is DayType.WORKDAY
    assert calendar.day_type(date(YEAR, 10, 11)) is DayType.WEEKEND
    assert calendar.describe(date(YEAR, 10, 1)) == "国庆节，法定放假"
    assert calendar.describe(date(YEAR, 10, 10)) == "调休上班日"
    assert calendar.describe(date(YEAR, 10, 12)) is None


def test_a_year_without_data_has_an_unknown_day_type_not_a_workday() -> None:
    calendar = _calendar()
    day = date(YEAR + 1, 3, 4)
    assert calendar.day_type(day) is None
    assert calendar.describe(day) is None
    conditions = dict(calendar.of_day(day))
    assert DAY_TYPE_KEY not in conditions  # 答不了的键不出现
    assert conditions[WEEKDAY_KEY] and conditions[DAY_OF_MONTH_KEY] == "4"
    # 建预期要名义值时另取，统计里不拿它充数
    assert nominal_day_type(day, weekend=(5, 6)) is (DayType.WORKDAY if day.weekday() < 5 else DayType.WEEKEND)


def test_calendar_is_a_fact_provider_with_month_end_and_weekday() -> None:
    calendar = _calendar()
    assert isinstance(calendar, FactProvider)
    conditions = dict(calendar.at(datetime(YEAR, 2, 28, 9, 0, tzinfo=CST)))
    assert conditions[MONTH_END_KEY] == "是"
    assert conditions[DAY_OF_MONTH_KEY] == "28"
    assert conditions[DAY_TYPE_KEY] in {item.value for item in DayType}
    assert "calendar:CN2099@t1" == calendar.version
    with pytest.raises(TypeError):
        calendar.at(datetime(YEAR, 2, 28, 9, 0))  # 不带时区


def test_empty_calendar_answers_dates_only() -> None:
    calendar = LocalCalendar()
    assert calendar.version == "calendar:none"
    assert calendar.day_type(date(2026, 10, 1)) is None
    assert DAY_TYPE_KEY not in dict(calendar.of_day(date(2026, 10, 1)))


@pytest.mark.parametrize(
    "overrides",
    [
        {"holidays": {f"{YEAR}-10-01": "国庆节"}, "makeup_workdays": [f"{YEAR}-10-01"]},  # 同一天又放又上
        {"holidays": {f"{YEAR + 1}-01-01": "元旦"}},  # 不在那一年
        {"makeup_workdays": [f"{YEAR}-10-10", f"{YEAR}-10-10"]},  # 重复
        {"source": ""},
        {"extra": 1},  # 未知字段
    ],
)
def test_bad_holiday_data_is_rejected(overrides: dict[str, object]) -> None:
    with pytest.raises(CalendarDataError):
        parse_holiday_year(json.dumps(_year(**overrides)))


def test_duplicate_keys_in_the_file_are_rejected() -> None:
    text = '{"region":"CN","region":"CN","year":2099,"source":"s","version":"v","weekend":[5,6],"holidays":{},"makeup_workdays":[]}'
    with pytest.raises(CalendarDataError):
        parse_holiday_year(text)


def test_the_calendar_directory_is_an_external_component(tmp_path: Path) -> None:
    assert load_holiday_table("CN").years == {}  # 不给目录：没有放假数据
    (tmp_path / f"{YEAR}.json").write_text(json.dumps(_year(version="t2", holidays={})), encoding="utf-8")
    table = load_holiday_table("CN", tmp_path)
    assert table.version == "CN2099@t2"
    assert table.day_type(date(YEAR, 10, 1)) is DayType.WORKDAY


def test_override_file_must_match_its_name_and_region(tmp_path: Path) -> None:
    (tmp_path / "2098.json").write_text(json.dumps(_year()), encoding="utf-8")
    with pytest.raises(CalendarDataError):
        load_holiday_table("CN", tmp_path)
    with pytest.raises(CalendarDataError):
        load_holiday_table("CN", tmp_path / "missing")


def test_table_rejects_mixed_regions_and_mismatched_years() -> None:
    cn = HolidayYear("CN", YEAR, "s", "v")
    with pytest.raises(CalendarDataError):
        HolidayTable({YEAR + 1: cn})
    with pytest.raises(CalendarDataError):
        HolidayTable({YEAR: cn, YEAR + 1: HolidayYear("HK", YEAR + 1, "s", "v")})


def test_regions_and_weekends_follow_where_the_person_is() -> None:
    """日历跟着这个人在哪：新加坡没有调休上班；周末是周五、周六的地区按数据算；州一级的地区码（US-CA）也认。"""

    singapore = parse_holiday_year(
        json.dumps(_year(region="SG", makeup_workdays=[], holidays={f"{YEAR}-10-12": "补假"}))
    )
    assert LocalCalendar(HolidayTable({YEAR: singapore})).day_type(date(YEAR, 10, 12)) is DayType.HOLIDAY
    friday_saturday = parse_holiday_year(
        json.dumps(_year(region="AE", weekend=[4, 5], holidays={}, makeup_workdays=[]))
    )
    calendar = LocalCalendar(HolidayTable({YEAR: friday_saturday}))
    assert calendar.day_type(date(YEAR, 10, 9)) is DayType.WEEKEND  # 周五
    assert calendar.day_type(date(YEAR, 10, 11)) is DayType.WORKDAY  # 周日
    assert parse_holiday_year(json.dumps(_year(region="US-CA", holidays={}, makeup_workdays=[]))).region == "US-CA"
    for bad in ({"region": "cn"}, {"region": "USA"}, {"weekend": [7]}, {"weekend": [5, 5]}):
        with pytest.raises(CalendarDataError):
            parse_holiday_year(json.dumps(_year(**bad)))


def test_expectations_get_a_nominal_day_type_when_the_holiday_data_cannot_say() -> None:
    """建预期用的日历条件：日型答不出时按名义周末给（统计用的 ``of_day`` 照样是未知）；有数据就用数据。"""

    from habitus.conditions import DAY_TYPE_KEY

    wednesday, saturday = date(2026, 7, 22), date(2026, 7, 25)
    plain = LocalCalendar()
    assert DAY_TYPE_KEY not in dict(plain.of_day(wednesday))
    assert dict(plain.for_expectations(wednesday))[DAY_TYPE_KEY] == "工作日"
    assert dict(plain.for_expectations(saturday))[DAY_TYPE_KEY] == "周末"
    friday_weekend = LocalCalendar(nominal_weekend=(4, 5))  # 周五周六休息的地方
    assert dict(friday_weekend.for_expectations(date(2026, 7, 24)))[DAY_TYPE_KEY] == "周末"
    year = HolidayYear("CN", 2026, "测试", "1", frozenset({5, 6}), {wednesday: "测试假"}, frozenset())
    assert dict(LocalCalendar(HolidayTable({2026: year})).for_expectations(wednesday))[DAY_TYPE_KEY] == "节假日"
    with pytest.raises(CalendarDataError):
        LocalCalendar(nominal_weekend=(7,))
    with pytest.raises(CalendarDataError):
        LocalCalendar(nominal_weekend=())


def test_the_day_note_says_unknown_only_when_the_day_type_cannot_be_told() -> None:
    from habitus.conditions import UNKNOWN_DAY_NOTE, day_note

    wednesday = date(2026, 7, 22)
    assert day_note(LocalCalendar(), wednesday) == UNKNOWN_DAY_NOTE == "日型未知"
    year = HolidayYear(
        "CN", 2026, "测试", "1", frozenset({5, 6}), {wednesday: "测试假"}, frozenset({date(2026, 7, 26)})
    )
    known = LocalCalendar(HolidayTable({2026: year}))
    assert day_note(known, wednesday) == "测试假，法定放假"
    assert day_note(known, date(2026, 7, 26)) == "调休上班日"
    assert day_note(known, date(2026, 7, 23)) is None  # 平常的工作日不说


def test_the_runtime_reader_skips_a_bad_year_and_reports_it(tmp_path: Path) -> None:
    """组合根用宽松读法：一份文件写错只跳过那一年（那一年日型未知），问题交出去，不拖垮进程（2026-10-18 四方评审）。"""

    from habitus.conditions.calendar_data import read_holiday_table

    (tmp_path / f"{YEAR}.json").write_text(json.dumps(_year()), encoding="utf-8")
    (tmp_path / f"{YEAR + 1}.json").write_text("{", encoding="utf-8")  # 写坏的一年
    (tmp_path / "2098.json").write_text(json.dumps(_year()), encoding="utf-8")  # 文件名与年份对不上
    table, problems = read_holiday_table("CN", tmp_path)
    assert set(table.years) == {YEAR} and table.day_type(date(YEAR, 10, 1)) is DayType.HOLIDAY
    assert len(problems) == 2 and any(f"{YEAR + 1}.json" in item for item in problems)
    empty, none = read_holiday_table("CN", None)
    assert not empty.years and none == ()
    missing_table, missing = read_holiday_table("CN", tmp_path / "missing")
    assert not missing_table.years and missing and "does not exist" in missing[0]
