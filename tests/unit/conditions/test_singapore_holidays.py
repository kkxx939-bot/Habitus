"""随包附带的新加坡 2025、2026 公共假期（W3）：照人力部（MOM）公布的原文录入，没有调休上班，落在周日的次日周一补休。"""

from __future__ import annotations

from datetime import date

import pytest

from habitus.conditions import DayType
from habitus.conditions.calendar_data import bundled_holiday_directory, load_holiday_table


@pytest.fixture(scope="module")
def singapore():
    directory = bundled_holiday_directory("SG")
    assert directory is not None
    return load_holiday_table("SG", directory)


@pytest.fixture(scope="module")
def mainland():
    directory = bundled_holiday_directory("CN")
    assert directory is not None
    return load_holiday_table("CN", directory)


def test_the_bundled_singapore_years_load_strictly_and_cite_the_ministry_of_manpower(singapore) -> None:
    assert sorted(singapore.years) == [2025, 2026]
    for year in singapore.years.values():
        assert "mom.gov.sg/employment-practices/public-holidays" in year.source
        assert year.weekend == frozenset({5, 6})
        assert year.makeup_workdays == frozenset()  # 新加坡没有调休上班
    # 2025：12 个公共假期（含 5 月 3 日大选投票日），没有落在周日的；2026：11 个，3 个落在周日，各补一个周一
    assert len(singapore.years[2025].holidays) == 12
    assert len(singapore.years[2026].holidays) == 14


@pytest.mark.parametrize(
    ("day", "expected"),
    [
        (date(2025, 1, 1), DayType.HOLIDAY),  # 元旦
        (date(2025, 1, 29), DayType.HOLIDAY),  # 农历新年
        (date(2025, 5, 3), DayType.HOLIDAY),  # 大选投票日（周六）
        (date(2025, 8, 9), DayType.HOLIDAY),  # 国庆日（周六）
        (date(2026, 1, 1), DayType.HOLIDAY),
        (date(2026, 2, 17), DayType.HOLIDAY),  # 农历新年
        (date(2026, 2, 18), DayType.HOLIDAY),
        (date(2026, 8, 9), DayType.HOLIDAY),  # 国庆日（周日）
        (date(2026, 8, 10), DayType.HOLIDAY),  # ……次日周一补休
        (date(2026, 6, 1), DayType.HOLIDAY),  # 卫塞节 5 月 31 日周日 → 6 月 1 日补休
        (date(2026, 11, 9), DayType.HOLIDAY),  # 屠妖节 11 月 8 日周日 → 11 月 9 日补休
        (date(2026, 3, 10), DayType.WORKDAY),  # 普通周二
        (date(2026, 3, 14), DayType.WEEKEND),  # 普通周六
        (date(2025, 10, 1), DayType.WORKDAY),  # 中国国庆，新加坡照常上班
    ],
)
def test_singapore_day_types_follow_the_published_holidays(singapore, day: date, expected: DayType) -> None:
    assert singapore.day_type(day) is expected


def test_observed_mondays_say_they_are_observed(singapore) -> None:
    for day in (date(2026, 6, 1), date(2026, 8, 10), date(2026, 11, 9)):
        name = singapore.holiday_name(day)
        assert name is not None and "observed" in name


def test_the_same_day_differs_between_singapore_and_mainland_china(singapore, mainland) -> None:
    # 2026-01-04 周日：中国元旦调休上班，新加坡是周末
    assert mainland.day_type(date(2026, 1, 4)) is DayType.MAKEUP_WORKDAY
    assert singapore.day_type(date(2026, 1, 4)) is DayType.WEEKEND
    # 2026-10-01：中国国庆放假，新加坡普通工作日
    assert mainland.day_type(date(2026, 10, 1)) is DayType.HOLIDAY
    assert singapore.day_type(date(2026, 10, 1)) is DayType.WORKDAY


def test_a_singapore_year_without_data_is_unknown_not_a_workday(singapore) -> None:
    assert singapore.day_type(date(2027, 3, 9)) is None
    assert singapore.day_type(date(2024, 3, 12)) is None
