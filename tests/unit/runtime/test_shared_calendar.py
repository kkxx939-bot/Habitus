"""组合根里那一份当地日历：语义树与预测层共用，每晚开头重读放假数据，年中补上的文件不用重启（2026-10-18 四方评审）。"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

from habitus.conditions import DAY_TYPE_KEY, DayType, DayTypeCalendar
from habitus.conditions.calendar_data import bundled_holiday_directory, load_holiday_table
from habitus.config.locale import LocaleConfig
from habitus.runtime.conditions import SharedCalendar, local_calendar


def _year(year: int) -> dict[str, object]:
    return {
        "region": "CN",
        "year": year,
        "source": "测试用的假想安排",
        "version": "t1",
        "weekend": [5, 6],
        "holidays": {f"{year}-01-01": "元旦"},
        "makeup_workdays": [],
    }


def test_a_year_added_later_is_picked_up_by_the_nightly_refresh(tmp_path: Path) -> None:
    calendar = SharedCalendar("CN", str(tmp_path), nominal_weekend=(5, 6))
    assert isinstance(calendar, DayTypeCalendar)
    new_year = date(2099, 1, 1)
    assert calendar.day_type(new_year) is None and calendar.problems == ()
    assert dict(calendar.for_expectations(new_year))[DAY_TYPE_KEY] == DayType.WORKDAY.value  # 没数据：按名义周末
    (tmp_path / "2099.json").write_text(json.dumps(_year(2099)), encoding="utf-8")
    assert calendar.refresh() is True  # 版本变了
    assert calendar.day_type(new_year) is DayType.HOLIDAY and calendar.describe(new_year) == "元旦，法定放假"
    assert calendar.version.endswith("2099@t1") and calendar.refresh() is False
    (tmp_path / "2100.json").write_text("{", encoding="utf-8")
    calendar.refresh()
    assert calendar.day_type(new_year) is DayType.HOLIDAY and len(calendar.problems) == 1  # 坏的那一年只跳过


def test_each_day_is_labelled_by_where_he_was_that_day(tmp_path: Path) -> None:
    """搬家后过去的日子不按新地区重打标签：那一天在别的地区、又没有那个地区的放假数据，日型就是答不出（S8 8.6）。"""

    (tmp_path / "2099.json").write_text(json.dumps(_year(2099)), encoding="utf-8")
    moved = date(2099, 6, 1)
    calendar = SharedCalendar(
        "CN", str(tmp_path), nominal_weekend=(5, 6), region_on=lambda day: "CN" if day < moved else "US"
    )
    assert calendar.day_type(date(2099, 1, 1)) is DayType.HOLIDAY  # 那天在中国：按中国的放假安排
    later = date(2099, 10, 1)
    assert calendar.day_type(later) is None and DAY_TYPE_KEY not in dict(
        calendar.of_day(later)
    )  # 在美国、没有美国的数据：不知道
    assert dict(calendar.of_day(later))["日历.周几"] == "周四"  # 周几照答
    assert dict(calendar.for_expectations(later))[DAY_TYPE_KEY] == DayType.WORKDAY.value  # 建预期按名义周末兜底


def test_the_bundled_mainland_data_follows_the_state_council_notices() -> None:
    """随包附带中国大陆 2025、2026 两年（裁定 80 第 2 条）：严格读法载入，日子与国务院办公厅通知原文一致
    （国办发明电〔2024〕12号、国办发明电〔2025〕7号）。"""

    directory = bundled_holiday_directory("CN")
    assert directory is not None
    table = load_holiday_table("CN", directory)
    assert sorted(table.years) == [2025, 2026]
    first, second = table.years[2025], table.years[2026]
    assert "国办发明电〔2024〕12号" in first.source and "国办发明电〔2025〕7号" in second.source
    assert (len(first.holidays), len(first.makeup_workdays)) == (28, 5)
    assert (len(second.holidays), len(second.makeup_workdays)) == (33, 6)
    # 2025：国庆、中秋合并放 8 天，10 月 11 日（周六）上班；春节 1 月 28 日至 2 月 4 日，1 月 26 日（周日）上班
    assert (
        table.day_type(date(2025, 10, 8)) is DayType.HOLIDAY
        and table.holiday_name(date(2025, 10, 8)) == "国庆节、中秋节"
    )
    assert table.day_type(date(2025, 10, 11)) is DayType.MAKEUP_WORKDAY
    assert (
        table.day_type(date(2025, 1, 26)) is DayType.MAKEUP_WORKDAY
        and table.day_type(date(2025, 2, 4)) is DayType.HOLIDAY
    )
    # 2026：春节 2 月 15 日至 23 日；国庆 10 月 1 日至 7 日，10 月 10 日（周六）上班；10 月 8 日是普通工作日
    assert (
        table.day_type(date(2026, 2, 23)) is DayType.HOLIDAY
        and table.day_type(date(2026, 2, 28)) is DayType.MAKEUP_WORKDAY
    )
    assert table.day_type(date(2026, 10, 10)) is DayType.MAKEUP_WORKDAY
    assert table.day_type(date(2026, 10, 8)) is DayType.WORKDAY
    assert bundled_holiday_directory("US") is None  # 没随包带的地区；SG 自 W3 起随包附带


def test_the_calendar_reads_the_bundled_data_unless_a_directory_is_configured(tmp_path: Path) -> None:
    """``locale.calendar_path`` 为空读随包附带的；配了目录只读配的那个；随包没带的地区日型答不出。"""

    national_day = date(2026, 10, 1)
    # W3 起默认地区是 SG：这里测的是"读随包的 CN 那一份"，地区显式写 CN
    default = local_calendar(SimpleNamespace(locale=LocaleConfig(region="CN")))  # type: ignore[arg-type]
    assert default.day_type(national_day) is DayType.HOLIDAY and default.problems == ()
    assert default.version.endswith("CN2025@2024-11-12+CN2026@2025-11-04")
    singapore = local_calendar(SimpleNamespace(locale=LocaleConfig()))  # type: ignore[arg-type]
    assert singapore.version.endswith("SG2025@2026-06-19+SG2026@2026-06-19") and singapore.problems == ()
    assert singapore.day_type(national_day) is DayType.WORKDAY  # 不写 locale：读随包的 SG，10 月 1 日照常上班
    (tmp_path / "2026.json").write_text(json.dumps(_year(2026)), encoding="utf-8")
    configured = local_calendar(SimpleNamespace(locale=LocaleConfig(region="CN", calendar_path=str(tmp_path))))  # type: ignore[arg-type]
    assert configured.day_type(national_day) is DayType.WORKDAY  # 配的那一份里 10 月 1 日不放假：只认配的
    assert configured.day_type(date(2026, 1, 1)) is DayType.HOLIDAY and configured.day_type(date(2025, 1, 1)) is None
    elsewhere = local_calendar(SimpleNamespace(locale=LocaleConfig(region="US")))  # type: ignore[arg-type]
    assert elsewhere.day_type(national_day) is None
