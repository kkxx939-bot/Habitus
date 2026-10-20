"""组合根里那一份当地日历：语义树与预测层共用，每晚开头重读放假数据，年中补上的文件不用重启（2026-10-18 四方评审）。"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from habitus.conditions import DAY_TYPE_KEY, DayType, DayTypeCalendar
from habitus.runtime.conditions import SharedCalendar


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
