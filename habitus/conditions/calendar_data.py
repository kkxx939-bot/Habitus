"""放假安排的数据文件：格式与读取（裁定 59）。

放假数据是**外接组件**：不随代码内置，由 ``locale.calendar_path`` 指向的目录提供（每年一份 ``<年>.json``，照这个人所在地
当年官方公布的放假安排录入、注明出处），和真实模型对照实验一起接入。没给目录就是空表：周几、几号、月末照常答，日型答不出（未知）。
地区码与 ``locale.region`` 一致（``CN``、``SG``、``US-CA``）。格式严格，未知字段与重复日子都拒：

    {
      "region": "CN",
      "year": 2026,
      "source": "国务院办公厅关于 2026 年部分节假日安排的通知（文号）",
      "version": "发布日期或修订号",
      "weekend": [5, 6],
      "holidays": {"2026-01-01": "元旦", ...},
      "makeup_workdays": ["2026-01-04", ...]
    }

新加坡、美国这类没有调休上班的地区 ``makeup_workdays`` 填 ``[]``；落在周日补周一的假期直接写进 ``holidays``。
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

from habitus.conditions.calendar import CalendarDataError, HolidayTable, HolidayYear

_FIELDS = {"region", "year", "source", "version", "weekend", "holidays", "makeup_workdays"}


def parse_holiday_year(text: str) -> HolidayYear:
    try:
        raw = json.loads(text, object_pairs_hook=_no_duplicates)
    except (json.JSONDecodeError, ValueError) as exc:
        raise CalendarDataError(f"calendar data is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict) or set(raw) != _FIELDS:
        raise CalendarDataError(f"calendar data must have exactly the fields {sorted(_FIELDS)}")
    holidays = raw["holidays"]
    makeup = raw["makeup_workdays"]
    if not isinstance(holidays, dict) or not isinstance(makeup, list):
        raise CalendarDataError("holidays must be an object and makeup_workdays a list")
    if len(set(makeup)) != len(makeup):
        raise CalendarDataError("makeup_workdays lists a day twice")
    weekend = raw["weekend"]
    if not isinstance(weekend, list) or len(set(weekend)) != len(weekend):
        raise CalendarDataError("weekend must be a list of distinct weekdays")
    return HolidayYear(
        region=raw["region"],
        year=raw["year"],
        source=raw["source"],
        version=raw["version"],
        weekend=frozenset(weekend),
        holidays={_day(key): value for key, value in holidays.items()},
        makeup_workdays=frozenset(_day(item) for item in makeup),
    )


def load_holiday_table(region: str, directory: str | Path | None = None) -> HolidayTable:
    """``directory`` 里这个地区的全部年份；不给目录就是空表（日型都答不出）。"""

    if directory is None:
        return HolidayTable()
    folder = Path(directory)
    if not folder.is_dir():
        raise CalendarDataError(f"calendar directory {folder} does not exist")
    years: dict[int, HolidayYear] = {}
    for path in sorted(folder.glob("*.json")):
        item = parse_holiday_year(path.read_text(encoding="utf-8"))
        if item.region != region or path.stem != str(item.year):
            raise CalendarDataError(f"{path.name} does not hold {region} data for the year in its name")
        years[item.year] = item
    return HolidayTable(years)


def read_holiday_table(region: str, directory: str | Path | None = None) -> tuple[HolidayTable, tuple[str, ...]]:
    """组合根用的宽松读法：读得了的年份进表，读不了的那一年跳过、问题交出去（那一年日型未知）。外接文件写错一个字
    不许拖垮整个进程（2026-10-18 四方评审）；严格读法 ``load_holiday_table`` 留给录入时的自检。"""

    if directory is None:
        return HolidayTable(), ()
    folder = Path(directory)
    if not folder.is_dir():
        return HolidayTable(), (f"calendar directory {folder} does not exist",)
    years: dict[int, HolidayYear] = {}
    problems: list[str] = []
    for path in sorted(folder.glob("*.json")):
        try:
            item = parse_holiday_year(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, CalendarDataError) as exc:
            problems.append(f"{path.name}: {exc}")
            continue
        if item.region != region or path.stem != str(item.year):
            problems.append(f"{path.name} does not hold {region} data for the year in its name")
            continue
        years[item.year] = item
    return HolidayTable(years), tuple(problems)


def _day(value: Any) -> date:
    if not isinstance(value, str):
        raise CalendarDataError(f"calendar day {value!r} must be an ISO date string")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise CalendarDataError(f"calendar day {value!r} is not an ISO date") from exc


def _no_duplicates(pairs: list[tuple[str, Any]]) -> Mapping[str, Any]:
    found: dict[str, Any] = {}
    for key, value in pairs:
        if key in found:
            raise ValueError(f"duplicate key {key!r}")
        found[key] = value
    return found


__all__ = ["load_holiday_table", "parse_holiday_year", "read_holiday_table"]
