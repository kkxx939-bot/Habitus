"""当地日历：这一天在当地是什么日子（周几、几号、是不是月末、日型）。裁定 59。

日型（工作日 / 周末 / 节假日 / 调休上班）只认**当地官方公布的放假安排**，不让模型判、不从行为推：日型是摆在判断者面前的
外部事实，拿行为推出来的日型再去解释行为是循环论证。"当地"是**这个人所在的地方**（``config.locale``），不是服务部署在哪：
中国按国务院通知（有调休上班），新加坡按人力部公布的公众假期（落在周日的补周一，记成节假日），美国按联邦 / 州的假日表
（州用 ``US-CA`` 这种地区码）。周末是哪几天也随地区，写在数据里。

- **周几、几号、月末**是日期的确定函数，任何一天都答得出。
- **日型**要那一年的放假安排；没有那一年的数据就答不出（``None``）——不知道不是"普通工作日"（外部条件契约第 2 条）。
  建预期时需要一个名义值的，由调用方用 ``nominal_day_type``（按配置的名义周末，``locale.nominal_weekend``）；做统计时未知的
  日子从日型这一族里扣掉。
- **不存每天的结果**：日型是"日期 + 数据版本"的确定函数，调休表还会改；读时算，出处只记 ``version``。

本模块只认日期：不读文件、不认识树（文件格式与读取在 ``conditions.calendar_data``）。
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from habitus.conditions.facts import Conditions, FactKey, require_local

WEEKDAY_NAMES = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")
#: 地区码：ISO 3166-1 两位国家码，或带 ISO 3166-2 的下一级（美国的州按州放假：US-CA）。
REGION_PATTERN = re.compile(r"[A-Z]{2}(-[A-Z0-9]{1,3})?")
WEEKDAY_KEY = "日历.周几"
DAY_OF_MONTH_KEY = "日历.几号"
MONTH_END_KEY = "日历.月末"
DAY_TYPE_KEY = "日历.日型"
YES = "是"
NO = "否"


class DayType(str, Enum):
    WORKDAY = "工作日"
    WEEKEND = "周末"
    HOLIDAY = "节假日"
    MAKEUP_WORKDAY = "调休上班"

    @property
    def working(self) -> bool:
        return self in (DayType.WORKDAY, DayType.MAKEUP_WORKDAY)


class CalendarDataError(ValueError):
    """放假安排的数据自相矛盾（日子不在那一年、同一天既放假又上班……）。"""


@dataclass(frozen=True)
class HolidayYear:
    """一年的放假安排：周末是周几（0=周一 … 6=周日）、放假的日子（日子 → 节名）、周末里要上班的日子（调休上班，没有这回事的
    地区留空）；``source`` 是官方出处，``version`` 是它的版本。"""

    region: str
    year: int
    source: str
    version: str
    weekend: frozenset[int] = frozenset({5, 6})
    holidays: Mapping[date, str] = field(default_factory=dict)
    makeup_workdays: frozenset[date] = frozenset()

    def __post_init__(self) -> None:
        if not isinstance(self.region, str) or REGION_PATTERN.fullmatch(self.region) is None:
            raise CalendarDataError("calendar region must be an ISO 3166 code such as CN, SG or US-CA")
        weekend = frozenset(self.weekend)
        if not weekend or any(
            isinstance(day, bool) or not isinstance(day, int) or not 0 <= day <= 6 for day in weekend
        ):
            raise CalendarDataError("calendar weekend names weekdays 0–6")
        object.__setattr__(self, "weekend", weekend)
        if isinstance(self.year, bool) or not isinstance(self.year, int):
            raise CalendarDataError("calendar year must be an integer")
        for label in ("source", "version"):
            value = getattr(self, label)
            if not isinstance(value, str) or not value.strip():
                raise CalendarDataError(f"calendar {label} must be non-empty text")
        holidays = dict(self.holidays)
        for day, name in holidays.items():
            if not isinstance(day, date) or isinstance(day, datetime) or day.year != self.year:
                raise CalendarDataError(f"holiday {day!r} is not a date in {self.year}")
            if not isinstance(name, str) or not name.strip():
                raise CalendarDataError(f"holiday {day} needs a name")
        makeup = frozenset(self.makeup_workdays)
        for day in makeup:
            if not isinstance(day, date) or isinstance(day, datetime) or day.year != self.year:
                raise CalendarDataError(f"makeup workday {day!r} is not a date in {self.year}")
        both = sorted(makeup & holidays.keys())
        if both:
            raise CalendarDataError(f"{both[0]} is both a holiday and a makeup workday")
        object.__setattr__(self, "holidays", MappingProxyType(holidays))
        object.__setattr__(self, "makeup_workdays", makeup)


class HolidayTable:
    """几年的放假安排。某一年不在表里，那一年的日型就答不出。"""

    def __init__(self, years: Mapping[int, HolidayYear] | None = None) -> None:
        found = dict(years or {})
        for year, item in found.items():
            if not isinstance(item, HolidayYear) or item.year != year:
                raise CalendarDataError(f"holiday table entry {year} does not hold that year's data")
        regions = {item.region for item in found.values()}
        if len(regions) > 1:
            raise CalendarDataError(f"holiday table mixes regions: {sorted(regions)}")
        self.years: Mapping[int, HolidayYear] = MappingProxyType(found)

    @property
    def version(self) -> str:
        return "+".join(f"{item.region}{year}@{item.version}" for year, item in sorted(self.years.items())) or "none"

    def day_type(self, day: date) -> DayType | None:
        item = self.years.get(day.year)
        if item is None:
            return None
        if day in item.holidays:
            return DayType.HOLIDAY
        if day in item.makeup_workdays:
            return DayType.MAKEUP_WORKDAY
        return nominal_day_type(day, weekend=item.weekend)

    def holiday_name(self, day: date) -> str | None:
        item = self.years.get(day.year)
        return None if item is None else item.holidays.get(day)


def nominal_day_type(day: date, *, weekend: Iterable[int]) -> DayType:
    """名义日型：不在 ``weekend`` 里的日子算工作日。有数据时用那一年的周末；缺数据时只给建预期的兜底用（周末取配置的
    ``locale.nominal_weekend``），统计里不用它（统计里未知就是未知）。"""

    return DayType.WEEKEND if day.weekday() in set(weekend) else DayType.WORKDAY


def month_end(day: date) -> bool:
    return (day + timedelta(days=1)).month != day.month


@runtime_checkable
class DayTypeCalendar(Protocol):
    """当地日历的读口。``describe`` 是给判断者读的一句话（"国庆节，法定放假"），平常的日子与答不出的日子都没话说。"""

    @property
    def version(self) -> str: ...

    def day_type(self, day: date) -> DayType | None: ...

    def describe(self, day: date) -> str | None: ...

    def of_day(self, day: date) -> Conditions: ...

    def for_expectations(self, day: date) -> Conditions: ...


class LocalCalendar:
    """按放假安排答日型的当地日历，同时是一个外部条件提供者（``FactProvider``）。表是空的就只答周几、几号、月末。"""

    def __init__(self, table: HolidayTable | None = None, *, nominal_weekend: Iterable[int] = (5, 6)) -> None:
        self.table = table if table is not None else HolidayTable()
        self.nominal_weekend = tuple(sorted(set(nominal_weekend)))
        if not self.nominal_weekend or any(not 0 <= day <= 6 for day in self.nominal_weekend):
            raise CalendarDataError("nominal_weekend names weekdays 0–6")

    @property
    def version(self) -> str:
        return f"calendar:{self.table.version}"

    def keys(self) -> tuple[FactKey, ...]:
        return (
            FactKey(DAY_OF_MONTH_KEY, "类别"),
            FactKey(DAY_TYPE_KEY, "类别"),
            FactKey(MONTH_END_KEY, "类别"),
            FactKey(WEEKDAY_KEY, "类别"),
        )

    def at(self, when: datetime) -> Conditions:
        return self.of_day(require_local(when).date())

    def of_day(self, day: date) -> Conditions:
        """这一天的日历条件；日型答不出的那一天不出现日型这个键。"""

        if not isinstance(day, date):
            raise TypeError("day must be a date")
        found = {
            WEEKDAY_KEY: WEEKDAY_NAMES[day.weekday()],
            DAY_OF_MONTH_KEY: str(day.day),
            MONTH_END_KEY: YES if month_end(day) else NO,
        }
        kind = self.table.day_type(day)
        if kind is not None:
            found[DAY_TYPE_KEY] = kind.value
        return tuple(sorted(found.items()))

    def for_expectations(self, day: date) -> Conditions:
        """建预期用的日历条件：日型答不出时按名义周末给一个（统计里不用它，统计用 ``of_day``，未知就是未知）。"""

        found = dict(self.of_day(day))
        if DAY_TYPE_KEY not in found:
            found[DAY_TYPE_KEY] = nominal_day_type(day, weekend=self.nominal_weekend).value
        return tuple(sorted(found.items()))

    def day_type(self, day: date) -> DayType | None:
        if not isinstance(day, date):
            raise TypeError("day must be a date")
        return self.table.day_type(day)

    def describe(self, day: date) -> str | None:
        kind = self.day_type(day)
        if kind is DayType.HOLIDAY:
            name = self.table.holiday_name(day)
            return f"{name}，法定放假" if name else "法定放假"
        if kind is DayType.MAKEUP_WORKDAY:
            return "调休上班日"
        return None


def day_note(calendar: DayTypeCalendar, day: date) -> str | None:
    """给判断者看的那一句：放假、调休照说；日型答得出的平常日子不说；答不出就说"日型未知"（缺放假数据，不当成工作日）。"""

    note = calendar.describe(day)
    if note is not None:
        return note
    return None if calendar.day_type(day) is not None else UNKNOWN_DAY_NOTE


#: 日型答不出时给判断者看的说法。
UNKNOWN_DAY_NOTE = "日型未知"


__all__ = [
    "DAY_OF_MONTH_KEY",
    "DAY_TYPE_KEY",
    "MONTH_END_KEY",
    "NO",
    "REGION_PATTERN",
    "UNKNOWN_DAY_NOTE",
    "WEEKDAY_KEY",
    "WEEKDAY_NAMES",
    "YES",
    "CalendarDataError",
    "DayType",
    "DayTypeCalendar",
    "HolidayTable",
    "HolidayYear",
    "LocalCalendar",
    "month_end",
    "day_note",
    "nominal_day_type",
]
