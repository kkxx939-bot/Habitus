"""外部条件：那一刻外面是什么样（裁定 53、59）。可插拔的事实来源，现在接日历，以后照同一协议接天气。

只依赖标准库，不认识任何树；语义树与预测层都从这里取。
"""

from habitus.conditions.calendar import (
    DAY_OF_MONTH_KEY,
    DAY_TYPE_KEY,
    MONTH_END_KEY,
    UNKNOWN_DAY_NOTE,
    WEEKDAY_KEY,
    WEEKDAY_NAMES,
    YES,
    CalendarDataError,
    DayType,
    DayTypeCalendar,
    HolidayTable,
    HolidayYear,
    LocalCalendar,
    day_note,
    month_end,
    nominal_day_type,
)
from habitus.conditions.facts import (
    FACT_KINDS,
    CompositeFacts,
    Conditions,
    FactKey,
    FactProvider,
    NoFacts,
    conditions_of,
    require_local,
)

__all__ = [
    "DAY_OF_MONTH_KEY",
    "DAY_TYPE_KEY",
    "FACT_KINDS",
    "MONTH_END_KEY",
    "WEEKDAY_KEY",
    "WEEKDAY_NAMES",
    "UNKNOWN_DAY_NOTE",
    "YES",
    "CalendarDataError",
    "CompositeFacts",
    "Conditions",
    "DayType",
    "DayTypeCalendar",
    "FactKey",
    "FactProvider",
    "HolidayTable",
    "HolidayYear",
    "LocalCalendar",
    "NoFacts",
    "conditions_of",
    "day_note",
    "month_end",
    "nominal_day_type",
    "require_local",
]
