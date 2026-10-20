"""外部条件在组合根的装配（裁定 59）：语义树与预测层用**同一个**当地日历，按 ``config.locale`` 读放假安排。

放假数据是外接组件、每年人手录一次，所以：组合根只建一份、两层共用（``SharedCalendar``）；每晚开头重读一遍目录
（``refresh``，用户 10-18：节假日在夜批时判断），年中补上明年的文件不用重启；某一年的文件读不了只跳过那一年（那一年日型未知），
问题挂在 ``problems`` 上给健康面，不拖垮进程（2026-10-18 四方评审）。

**每一天按他那天人在哪儿打标签**（裁定 59 第 7 条）：``region_on`` 给出某一天他在哪个地区，那天的周几、日型按那个地区的日历答；
没有那个地区放假数据的日子日型答不出（不知道），绝不拿别的地区的放假安排去答——不然改了配置里的地区，过去在新加坡的日子会被按
美国的假期重新打标签，历史就被改写了。"那一天在哪个地区"从哪来还没定（待办「人到外地日历是否跟着换」），不给就是一直在配置的地区。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

from habitus.conditions import Conditions, DayType, LocalCalendar
from habitus.conditions.calendar_data import read_holiday_table
from habitus.config import HabitusConfig


class SharedCalendar:
    """一个可重读的当地日历：答法全部转给当前那一份 ``LocalCalendar``，``refresh`` 换成重读出来的一份。"""

    def __init__(
        self,
        region: str,
        directory: str | None,
        *,
        nominal_weekend: tuple[int, ...],
        region_on: Callable[[date], str] | None = None,
    ) -> None:
        self.region = region
        self.directory = directory
        self._weekend = tuple(nominal_weekend)
        self._current = LocalCalendar(nominal_weekend=self._weekend)
        #: 没有放假数据的地区用的日历：周几、几号、月末照答，日型答不出。
        self._elsewhere = LocalCalendar(nominal_weekend=self._weekend)
        self.region_on = region_on
        self.problems: tuple[str, ...] = ()
        self.refresh()

    def refresh(self) -> bool:
        """重读放假数据；数据版本变了返回 True。"""

        table, problems = read_holiday_table(self.region, self.directory)
        before = self._current.version
        self._current = LocalCalendar(table, nominal_weekend=self._weekend)
        self.problems = problems
        return self._current.version != before

    @property
    def version(self) -> str:
        return self._current.version

    @property
    def nominal_weekend(self) -> tuple[int, ...]:
        return self._current.nominal_weekend

    def day_type(self, day: date) -> DayType | None:
        return self._on(day).day_type(day)

    def describe(self, day: date) -> str | None:
        return self._on(day).describe(day)

    def of_day(self, day: date) -> Conditions:
        return self._on(day).of_day(day)

    def for_expectations(self, day: date) -> Conditions:
        return self._on(day).for_expectations(day)

    def _on(self, day: date) -> LocalCalendar:
        """那一天他所在地区的日历：在配置的地区就是读到的放假数据，在别处就是没有放假数据的那一份。"""

        if self.region_on is None or self.region_on(day) == self.region:
            return self._current
        return self._elsewhere


def local_calendar(config: HabitusConfig) -> SharedCalendar:
    """放假安排是外接组件：``locale.calendar_path`` 给了目录就读它；没给就只答周几、几号、月末，日型未知。"""

    return SharedCalendar(
        config.locale.region, config.locale.calendar_path, nominal_weekend=config.locale.nominal_weekend
    )


__all__ = ["SharedCalendar", "local_calendar"]
