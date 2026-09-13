"""当地日历：这一天在当地是什么日子（补班、节假日、还是普通的周几）。

**为什么住在语义层**：日型是"这一天"的一个外部约束事实——确定性的、由日历数据给出的，不是从
行为里推出来的状态。时间预测树是零语义的钟面计数，键就是 (周几 × 槽)，它不该知道日历，也
**不按日型换键**（补班的周六仍然按周六查：那天他到底像周一上班还是像周六，交给判断者对着当天
已经发生的几步与相似情景去判，系统不替他决定）。日型放在这里，三个历史维度就都带得上它，
"上次补班日这个时段他做了什么"才问得出来；放在预测层只能得到"今天补班"一句，没有历史可比。

**读时算，不进文档**：日历是 ``date → 日型`` 的确定函数，而调休表每年发布、还会改。存进情景
文档就会留下"归组那天的旧值"，成了第二个真源；读侧每次按当前日历重新标注，永远与日历一致。
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable


@runtime_checkable
class DayTypeCalendar(Protocol):
    """一天一句话，或者没话说。

    返回的文本原样进上下文视图，给判断者读——所以写"补班日，按周一上班"这种人看得懂的句子，
    不要写枚举码。本层不规定词表：当地日历的说法各地不同，那是数据的事，不是接口的事。
    """

    def describe(self, day: date) -> str | None: ...


class NominalCalendar:
    """没有当地日历数据时的实现：对任何一天都不多说一句话。

    这是**显式的零修正**，不是占位：没有日历文件时，"今天是普通的周三"本来就是我们能说的
    全部，视图上的 ``day_note`` 为空表达的正是这个。
    """

    def describe(self, day: date) -> str | None:
        if not isinstance(day, date):
            raise TypeError("day must be a date")
        return None


__all__ = ["DayTypeCalendar", "NominalCalendar"]
