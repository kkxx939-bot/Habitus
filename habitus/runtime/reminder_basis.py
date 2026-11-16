"""健康面的 ``reminder_basis``：提醒是不是按旧的习惯给的（用户 2026-11-16「预测不要被换代卡住」）。

今天这一代没做成时，预测拍用上一张做完了的表建预期（``runtime.foresight_forming.usable_tables``）；健康面据此报出用的表旧了几天
（旧了 ``REPORT_AFTER_DAYS`` 天起才报），换代追上后不再报。每次查健康面按当地今天现算：哪一天第一次运行，那一刻起就报。那句话会原样显示给用户（插件 doctor 把不健康的检查项列给人看），所以是大白话，不出现内部的词
（用户 2026-11-16「这个提醒要口语化」）。
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

from habitus.foresight import ForesightError
from habitus.runtime.foresight_forming import usable_tables
from habitus.runtime.foresight_opportunities import OpportunityStore

#: 建预期用的表比今天旧几天起才报（用户 2026-11-16「不应该是差两天 应该要差3天才可以」）：差 1、2 天是每天早上换代做成之前的
#: 常态，报了只是噪音。
REPORT_AFTER_DAYS = 3
#: 给人看的那句话里怎么称呼一支（不用"会话 lane"这种内部的词）。
_LANE_WORDS = {"session": "和编程助手对话的提醒", "physical": "穿戴设备记录的提醒"}


def basis_age(store: OpportunityStore, lane: str, today: date, finished: Callable[[date], bool | None]) -> int | None:
    """建预期用的表比今天旧几天（今天的是 0）；一张做完了的表都没有、表读不了是 None（``opportunities``、``generation_marks``
    那几项另报）。"""

    try:
        table = usable_tables(store, lane, today, finished=finished).forming
    except ForesightError:
        return None
    return None if table is None else (today - table.night).days


def told(lane: str, days: int) -> str:
    """给人看的那一句。"""

    who = _LANE_WORDS.get(lane, "提醒")
    return f"{who}：最近 {days} 天的记录还没整理完，现在的提醒是按 {days} 天前的习惯给的。"


__all__ = ["REPORT_AFTER_DAYS", "basis_age", "told"]
