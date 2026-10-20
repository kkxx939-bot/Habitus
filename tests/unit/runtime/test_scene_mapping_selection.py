"""这一晚映射哪几天：今晚的总在，其余按预算；重试账里退避中或已放弃的不占预算（26 号清单 2.4.6）。"""

from __future__ import annotations

from datetime import date, timedelta

from habitus.runtime.scene_night import select_mapping_days

TODAY = date(2026, 10, 18)
DAYS = [TODAY - timedelta(days=offset) for offset in range(6)]  # 最新的先


def test_blocked_days_do_not_take_the_budget() -> None:
    blocked = {DAYS[1], DAYS[2]}
    chosen, waiting = select_mapping_days(DAYS, today=TODAY, budget=2, blocked=lambda day: day in blocked)
    assert chosen == [TODAY, DAYS[3], DAYS[4]]
    assert waiting == [DAYS[1], DAYS[2]]


def test_tonight_is_mapped_even_if_its_key_is_blocked() -> None:
    chosen, waiting = select_mapping_days(DAYS, today=TODAY, budget=1, blocked=lambda _day: True)
    assert chosen == [TODAY]
    assert waiting == DAYS[1:]


def test_without_blocks_the_old_rule_holds() -> None:
    chosen, waiting = select_mapping_days(DAYS, today=TODAY, budget=3, blocked=lambda _day: False)
    assert chosen == DAYS[:4] and waiting == []
