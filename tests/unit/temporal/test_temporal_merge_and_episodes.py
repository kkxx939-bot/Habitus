"""扎堆合并按开始时刻的秒数（45 分钟）链式连接；一次是不是自然发生看第一条；按天归的发生。"""

from __future__ import annotations

from datetime import date

import pytest

from habitus.temporal import absolute_slot, episodes_of, merge_starts, occurrence_days, slot_index
from tests.unit.temporal.fixtures import at, record

WED = date(2026, 1, 7)
WINDOW = 2700


def test_merge_is_by_seconds_and_chains() -> None:
    assert merge_starts([at(WED, 17, 0), at(WED, 17, 44)], window_seconds=WINDOW) == ((0, 1),)
    # 46 分钟：按秒不合（按"槽号差 ≤3"会合，17:00 与 17:46 是第 68、70 槽）
    assert merge_starts([at(WED, 17, 0), at(WED, 17, 46)], window_seconds=WINDOW) == ((0,), (1,))
    chained = [at(WED, 17, 0), at(WED, 17, 40), at(WED, 18, 20)]
    assert merge_starts(chained, window_seconds=WINDOW) == ((0, 1, 2),)
    assert merge_starts([at(WED, 17, 0), at(WED, 18, 30)], window_seconds=WINDOW) == ((0,), (1,))
    with pytest.raises(ValueError):
        merge_starts([at(WED, 18, 0), at(WED, 17, 0)], window_seconds=WINDOW)


def test_an_episode_is_natural_when_its_first_record_was_not_prompted() -> None:
    records = [
        record(at(WED, 17, 0), reminded=True),
        record(at(WED, 17, 10)),  # 同一次里接着做的，跟着第一条走
        record(at(date(2026, 1, 14), 17, 0)),
    ]
    found = episodes_of(records, window_seconds=WINDOW)
    assert [item.natural for item in found] == [False, True]
    assert len(found[0].uris) == 2 and found[0].last_seen == at(WED, 17, 20)
    days = occurrence_days(found)
    assert [(item.day, item.count, item.natural) for item in days] == [(WED, 1, 0), (date(2026, 1, 14), 1, 1)]
    assert days[0].natural_uri is None and days[1].natural_at == at(date(2026, 1, 14), 17, 0)


def test_episodes_merge_a_group_of_kinds() -> None:
    """汇总概念：几个成员类的发生合在一起再扎堆。"""

    found = episodes_of([record(at(WED, 9)), record(at(WED, 9, 20), "s-k0002")], window_seconds=WINDOW)
    assert len(found) == 1 and len(found[0].uris) == 2


def test_slot_arithmetic() -> None:
    assert slot_index(at(WED, 17, 14), slot_minutes=15) == 68
    assert absolute_slot(at(WED, 0, 0), slot_minutes=15) == WED.toordinal() * 96


def test_series_gives_one_kinds_timeline_without_pending_records() -> None:
    from dataclasses import replace

    from tests.unit.temporal.fixtures import series

    pending = replace(record(at(WED, 9), "s-待定"), classified=False)
    found = series([record(at(WED, 17)), record(at(WED, 18), "s-k0002"), pending], cutoff=date(2026, 1, 8))
    assert [item.kind_token for item in found.records_of("s-k0001")] == ["s-k0001"]
    assert found.records_of("s-待定") == ()
