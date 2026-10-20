"""间隔按日历日；跨空白的按漏看概率加权不剔除；被提醒结尾的截尾；成簇沿用旧规则（在险 ≥2）；久别样本少不判。"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from habitus.temporal import (
    cluster_at,
    coverage_of,
    day_intervals,
    episodes_of,
    median_days,
    occurrence_days,
)
from tests.unit.temporal.fixtures import at, gap, record, series

WED = date(2026, 1, 7)
WINDOW = 2700


def _intervals(records, *, cutoff, gaps=None, share=None):
    days = occurrence_days(episodes_of(records, window_seconds=WINDOW))
    coverage = coverage_of(series(records, cutoff=cutoff, gaps=gaps), "session", slot_minutes=15)
    return days, day_intervals(days, coverage, profile=None, occurrence_day_share=share)


def _weekly(gaps_in_days, *, reminded_at=()):
    day, found = WED, [record(at(WED, 17, 0))]
    for index, gap_days in enumerate(gaps_in_days, start=1):
        day += timedelta(days=gap_days)
        found.append(record(at(day, 17, 0), reminded=index in reminded_at))
    return found, day


def test_interval_is_counted_in_calendar_days() -> None:
    records = [record(at(WED, 17, 0)), record(at(WED + timedelta(days=7), 16, 59))]  # 6.99 天
    _days, found = _intervals(records, cutoff=WED + timedelta(days=8))
    assert [item.days for item in found] == [7]


def test_an_interval_across_a_gap_is_kept_with_its_seen_weight() -> None:
    records, last = _weekly([7, 7])
    blank = [gap(at(WED + timedelta(days=9), 0, 0), at(WED + timedelta(days=10), 0, 0))]  # 第二段中间整天没在看
    _days, found = _intervals(records, cutoff=last + timedelta(days=1), gaps=blank, share=0.2)
    assert len(found) == 2  # 不剔除
    assert found[0].seen == 1.0 and found[1].seen == pytest.approx(0.8)


def test_reminded_ending_is_censored() -> None:
    records, last = _weekly([7, 7, 7], reminded_at={2})
    _days, found = _intervals(records, cutoff=last + timedelta(days=1))
    assert [item.censored for item in found] == [False, True, False]
    reading = cluster_at(found, 7)
    assert reading is not None
    assert len(reading.hits) == 2 and reading.opportunities == 2  # 截尾那段不算命中，也从机会里拿掉
    assert median_days(found) == 7.0


def test_cluster_needs_two_at_risk_and_handles_a_skipped_week() -> None:
    records, last = _weekly([7, 14, 7, 28])
    _days, found = _intervals(records, cutoff=last + timedelta(days=1))
    skipped = cluster_at(found, 14)
    assert skipped is not None and skipped.cluster == (14,) and skipped.opportunities == 2  # 跳一周照样有读数
    assert cluster_at(found, 28) is None  # 不短于 28 天的只有一段：在险不到 2 个
    assert cluster_at(found, 10) is None  # 历史上没隔过 10 天
    # 在险按扣掉截尾之后数：两段 7 天，一段是被提醒才做的，剩一段不够读数（2026-10-18 四方评审）
    records, last = _weekly([7, 7], reminded_at={2})
    _days, half = _intervals(records, cutoff=last + timedelta(days=1))
    assert [item.censored for item in half] == [False, True] and cluster_at(half, 7) is None


def test_common_interval() -> None:
    from habitus.temporal import common_interval_days, gap_days

    days = [WED + timedelta(days=7 * week) for week in range(4)]
    assert gap_days(days + [days[0]]) == (7, 7, 7)
    assert common_interval_days(days) == 7.0
    assert common_interval_days(days[:1]) is None
    # 与 median_days 同一个下中位数：7 与 28 → 7，不是 17.5
    assert common_interval_days([WED, WED + timedelta(days=7), WED + timedelta(days=35)]) == 7.0
