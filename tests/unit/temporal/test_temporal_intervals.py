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
    overdue_at,
)
from tests.unit.temporal.fixtures import at, gap, record, series, with_sessions

WED = date(2026, 1, 7)
WINDOW = 2700
LINE = 0.67  # 账的线 θ（产品默认 1 − 0.33）


def _intervals(records, *, cutoff, gaps=None, share=None):
    days = occurrence_days(episodes_of(records, window_seconds=WINDOW))
    coverage = coverage_of(
        series(with_sessions(records, cutoff=cutoff), cutoff=cutoff, gaps=gaps), "session", slot_minutes=15
    )
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
    reading = cluster_at(found, 7, line=LINE)
    assert reading is not None
    assert len(reading.hits) == 2 and reading.opportunities == 2  # 截尾那段不算命中，也从机会里拿掉
    assert median_days(found) == 7.0


def test_cluster_needs_two_at_risk_and_handles_a_skipped_week() -> None:
    records, last = _weekly([7, 14, 7, 28])
    _days, found = _intervals(records, cutoff=last + timedelta(days=1))
    skipped = cluster_at(found, 14, line=LINE)
    assert skipped is not None and skipped.cluster == (14,) and skipped.opportunities == 2  # 跳一周照样有读数
    assert cluster_at(found, 28, line=LINE) is None  # 不短于 28 天的只有一段：在险不到 2 个
    assert cluster_at(found, 10, line=LINE) is None  # 历史上没隔过 10 天
    # 在险按扣掉截尾之后数：两段 7 天，一段是被提醒才做的，剩一段不够读数（2026-10-18 四方评审）
    records, last = _weekly([7, 7], reminded_at={2})
    _days, half = _intervals(records, cutoff=last + timedelta(days=1))
    assert [item.censored for item in half] == [False, True] and cluster_at(half, 7, line=LINE) is None


def test_common_interval() -> None:
    from habitus.temporal import common_interval_days, gap_days

    days = [WED + timedelta(days=7 * week) for week in range(4)]
    assert gap_days(days + [days[0]]) == (7, 7, 7)
    assert common_interval_days(days) == 7.0
    assert common_interval_days(days[:1]) is None
    # 与 median_days 同一个下中位数：7 与 28 → 7，不是 17.5
    assert common_interval_days([WED, WED + timedelta(days=7), WED + timedelta(days=35)]) == 7.0


def test_the_weighted_median_of_two_equal_intervals_is_the_lower_one() -> None:
    """两个没截尾、看全了的间隔 2 天与 4 天：加权中位数取下中位数 2（累积权重恰好到一半就停），与 ``common_interval_days`` 同口径。"""

    records = [record(at(WED, 17)), record(at(WED + timedelta(days=2), 17)), record(at(WED + timedelta(days=6), 17))]
    _days, intervals = _intervals(records, cutoff=WED + timedelta(days=7))
    assert sorted(item.days for item in intervals) == [2, 4] and all(item.seen == 1.0 for item in intervals)
    assert median_days(intervals) == 2.0


def _daily(gaps_in_days):
    day, found = WED, [record(at(WED, 8))]
    for gap_days in gaps_in_days:
        day += timedelta(days=gap_days)
        found.append(record(at(day, 8)))
    return found, day


def test_the_cluster_reaches_the_next_day_only_when_the_day_itself_falls_short_of_the_line() -> None:
    """裁定 89 第 1 条：到离上次 1 天为止他来了的超过线 θ（0.67），簇只有 1——那天没来就是漏做；没超过才把第 2 天接进来。
    整簇的身份（``block``）照旧按全部连着的间隔算。"""

    records, last = _daily([1, 2] * 3 + [1] * 4)  # 7 次隔 1 天、3 次隔 2 天：7 / 10 = 0.70
    _days, found = _intervals(records, cutoff=last + timedelta(days=1))
    tight = cluster_at(found, 1, line=LINE)
    assert tight is not None and tight.cluster == (1,) and tight.block == (1, 2)
    assert len(tight.hits) == 7 and tight.opportunities == 10  # 隔 2 天的那 3 次照旧是在险的机会，只是没说中
    records, last = _daily([1, 2] * 3 + [1] * 3)  # 6 次隔 1 天、3 次隔 2 天：6 / 9 ≈ 0.667 < 0.67
    _days, found = _intervals(records, cutoff=last + timedelta(days=1))
    wide = cluster_at(found, 1, line=LINE)
    assert wide is not None and wide.cluster == (1, 2) and len(wide.hits) == 9
    # 同一份历史，线定得更低（能接受更多误提醒）就不接；恰好在线上（6 / 9 = 2 / 3）宁可不提醒，照样接
    lower = cluster_at(found, 1, line=0.6)
    assert lower is not None and lower.cluster == (1,)
    on_the_line = cluster_at(found, 1, line=2 / 3)
    assert on_the_line is not None and on_the_line.cluster == (1, 2)
    with pytest.raises(ValueError, match="line"):
        cluster_at(found, 1, line=1.0)


def test_a_two_day_gap_that_was_not_seen_does_not_widen_the_cluster() -> None:
    """隔 2 天的那几段中间那天整天没在看（``seen`` 为 0）：不知道他是不是隔了两天，不算"到第 1 天没来"，簇不往后接。"""

    gaps = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
    records, last = _daily(gaps)
    _days, seen = _intervals(records, cutoff=last + timedelta(days=1))
    assert cluster_at(seen, 1, line=LINE).cluster == (1, 2)  # type: ignore[union-attr]
    skipped, day = [], WED
    for gap_days in gaps:
        if gap_days == 2:
            skipped.append(gap(at(day + timedelta(days=1), 0, 0), at(day + timedelta(days=2), 0, 0)))
        day += timedelta(days=gap_days)
    _days, unseen = _intervals(records, cutoff=last + timedelta(days=1), gaps=skipped)
    assert all(item.seen == 0.0 for item in unseen if item.days == 2)
    reading = cluster_at(unseen, 1, line=LINE)
    assert reading is not None and reading.cluster == (1,)


# ── 裁定 90：连漏期间照常建预期 ──────────────────────────────────────────


def test_a_streak_past_the_history_goes_on_at_the_typical_interval() -> None:
    """离上次已比历史上（够得上读数的）间隔都久：常规读法读不出，按常见间隔继续——天天做的每天一条，每周一次的每 7 天一条，
    纯隔天的每 2 天一条；每条只管那一天，命中与机会借常见间隔那一天的读数。"""

    records, last = _daily([1] * 19)
    _days, daily = _intervals(records, cutoff=last + timedelta(days=1))
    assert cluster_at(daily, 2, line=LINE) is None
    for elapsed in (2, 3, 5, 9):
        reading = overdue_at(daily, elapsed, line=LINE)
        assert reading is not None
        assert (reading.cluster, reading.block, reading.opened) == ((elapsed,), (elapsed,), elapsed)
        assert len(reading.hits) == 19 and reading.opportunities == 19
    assert overdue_at(daily, 1, line=LINE) is None  # 历史还读得到：归 cluster_at

    records, last = _weekly([7] * 8)
    _days, weekly = _intervals(records, cutoff=last + timedelta(days=1))
    assert [elapsed for elapsed in range(1, 30) if overdue_at(weekly, elapsed, line=LINE)] == [14, 21, 28]

    records, last = _daily([2] * 10)
    _days, every_other = _intervals(records, cutoff=last + timedelta(days=1))
    assert [elapsed for elapsed in range(1, 9) if overdue_at(every_other, elapsed, line=LINE)] == [4, 6, 8]
    with pytest.raises(ValueError, match="line"):
        overdue_at(daily, 3, line=0.0)


def test_a_single_longer_gap_is_not_enough_history_to_hold_the_streak_back() -> None:
    """只漏过一次（一段隔 2 天）的天天做：离上次 2 天时不短于它的只有那一段，常规读法读不出（样本少）——这就已经超过了历史，
    连漏第 2 天照样有一条；每周一次跳过一周的（一段 14 天），连漏第 2 周（14 天）也有。不短于它的有 2 段起归常规读法。"""

    records, last = _daily([1] * 9 + [2] + [1] * 8)
    _days, once = _intervals(records, cutoff=last + timedelta(days=1))
    assert cluster_at(once, 2, line=LINE) is None
    assert overdue_at(once, 2, line=LINE) is not None and overdue_at(once, 3, line=LINE) is not None

    records, last = _weekly([7, 7, 14, 7, 7, 7])
    _days, skipped = _intervals(records, cutoff=last + timedelta(days=1))
    assert cluster_at(skipped, 14, line=LINE) is None
    assert [elapsed for elapsed in range(1, 22) if overdue_at(skipped, elapsed, line=LINE)] == [14, 21]

    records, last = _daily([1] * 9 + [2, 2] + [1] * 8)
    _days, twice = _intervals(records, cutoff=last + timedelta(days=1))
    assert overdue_at(twice, 2, line=LINE) is None and cluster_at(twice, 2, line=LINE) is not None
    assert overdue_at(twice, 3, line=LINE) is not None


def test_overdue_needs_a_reading_at_the_typical_interval() -> None:
    """常见间隔那一天本身读不出（隔 7 天的两段一段是被提醒才做的）就没有读数：样本少就没有读数。历史不到两段也没有。"""

    records, last = _weekly([7, 7], reminded_at={2})
    _days, half = _intervals(records, cutoff=last + timedelta(days=1))
    assert overdue_at(half, 14, line=LINE) is None
    records, last = _weekly([7])
    _days, one = _intervals(records, cutoff=last + timedelta(days=1))
    assert overdue_at(one, 14, line=LINE) is None


def test_the_day_belongs_to_the_window_that_covers_it() -> None:
    """``opened``：簇被线截成几段时，离上次 e 天归盖住它的那一段的第一天。天天做、漏过 2 次的事：离上次 1 天只管当天，离上次
    2 天另开一段（连漏第 2 天是新的一条）；1、2 天真混着的离上次 1 天那段盖住第 2 天，第 2 天仍是第 1 天那段（同一个身份）。"""

    records, last = _daily([1] * 7 + [2] + [1] * 7 + [2] + [1] * 4)
    _days, missed_twice = _intervals(records, cutoff=last + timedelta(days=1))
    first, second = cluster_at(missed_twice, 1, line=LINE), cluster_at(missed_twice, 2, line=LINE)
    assert first is not None and second is not None
    assert (first.cluster, first.block, first.opened) == ((1,), (1, 2), 1)
    assert (second.cluster, second.block, second.opened) == ((2,), (1, 2), 2)

    records, last = _daily([1, 2] * 6)
    _days, mixed = _intervals(records, cutoff=last + timedelta(days=1))
    first, second = cluster_at(mixed, 1, line=LINE), cluster_at(mixed, 2, line=LINE)
    assert first is not None and second is not None
    assert first.cluster == (1, 2) and first.opened == 1 and second.opened == 1
