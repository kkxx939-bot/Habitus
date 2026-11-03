"""观测覆盖按"在看"口径；密度与日内分布只看最近 M 个自然发生日；日内分布按"做的那天几点做"，峰要至少两个不同的日子。"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from habitus.temporal import (
    Rulers,
    blind_days,
    coverage_of,
    day_profile,
    density_of,
    episodes_of,
    occurrence_days,
)
from tests.unit.temporal.fixtures import at, gap, record, series, with_sessions

START = date(2026, 1, 5)  # 周一
WINDOW = 2700


def _profile(records, cutoff, *, gaps=None, recent=35):
    found = episodes_of(records, window_seconds=WINDOW)
    days = occurrence_days(found)
    coverage = coverage_of(
        series(with_sessions(records, cutoff=cutoff), cutoff=cutoff, gaps=gaps), "session", slot_minutes=15
    )
    return day_profile(
        found,
        days,
        coverage,
        pool_half_width=3,
        shrink_slot_to_pool=5.0,
        shrink_pool_to_all_day=5.0,
        min_peak_mass=0.10,
        recent=recent,
    )


def test_coverage_counts_every_calendar_day_and_subtracts_gaps_by_share() -> None:
    """每个日历日都在覆盖里、空白按比例扣；一整天没记录：会话 lane 是没在看（裁定 89 第 2 条），物理 lane 照旧在看。"""

    from dataclasses import replace

    records = [record(at(START, 9)), record(at(START + timedelta(days=1), 9)), record(at(START + timedelta(days=3), 9))]
    gaps = [gap(at(START + timedelta(days=1), 10, 0), at(START + timedelta(days=1), 10, 7))]
    found = coverage_of(series(records, cutoff=START + timedelta(days=5), gaps=gaps), "session", slot_minutes=15)
    assert list(found.days) == [START + timedelta(days=offset) for offset in range(5)]
    second = found.days[START + timedelta(days=1)]
    assert second[40] == pytest.approx(1 - 7 / 15)  # 10:00 那一槽被盖住 7 分钟
    assert found.day_share(START) == 1.0
    # 会话 lane：2 号、4 号一整天没会话，等同上游报了全天「未观测」
    assert found.day_share(START + timedelta(days=2)) == 0.0 and found.day_share(START + timedelta(days=4)) == 0.0
    # 那天有一条「非事件」也算开过会话：只看那天有没有记录，不看进没进序列
    from habitus.series import SkippedRecord, SkipReason

    skipped = SkippedRecord("behavior://s-非事件/x", START + timedelta(days=2), SkipReason.NOT_EVENT)
    with_skip = replace(series(records, cutoff=START + timedelta(days=5), gaps=gaps), skipped=(skipped,))
    assert coverage_of(with_skip, "session", slot_minutes=15).day_share(START + timedelta(days=2)) == 1.0
    # 物理 lane：上游没在看时自己报空白，空着的一天照旧算在看（那天就是没做）
    physical = [replace(item, lane="physical", kind_token="p-k0001") for item in records]
    worn = coverage_of(series(physical, cutoff=START + timedelta(days=5), gaps=gaps), "physical", slot_minutes=15)
    assert worn.day_share(START + timedelta(days=2)) == 1.0 and worn.days[START + timedelta(days=1)][40] == second[40]
    assert coverage_of(series([], cutoff=START), "session", slot_minutes=15).days == {}


def test_coverage_starts_on_the_earliest_local_day_whatever_the_offsets() -> None:
    """记录按时刻排：+09:00 的 18 号 00:30 早于 +08:00 的 17 号 23:50。覆盖从日期最早的 17 号起，不从时刻最早那条的日期起。"""

    from datetime import datetime, timezone

    tokyo = timezone(timedelta(hours=9))
    later_day = record(datetime(2026, 8, 18, 0, 30, tzinfo=tokyo))
    earlier_day = record(at(date(2026, 8, 17), 23, 50))
    found = coverage_of(series([later_day, earlier_day], cutoff=date(2026, 8, 19)), "session", slot_minutes=60)
    assert list(found.days) == [date(2026, 8, 17), date(2026, 8, 18)]


def test_overlapping_gaps_are_subtracted_once() -> None:
    """「未观测」与「没读懂」都盖住 10:00–10:30、另一段「没读懂」10:15–10:45（空白不分 lane，会彼此重叠）：10 点那一槽按并集
    扣掉 45 分钟，还剩四分之一在看（重复扣会扣到 0）。"""

    from habitus.series import GapKind

    day = START
    gaps = [
        gap(at(day, 10, 0), at(day, 10, 30)),
        gap(at(day, 10, 0), at(day, 10, 30), GapKind.UNREADABLE),
        gap(at(day, 10, 15), at(day, 10, 45), GapKind.UNREADABLE),
    ]
    found = coverage_of(series([record(at(day, 8))], cutoff=day + timedelta(days=1), gaps=gaps), "session", slot_minutes=60)
    assert found.days[day][10] == pytest.approx(0.25)


def test_density_counts_from_the_first_of_its_latest_occurrences() -> None:
    other = [record(at(START, 8), "s-k0009")]  # 这条 lane 从 1 月就有记录
    june = date(2026, 6, 1)
    guitar = [record(at(june + timedelta(days=7 * week), 19), "s-k0002") for week in range(6)]
    cutoff = june + timedelta(days=42)
    coverage = coverage_of(
        series(with_sessions(other + guitar, cutoff=cutoff), cutoff=cutoff), "session", slot_minutes=15
    )
    found = density_of(occurrence_days(episodes_of(guitar, window_seconds=WINDOW)), coverage, recent=35)
    assert found is not None
    assert found.days_per_occurrence == pytest.approx(7.0)  # 不是从 1 月算起的三十几天
    assert found.occurrence_day_share == pytest.approx(1 / 7)


def test_density_and_profile_read_only_the_latest_occurrence_days() -> None:
    """26 号清单 4.7：一年前每周一次、19:00；后来改成每两天一次、07:00。只看最近 M 个自然发生日：频率是现在的两天一次，
    峰在 07:00；看全历史时 19:00 那一段还在。"""

    weekly = [record(at(START + timedelta(days=7 * week), 19)) for week in range(30)]
    switched = START + timedelta(days=7 * 30)
    often = [record(at(switched + timedelta(days=2 * index), 7)) for index in range(40)]
    records = weekly + often
    cutoff = switched + timedelta(days=80)
    days = occurrence_days(episodes_of(records, window_seconds=WINDOW))
    coverage = coverage_of(series(with_sessions(records, cutoff=cutoff), cutoff=cutoff), "session", slot_minutes=15)
    recent = density_of(days, coverage, recent=35)
    assert recent is not None and recent.occurrences == 35 and recent.days_per_occurrence == pytest.approx(2.0, abs=0.1)
    whole = density_of(days, coverage, recent=1000)
    assert whole is not None and whole.days_per_occurrence > 3.5
    latest = _profile(records, cutoff, recent=35)
    assert latest is not None and [peak.centre_slot // 4 for peak in latest.peaks] == [7]
    full = _profile(records, cutoff, recent=1000)
    assert full is not None and sorted(peak.centre_slot // 4 for peak in full.peaks) == [7, 19]
    with pytest.raises(ValueError, match="recent"):
        density_of(days, coverage, recent=0)
    with pytest.raises(ValueError, match="recent"):
        _profile(records, cutoff, recent=0)
    with pytest.raises(ValueError, match="recent"):
        Rulers(15, WINDOW, 3, 5.0, 5.0, 0.10, 0)


def test_blind_days_leave_both_numerator_and_denominator() -> None:
    records = [record(at(START, 9)), record(at(START + timedelta(days=1), 9), reminded=True)]
    days = occurrence_days(episodes_of(records, window_seconds=WINDOW))
    assert blind_days(days) == frozenset({START + timedelta(days=1)})
    coverage = coverage_of(series(records, cutoff=START + timedelta(days=2)), "session", slot_minutes=15)
    found = density_of(days, coverage, recent=35)
    assert found is not None and found.watched_days == pytest.approx(1.0) and found.occurrences == 1


def test_profile_finds_the_time_on_the_days_it_happens() -> None:
    weeks = [record(at(START + timedelta(days=2 + 7 * week), 17, 0)) for week in range(10)]  # 每周三 17:00
    found = _profile(weeks, START + timedelta(days=80))
    assert found is not None and len(found.peaks) == 1
    peak = found.peaks[0]
    assert peak.first_slot <= 68 <= peak.last_slot and peak.share > 0.5 and peak.days == 10
    assert sum(found.shares) == pytest.approx(1.0)


def test_one_off_times_make_no_peak() -> None:
    """三次各在不同时段：每段只有一天，不报峰（样本少就没有读数）。"""

    scattered = [
        record(at(START, 9)),
        record(at(START + timedelta(days=40), 13)),
        record(at(START + timedelta(days=90), 17)),
    ]
    found = _profile(scattered, START + timedelta(days=100))
    assert found is not None and found.peaks == ()


def test_a_peak_across_midnight_is_one_peak() -> None:
    late = []
    for offset in range(8):
        day = START + timedelta(days=offset * 2)
        late.append(record(at(day, 23, 45)))
        late.append(record(at(day + timedelta(days=1), 0, 50), number=1))
    found = _profile(late, START + timedelta(days=20))
    assert found is not None and len(found.peaks) == 1
    peak = found.peaks[0]
    assert peak.last_slot >= 96  # 尾巴落在次日


def test_a_lane_whose_only_record_is_on_the_last_sealed_day_has_that_day_covered() -> None:
    """唯一一条记录就在最后一个已封口的日子：覆盖里有这一天（起点等于最后一天也算）。"""

    day = date(2026, 8, 17)
    found = coverage_of(series([record(at(day, 9))], cutoff=day + timedelta(days=1)), "session", slot_minutes=60)
    assert list(found.days) == [day]


def test_a_flat_curve_has_no_peak() -> None:
    """峰是"高于均值"的一段：每个槽都一样高时没有哪一段高于均值，没有峰。"""

    from habitus.temporal import peaks_of

    assert peaks_of([0.25, 0.25, 0.25, 0.25], floor=0.1) == ()


def test_a_slot_next_to_the_usual_time_borrows_from_its_neighbourhood() -> None:
    """收缩先向邻域、再向全天：三天都在 10:00 做，10:15 那一槽（邻域里有 10:00）比 15:00 那一槽（邻域里什么都没有）占比高。"""

    records = [record(at(START + timedelta(days=offset), 10)) for offset in range(3)]
    profile = _profile(records, START + timedelta(days=3))
    assert profile is not None
    assert profile.shares[41] > profile.shares[60]
