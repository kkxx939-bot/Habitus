"""判缺失：这一次的空白比历史上最长的间隔还长；长度按看着的日子算，截尾作下界，样本不够不判。
账的久别另要空白不短于 H 个典型间隔（历史间隔中位数，裁定 88 第 2 条）。"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from habitus.temporal.absence import (
    AbsenceVerdict,
    GapSummary,
    Visit,
    effective_gaps,
    exceeds_longest,
    gap_between,
    long_absence,
    open_gap,
)

WED = date(2026, 1, 7)


def _weekly(weeks: int) -> list[Visit]:
    return [Visit(WED + timedelta(days=7 * week)) for week in range(weeks)]


def test_a_gap_longer_than_all_before_exceeds() -> None:
    gaps = effective_gaps(_weekly(5))
    current = open_gap(WED + timedelta(days=28), WED + timedelta(days=40))
    assert current is not None and current.length == 12
    assert exceeds_longest(gaps, current, min_samples=2) is AbsenceVerdict.EXCEEDS
    within = open_gap(WED + timedelta(days=28), WED + timedelta(days=33))
    assert within is not None and exceeds_longest(gaps, within, min_samples=2) is AbsenceVerdict.WITHIN


def test_too_few_samples_do_not_decide() -> None:
    gaps = effective_gaps(_weekly(2))
    current = open_gap(WED + timedelta(days=7), WED + timedelta(days=200))
    assert current is not None
    assert exceeds_longest(gaps, current, min_samples=2) is AbsenceVerdict.UNDECIDED
    assert exceeds_longest(gaps, current, min_samples=1) is AbsenceVerdict.EXCEEDS


def test_unwatched_days_do_not_stretch_the_longest_gap() -> None:
    """中间 60 天没在看：那段 91 天的间隔有效长度只有 31，之后空 40 天就算超过（按日历日算要 92 天）。"""

    start = WED
    blind = {start + timedelta(days=d) for d in range(10, 70)}
    visits = [Visit(start), Visit(start + timedelta(days=91)), Visit(start + timedelta(days=98))]

    def watched(day: date) -> float:
        return 0.0 if day in blind else 1.0

    gaps = effective_gaps(visits, watched=watched)
    assert [gap.length for gap in gaps] == [31.0, 7.0]
    current = open_gap(start + timedelta(days=98), start + timedelta(days=138), watched=watched)
    assert current is not None and exceeds_longest(gaps, current, min_samples=2) is AbsenceVerdict.EXCEEDS


def test_only_counted_day_types_make_length() -> None:
    """只数工作日：国庆七天不算长度。"""

    workdays = {WED + timedelta(days=d) for d in range(60) if (WED + timedelta(days=d)).weekday() < 5}
    holiday = {WED + timedelta(days=d) for d in range(30, 37)}

    def counts(day: date) -> bool:
        return day in workdays and day not in holiday

    gap = gap_between(WED + timedelta(days=29), WED + timedelta(days=38), counts=counts)
    assert gap.length == sum(1 for d in range(30, 39) if counts(WED + timedelta(days=d)))


def test_a_prompted_ending_is_censored_and_ignored_as_a_sample() -> None:
    visits = [*_weekly(3), Visit(WED + timedelta(days=60), prompted=True)]
    gaps = effective_gaps(visits)
    assert [gap.censored for gap in gaps] == [False, False, True]
    current = open_gap(WED + timedelta(days=60), WED + timedelta(days=70))
    assert current is not None
    # 截尾那段（46 天）不进"历史最长"：10 天的空白已经比 7 天长
    assert exceeds_longest(gaps, current, min_samples=2) is AbsenceVerdict.EXCEEDS


def test_one_natural_visit_on_a_day_makes_it_uncensored() -> None:
    day = WED + timedelta(days=7)
    gaps = effective_gaps([Visit(WED), Visit(day, prompted=True), Visit(day, prompted=False)])
    assert gaps[0].censored is False


def test_bad_inputs() -> None:
    with pytest.raises(ValueError):
        exceeds_longest((), gap_between(WED, WED + timedelta(days=1)), min_samples=0)
    with pytest.raises(ValueError):
        gap_between(WED, WED + timedelta(days=2), watched=lambda _d: 2.0)
    assert open_gap(WED, WED) is None


def test_gap_shape_is_validated() -> None:
    from habitus.temporal.absence import EffectiveGap

    with pytest.raises(ValueError):
        EffectiveGap(WED, WED, 1.0, False)
    with pytest.raises(ValueError):
        EffectiveGap(WED, WED + timedelta(days=1), -1.0, False)


# --- 账的久别：比历史最长还长，且至少 H 个典型间隔（裁定 88 第 2 条）-------------------------------------


def test_the_typical_gap_is_the_median_of_the_uncensored_gaps() -> None:
    assert GapSummary().count == 0 and GapSummary().longest == 0.0 and GapSummary().typical == 0.0
    odd = GapSummary().including(effective_gaps([Visit(WED + timedelta(days=d)) for d in (0, 1, 2, 9)]))
    assert odd.lengths == ((1.0, 2), (7.0, 1)) and odd.count == 3 and odd.longest == 7.0 and odd.typical == 1.0
    even = GapSummary(((1.0, 1), (3.0, 1), (5.0, 1), (40.0, 1)))
    assert even.typical == 4.0  # 中间两个的平均
    prompted = [Visit(WED), Visit(WED + timedelta(days=1)), Visit(WED + timedelta(days=30), prompted=True)]
    assert GapSummary().including(effective_gaps(prompted)).lengths == ((1.0, 1),)  # 截尾那段不进
    assert odd.including(()) is odd


def test_a_gap_summary_refuses_unordered_repeated_or_empty_entries() -> None:
    for lengths in (((3.0, 1), (1.0, 1)), ((1.0, 1), (1.0, 2)), ((1.0, 0),), ((-1.0, 1),)):
        with pytest.raises(ValueError):
            GapSummary(lengths)
    with pytest.raises(IndexError):
        GapSummary(((1.0, 1),))._ranked(1)


def test_a_daily_habit_missing_one_day_is_not_a_long_absence_but_eight_typical_gaps_are() -> None:
    """天天做：漏一天（空白 2 天）比历来都长，但不到 8 个典型间隔；空白 7 天不算、8 天算。"""

    gaps = effective_gaps([Visit(WED + timedelta(days=d)) for d in range(20)])
    last = WED + timedelta(days=19)

    def verdict(blank: int) -> AbsenceVerdict:
        current = open_gap(last, last + timedelta(days=blank))
        assert current is not None
        return long_absence(gaps, current, min_samples=2, typical_multiple=8.0)

    assert verdict(2) is AbsenceVerdict.WITHIN
    assert verdict(7) is AbsenceVerdict.WITHIN
    assert verdict(8) is AbsenceVerdict.EXCEEDS


def test_a_weekly_habit_stopped_three_weeks_is_not_a_long_absence_but_eight_weeks_is() -> None:
    gaps = effective_gaps(_weekly(8))
    last = WED + timedelta(days=49)
    for weeks, expected in ((3, AbsenceVerdict.WITHIN), (8, AbsenceVerdict.EXCEEDS)):
        current = open_gap(last, last + timedelta(weeks=weeks))
        assert current is not None
        assert long_absence(gaps, current, min_samples=2, typical_multiple=8.0) is expected
    almost = open_gap(last, last + timedelta(days=55))
    assert almost is not None and long_absence(gaps, almost, min_samples=2, typical_multiple=8.0) is AbsenceVerdict.WITHIN


def test_a_long_absence_still_needs_to_beat_the_longest_gap_and_enough_samples() -> None:
    """一次很长的旧间隔把"历史最长"抬上去：够 8 个典型间隔也还要比它长；样本不够不判。"""

    gaps = effective_gaps([Visit(WED + timedelta(days=d)) for d in (0, 1, 2, 3, 23, 24, 25)])  # 中间停过 20 天
    last = WED + timedelta(days=25)
    for blank, expected in ((12, AbsenceVerdict.WITHIN), (20, AbsenceVerdict.WITHIN), (21, AbsenceVerdict.EXCEEDS)):
        current = open_gap(last, last + timedelta(days=blank))
        assert current is not None
        assert long_absence(gaps, current, min_samples=2, typical_multiple=8.0) is expected
    current = open_gap(last, last + timedelta(days=60))
    assert current is not None
    assert long_absence(gaps[:1], current, min_samples=2, typical_multiple=8.0) is AbsenceVerdict.UNDECIDED


def test_a_long_absence_reads_the_typical_gap_across_the_compacted_part() -> None:
    """压实那一段天天做、窗口里只剩几次每三天一次：典型间隔按全历史（1 天）算，空白 10 天算久别；只看窗口（3 天）就不算。"""

    earlier = GapSummary(((1.0, 40),))
    window = effective_gaps([Visit(WED + timedelta(days=d)) for d in (0, 3, 6, 9)])
    current = open_gap(WED + timedelta(days=9), WED + timedelta(days=19))
    assert current is not None
    assert long_absence(window, current, min_samples=2, typical_multiple=8.0) is AbsenceVerdict.WITHIN
    verdict = long_absence(window, current, min_samples=2, typical_multiple=8.0, earlier=earlier)
    assert verdict is AbsenceVerdict.EXCEEDS


def test_the_typical_multiple_must_be_a_positive_finite_number() -> None:
    current = gap_between(WED, WED + timedelta(days=1))
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            long_absence((), current, min_samples=2, typical_multiple=bad)
    with pytest.raises(ValueError):
        long_absence((), current, min_samples=0, typical_multiple=8.0)
