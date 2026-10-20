"""判缺失：这一次的空白比历史上最长的间隔还长；长度按看着的日子算，截尾作下界，样本不够不判。"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from habitus.temporal.absence import (
    AbsenceVerdict,
    Visit,
    effective_gaps,
    exceeds_longest,
    gap_between,
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
