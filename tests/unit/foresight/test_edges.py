"""单元的边（裁定 61 第 1、3、7 条）：权重从单元历史的各刻按边 fold——只读最近 M 刻、按名次遗忘、规律断了打折；
一刻只记给签名里的那几条边；增量形式 w ← w + η·s·(y − w) 与 fold 出来的均值一致。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from habitus.foresight import Evidence, account_key
from habitus.foresight.edges import DOWN, PastMoment, edge_moments, edge_reading, fold_sequence, ordered
from habitus.foresight.errors import ForesightError
from habitus.foundation.lifecycle import decay

CST = timezone(timedelta(hours=8))
NOW = datetime(2026, 9, 1, 21, 0, tzinfo=CST)
EVIDENCE = Evidence(alpha=0.05, half_life=8.0)
FLIGHT = account_key("s-k0010", "事件", "session|flight|luggage|days_2_7")
HOTEL = account_key("s-k0010", "事件", "session|hotel|luggage|days_2_7")
CANCEL = f"{DOWN}session|cancel|luggage|next_day"


def at(days_ago: float) -> datetime:
    return NOW - timedelta(days=days_ago)


def test_an_edge_is_the_moments_whose_signature_carries_it() -> None:
    both = PastMoment(at(5), frozenset({FLIGHT, HOTEL}), True)
    flight_only = PastMoment(at(4), frozenset({FLIGHT}), False)
    suppressed = PastMoment(at(3), frozenset({FLIGHT, CANCEL}), False)
    hotel_only = PastMoment(at(2), frozenset({HOTEL}), True)
    history = (hotel_only, suppressed, both, flight_only)
    assert edge_moments(history, FLIGHT) == (both, flight_only, suppressed)
    assert edge_moments(history, HOTEL) == (both, hotel_only)
    assert edge_moments(history, CANCEL) == (suppressed,)
    assert edge_moments(history, "nothing") == ()
    # 同一时刻的几刻按在场那一组排先后：读数与输入顺序无关
    same = PastMoment(at(5), frozenset({HOTEL}), False)
    assert ordered((both, same)) == ordered((same, both))


def test_fold_reads_only_the_window_and_forgets_by_rank() -> None:
    hits = [PastMoment(at(100 - i), frozenset({FLIGHT}), True) for i in range(100)]
    windowed = fold_sequence(hits, alpha=0.05, half_life=8.0, window=EVIDENCE.window)
    assert windowed.count == EVIDENCE.window and windowed.break_at is None and windowed.discount == 1.0
    expected = sum(decay(EVIDENCE.window - 1 - index, half_life=8.0) for index in range(EVIDENCE.window))
    assert windowed.reading.n == pytest.approx(expected) and windowed.reading.k == pytest.approx(expected)
    # 窗外一刻的分量已低于 α：读全部 100 刻与读最近 35 刻的读数相差不到 α
    whole = fold_sequence(hits, alpha=0.05, half_life=8.0, window=1000).reading
    assert abs(whole.lower - windowed.reading.lower) < 0.05 and whole.n - windowed.reading.n < 1.0
    # 分量乘进去；来没来进 k
    mixed = [PastMoment(at(3), frozenset({FLIGHT}), True, 0.5), PastMoment(at(1), frozenset({FLIGHT}), False, 0.25)]
    folded = fold_sequence(mixed, alpha=0.05, half_life=8.0, window=35).reading
    assert folded.n == pytest.approx(0.5 * decay(1, half_life=8.0) + 0.25)
    assert folded.k == pytest.approx(0.5 * decay(1, half_life=8.0))
    with pytest.raises(ForesightError):
        fold_sequence(mixed, alpha=0.05, half_life=8.0, window=0)


def test_the_incremental_update_equals_the_folded_mean() -> None:
    """方案第三节：每来一刻 w ← w + η·s·(y − w)，η = s ÷ (有效样本 + s)——与"全部刻 fold 出来的加权均值"是同一个数。"""

    sequence = [PastMoment(at(30 - i), frozenset({FLIGHT}), i % 4 != 3, 1.0 if i % 5 else 0.6) for i in range(30)]
    factor = decay(1, half_life=8.0)
    w = n = 0.0
    for moment in sequence:
        n *= factor
        s, y = moment.weight, 1.0 if moment.came else 0.0
        eta = s / (n + s)
        w = w + eta * (y - w)
        n += s
    folded = fold_sequence(sequence, alpha=0.05, half_life=8.0, window=100).reading
    assert folded.k / folded.n == pytest.approx(w)
    assert folded.n == pytest.approx(n)


def test_edge_reading_cuts_at_the_visible_moment_for_diagnosis() -> None:
    history = [PastMoment(at(20 - i), frozenset({FLIGHT}), i < 15) for i in range(20)]
    now = fold_sequence(history, alpha=0.05, half_life=8.0, window=35)
    then = edge_reading(history, FLIGHT, alpha=0.05, half_life=8.0, window=35, visible_at=at(5))
    assert then.count == 15 and then.reading.lower > now.reading.lower
    assert now.break_at is not None and then.break_at is None


def test_moments_refuse_nonsense() -> None:
    with pytest.raises(ForesightError):
        PastMoment(NOW.replace(tzinfo=None), frozenset({FLIGHT}), True)
    with pytest.raises(ForesightError):
        PastMoment(NOW, frozenset({FLIGHT}), True, 1.5)
    assert PastMoment(NOW, frozenset({FLIGHT, CANCEL}), True).ups == 1
    assert PastMoment(NOW, frozenset({FLIGHT, CANCEL}), True).downs == 1
