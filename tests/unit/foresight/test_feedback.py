"""单元的提醒反馈（裁定 61 第 4 条；方案 30 第四节）：说 + 做合判——做了是好、没回应或负又没做是坏、中又没做不写；
读数单位权重、上界掉到线下单元回影子。"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from habitus.foresight import Evidence, Tolerance
from habitus.foresight.edges import PastMoment
from habitus.foresight.feedback import feedback_reading, latest_polarity, reminder_feedback
from habitus.foresight.history import Moment
from habitus.foresight.ladder import Standing
from habitus.foresight.ledger import Expectation, Response, Settlement, UnitFeedback
from habitus.foresight.reliability import reading
from habitus.foresight.units import Input, Level, UnitReading, unit_level

CST = timezone(timedelta(hours=8))
NOW = datetime(2026, 9, 1, 21, 0, tzinfo=CST)
EVIDENCE = Evidence(alpha=0.05, half_life=8.0)
LOSS = Tolerance(line=2 / 3)
HOTEL = "s-k0002"


def expectation(identity: str = "hotel") -> Expectation:
    return Expectation(
        expectation_id=identity,
        kind_token=HOTEL,
        day=NOW.date(),
        formed_at=NOW,
        source="事件",
        trigger="session|flight|hotel|chain",
        trigger_instance="behavior://flight.md",
        window=(NOW + timedelta(minutes=5), NOW + timedelta(minutes=50)),
        deadline=NOW + timedelta(hours=3),
        basis=(),
        opportunities=13,
        hits=10,
        observed_from=NOW,
    )


def settled(outcome: str) -> Settlement:
    found = outcome != "过期待决"
    return Settlement(
        "hotel",
        HOTEL,
        NOW.date(),
        NOW + timedelta(hours=4),
        outcome,
        "behavior://hotel.md" if found else None,
        15 if outcome in ("提醒后发生", "早到", "晚了但做了") else None,
    )


def said(polarity: str, minutes: int = 30) -> Response:
    return Response("hotel", NOW + timedelta(minutes=minutes), polarity, f"conv#{minutes}")


@pytest.mark.parametrize(
    ("outcome", "responses", "expected"),
    [
        ("提醒后发生", (), True),  # 说了 → 做了
        ("提醒后发生", (said("负"),), True),  # 说什么都一样
        ("提醒后发生", (said("中"),), True),  # 中 → 时限前做了
        ("准时", (), None),  # 说之前就做了（结算只把说出之后的发生记成提醒后发生）：这次机会不算提醒的（裁定 65）
        ("早到", (said("正"),), None),
        ("过期待决", (), False),  # 没回应、没做：误提醒
        ("过期待决", (said("负"),), False),  # 这次不用、没做
        ("过期待决", (said("正"), said("负", 60)), False),  # 最后一次判出的方向算数
        ("过期待决", (said("中"),), None),  # 中、没做：不写、不算机会
    ],
)
def test_say_and_do_are_judged_together_with_doing_first(outcome: str, responses, expected) -> None:  # type: ignore[no-untyped-def]
    feedback = reminder_feedback(
        expectation(), settled(outcome), spoken_at=NOW + timedelta(minutes=50), responses=responses
    )
    if expected is None:
        assert feedback is None
    else:
        assert feedback is not None and feedback.good is expected
        assert (feedback.kind_token, feedback.at, feedback.expectation_id) == (HOTEL, NOW, "hotel")
        assert feedback.polarity == latest_polarity(responses)


def test_nothing_is_written_when_nothing_was_said() -> None:
    assert reminder_feedback(expectation(), settled("过期待决"), spoken_at=None, responses=()) is None
    assert reminder_feedback(expectation(), settled("准时"), spoken_at=None, responses=(said("正"),)) is None


def test_the_feedback_reading_is_unit_weight_and_pulls_the_unit_back_to_shadow() -> None:
    bad = [UnitFeedback(HOTEL, NOW - timedelta(days=i), False, f"e{i}") for i in range(5)]
    ignored = feedback_reading(bad, alpha=EVIDENCE.alpha, half_life=EVIDENCE.half_life, window=EVIDENCE.window)
    assert ignored.k == 0.0 and ignored.n > 3.0 and ignored.upper < LOSS.threshold
    good = [UnitFeedback(HOTEL, NOW - timedelta(days=i), True, f"g{i}", "正") for i in range(5)]
    accepted = feedback_reading(good, alpha=EVIDENCE.alpha, half_life=EVIDENCE.half_life, window=EVIDENCE.window)
    assert accepted.upper > LOSS.threshold
    standing = Standing(level=1, threshold=LOSS.threshold, prediction=reading(9.0, 9.5, alpha=0.05))
    chosen = Input(expectation(), standing, status="established")
    strong = UnitReading(Level.SAME_SET, reading(9.0, 9.5, alpha=0.05), (10, 10, 10))
    assert unit_level(chosen, strong, previous=1) == 1
    assert unit_level(chosen, strong, previous=1, feedback=ignored) == 0  # 说了他老不做：照降
    assert unit_level(chosen, strong, previous=1, feedback=accepted) == 1
    # 「以后都提醒我」只跳过攒把握（裁定 60 第 8 条）：说了老不做照样降
    weak = replace(strong, reading=reading(1.0, 4.0, alpha=0.05))
    assert unit_level(chosen, weak, previous=None, setting="以后都提醒我") == 1
    assert unit_level(chosen, weak, previous=None, setting="以后都提醒我", feedback=ignored) == 0
    assert unit_level(chosen, strong, previous=1, setting="以后别提了") == 0
    assert PastMoment(NOW, frozenset({"a"}), True).weight == 1.0


def test_the_first_closing_response_decides_and_a_later_neutral_cannot_undo_it() -> None:
    """正 / 负关闭、之后不再判：两份会话源乱序融合时，晚判的中不能盖掉早说的负；没有关闭的才取最晚的中。"""

    assert latest_polarity((said("负", 30), said("中", 60))) == "负"
    assert latest_polarity((said("中", 60), said("负", 30))) == "负"
    assert latest_polarity((said("正", 60), said("负", 30))) == "负"  # 最早关闭的那条
    assert latest_polarity((said("中", 30), said("中", 60))) == "中"
    assert latest_polarity(()) is None


def test_the_feedback_clock_advances_by_the_units_silent_misses_so_a_pushed_back_unit_can_recover() -> None:
    """裁定 60 第 4 条：回了影子以后没有"说"、不再写反馈，但他每一次到时窗末还没做的机会推一格名次，遗忘把上界带回线上；
    记过反馈的那几刻、他自己做了的那几刻不推。"""

    rows = [UnitFeedback(HOTEL, NOW + timedelta(days=index), index < 12, f"e{index}") for index in range(16)]
    options = dict(alpha=EVIDENCE.alpha, half_life=EVIDENCE.half_life, window=EVIDENCE.window)
    stuck = feedback_reading(rows, **options)
    assert stuck.upper < LOSS.threshold
    edge = frozenset({"a"})

    def later(count: int, came: bool) -> tuple[Moment, ...]:
        return tuple(
            Moment(PastMoment(NOW + timedelta(days=16 + index), edge, came, 1.0), (f"m{index}",))
            for index in range(count)
        )

    # 记过反馈的那几刻（同一条预期）不另数一遍；他自己做了的那几刻不是"还没做"的机会
    same = tuple(Moment(PastMoment(row.at, edge, row.good, 1.0), (row.expectation_id,)) for row in rows)
    assert feedback_reading(rows, moments=same, **options) == stuck
    assert feedback_reading(rows, moments=later(50, True), **options) == stuck
    # 没做的机会推着时钟走（分量 0、不添计数）：上界回到线上，单元才有再开口的一天；推满 M 格旧的全忘光
    readings = [
        feedback_reading(rows, moments=later(count, False), **options) for count in range(1, EVIDENCE.window + 1)
    ]
    assert readings[-1].n == 0.0
    recovered = next(count for count, found in enumerate(readings, start=1) if found.upper >= LOSS.threshold)
    assert 1 < recovered <= EVIDENCE.window
    standing = Standing(level=1, threshold=LOSS.threshold, prediction=reading(9.0, 9.5, alpha=0.05))
    chosen = Input(expectation(), standing, status="established")
    strong = UnitReading(Level.SAME_SET, reading(9.0, 9.5, alpha=0.05), (10, 10, 10))
    assert unit_level(chosen, strong, previous=0, feedback=stuck) == 0
    assert unit_level(chosen, strong, previous=0, feedback=readings[recovered - 1]) == 1
