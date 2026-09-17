"""树发布的数：逐字取自 ``prediction.query``、一个不重算；复发的"距上次"只从今天算。"""

from __future__ import annotations

from datetime import timedelta

import pytest

from habitus.foresight import ForesightError, UnsealedRow, candidate_numbers, render_candidate
from habitus.prediction import query
from habitus.prediction.model import SlotKey
from tests.unit.foresight.fixtures import MONDAY, Ground, at, slot_of

NOW = MONDAY + timedelta(days=28)


def weekly_ground(tmp_path) -> Ground:
    ground = Ground(tmp_path, now=at(NOW, 23, 0))
    for week in range(4):
        ground.record(MONDAY + timedelta(days=7 * week), "打球", 19, 0, kind="打球")
    return ground


def test_the_published_numbers_are_the_trees_own_word_for_word(tmp_path) -> None:
    ground = weekly_ground(tmp_path)
    tree = ground.tree()
    slot = SlotKey(weekday=0, slot=slot_of(19, 0))
    expected = query.node_at(tree, slot, "打球")
    assert expected is not None

    numbers = candidate_numbers(tree, slot, "打球", done_today=0, elapsed_seconds=None)

    assert (numbers.marginal, numbers.hazard, numbers.cumulative) == (
        expected.marginal,
        expected.hazard,
        expected.cumulative,
    )
    assert (numbers.lift_all_day, numbers.lift_weekday, numbers.count, numbers.n_eff, numbers.trend) == (
        expected.lift_all_day,
        expected.lift_weekday,
        expected.count,
        expected.n_eff,
        expected.trend,
    )
    assert numbers.trend_n_eff == tree.curves[(0, "打球")].trend_n_eff
    assert numbers.recurrence is not None and numbers.recurrence.p50 == pytest.approx(7 * 86_400.0)
    assert numbers.recurrence.overdue is None and numbers.done_today == 0


def test_overdue_is_only_measured_from_todays_own_last_time(tmp_path) -> None:
    """今天没做过就是 None，不拿昨天那次凑；今天做过一次，距上次就从那一次起算。"""

    ground = weekly_ground(tmp_path)
    tree = ground.tree()
    slot = SlotKey(weekday=0, slot=slot_of(19, 0))
    half_a_period = 3.5 * 86_400.0

    numbers = candidate_numbers(tree, slot, "打球", done_today=1, elapsed_seconds=half_a_period)

    assert numbers.recurrence is not None and numbers.recurrence.overdue == pytest.approx(0.5)
    assert numbers.done_today == 1
    with pytest.raises(ForesightError, match="non-negative"):
        candidate_numbers(tree, slot, "打球", done_today=0, elapsed_seconds=-1.0)
    with pytest.raises(ForesightError, match="no curve"):
        candidate_numbers(tree, SlotKey(weekday=3, slot=slot_of(19, 0)), "打球", done_today=0, elapsed_seconds=None)


def test_the_pack_measures_elapsed_from_todays_last_start_including_the_unsealed(tmp_path) -> None:
    """此刻 19:05；树上今天 18:00 做过一次，判断存储里 18:40 还有一次没封口——距上次从 18:40 算。"""

    ground = weekly_ground(tmp_path)
    ground.record(NOW, "打球", 18, 0, kind="打球")
    unsealed = (
        UnsealedRow(name="打球", kind_token="打球", started_at=at(NOW, 18, 40), last_observed_at=at(NOW, 19, 0), summary="又去了"),
    )

    pack = ground.pack(at(NOW, 19, 5), unsealed=unsealed)

    candidate = next(item for item in pack.candidates if item.kind_token == "打球")
    assert pack.now.done_today["打球"] == 2 and pack.now.elapsed_seconds("打球") == 25 * 60.0
    assert candidate.numbers.done_today == 2
    assert candidate.numbers.recurrence is not None
    assert candidate.numbers.recurrence.overdue == pytest.approx(25 * 60.0 / (7 * 86_400.0))
    assert pack.now.elapsed_seconds("洗澡") is None


def test_the_published_numbers_are_rendered_after_the_four_layers(tmp_path) -> None:
    ground = weekly_ground(tmp_path)
    (candidate,) = ground.pack(at(NOW, 19, 5)).expanded
    text = render_candidate(candidate)
    table_end = text.index("| 全天 |")
    published = text.index("发布的率：边际")
    assert table_end < published < text.index("### 历史")
    assert "复发：p10 7.0 · p50 7.0 · p90 7.0 天" in text and "今天还没做" in text
