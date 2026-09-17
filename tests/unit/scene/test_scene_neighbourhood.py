"""槽位邻域序列：历史侧从开始前 k 槽到最后所见后 k 槽，此刻侧从当前槽前 k 槽到此刻。全部机械、零 LLM。

守的是三处与预测树同口径的对齐点：槽的起点算法、半宽的含义、跨午夜不做环形而是去邻日取行。
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from habitus.prediction.model import SlotKey
from habitus.prediction.nodes import pool_indexes
from habitus.scene.views import DayIndexCache, slot_neighbourhood, slot_neighbourhood_until, slot_window
from habitus.scene.views.clock import slot_floor, slot_index
from tests.unit.scene.fixtures import DAY1, DAY2, SUBJECT, Site, at, publish


def site(tmp_path) -> Site:
    return Site(tmp_path, now=at(DAY2, 23, 0))


def cache_for(ground: Site) -> DayIndexCache:
    return DayIndexCache(ground.behavior_tree, subject=SUBJECT)


# ── 槽的起止 ─────────────────────────────────────────────────────────────────


def test_slot_floor_uses_the_prediction_trees_slot_arithmetic() -> None:
    """槽 = 本地时分 // slot_minutes，与 ``SlotKey.of`` 同一公式；秒与微秒抹掉，时区保留。"""

    moment = at(DAY1, 16, 41).replace(second=37, microsecond=5)
    assert slot_floor(moment, slot_minutes=15) == at(DAY1, 16, 30)
    assert slot_floor(moment, slot_minutes=15).tzinfo == moment.tzinfo


def test_the_window_covers_exactly_the_trees_pooling_slots() -> None:
    """窗口里的每一分钟所在的槽，恰好是预测树 ``pool_indexes`` 那几格（不跨午夜时）；两处口径同一。"""

    moment = at(DAY1, 16, 41)
    start, end = slot_window(moment, slot_minutes=15, half_width=3)
    covered = sorted({slot_index(start + timedelta(minutes=m), slot_minutes=15) for m in range(int((end - start).total_seconds() // 60))})
    assert covered == sorted(pool_indexes(SlotKey.of(moment, slot_minutes=15).slot, 3, 96))


def test_slot_window_is_left_closed_right_open_and_crosses_midnight() -> None:
    start, end = slot_window(at(DAY2, 0, 5), slot_minutes=15, half_width=1)
    assert (start, end) == (at(DAY1, 23, 45), at(DAY2, 0, 30))


@pytest.mark.parametrize(
    ("slot_minutes", "half_width", "message"),
    [(7, 1, "divisor of 1440"), (0, 1, "divisor of 1440"), (15, -1, "non-negative")],
)
def test_the_window_refuses_a_bad_clock_face(slot_minutes: int, half_width: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        slot_window(at(DAY1, 16, 0), slot_minutes=slot_minutes, half_width=half_width)


def test_the_window_refuses_a_naive_moment() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        slot_window(at(DAY1, 16, 0).replace(tzinfo=None), slot_minutes=15, half_width=1)


# ── 历史侧 ───────────────────────────────────────────────────────────────────


def test_the_after_part_is_measured_from_the_last_sighting_not_the_start(tmp_path) -> None:
    """打球一个半小时，回来洗澡在开始后两小时：按开始 ±k 会切掉，按最后所见 +k 留得住。

    k=2、15 分钟槽：之前段从 16:30 − 30 = 16:00 起，之后段到 floor(18:10) + 3×15 = 18:45 止（开区间）。
    """

    ground = site(tmp_path)
    publish(ground.behavior_tree, DAY1, "换衣服", 15, 55)  # 16:00 之前，不在
    early = publish(ground.behavior_tree, DAY1, "收拾球包", 16, 5)
    own = publish(ground.behavior_tree, DAY1, "打球", 16, 40, lasts_minutes=90)  # 最后所见 18:10
    shower = publish(ground.behavior_tree, DAY1, "洗澡", 18, 40)
    publish(ground.behavior_tree, DAY1, "开电脑", 18, 45)  # 正好在右开边界上，不在
    publish(ground.behavior_tree, DAY1, "打游戏", 18, 50)

    rows = slot_neighbourhood(own, cache_for(ground), slot_minutes=15, half_width=2)

    assert [row.uri for row in rows] == [early, own, shower]
    assert [row.name for row in rows] == ["收拾球包", "打球", "洗澡"]
    assert rows[1].kind_token == "打球" and rows[1].at == at(DAY1, 16, 40)


def test_the_before_part_crosses_midnight_into_the_previous_day(tmp_path) -> None:
    """不做环形：00:05 的行为往前 1 槽是前一天 23:45 起，前一天目录里的行要取到。"""

    ground = site(tmp_path)
    late = publish(ground.behavior_tree, DAY1, "刷牙", 23, 50)
    publish(ground.behavior_tree, DAY1, "看手机", 23, 40)  # 23:45 之前，不在
    own = publish(ground.behavior_tree, DAY2, "关灯", 0, 5)

    rows = slot_neighbourhood(own, cache_for(ground), slot_minutes=15, half_width=1)

    assert [row.uri for row in rows] == [late, own]


def test_the_after_part_crosses_midnight_into_the_next_day(tmp_path) -> None:
    """23:50 开始、最后所见 23:58，k=1 → 之后段到次日 00:15；后一天目录里的行要取到。"""

    ground = site(tmp_path)
    own = publish(ground.behavior_tree, DAY1, "刷牙", 23, 50, lasts_minutes=8)
    early = publish(ground.behavior_tree, DAY2, "关灯", 0, 5)
    publish(ground.behavior_tree, DAY2, "睡觉", 0, 20)  # 00:15 之后，不在

    rows = slot_neighbourhood(own, cache_for(ground), slot_minutes=15, half_width=1)

    assert [row.uri for row in rows] == [own, early]


def test_rows_carry_their_own_days_count_of_their_kind(tmp_path) -> None:
    """跨日的行各带各天的次数：周日 23:50 的刷牙数的是周日全天，不借周一的。"""

    ground = site(tmp_path)
    publish(ground.behavior_tree, DAY1, "刷牙", 8, 0)
    publish(ground.behavior_tree, DAY1, "刷牙", 23, 50)
    own = publish(ground.behavior_tree, DAY2, "刷牙", 0, 5)

    rows = slot_neighbourhood(own, cache_for(ground), slot_minutes=15, half_width=1)

    assert [(row.at.day, row.day_count) for row in rows] == [(DAY1.day, 2), (DAY2.day, 1)]
    assert rows[1].last_observed_at == at(DAY2, 0, 15)


def test_a_zero_half_width_keeps_only_the_own_slots(tmp_path) -> None:
    ground = site(tmp_path)
    publish(ground.behavior_tree, DAY1, "收拾球包", 16, 25)
    own = publish(ground.behavior_tree, DAY1, "打球", 16, 40)  # 最后所见 16:50，窗口 [16:30, 17:00)
    same_slot = publish(ground.behavior_tree, DAY1, "喝水", 16, 55)

    rows = slot_neighbourhood(own, cache_for(ground), slot_minutes=15, half_width=0)

    assert [row.uri for row in rows] == [own, same_slot]


def test_an_unknown_occurrence_is_refused(tmp_path) -> None:
    ground = site(tmp_path)
    missing = "behavior://occurrences/2026/08/15/打球--20260815T164000000000%2B0800.md"
    with pytest.raises(KeyError, match="not on the behaviour tree"):
        slot_neighbourhood(missing, cache_for(ground), slot_minutes=15, half_width=1)


# ── 此刻侧 ───────────────────────────────────────────────────────────────────


def test_the_now_side_runs_from_k_slots_back_up_to_this_very_moment(tmp_path) -> None:
    """[floor(now) − k 槽, now]：右端闭，此刻这一秒发生的算，之后的不算。

    now = 16:15 正好是槽起点，k=2 → 窗口 [15:45, 16:15]。
    """

    ground = site(tmp_path)
    publish(ground.behavior_tree, DAY1, "换衣服", 15, 40)  # 15:45 之前，不在
    call = publish(ground.behavior_tree, DAY1, "与朋友通电话", 15, 50)
    pack = publish(ground.behavior_tree, DAY1, "收拾球包", 16, 5)
    door = publish(ground.behavior_tree, DAY1, "出门", 16, 15)  # 此刻这一分钟，在
    publish(ground.behavior_tree, DAY1, "打球", 16, 16)  # 此刻之后，不在

    rows = slot_neighbourhood_until(at(DAY1, 16, 15), cache_for(ground), slot_minutes=15, half_width=2)

    assert [row.uri for row in rows] == [call, pack, door]


def test_the_now_side_orders_rows_by_instant_across_days(tmp_path) -> None:
    ground = site(tmp_path)
    yesterday = publish(ground.behavior_tree, DAY1, "刷牙", 23, 50)
    today = publish(ground.behavior_tree, DAY2, "关灯", 0, 5)

    rows = slot_neighbourhood_until(at(DAY2, 0, 10), cache_for(ground), slot_minutes=15, half_width=1)

    assert [row.uri for row in rows] == [yesterday, today]
    assert rows[0].at < rows[1].at and rows[1].at - rows[0].at == timedelta(minutes=15)
