"""四层拆解与它们各自的出处日，以及"每层只按自己那批日子取背景"。

这里钉的是**对应关系**，不是数值精度：数字在预测树的单测里已经手算过了，本文件要证明的是
每一层配到的日子恰好是算出那个数的那几天，一天不多、一天不少。
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from habitus.foresight import CellIndex, ForesightError, Layer, layer_background, provenance
from habitus.prediction.model import SlotKey
from habitus.prediction.query import neighbourhood
from tests.unit.foresight.fixtures import Ground, at, slot_of

MONDAY = date(2026, 8, 3)


def days(*offsets: int) -> tuple[date, ...]:
    return tuple(MONDAY + timedelta(days=offset) for offset in offsets)


def ground_with_a_weekly_habit(tmp_path) -> Ground:
    """周一 19:00 打球三周，另外三条各自落在不同的层上：

    - 周一 19:30：同一个周几、在 ±2 槽的邻域里 → 只有邻域层看得见；
    - 周三 19:00：同一个槽、不同周几        → 只有跨周几层看得见；
    - 周一 08:00：同一个周几、离得很远的槽    → 只有全天层看得见。
    """

    ground = Ground(tmp_path, now=at(MONDAY + timedelta(days=30), 12, 0))
    for week in range(3):
        ground.record(MONDAY + timedelta(days=7 * week), "打球", 19, 0, kind="打球")
    ground.record(MONDAY + timedelta(days=21), "打球", 19, 30, kind="打球")
    ground.record(MONDAY + timedelta(days=2), "打球", 19, 0, kind="打球")
    ground.record(MONDAY + timedelta(days=28), "打球", 8, 0, kind="打球")
    return ground


def test_each_layer_carries_exactly_the_days_its_own_number_came_from(tmp_path) -> None:
    """四层的日子照着**那一层实际怎么算的**去并，不多不少。

    并法错一点就会让判断者看着邻域的数字读全天的背景——而四层拆解存在的全部理由正是让他
    看清"这个 0.5 是从哪几天来的"。
    """

    tree = ground_with_a_weekly_habit(tmp_path).tree()
    cells = CellIndex.of(tree)
    slot = SlotKey(weekday=0, slot=slot_of(19, 0))
    layers = provenance(cells, "打球", slot, half_width=2, grouped=lambda day: True)

    assert layers.slot.days == days(0, 7, 14)  # 就是这一格
    assert layers.pool.days == days(0, 7, 14, 21)  # 加上 19:30 那次
    # 链上第三层只把周几从 1 扩到 7、**时间窗保持 ±k 不变**（见 nodes._dense_chain 的 cross），
    # 所以它是 pool ∪ 周三那次，而不是"七个周几在精确槽上"。
    assert layers.cross_weekday.days == days(0, 2, 7, 14, 21)
    assert layers.all_day.days == days(0, 2, 7, 14, 21, 28)  # 全部
    assert [layer.name for layer in layers] == ["slot", "pool", "cross_weekday", "all_day"]


def test_the_three_chain_layers_expose_the_raw_ledger_and_all_day_does_not(tmp_path) -> None:
    """本槽 / 邻域 / 跨周几都给裸账本（分子分母摆出来），只有全天是树已发布的率。

    前三层都能从格子与曝光精确复原成收缩链在那一层看到的证据；全天在树上就只有一个率，
    编一个裸账本出来会让判断者以为那也是能自己掂量的比值。
    """

    tree = ground_with_a_weekly_habit(tmp_path).tree()
    cells = CellIndex.of(tree)
    slot = SlotKey(weekday=0, slot=slot_of(19, 0))
    layers = provenance(cells, "打球", slot, half_width=2, grouped=lambda day: True)

    cell = tree.nodes[(slot, "打球")]
    assert layers.slot.hits == pytest.approx(cell.counts.occurred_days)
    assert layers.slot.exposure == pytest.approx(tree.exposure[slot].observed_days)
    assert layers.slot.value == pytest.approx(layers.slot.hits / layers.slot.exposure)
    assert layers.pool.hits is not None and layers.pool.hits > layers.slot.hits  # 邻域多罩进 19:30 那次
    assert layers.pool.exposure is not None and layers.pool.exposure > layers.slot.exposure
    # 跨周几只扩周几：分子分母都比邻域大（多了另外六个周几），窗口宽度不变。
    assert layers.cross_weekday.exposure is not None and layers.cross_weekday.exposure > layers.pool.exposure
    assert layers.all_day.hits is None and layers.all_day.exposure is None
    assert layers.all_day.value == pytest.approx(tree.baselines["打球"])


def test_the_cross_weekday_layer_is_the_shrinkage_chains_third_layer(tmp_path) -> None:
    """跨周几必须是链上真正的第三层，不是 ``weekday_baselines``。

    链上是 ``Σ_w pooled_top[w][slot] / Σ_w pooled_bottom[w][slot]``（含 ±k 邻域、不平滑）；
    ``weekday_baselines`` 是按精确槽归并、带 Laplace 的 ``lift_周几`` 分母，**从来没有传进
    ``_dense_chain``**。取错了会把周规律的证据抹掉一个池宽的量级——12 周每周一 19:00 的数据上，
    邻域相对链上第三层是 7 倍，相对 ``weekday_baselines`` 只剩 1.35 倍，读出来就成了
    "这个时段大家都忙，周一不特别"。
    """

    tree = ground_with_a_weekly_habit(tmp_path).tree()
    slot = SlotKey(weekday=0, slot=slot_of(19, 0))
    layers = provenance(CellIndex.of(tree), "打球", slot, half_width=2, grouped=lambda day: True)
    pool = neighbourhood(tree, slot, 2)
    expected_top = sum(
        tree.nodes[(SlotKey(weekday=weekday, slot=key.slot), "打球")].counts.occurred_days
        for weekday in range(7)
        for key in pool
        if (SlotKey(weekday=weekday, slot=key.slot), "打球") in tree.nodes
    )
    expected_bottom = sum(
        tree.exposure[SlotKey(weekday=weekday, slot=key.slot)].observed_days
        for weekday in range(7)
        for key in pool
        if SlotKey(weekday=weekday, slot=key.slot) in tree.exposure
    )
    assert layers.cross_weekday.hits == pytest.approx(expected_top)
    assert layers.cross_weekday.exposure == pytest.approx(expected_bottom)
    assert layers.cross_weekday.value != pytest.approx(tree.weekday_baselines["打球"][slot.slot])


def test_the_neighbourhood_wraps_inside_one_weekday(tmp_path) -> None:
    """钟面环形但**不跨周几**：周一 23:45 的邻域含周一 00:05，不含周二 00:05。

    节点那边的池化就是按周几逐行做环形窗口和的；这里换一个口径，数字与背景当场对不上。
    """

    ground = Ground(tmp_path, now=at(MONDAY + timedelta(days=30), 12, 0))
    ground.record(MONDAY, "夜宵", 23, 45, kind="夜宵")
    ground.record(MONDAY + timedelta(days=7), "夜宵", 0, 5, kind="夜宵")  # 周一凌晨
    ground.record(MONDAY + timedelta(days=1), "夜宵", 0, 5, kind="夜宵")  # 周二凌晨
    cells = CellIndex.of(ground.tree())
    late = SlotKey(weekday=0, slot=slot_of(23, 45))
    layers = provenance(cells, "夜宵", late, half_width=2, grouped=lambda day: True)
    assert layers.slot.days == days(0)
    assert layers.pool.days == days(0, 7)  # 绕过午夜，但仍在周一那一行
    assert days(1)[0] not in layers.pool.days


def test_days_with_numbers_but_no_scene_are_named_not_dropped(tmp_path) -> None:
    """出处日里情景树没归组的那几天要明说"有数、没背景"。

    树读整棵行为树，语义侧只看得到已归组的日子。不说出来，判断者就分不清"这个数字只有两天"
    和"有三天、其中一天没背景"——后者该让他更相信这个规律，不是更不相信。
    """

    ground = ground_with_a_weekly_habit(tmp_path)
    cells = CellIndex.of(ground.tree())
    slot = SlotKey(weekday=0, slot=slot_of(19, 0))
    missing = {MONDAY + timedelta(days=14)}
    layers = provenance(cells, "打球", slot, half_width=2, grouped=lambda day: day not in missing)

    assert layers.slot.days == days(0, 7, 14)
    assert layers.slot.ungrouped == days(14)
    assert layers.slot.grouped == days(0, 7)
    assert layers.ungrouped == days(14)  # 四层合起来，去重


def test_a_candidate_that_never_hit_this_cell_has_an_empty_slot_layer(tmp_path) -> None:
    """这一格从没发生过是**正常答案**：本槽空着，上面几层照样有数与日子。"""

    tree = ground_with_a_weekly_habit(tmp_path).tree()
    cells = CellIndex.of(tree)
    layers = provenance(
        cells, "打球", SlotKey(weekday=0, slot=slot_of(3, 0)), half_width=2, grouped=lambda day: True
    )
    assert layers.slot.days == () and layers.slot.hits == 0.0 and layers.slot.value == 0.0
    assert layers.pool.days == ()
    assert layers.all_day.days == days(0, 2, 7, 14, 21, 28)


def test_layer_guards(tmp_path) -> None:
    tree = ground_with_a_weekly_habit(tmp_path).tree()
    cells = CellIndex.of(tree)
    slot = SlotKey(weekday=0, slot=slot_of(19, 0))
    with pytest.raises(ForesightError):
        provenance(cells, "", slot, half_width=2, grouped=lambda day: True)
    with pytest.raises(ForesightError):
        provenance(cells, "打球", slot, half_width=2, grouped=None)  # type: ignore[arg-type]
    with pytest.raises(ForesightError):
        Layer(name="不认识的层", value=0.0, days=(), ungrouped=())
    with pytest.raises(ForesightError):
        Layer(name="slot", value=0.0, days=days(0, 0), ungrouped=())
    with pytest.raises(ForesightError):
        Layer(name="slot", value=0.0, days=days(0), ungrouped=days(7))


# --- 语义侧：每层按自己的日子取背景 -----------------------------------------------------


def test_each_layer_reads_its_own_days_with_its_own_slot_filter(tmp_path) -> None:
    """**四层各取各的**：本槽单格、邻域 ±k、跨周几 ±k 但跨七个周几、全天不过滤时刻。

    这四条必须能互相区分开。现场刻意让每一条多罩进一条别的层罩不到的历史：19:30 那次只有
    邻域及以上看得见、周三那次只有跨周几及以上看得见、早上八点那次只有全天看得见。少了任何
    一条，把跨周几的半宽写成 0、或把邻域的写成 0，测试都照样全绿——背景就会比它要解释的
    数字宽一截或窄一截，而没人看得出来。
    """

    ground = Ground(tmp_path, now=at(MONDAY + timedelta(days=30), 12, 0))
    ground.record(MONDAY, "打球", 19, 0, kind="打球")
    ground.record(MONDAY + timedelta(days=7), "打球", 19, 30, kind="打球")  # 同周几、在 ±3 邻域里
    ground.record(MONDAY + timedelta(days=2), "打球", 19, 0, kind="打球")  # 周三同一时刻
    ground.record(MONDAY + timedelta(days=14), "打球", 8, 0, kind="打球")  # 同周几、离得很远的槽
    grouped_days = (MONDAY, MONDAY + timedelta(days=2), MONDAY + timedelta(days=7), MONDAY + timedelta(days=14))
    ground.group(*grouped_days)
    cells = CellIndex.of(ground.tree())
    slot = SlotKey(weekday=0, slot=slot_of(19, 0))
    cache = ground.cache()
    layers = provenance(cells, "打球", slot, half_width=3, grouped=cache.covered)

    def background(layer):
        return layer_background(
            layer,
            "打球",
            cache,
            slot_minutes=15,
            slot_index=slot.slot,
            half_width=3,
            window_days=30,
            transition_window_seconds=7_200.0,
            max_days=10,
        )

    def shown(layer):
        return sorted((view.day, view.at.hour, view.at.minute) for view in background(layer).views)

    monday_evening = (MONDAY, 19, 0)
    monday_half_past = (MONDAY + timedelta(days=7), 19, 30)
    wednesday_evening = (MONDAY + timedelta(days=2), 19, 0)
    monday_morning = (MONDAY + timedelta(days=14), 8, 0)

    assert shown(layers.slot) == [monday_evening]
    assert shown(layers.pool) == sorted([monday_evening, monday_half_past])
    assert shown(layers.cross_weekday) == sorted([monday_evening, wednesday_evening, monday_half_past])
    assert shown(layers.all_day) == sorted(
        [monday_evening, wednesday_evening, monday_half_past, monday_morning]
    )
    assert background(layers.slot).dropped_days == 0


def test_the_protective_limit_says_how_many_days_it_left_out(tmp_path) -> None:
    """保护闸截掉更早的日子时要报出来，否则"给你看的这几天"会被读成"一共就这几天"。"""

    ground = Ground(tmp_path, now=at(MONDAY + timedelta(days=30), 12, 0))
    for week in range(4):
        ground.record(MONDAY + timedelta(days=7 * week), "打球", 19, 0, kind="打球")
    ground.group(*(MONDAY + timedelta(days=7 * week) for week in range(4)))
    cells = CellIndex.of(ground.tree())
    slot = SlotKey(weekday=0, slot=slot_of(19, 0))
    cache = ground.cache()
    layers = provenance(cells, "打球", slot, half_width=2, grouped=cache.covered)
    background = layer_background(
        layers.slot,
        "打球",
        cache,
        slot_minutes=15,
        slot_index=slot.slot,
        half_width=2,
        window_days=30,
        transition_window_seconds=7_200.0,
        max_days=2,
    )
    assert background.dropped_days == 2
    assert [view.day for view in background.views] == list(days(14, 21))  # 留最近的两天
    with pytest.raises(ForesightError):
        layer_background(
            layers.slot, "打球", cache, slot_minutes=15, slot_index=slot.slot, half_width=2,
            window_days=30, transition_window_seconds=7_200.0, max_days=0,
        )
