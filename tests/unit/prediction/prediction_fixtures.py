"""时间预测树测试共用的构造件。

时间一律是**本地时刻 + 显式偏移**（与行为树同一约定）；钟面槽位按本地时分映射。
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta, timezone

from prediction import builder, query
from prediction.config import PredictionTreeConfig
from prediction.model import BehaviorSnapshot, ObservedAction, ObservedGap, PredictionTree, SlotKey

CST = timezone(timedelta(hours=8))
# 2026-08-03 是周一，便于把"周几"算清楚。
FIRST_DAY = date(2026, 8, 3)


def config(**overrides) -> PredictionTreeConfig:
    """一组便于手算的参数：收缩几乎关掉，好让计数与曝光的比值能直接对上。

    **它不是生产档**。真实启动档在 ``Config/example.yaml``（收缩强度 5、ε=0.5），两者的
    收缩强度差 5000 倍，所以这组参数下跑绿并不代表收缩链本身被验证过——收缩相关的行为
    一律用 ``production_config()``。
    """

    values = dict(
        slot_minutes=15,
        decay_half_life_days=3_650.0,  # 上限；测试跨度内几乎不衰减，计数可以手算
        recent_half_life_days=14.0,
        recurrence_half_life_days=3_650.0,  # 同 decay：测试跨度内几乎不衰减，间隔可手算
        pool_half_width=0,  # 默认不池化，收缩链退化成两层，便于逐项验证
        shrink_slot_to_pool=0.001,
        shrink_pool_to_weekday=0.001,
        shrink_weekday_to_all_day=0.001,
        laplace_epsilon=0.001,
        transition_window_seconds=7_200.0,
        shrink_edge=0.001,
        recurrence_window_days=90.0,
        rebuild_interval_seconds=86_400.0,
        published_generations=3,
    )
    values.update(overrides)
    return PredictionTreeConfig(**values)


def production_config(**overrides) -> PredictionTreeConfig:
    """与 ``Config/example.yaml`` 的启动档一致的参数。

    收缩链、池化、lift 这些"参数一变行为就变"的东西必须在这组数值下验证：用几乎关掉收缩的
    夹具去测收缩，等于什么都没测（变异测试实证：把收缩顺序整个倒过来，全部单测照样全绿）。
    """

    values = dict(
        slot_minutes=15,
        decay_half_life_days=60.0,
        recent_half_life_days=14.0,
        recurrence_half_life_days=365.0,
        pool_half_width=2,
        shrink_slot_to_pool=5.0,
        shrink_pool_to_weekday=5.0,
        shrink_weekday_to_all_day=5.0,
        laplace_epsilon=0.5,
        transition_window_seconds=1_800.0,
        shrink_edge=5.0,
        recurrence_window_days=90.0,
        rebuild_interval_seconds=86_400.0,
        published_generations=7,
    )
    values.update(overrides)
    return PredictionTreeConfig(**values)


def at(day_offset: int, hour: int, minute: int = 0, second: int = 0) -> datetime:
    """从 FIRST_DAY 起第 day_offset 天的本地时刻。"""

    day = FIRST_DAY + timedelta(days=day_offset)
    return datetime(day.year, day.month, day.day, hour, minute, second, tzinfo=CST)


def action(name: str, day_offset: int, hour: int, minute: int = 0) -> ObservedAction:
    moment = at(day_offset, hour, minute)
    return ObservedAction(action=name, started_at=moment, day=moment.date())


def gap(
    day_offset: int, start_hour: int, end_hour: int, *, watched: bool = True
) -> ObservedGap:
    """默认造「没读懂」那一类（我们在看、只是读不出）——树里目前只有这一类。"""

    return ObservedGap(
        started_at=at(day_offset, start_hour),
        ended_at=at(day_offset, end_hour),
        watched=watched,
    )


def reference(day_offset: int) -> date:
    return FIRST_DAY + timedelta(days=day_offset)


def daily(name: str, days: int, hour: int, minute: int = 0, *, start: int = 0) -> list[ObservedAction]:
    """连续若干天在同一时刻做同一件事。"""

    return [action(name, start + offset, hour, minute) for offset in range(days)]


def weekly(
    name: str, weeks: int, weekday: int, hour: int, minute: int = 0
) -> list[ObservedAction]:
    """每周固定某一天做一件事；weekday 0=周一（FIRST_DAY 就是周一）。"""

    return [action(name, weekday + 7 * week, hour, minute) for week in range(weeks)]


class Published:
    """按 ``[(SlotKey, 动作)]`` 读一棵已建好的树——走的就是 ``query`` 组装候选的那条路。

    率与 lift 都住在 (周几, 动作) 的曲线与基线上，格子只留原始账本与这一格的机会数；测试
    要断言的是**发布出去的答案**，所以这里不重新拼一遍，直接调 ``query.node_at``。
    """

    def __init__(self, tree: PredictionTree) -> None:
        self.tree = tree

    def __getitem__(self, key: tuple[SlotKey, str]):
        candidate = query.node_at(self.tree, key[0], key[1])
        if candidate is None:
            raise KeyError(key)
        return candidate

    def __contains__(self, key: tuple[SlotKey, str]) -> bool:
        return key in self.tree.nodes


def publish(actions, gaps=(), *, config: PredictionTreeConfig, reference_day: date) -> Published:
    """把一批行为按夜批的真实顺序建成树，再包成按格子读的视图。"""

    ordered = tuple(sorted(actions, key=lambda item: item.started_at))
    snapshot = BehaviorSnapshot(
        actions=ordered, gaps=tuple(gaps), concurrent=(), skipped_duplicates=0
    )
    tree = builder.build(
        snapshot,
        config=config,
        reference=reference_day,
        built_at=datetime(2026, 12, 31, tzinfo=UTC),
    )
    return Published(tree)


__all__ = [
    "CST",
    "FIRST_DAY",
    "Published",
    "action",
    "at",
    "config",
    "daily",
    "gap",
    "publish",
    "reference",
    "weekly",
]
