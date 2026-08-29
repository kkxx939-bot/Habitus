"""时间预测树测试共用的构造件。

时间一律是**本地时刻 + 显式偏移**（与行为树同一约定）；钟面槽位按本地时分映射。
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

from prediction.config import PredictionTreeConfig
from prediction.model import ObservedAction, ObservedGap

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


def gap(day_offset: int, start_hour: int, end_hour: int) -> ObservedGap:
    return ObservedGap(started_at=at(day_offset, start_hour), ended_at=at(day_offset, end_hour))


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


__all__ = [
    "CST",
    "FIRST_DAY",
    "action",
    "at",
    "config",
    "daily",
    "gap",
    "reference",
    "weekly",
]
