"""runtime 接线测试共用的现场常量。

放在夹具模块而不是某个测试文件里：跨测试文件 import 会造成循环（集成测试与单元测试互相引用），
而且它让"谁是谁的现场"变得不可读。
"""

from __future__ import annotations

#: 预测夜批的十三个启动档 + 开关。配置层要求开着就必须全给，所以任何要开预测层的现场都要这一份。
STARTUP_PARAMETERS: dict[str, object] = {
    "enabled": True,
    "slot_minutes": 15,
    "decay_half_life_days": 60,
    "recent_half_life_days": 14,
    "recurrence_half_life_days": 365,
    "pool_half_width": 3,
    "shrink_slot_to_pool": 5,
    "shrink_pool_to_weekday": 5,
    "shrink_weekday_to_all_day": 5,
    "laplace_epsilon": 0.5,
    "transition_window_seconds": 1800,
    "shrink_edge": 5,
    "recurrence_window_days": 90,
    "rebuild_interval_seconds": 86400,
    "published_generations": 7,
}

__all__ = ["STARTUP_PARAMETERS"]
