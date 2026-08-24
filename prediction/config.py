"""时间预测树的 13 个参数（其中 11 个是估计参数，2 个是运维节奏）。

**全部必填、不给 Python 默认值**——这是用户裁定的纪律：它们要在我们自己的数据上定档，
给默认值等于把"没有依据的初值"伪装成"合理缺省"，而且违反"可调运维参数不得藏在构造器里"。
运行值从 ``Config.prediction``（YAML）注入；``Config/example.yaml`` 里给的是一组明确标注
"待定档"的启动值。

# TODO(PRED-TUNING-001): 11 个估计参数**全部没有数据依据**，当前值只是能让管线跑起来的启动档
# （另外两个 rebuild_interval_seconds / published_generations 是运维数，不进 estimation_parameters()）。
# - 定档方法（已定）：``prediction/evaluation.py`` 的留出回测——用前 N−k 天建树、在后 k 天上
#   评估，逐项调参。判据以**校准**为主（预测 P=0.7 的格子实际是不是真的 70% 发生），辅以
#   log-loss / escape_rate / ECE / 覆盖率-精确率曲线。**任何调整不涨指标不合入。**
# - 各参数的定档判据分别是：槽宽 = min(提醒精度目标, 链内相邻间隔低分位)；长 τ = 行为分布的
#   稳定期长度；短 τ = 能多快认出习惯变化而不被单周波动骗到；池化半宽 = 同一习惯的日间时间
#   抖动；三个收缩强度与边收缩 = 留出集上最小化对数损失；转移窗口 = 链内/链间间隔的**双峰谷底**
#   （最依赖真实数据的一个）；复发窗口 = 超过多久算"重新开始"而不是"复发"。
# - **已实测的观察点（定档时首要看这个）**：启动档的三个收缩强度都是 5，而 30 天里同一周几
#   只有 4–5 天，样本量与强度同量级——一个 30/30 的完美习惯只发布成 P≈0.57。排序与 lift 都
#   还对（有测试守着），但绝对值被压得很低，直接拿 P 去卡阈值会漏掉真习惯。收缩强度、池化
#   半宽、槽宽三者要一起定，不能单调一个。
# - 时机：behavior 数据链路产出第一批真实 occurrence 之后。在此之前不要为了"更合理"提前改。
"""

from __future__ import annotations

from dataclasses import dataclass

from prediction.errors import PredictionTreeError
from prediction.model import MINUTES_PER_DAY


def _positive(value: object, name: str, *, maximum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise PredictionTreeError(f"{name} must be a positive number")
    resolved = float(value)
    if maximum is not None and resolved > maximum:
        raise PredictionTreeError(f"{name} must not exceed {maximum:g}")
    return resolved


def _non_negative_int(value: object, name: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
        raise PredictionTreeError(f"{name} must be an integer between 0 and {maximum}")
    return value


@dataclass(frozen=True)
class PredictionTreeConfig:
    """一次重建的全部可调量；构造时全部必填。"""

    slot_minutes: int
    decay_half_life_days: float
    recent_half_life_days: float
    pool_half_width: int
    shrink_slot_to_pool: float
    shrink_pool_to_weekday: float
    shrink_weekday_to_all_day: float
    laplace_epsilon: float
    transition_window_seconds: float
    shrink_edge: float
    recurrence_window_days: float
    rebuild_interval_seconds: float
    published_generations: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.slot_minutes, bool)
            or not isinstance(self.slot_minutes, int)
            or self.slot_minutes <= 0
            or MINUTES_PER_DAY % self.slot_minutes
        ):
            raise PredictionTreeError("slot_minutes must be a positive divisor of 1440")
        _positive(self.decay_half_life_days, "decay_half_life_days", maximum=3_650.0)
        _positive(self.recent_half_life_days, "recent_half_life_days", maximum=3_650.0)
        _non_negative_int(self.pool_half_width, "pool_half_width", maximum=48)
        # 池化是环形的，半宽到了半天就开始绕回来重复数同一个槽：有效样本量凭空翻倍，
        # 而收缩强度还是那个数。这是我们自己两个参数的矛盾，硬拒。
        if 2 * self.pool_half_width >= self.slots_per_day:
            raise PredictionTreeError(
                "pool_half_width must stay below half a day of slots; a wider circular "
                "neighbourhood wraps around and counts the same slots more than once"
            )
        for name in (
            "shrink_slot_to_pool",
            "shrink_pool_to_weekday",
            "shrink_weekday_to_all_day",
            "shrink_edge",
        ):
            _positive(getattr(self, name), name, maximum=10_000.0)
        _positive(self.laplace_epsilon, "laplace_epsilon", maximum=100.0)
        _positive(self.transition_window_seconds, "transition_window_seconds", maximum=86_400.0)
        _positive(self.recurrence_window_days, "recurrence_window_days", maximum=3_650.0)
        _positive(self.rebuild_interval_seconds, "rebuild_interval_seconds", maximum=604_800.0)
        _non_negative_int(self.published_generations, "published_generations", maximum=365)
        if self.published_generations < 1:
            raise PredictionTreeError("published_generations must keep at least one generation")
        # 自洽校验：短窗必须真的比长窗短，否则"趋势"这个量没有意义（它比的是两个同样的东西）。
        if self.recent_half_life_days >= self.decay_half_life_days:
            raise PredictionTreeError(
                "recent_half_life_days must be shorter than decay_half_life_days; "
                "the trend signal compares a short window against a long one"
            )

    @property
    def slots_per_day(self) -> int:
        return MINUTES_PER_DAY // self.slot_minutes

    def estimation_parameters(self) -> dict[str, float | int]:
        """只含**会改变数字**的参数；发布指纹以此为准。

        ``rebuild_interval_seconds``（多久重建一次）与 ``published_generations``（留几代）
        是运维节奏，不进指纹：把它们算进去，改一次保留代数就会让全部已发布的树看起来
        "出自另一套统计"而被读侧拒绝，而它们其实一个数字都没变。
        """

        return {
            name: getattr(self, name)
            for name in (
                "slot_minutes",
                "decay_half_life_days",
                "recent_half_life_days",
                "pool_half_width",
                "shrink_slot_to_pool",
                "shrink_pool_to_weekday",
                "shrink_weekday_to_all_day",
                "laplace_epsilon",
                "transition_window_seconds",
                "shrink_edge",
                "recurrence_window_days",
            )
        }


__all__ = ["PredictionTreeConfig"]
