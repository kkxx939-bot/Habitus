"""预测层的运行配置（``19-预测层方案.md`` 第六节）。

这里没有算法参数：时窗来自历史那几次的实际时刻，回看多远由每条关系自己的实测跨度定，转移窗就是预测树的
``transition_window_slots``；升不升从账上算（裁定 49），α 借关系引擎的 ``forward_alpha``、遗忘借预测树的
``decay_half_life_days``（遗忘随统一生命周期再定，裁定 15）。这里只有四类东西：开关、可接受的误提醒率（产品决定，不随用户变，
裁定 52）、重轻多久重判、保护闸。没定的几项先给默认值、不挡开工（裁定 46）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from habitus.config.loader import construct_config

_INT_BOUNDS = (
    ("daily_reminder_cap", 1, 100),
    ("judge_transient_retries", 0, 20),
)
_FLOAT_BOUNDS = (
    ("holdout_rate", 0.0, 0.5),
    ("holdout_rate_heavy", 0.0, 0.5),
    ("unit_prior_strength", 0.0, 1_000.0),
    ("line_floor", 0.01, 0.99),
    ("line_ceiling", 0.01, 0.99),
    ("cost_ratio_unjudged", 0.1, 10.0),
    ("severity_review_days", 1.0, 3_650.0),
    ("tick_interval_seconds", 60.0, 86_400.0),
    ("judge_transient_retry_delay_seconds", 0.0, 600.0),
    ("worker_shutdown_timeout_seconds", 1.0, 3_600.0),
)
_FLAGS = ("enabled", "shadow_uses_unestablished")


@dataclass(frozen=True)
class ForesightConfig:
    """预测层是否启用、线的上下限、重轻多久重判、几道保护闸。"""

    enabled: bool = False
    # ── 影子 → 提醒的线：每本账一条，在上下限之间波动（裁定 58 第 1 条）──
    # 线 = 1 / (1 + c × e)：c 是这件事本身的代价比（漏一次相当于错提醒几次，模型按账判），e 是提醒的效果上界（提醒组 − 扣发组）。
    # 上下限是产品的保护闸；没判过的按 ``cost_ratio_unjudged``（0.5 → 线 0.67，按轻的那一头）。暂定值，回放后定。
    line_floor: float = 0.2
    line_ceiling: float = 0.9
    cost_ratio_unjudged: float = 0.5
    # 重轻按账判一次，过这么多天重判（归统一生命周期，裁定 15；在那之前做成配置）。
    severity_review_days: float = 60.0
    # ── 没定的，先给默认值（裁定 46；常识冷启动没有消费者，开关删了，要做时见待办）──
    # 影子预期可来自历史上跟过、但语义树还没成立的关系；开口仍只用成立的。默认开（裁定 56 第 6 条：成立关系上半年几乎
    # 没有——甲盘 3 月底 0 条、5 月底 10 条——候选进影子让账早几个月开始攒；影子不开口）。
    shadow_uses_unestablished: bool = True
    # ── 保护闸 ──
    # 每天最多主动提醒几次；应当很少触发，常触发说明升级条件该重看。
    daily_reminder_cap: int = 6
    # 随机扣发的比例：按已知概率不说一部分提醒，是估提醒效应的唯一因果手段（裁定 50 默认开）。最轻一级与最重一级各一个，
    # 中间按级插；重的也扣、比例更小（裁定 58 第 7 条：不扣就估不出提醒在重的事上有没有用）。暂定值，回放后定。
    holdout_rate: float = 0.1
    holdout_rate_heavy: float = 0.05
    # ── 行为单元（裁定 56 第 12 条）──
    # 单元读数分层退（同一组 → 同样多 → 最强那一个），每一层以下一层的读数为先验，先验的分量最多这么多（还不超过下一层自己的
    # 有效样本）：没见过的组合贴着更粗的那一层，见得越多越按它自己。暂定值，回放后定。
    unit_prior_strength: float = 8.0
    # 模型对传输层瞬态错的有界重试。
    judge_transient_retries: int = 1
    judge_transient_retry_delay_seconds: float = 5.0
    # 常驻 worker：白天多久一拍（一拍 = 建预期、到时窗的决定说不说；默认一个槽），与停机时限。
    tick_interval_seconds: float = 900.0
    worker_shutdown_timeout_seconds: float = 30.0

    def __post_init__(self) -> None:
        for name in _FLAGS:
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"foresight.{name} must be a boolean")
        for name, lower, upper in _INT_BOUNDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"foresight.{name} must be an integer between {lower} and {upper}")
        for name, low, high in _FLOAT_BOUNDS:
            number = getattr(self, name)
            if isinstance(number, bool) or not isinstance(number, int | float) or not low <= float(number) <= high:
                raise ValueError(f"foresight.{name} must be a number between {low:g} and {high:g}")
        if self.line_floor >= self.line_ceiling:
            raise ValueError("foresight.line_floor must lie below foresight.line_ceiling")
        if self.holdout_rate_heavy > self.holdout_rate:
            raise ValueError("foresight.holdout_rate_heavy must not exceed foresight.holdout_rate")

    @classmethod
    def from_mapping(cls, value: Any) -> ForesightConfig:
        return construct_config(cls, value, "config.foresight")


__all__ = ["ForesightConfig"]
