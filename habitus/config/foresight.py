"""预测层的运行配置（``19-预测层方案.md`` 第六节）。

这里没有算法参数：时窗来自历史那几次的实际时刻，回看多远由每条关系自己的实测跨度定，时间尺子在 ``config.temporal``；
升不升从账上算（裁定 49），α 借关系引擎的 ``forward_alpha``。这里是开关、可接受的误提醒率（产品决定，不随用户变，裁定 52）、
保护闸，以及账本遗忘的半衰期（按这本账自己的机会数，裁定 57；临时值，回放后定）。没定的几项先给默认值、不挡开工（裁定 46）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from habitus.config.loader import construct_config

_INT_BOUNDS = (
    ("daily_reminder_cap", 1, 100),
    ("judge_transient_retries", 0, 20),
    ("open_reminders_judged", 1, 50),
)
_FLOAT_BOUNDS = (
    ("false_alarm_rate", 0.01, 0.99),
    ("tick_interval_seconds", 60.0, 86_400.0),
    ("judge_transient_retry_delay_seconds", 0.0, 600.0),
    ("worker_shutdown_timeout_seconds", 1.0, 3_600.0),
    ("half_life_opportunities", 1.0, 200.0),
)
_FLAGS = ("enabled", "shadow_uses_unestablished")


@dataclass(frozen=True)
class ForesightConfig:
    """预测层是否启用、可接受的误提醒率、几道保护闸。"""

    enabled: bool = False
    # ── 影子 → 提醒的线 = 1 − 可接受的误提醒率，所有账同一条（裁定 58 第 1 条：扣发与急缓这一版都不做）──
    # 下界过线 = 有 1 − α 的把握说出口的那几次里他不做的不超过这个比例。0.33 → 线 0.67。暂定值，回放后定。
    false_alarm_rate: float = 0.33
    # ── 没定的，先给默认值（裁定 46；常识冷启动没有消费者，开关删了，要做时见待办）──
    # 影子预期可来自历史上跟过、但语义树还没成立的关系；开口仍只用成立的。默认开（裁定 56 第 6 条：成立关系上半年几乎
    # 没有——甲盘 3 月底 0 条、5 月底 10 条——候选进影子让账早几个月开始攒；影子不开口）。
    shadow_uses_unestablished: bool = True
    # ── 保护闸 ──
    # 每天最多主动提醒几次；应当很少触发，常触发说明升级条件该重看。
    daily_reminder_cap: int = 6
    # 每段会话进融合时最多带几条开着的提醒给模型判回应（按截止由近到远；没带进去的只按做结算；裁定 61 第 5 条）。
    # 启动值，以后从回放定。
    open_reminders_judged: int = 5
    # ── 行为单元（裁定 56 第 12 条）──
    # 单元读数分层退（同一组 → 同样多 → 最强那一个），"同一组"以"同样多"里别的组合为先验；先验的分量 = 有效样本上限
    # 1 / (1 − 2^(−1/H)) 的这个比例（还不超过先验自己的有效样本）：没见过的组合贴着更粗的那一层，见得越多越按它自己。
    # 按机会数遗忘后有效样本有上限（H = 8 时约 12），分量写成固定数会占到四成、让 100/100 的组合也过不了线。暂定值，回放后定。
    unit_prior_share: float = 1 / 3
    # 模型对传输层瞬态错的有界重试。
    judge_transient_retries: int = 1
    judge_transient_retry_delay_seconds: float = 5.0
    # 常驻 worker：白天多久一拍（一拍 = 建预期、到时窗的决定说不说；默认一个槽），与停机时限。
    tick_interval_seconds: float = 900.0
    worker_shutdown_timeout_seconds: float = 30.0
    # ── 账本遗忘的半衰期（几次机会，裁定 57）──
    # 一笔证据的分量只看它之后这本账又来了几次机会（边数这本账的全部机会，单元的提醒反馈只数到时窗末还没做的，裁定 60 第 4 条），不看隔了几天：
    # 一个月一次的事不再被压到只剩两三次有效样本。行为单元按每一层自己的刻数、temporal 的时段读数按最近 M 次用同一个 H，
    # M = ⌈H · log₂(1/α)⌉ 由组合根算。暂定值，回放后定。
    half_life_opportunities: float = 8.0
    # ── 没见过的日型怎么建预期（裁定 59 第 7 条的待定项，用户定之前默认不外推）──
    # "skip"：日历关系只在见过他做的那几种日型上建（只在工作日的周三做过，国庆的周三不建）；
    # "official"：没见过的日型按外接放假数据的官方含义归组（调休上班算上班、节假日算休息）。只管建预期，不管关系检验。
    unseen_day_types: str = "skip"

    def __post_init__(self) -> None:
        if self.unseen_day_types not in ("skip", "official"):
            raise ValueError("foresight.unseen_day_types must be 'skip' or 'official'")
        for name in _FLAGS:
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"foresight.{name} must be a boolean")
        for name, lower, upper in _INT_BOUNDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"foresight.{name} must be an integer between {lower} and {upper}")
        share = self.unit_prior_share
        if isinstance(share, bool) or not isinstance(share, int | float) or not 0.0 < float(share) < 1.0:
            raise ValueError("foresight.unit_prior_share must be a number strictly between 0 and 1")
        for name, low, high in _FLOAT_BOUNDS:
            number = getattr(self, name)
            if isinstance(number, bool) or not isinstance(number, int | float) or not low <= float(number) <= high:
                raise ValueError(f"foresight.{name} must be a number between {low:g} and {high:g}")

    @classmethod
    def from_mapping(cls, value: Any) -> ForesightConfig:
        return construct_config(cls, value, "config.foresight")


__all__ = ["ForesightConfig"]
