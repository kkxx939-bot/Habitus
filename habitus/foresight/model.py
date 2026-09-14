"""预测层的产物类型里**不带语义背景**的那些：层、四层拆解、此刻。

本模块只 import ``datetime``。带着情景视图的产物（``LayerBackground``、``CandidateEvidence``）
住在 ``context`` 与 ``assemble``，因为它们引用 ``scene.views`` 的类型——让数字这一侧也认识
读侧，"出处从预测树来、不从别处重推"这条就没有边界钉着了。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from habitus.foresight.errors import ForesightError

# 收缩链的四层，从假设最弱的借用开始（与 ``prediction.nodes.derive_all`` 同序）。
LAYER_NAMES = ("slot", "pool", "cross_weekday", "all_day")
LAYER_LABELS = {
    "slot": "本槽",
    "pool": "邻域",
    "cross_weekday": "跨周几",
    "all_day": "全天",
}


@dataclass(frozen=True)
class Layer:
    """收缩链上的一层：这一层给出的数、它是哪几天攒出来的、其中哪几天语义侧看不见。

    ``value`` 在四层上不是同一种东西，这是树的发布形状决定的，不是本层的选择：

    - ``slot`` / ``pool`` / ``cross_weekday``：``hits / exposure`` 的**裸比值**（分子分母都摆在
      这里），没有收缩、没有平滑——与收缩链在那一层看到的证据逐位一致。判断者要能自己掂量
      这个数薄不薄，而不是只拿到一个被链揉过、来龙去脉看不见的 0.5。
      **这两个是衰减加权量**（每天贡献 ``时间衰减 × 覆盖比例`` ≤ 1），不是次数：12 个周一可能
      写成 5.74/5.74，那不是"5.74 次"。``exposure`` 为 0 表示这一格**根本没看过**（那个周几
      还没进过观测跨度），不是"看了没发生"。
    - ``all_day``：树**已发布的率**（含 Laplace 平滑），树上没有对应的裸账本，所以 ``hits`` /
      ``exposure`` 为 None。注意它的分母是**全部日历日的曝光**，而 ``days`` 只是分子那边的
      日子——"他那天没做"的那些天不在 ``days`` 里。

    ``days`` 是这一层的出处，直接来自树上的 ``cell_days``，**不由本层按覆盖日重推**——树读
    整棵行为树、按覆盖扣、按衰减加权，重推会得到另一份事实。``unassociated`` 是这批日子里语义层
    还没关联完成的那几天：它们有数、没背景，必须明说，否则判断者分不清"这个数字只有 6 天"和
    "有 8 天、其中 2 天没背景"。
    """

    name: str
    value: float
    days: tuple[date, ...]
    unassociated: tuple[date, ...]
    hits: float | None = None
    exposure: float | None = None

    def __post_init__(self) -> None:
        if self.name not in LAYER_NAMES:
            raise ForesightError(f"unknown shrinkage layer: {self.name!r}")
        for label, series in (("days", self.days), ("unassociated", self.unassociated)):
            if any(later <= earlier for earlier, later in zip(series, series[1:], strict=False)):
                raise ForesightError(f"layer {self.name} lists its {label} out of order or twice")
        if not set(self.unassociated) <= set(self.days):
            raise ForesightError(f"layer {self.name} counts unassociated days it did not come from")
        # 裸账本要么两个都给、要么都不给：只给一半的话，"分子分母摆出来让人自己掂量"这件事
        # 就做了一半，而读的人无从知道缺的是哪一半。
        if (self.hits is None) != (self.exposure is None):
            raise ForesightError(f"layer {self.name} must publish both hits and exposure, or neither")
        # 给了裸账本，那个比值就必须真的是这两个数算出来的——三个数互相印证正是它们的全部意义。
        if self.hits is not None and self.exposure is not None:
            expected = self.hits / self.exposure if self.exposure > 0.0 else 0.0
            if abs(self.value - expected) > 1e-9 * max(1.0, abs(expected)):
                raise ForesightError(f"layer {self.name} reports a value its own ledger does not produce")

    @property
    def label(self) -> str:
        return LAYER_LABELS[self.name]


        missing = set(self.unassociated)
        return tuple(day for day in self.days if day not in missing)


@dataclass(frozen=True)
class Provenance:
    """一个候选在这一刻的四层拆解：每层的数字连着它自己的那批日子。

    判断者据此看到的不是一个光秃秃的概率，而是"本槽 2/4、邻域 6/20、跨周几 0.1、全天 0.05"，
    并且每一层配的历史背景都恰好来自**算出那个数的那几天**。
    """

    slot: Layer
    pool: Layer
    cross_weekday: Layer
    all_day: Layer

    def __iter__(self):
        return iter((self.slot, self.pool, self.cross_weekday, self.all_day))

    @property
    def unassociated(self) -> tuple[date, ...]:
        """四层合起来有数、却没有语义背景的日子（升序去重）。"""

        return tuple(sorted({day for layer in self for day in layer.unassociated}))


@dataclass(frozen=True)
class Moment:
    """"此刻"在钟面上的位置，外加当地日历对今天的说法。

    契约：``at`` 必须是**主体的本地时刻**（与行为树上 occurrence 同一偏移）。今天是哪一天、
    此刻在哪个槽，全部从它的本地时分算——传 UTC 进来会读错一天、也会读错槽。``slot`` 与
    ``weekday`` 由装配层用树自己的槽宽映射出来，不在这里重算，否则钟面就有了两种算法。
    """

    at: datetime
    slot: int
    weekday: int
    day_note: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.at, datetime) or self.at.utcoffset() is None:
            raise ForesightError("moment must be a timezone-aware local datetime")

    @property
    def day(self) -> date:
        return self.at.date()


__all__ = ["LAYER_LABELS", "LAYER_NAMES", "Layer", "Moment", "Provenance"]
