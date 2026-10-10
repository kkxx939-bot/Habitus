"""关系检验的全部可调量（语义树新方案 ``13`` ③、效果评估标准 ``10`` 4.1）。

两类：

- **和时间有关的不在这里写死**：槽宽、转移窗口（"同一条链"）、复发窗口（"久别重来"）是预测树的参数
  （``prediction.slot_minutes`` / ``transition_window_slots`` / ``recurrence_window_days``），组合根抄过来；块长每晚从数据算。
- **统计门槛是默认值**（``RelationThresholds``）：FDR 10%、前向验证 5 次 / 5%、影响够大的 5 个百分点 + 20%、维持检验
  最近 8 个块。用户 10-07 定的起点，跑真实数据后再校；可由 ``config.scene.relations`` 覆盖（裁定 26）。另有几个实现用的数
  （异质性检验的显著性、区间的置信度与重抽样次数、精确检验的规模上限），也在这里、也是默认值。
"""

from __future__ import annotations

from dataclasses import dataclass, field

MINUTES_PER_DAY = 24 * 60


class RelationConfigError(ValueError):
    """参数自相矛盾。"""


@dataclass(frozen=True)
class RelationThresholds:
    """统计门槛：全是默认值，可由 ``config.scene.relations`` 覆盖（组合根换算）。和时间有关的不在这里。"""

    #: 第 1 道：当晚检验族的假发现率。
    fdr: float = 0.10
    #: 前向验证：发现之后新数据里至少攒够几次独立的前因才验；单侧显著性。
    forward_min_antecedents: int = 5
    forward_alpha: float = 0.05
    #: 前向验证要攒几次按功效定（裁定 27 第 3 条）：发现那晚按效应区间里离 0 近的一端算，攒到"真关系有这么大把握过关"再判，
    #: 至多攒这么多次（效应小到攒这么多也不够的，到这里就判）。
    forward_power: float = 0.80
    forward_max_antecedents: int = 40
    #: 第 3 道：差值区间的下限同时满足绝对差与相对差。
    min_absolute_lift: float = 0.05
    min_relative_lift: float = 0.20
    #: 维持检验：每攒满这么多个块的新数据（与上一段不重叠）判一次；连续两段都说没了才失效（裁定 27 第 3 条）。
    maintenance_blocks: int = 8
    #: 第 2 道：各块效应异质性检验的显著性（p 低于它算"各块不一样"）。
    heterogeneity_alpha: float = 0.05
    #: 区间：置信度与按块重抽样的次数。
    interval_level: float = 0.95
    bootstrap_rounds: int = 400
    #: 有信息的层（对照里有阳性也有阴性）不超过这么多时 p 值精确算，超过用正态近似（带连续性校正）。
    exact_strata_limit: int = 400
    #: 长跨度"新类第一次出现"：记录开始后至少过了 max(这么多天, 这一类的这么多个块长) 才算第一次（保护闸，乘的是行为自己的节奏）。
    first_seen_min_days: int = 14
    first_seen_min_blocks: int = 3
    #: 调节条件（含组合前因）只在独立样本至少这么多的前因上开检验。
    moderation_min_antecedents: int = 40
    #: 后果是细分概念时，"判不了"的窗口占比超过这个就不进检验族（裁定 27 第 6 条：盲的多了，上下界宽到什么都说明不了，
    #: 该回去改触点①的区别句）。
    max_blind_share: float = 0.5

    def __post_init__(self) -> None:
        for label in ("fdr", "forward_alpha", "forward_power", "heterogeneity_alpha", "interval_level"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 < value < 1.0:
                raise RelationConfigError(f"{label} must lie strictly between 0 and 1")
        for label in ("min_absolute_lift", "min_relative_lift"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= value < 1.0:
                raise RelationConfigError(f"{label} must lie in [0, 1)")
        for label in (
            "forward_min_antecedents",
            "forward_max_antecedents",
            "maintenance_blocks",
            "bootstrap_rounds",
            "exact_strata_limit",
            "first_seen_min_days",
            "first_seen_min_blocks",
            "moderation_min_antecedents",
        ):
            if not _positive_int(getattr(self, label)):
                raise RelationConfigError(f"{label} must be a positive integer")
        if isinstance(self.max_blind_share, bool) or not isinstance(self.max_blind_share, int | float) or not 0.0 < self.max_blind_share <= 1.0:
            raise RelationConfigError("max_blind_share must lie in (0, 1]")
        if self.forward_max_antecedents < self.forward_min_antecedents:
            raise RelationConfigError("forward_max_antecedents must not be below forward_min_antecedents")


@dataclass(frozen=True)
class RelationConfig:
    slot_minutes: int
    #: 转移窗口（同一条链）有几槽，= 预测树的 ``transition_window_slots``。
    transition_window_slots: int
    #: 隔多久再出现算"久别重来"，= 预测树的 ``recurrence_window_days``（与预测树认的"重新开始"同一个判断）。
    recurrence_window_days: float
    thresholds: RelationThresholds = field(default_factory=RelationThresholds)

    def __post_init__(self) -> None:
        if not _positive_int(self.slot_minutes) or MINUTES_PER_DAY % self.slot_minutes:
            raise RelationConfigError("slot_minutes must be a positive divisor of 1440")
        if not _positive_int(self.transition_window_slots) or self.transition_window_slots >= self.slots_per_day:
            raise RelationConfigError("transition_window_slots must be a positive number of slots below a day")
        if (
            isinstance(self.recurrence_window_days, bool)
            or not isinstance(self.recurrence_window_days, int | float)
            or self.recurrence_window_days <= 0
        ):
            raise RelationConfigError("recurrence_window_days must be a positive number of days")
        if not isinstance(self.thresholds, RelationThresholds):
            raise RelationConfigError("thresholds must be RelationThresholds")

    @property
    def slots_per_day(self) -> int:
        return MINUTES_PER_DAY // self.slot_minutes


def _positive_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


__all__ = ["MINUTES_PER_DAY", "RelationConfig", "RelationConfigError", "RelationThresholds"]
