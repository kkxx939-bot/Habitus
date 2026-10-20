"""关系检验的全部可调量（语义树新方案 ``13`` ③、效果评估标准 ``10`` 4.1）。

两类：

- **和时间有关的不在这里写死**：槽宽、转移窗口（"同一条链"）是 ``config.temporal`` 的参数，组合根抄过来；
  块长与"久别重来"每晚从数据算（``temporal.common_interval_days`` / ``temporal.absence``，裁定 59）。
- **统计门槛是默认值**（``RelationThresholds``）：FDR 10%、前向验证与维持检验按 α = 5% 与功效 80% 定次数（裁定 60 第 9 条：
  没有"至少 5 次""每段 8 块""调节前因 ≥ 40"这类固定数）、影响够大的 5 个百分点 + 20%。用户 10-07 定的起点，跑真实数据后
  再校；可由 ``config.scene.relations`` 覆盖（裁定 26）。另有几个实现用的数
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
    #: 前向验证与维持检验的单侧显著性（调节条件开不开检验、久别重来的样本数也用它）。
    forward_alpha: float = 0.05
    #: 前向验证要攒几次按功效定（裁定 27 第 3 条；裁定 60 第 9 条去掉了"至少 5 次"的下限）：发现那晚按效应区间里离 0 近的
    #: 一端算，攒到"真关系有这么大把握过关"再判，至多攒这么多次（效应小到攒这么多也不够的，到这里就判）。维持检验每段攒几次
    #: 同一对功效与上限，成立那晚按"回落到对照率"算好存下。
    forward_power: float = 0.80
    forward_max_antecedents: int = 40
    #: 第 3 道：差值区间的下限同时满足绝对差与相对差。
    min_absolute_lift: float = 0.05
    min_relative_lift: float = 0.20
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
    #: 问模型"这个前因在什么条件下更常接着做"的门槛：前因攒到这么多次才问（用户 10-19 定，保留 40）。机械列举的调节条件
    #: 不用它：每条带条件的关系按自己的最小可达 p 定开不开检验（``conditions.ready_to_test``，裁定 60 第 9 条）。
    condition_ask_min_antecedents: int = 40
    #: 后果是细分概念时，"判不了"的窗口占比超过这个就不进检验族（裁定 27 第 6 条：盲的多了，上下界宽到什么都说明不了，
    #: 该回去改触点①的区别句）。
    max_blind_share: float = 0.5
    #: 一条关系最多挂几个条件（裁定 61 第 6 条、62 第 1 条：第二个条件起只长在已成立的带条件关系上；保护闸，启动值 3，
    #: 以后从回放定）。
    max_combined_conditions: int = 3

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
            "forward_max_antecedents",
            "bootstrap_rounds",
            "exact_strata_limit",
            "first_seen_min_days",
            "first_seen_min_blocks",
            "condition_ask_min_antecedents",
            "max_combined_conditions",
        ):
            if not _positive_int(getattr(self, label)):
                raise RelationConfigError(f"{label} must be a positive integer")
        if (
            isinstance(self.max_blind_share, bool)
            or not isinstance(self.max_blind_share, int | float)
            or not 0.0 < self.max_blind_share <= 1.0
        ):
            raise RelationConfigError("max_blind_share must lie in (0, 1]")


@dataclass(frozen=True)
class RelationConfig:
    slot_minutes: int
    #: 转移窗口（同一条链）有几槽，由 ``config.temporal`` 的合并窗口与槽宽换算（裁定 59）。
    transition_window_slots: int
    thresholds: RelationThresholds = field(default_factory=RelationThresholds)

    def __post_init__(self) -> None:
        if not _positive_int(self.slot_minutes) or MINUTES_PER_DAY % self.slot_minutes:
            raise RelationConfigError("slot_minutes must be a positive divisor of 1440")
        if not _positive_int(self.transition_window_slots) or self.transition_window_slots >= self.slots_per_day:
            raise RelationConfigError("transition_window_slots must be a positive number of slots below a day")
        if not isinstance(self.thresholds, RelationThresholds):
            raise RelationConfigError("thresholds must be RelationThresholds")

    @property
    def slots_per_day(self) -> int:
        return MINUTES_PER_DAY // self.slot_minutes


def _positive_int(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


__all__ = ["MINUTES_PER_DAY", "RelationConfig", "RelationConfigError", "RelationThresholds"]
