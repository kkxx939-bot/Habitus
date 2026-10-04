"""读侧的数值配置：累积度门槛、分层门槛、区间水平、类型判据、新情境影响的分界。全部是待定值。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ViewsConfig:
    """全部是待定值，在重放上定；这里只是默认。"""

    #: 累积度门槛（用户 09-27 定成 3 次 / 跨 2 块）：过了就出数，"这个数信不信"归预测层判。
    #: 实测代价（对照 p0=0.50）：n=3 时真有效应抓到 55.5%、真没效应误报 25%——误报率跟着离散格子跳，
    #: 不是单调的（n=3 只有 k=3 或 k=0 才够窄，真没效应时那两种合起来正好 2/8）。定这么低的前提是
    #: 基准写假设时不穷举连线：常识猜对七成时 n=3 出数里假的只占 16%，纯随机配对则是 90%。
    min_count: int = 3
    #: 块那一维留着：挡掉"3 次全挤在同一天"那类（用户：再怎么高频也不能在一天内说明什么）。
    #: 块长 = max(1 天, 前件复发间隔 p50)，所以一天一次或更慢的行为就是"至少跨两天"。
    min_blocks: int = 2
    split_min: int = 3
    split_min_blocks: int = 2
    interval_level: float = 0.90
    bootstrap_samples: int = 1000
    enabling_pn: float = 0.5
    promoting_ps: float = 0.5
    #: 情境"影响大 / 小"的分界（2026-09-30 裁定八 ③）：在场与不在场两层区间分开、且点估计差到这么多才算"大"。
    #: 按读数单位各一个：概率（比例）、时刻（分钟）、次数（次）。**待定值**，重跑后定。
    influence_gap_probability: float = 0.15
    influence_gap_minutes: float = 30.0
    influence_gap_times: float = 0.5

    def influence_gap(self, unit: str) -> float:
        """这个单位的读数，两层差多少算"影响大"。无节律型（兑现率）按比例那条。"""

        if unit == "minutes":
            return self.influence_gap_minutes
        if unit == "times":
            return self.influence_gap_times
        return self.influence_gap_probability

    def __post_init__(self) -> None:
        for label in ("min_count", "min_blocks", "split_min", "split_min_blocks", "bootstrap_samples"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{label} must be a positive integer")
        for label in ("interval_level", "enabling_pn", "promoting_ps", "influence_gap_probability", "influence_gap_minutes", "influence_gap_times"):
            value = getattr(self, label)
            if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
                raise ValueError(f"{label} must be a positive number")
        if not self.interval_level < 1.0:
            raise ValueError("interval_level lies in (0, 1)")

    @property
    def main_gate(self) -> tuple[int, int]:
        return (self.min_count, self.min_blocks)

    @property
    def split_gate(self) -> tuple[int, int]:
        return (self.split_min, self.split_min_blocks)



__all__ = ["ViewsConfig"]
