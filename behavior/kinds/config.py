"""行为类型词表的运维参数：容量边界、归一批形状、存活期。

数值由 ``Config.behavior`` 的 ``kinds_*`` 字段经组合根注入（``Runtime/behavior.py``）；这里的默认值
是代码内唯一出处，YAML 留空即取之。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BehaviorKindConfig:
    """词表的运维边界。

    - 容量：WP4 之后一天约二百多个不同名字、大多跨天重复，一年量级几千条；上限只是防失控的护栏。
    - 归一批形状：一次调用判几个名字、给每个名字多少候选（向量最近邻 ∪ 高频）。
    - 存活期：``last_hit_day + max(base_days, gap_multiplier × max_gap_days)``——一次性名字基础期
      后删，周期行为按自己量出来的间隔续命（周频→≥21 天、月频→90、季频→270）。
    """

    max_kinds: int = 10_000
    max_aliases_per_kind: int = 200
    max_encoded_bytes: int = 8 * 1024 * 1024
    batch_size: int = 10
    vector_candidates: int = 30
    frequent_candidates: int = 20
    literal_candidates: int = 30
    base_days: int = 30
    gap_multiplier: int = 3
    # 批量判定里违约名字的重问轮数（每轮只重问违约的那些）；耗尽后当 null 新建并留信号。
    validation_rounds: int = 2
    # 单次归一调用对瞬态错误（超时、断连）的有界重试：一次 sweep 几十次调用，一次断连不该让整轮作废。
    transient_retries: int = 5
    transient_retry_delay_seconds: float = 5.0

    def __post_init__(self) -> None:
        for name, lower, upper in (
            ("max_kinds", 1, 1_000_000),
            ("max_aliases_per_kind", 1, 100_000),
            ("max_encoded_bytes", 1, 256 * 1024 * 1024),
            ("batch_size", 1, 100),
            ("vector_candidates", 0, 500),
            ("frequent_candidates", 0, 500),
            ("literal_candidates", 0, 500),
            ("base_days", 1, 3_650),
            ("gap_multiplier", 1, 100),
            ("validation_rounds", 0, 10),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer between {lower} and {upper}")
        if isinstance(self.transient_retries, bool) or not isinstance(self.transient_retries, int) or not 0 <= self.transient_retries <= 20:
            raise ValueError("transient_retries must be an integer between 0 and 20")
        if isinstance(self.transient_retry_delay_seconds, bool) or not isinstance(self.transient_retry_delay_seconds, int | float) or not 0.0 <= float(self.transient_retry_delay_seconds) <= 600.0:
            raise ValueError("transient_retry_delay_seconds must be between 0 and 600")


__all__ = ["BehaviorKindConfig"]
