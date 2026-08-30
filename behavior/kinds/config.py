"""行为类型词表的容量边界。

TODO(BHV-FUSION-003·余项): BHV-RUNTIME-001 已完成，本容量组**未**随之并入 ``Config/`` 边界
（当批只收口了窗口参数）——留待与 BHV-LIFECYCLE-001 的存储生命周期改造同批。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BehaviorKindConfig:
    """词表的运维容量；类型数天然收敛（几百封顶），上限只是防失控的护栏。"""

    max_kinds: int = 500
    max_aliases_per_kind: int = 64
    max_encoded_bytes: int = 262_144
    # 单次归一调用对瞬态错误（超时、断连）的有界重试：一次 sweep 近三千次串行调用，一次断连
    # 不该让整轮作废（BHV-REALDATA-001）。
    transient_retries: int = 5
    transient_retry_delay_seconds: float = 5.0

    def __post_init__(self) -> None:
        for name, upper in (
            ("max_kinds", 100_000),
            ("max_aliases_per_kind", 10_000),
            ("max_encoded_bytes", 64 * 1024 * 1024),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper:
                raise ValueError(f"{name} must be an integer between 1 and {upper}")


__all__ = ["BehaviorKindConfig"]
