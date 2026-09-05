"""行为目录语义层（L0/L1）的显式边界配置。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BehaviorSemanticConfig:
    """限制单次目录刷新的资源使用；边界**种类**沿 memory 语义层，数值按行为侧另定。

    已知的极端组合：条目数 × 单条摘要上限可超过 overview 渲染上限（512×500 > 64k）——命中时
    该目录的刷新每轮降级成 "semantic refresh failed" 信号、摘要不物化，等人调界。日目录的真实
    条目量是几十不是几百，先不为它加交叉校验。
    """

    max_direct_entries: int = 512
    max_prompt_chars: int = 120_000
    max_narrative_chars: int = 4_000
    max_entry_summary_chars: int = 500
    max_overview_chars: int = 64_000
    max_abstract_chars: int = 800

    def __post_init__(self) -> None:
        for name in (
            "max_direct_entries",
            "max_prompt_chars",
            "max_narrative_chars",
            "max_entry_summary_chars",
            "max_overview_chars",
            "max_abstract_chars",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_abstract_chars > self.max_overview_chars:
            raise ValueError("max_abstract_chars must not exceed max_overview_chars")


__all__ = ["BehaviorSemanticConfig"]
