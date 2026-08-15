"""行为融合的容量边界。

这些是纯运维参数：只回答"一次融合最多接受多少"，不回答"这组动作算不算一件事"。
后者是判据，属于不可配置的代码与提示词不变量。

TODO(BHV-FUSION-003): 与 ``behavior/observation/config.py`` 同样尚未接入唯一的 ``Config/``
边界；behavior 接入 Runtime 组合根（``TODO(BHV-RUNTIME-001)``）时一并纳入并改为必填。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BehaviorFusionConfig:
    # 切段：只按容量与跨度分组，不判断行为是否结束。必须切时在尾部这么多条里挑最大空白下刀，
    # 以降低把一次行为拦腰切断的概率——这仍然是确定性的，不含语义判断。
    max_fragments_per_segment: int = 512
    max_segment_span_seconds: int = 1_800
    boundary_search_fragments: int = 64

    # 一次融合最多接受多少条判断；一条判断最多分解成多少条行为事实。
    max_judgements: int = 96
    max_basis_facts: int = 64
    max_text_chars: int = 2_000
    max_name_chars: int = 200

    # 回执只记处置不复制正文，所以很小；判断携带完整语义，单独给一档。
    # 两者的上限必须**不小于**作业存储的上限（``BehaviorFusionJobConfig.max_file_bytes``）——
    # 作业在 stage 时把它们整个装进去，若作业放得下而这里放不下，检查点会过、落盘会失败，
    # 留下一批没有回执的判断且队列永久卡死。
    max_receipt_file_bytes: int = 4_194_304
    max_receipt_files: int = 100_000
    max_judgement_file_bytes: int = 4_194_304
    max_judgement_files: int = 1_000_000

    def __post_init__(self) -> None:
        for name, value in {
            "max_fragments_per_segment": self.max_fragments_per_segment,
            "max_segment_span_seconds": self.max_segment_span_seconds,
            "boundary_search_fragments": self.boundary_search_fragments,
            "max_judgements": self.max_judgements,
            "max_basis_facts": self.max_basis_facts,
            "max_text_chars": self.max_text_chars,
            "max_name_chars": self.max_name_chars,
            "max_receipt_file_bytes": self.max_receipt_file_bytes,
            "max_receipt_files": self.max_receipt_files,
            "max_judgement_file_bytes": self.max_judgement_file_bytes,
            "max_judgement_files": self.max_judgement_files,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


__all__ = ["BehaviorFusionConfig"]
