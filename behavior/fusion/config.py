"""行为融合的容量边界。

这些是纯运维参数：只回答"一次融合最多接受多少"，不回答"这组动作算不算一件事"。
后者是判据，属于不可配置的代码与提示词不变量。

TODO(BHV-FUSION-003·余项): 窗口参数已随 BHV-RUNTIME-001 经 ``Config.behavior`` 注入（见下方
常量注释）；本 dataclass 的**容量类边界**仍是代码侧默认、未并入 ``Config/`` 边界——留待与
BHV-LIFECYCLE-001 的存储生命周期改造同批（届时保留期/容量要一起进配置）。
"""

from __future__ import annotations

from dataclasses import dataclass

# 融合取"先前的判断"上下文的窗口：最多几条、往回看多久。lookback 同时定义了归约层的封口视界
# ——一条判断的 evidence_ready_at 老出该窗口后，任何未来融合都不可能再引用（continues/supersedes）
# 它，链因此机械闭合。两处必须是同一个数：各写一个会让"融合还能续"与"归约已封口"静默分叉，
# 所以归约层只准从这里 import，不准自带默认值。
# 1 小时是用户裁定（原 6 小时"事件粒度太粗"）：行为被打断后的真实恢复间隔是分钟级，隔更久回来
# 按融合自身纪律就是新的一件事；窗口越短，行为进树的延迟也越短（封口要等出窗）。
FUSION_CONTEXT_LIMIT = 8
FUSION_CONTEXT_LOOKBACK_SECONDS = 3_600.0
# BHV-RUNTIME-001 已闭合窗口注入：组合根（Runtime/behavior.py）把 Config.behavior 解析出的
# **同一份** lookback/limit 喂给融合与归约两处，配置层不复制默认值（缺省 None → 取此处）。
# BHV-FUSION-003 余项：容量类边界（本 dataclass）仍是代码侧默认，未并入 Config 边界。


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


__all__ = [
    "FUSION_CONTEXT_LIMIT",
    "FUSION_CONTEXT_LOOKBACK_SECONDS",
    "BehaviorFusionConfig",
]
