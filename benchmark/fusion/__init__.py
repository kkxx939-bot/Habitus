"""行为融合的数据集驱动评测。

与同目录下的长期记忆基准并列，但**不共享数据协议**：那边是"会话 + 问答"，这边是"观测片段序列 →
判断"。共享的只有入口、真实配置加载与报告形态。

只测确定性可判的性质（结构合规、关系命中与误标、主体完整、读不懂、粒度）。需要语义评判的部分
（目标准不准、分解合不合理）等这一层跑稳、看清哪里真的不稳定之后再加 Judge。

每一类都配对照组：只量"该标的标了没有"会漏掉滥用，只量"不该标的没标"会漏掉能力被压死——后者
实测发生过，一句保守的措辞就让模型对所有真因果都不敢标。
"""

from benchmark.fusion.dataset import (
    FUSION_CASE_SCHEMA,
    FusionBenchmarkError,
    FusionCase,
    FusionExpectation,
    FusionFragment,
    load_cases,
)
from benchmark.fusion.report import aggregate, evaluate, render_markdown
from benchmark.fusion.runner import FusionCaseRun, run_case

__all__ = [
    "FUSION_CASE_SCHEMA",
    "FusionBenchmarkError",
    "FusionCase",
    "FusionCaseRun",
    "FusionExpectation",
    "FusionFragment",
    "aggregate",
    "evaluate",
    "load_cases",
    "render_markdown",
    "run_case",
]
