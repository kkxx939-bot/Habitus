"""与公开记忆基准对齐的可审计运行协议。"""

from __future__ import annotations

from enum import Enum

from habitus.benchmark.model import BenchmarkDatasetName

# 对齐时实际核对的 OpenViking main。运行清单会记录该值，避免“对齐”变成无版本口号。
OPENVIKING_REFERENCE_REVISION = "291e9c580ffb71c3a17304fd0ebf175a144118c3"
OPENVIKING_PUBLIC_RETRIEVAL_LIMIT = 10
OPENVIKING_PUBLIC_ANSWER_CONTEXT_CHARS = 4_000


class BenchmarkJudgePolicy(str, Enum):
    """公开数据集可复现 OpenViking 默认口径，也可显式采用严格口径。"""

    DATASET_DEFAULT = "dataset-default"
    OPENVIKING_DEFAULT = "openviking-default"
    STRICT = "strict"


def resolve_judge_policy(
    dataset: str | BenchmarkDatasetName,
    requested: str | BenchmarkJudgePolicy,
) -> BenchmarkJudgePolicy:
    """把自动策略解析为真正写入 Judge 清单的固定策略。"""

    dataset_name = BenchmarkDatasetName(dataset)
    policy = BenchmarkJudgePolicy(requested)
    if policy is BenchmarkJudgePolicy.OPENVIKING_DEFAULT and dataset_name is BenchmarkDatasetName.HABITUS:
        raise ValueError("openviking-default judge policy is only defined for LoCoMo and LongMemEval")
    if policy is not BenchmarkJudgePolicy.DATASET_DEFAULT:
        return policy
    if dataset_name in {BenchmarkDatasetName.LOCOMO, BenchmarkDatasetName.LONGMEMEVAL}:
        return BenchmarkJudgePolicy.OPENVIKING_DEFAULT
    return BenchmarkJudgePolicy.STRICT


__all__ = [
    "BenchmarkJudgePolicy",
    "OPENVIKING_PUBLIC_ANSWER_CONTEXT_CHARS",
    "OPENVIKING_PUBLIC_RETRIEVAL_LIMIT",
    "OPENVIKING_REFERENCE_REVISION",
    "resolve_judge_policy",
]
