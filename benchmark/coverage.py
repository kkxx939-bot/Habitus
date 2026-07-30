"""把 benchmark 代码能力与实际数据场景分开审计，避免用“支持”冒充“已覆盖”。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from benchmark.model import BenchmarkDataset
from benchmark.protocol import (
    OPENVIKING_PUBLIC_ANSWER_CONTEXT_CHARS,
    OPENVIKING_PUBLIC_RETRIEVAL_LIMIT,
    OPENVIKING_REFERENCE_REVISION,
)

REQUIRED_NATIVE_SCENARIOS = (
    "profile",
    "preference",
    "entity",
    "tool_success",
    "tool_failure_recovery",
    "event",
    "intention_open",
    "intention_waiting",
    "intention_blocked",
    "intention_completed",
    "memory_update",
    "memory_correction",
    "explicit_forget",
    "fully_invalidated",
    "duplicate_merge",
    "cross_conversation_order",
    "relation_add",
    "relation_remove",
    "relation_merge_migration",
    "hierarchical_retrieval",
    "relation_one_hop",
    "completed_intention_history",
    "summary_fallback",
    "insufficient_no_answer",
    "temporal_reasoning",
    "multilingual",
    "long_context",
    "adversarial",
)


def benchmark_suite_matrix() -> tuple[Mapping[str, object], ...]:
    """返回与 OpenViking 对标后的适用边界和 m2bOS 增补项。"""

    return (
        _suite(
            "long_term_memory_qa",
            "aligned",
            "LoCoMo、LongMemEval；Top-10、4000 字符回答预算、数据集专用 Answer/Judge 协议",
        ),
        _suite("retrieval_effectiveness", "implemented", "证据 Recall、Answer Judge、关系和 Summary 路由"),
        _suite("retrieval_performance", "implemented", "稳态并发阶梯、Recall、QPS、p50/p95/p99 与饱和拐点"),
        _suite("vector_backend", "implemented", "按配置容量比例扩展、并发阶梯、过滤 Recall、增量更新删除"),
        _suite("runtime_contention", "implemented", "读写比例矩阵、MemoryJob 队列深度/年龄、排空和资源增长"),
        _suite("http_service", "implemented", "真实 HTTP 预热、稳态并发、写入任务跟踪和端到端长尾"),
        _suite("conversation_lifecycle", "implemented", "正式策略下按 Session 规模测量压缩、归档、清理和磁盘变化"),
        _suite("memory_semantic_evolution", "implemented", "仓库 native 数据覆盖六类节点、更新删除合并和关系"),
        _suite("failure_recovery", "implemented", "连续增加瞬态 VectorStore 故障直到正式 Job 重试预算耗尽"),
        _suite("boundary_aggregation", "implemented", "显式预算、独立运行 median/MAD、并发和规模双边界判定"),
        _suite("resource_rag", "not_applicable", "Resource 已退出当前 m2bOS 主链"),
        _suite("grep_bm25", "not_applicable", "当前记忆检索没有 grep/BM25 公共能力"),
        _suite("skillsbench", "not_applicable", "Skill 已退出当前 m2bOS 主链"),
        _suite("tau2_agent_trajectory", "not_applicable", "Agent/trajectory 不属于当前仓库能力"),
        _suite("cuvs_local_index", "not_applicable", "m2bOS 只支持远程 VectorStore Adapter"),
    )


def audit_dataset_coverage(dataset: BenchmarkDataset) -> Mapping[str, object]:
    """统计一份实际数据覆盖了哪些复杂记忆场景；未标注就明确视为未覆盖。"""

    if not isinstance(dataset, BenchmarkDataset):
        raise TypeError("dataset must be BenchmarkDataset")
    scenarios: Counter[str] = Counter()
    question_types: Counter[str] = Counter()
    for sample in dataset.samples:
        for question in sample.questions:
            question_types[question.question_type] += 1
            question_scenarios: set[str] = set()
            declared = question.metadata.get("scenarios", ())
            if isinstance(declared, str):
                declared = (declared,)
            if isinstance(declared, list | tuple):
                for value in declared:
                    if isinstance(value, str) and value.strip():
                        question_scenarios.add(value.strip())
            if question.question_type in REQUIRED_NATIVE_SCENARIOS:
                question_scenarios.add(question.question_type)
            scenarios.update(question_scenarios)
    missing = tuple(name for name in REQUIRED_NATIVE_SCENARIOS if scenarios[name] == 0)
    return {
        "schema_version": "m2bos_benchmark_coverage_v2",
        "openviking_reference_revision": OPENVIKING_REFERENCE_REVISION,
        "public_protocol": {
            "retrieval_limit": OPENVIKING_PUBLIC_RETRIEVAL_LIMIT,
            "answer_context_chars": OPENVIKING_PUBLIC_ANSWER_CONTEXT_CHARS,
            "judge_policies": ["openviking-default", "strict"],
        },
        "dataset": dataset.name.value,
        "sample_count": len(dataset.samples),
        "question_count": sum(question_types.values()),
        "question_types": dict(sorted(question_types.items())),
        "required_native_scenario_count": len(REQUIRED_NATIVE_SCENARIOS),
        "covered_native_scenario_count": len(REQUIRED_NATIVE_SCENARIOS) - len(missing),
        "scenario_counts": {name: scenarios[name] for name in REQUIRED_NATIVE_SCENARIOS},
        "missing_native_scenarios": list(missing),
        "native_scenarios_complete": not missing,
        "suite_matrix": list(benchmark_suite_matrix()),
    }


def _suite(name: str, status: str, evidence: str) -> Mapping[str, object]:
    return {"suite": name, "status": status, "evidence": evidence}


__all__ = [
    "REQUIRED_NATIVE_SCENARIOS",
    "audit_dataset_coverage",
    "benchmark_suite_matrix",
]
