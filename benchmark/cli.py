"""m2bOS 数据集记忆基准的一级命令入口。"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from benchmark.boundary import (
    BoundaryPolicy,
    BoundaryProfile,
    BoundaryProfileName,
    aggregate_boundary_runs,
)
from benchmark.coverage import audit_dataset_coverage
from benchmark.datasets import load_dataset
from benchmark.evaluation import judge_answers
from benchmark.http_boundary import HTTPBoundaryBenchmark
from benchmark.lifecycle import LifecycleBenchmark
from benchmark.lifecycle_boundary import LifecycleBoundaryBenchmark
from benchmark.load import RuntimeLoadBenchmark
from benchmark.model import BenchmarkDataset, BenchmarkDatasetName, BenchmarkJudgeRecord
from benchmark.protocol import (
    OPENVIKING_PUBLIC_ANSWER_CONTEXT_CHARS,
    OPENVIKING_PUBLIC_RETRIEVAL_LIMIT,
    BenchmarkJudgePolicy,
)
from benchmark.recovery_boundary import RecoveryBoundaryBenchmark
from benchmark.reliability import RecoveryBenchmark
from benchmark.report import build_report
from benchmark.runner import BenchmarkRunner
from benchmark.runtime_boundary import RuntimeBoundaryBenchmark
from benchmark.vector import VectorBenchmarkRunner, load_vector_dataset
from benchmark.vector_boundary import VectorBoundaryBenchmark
from Config import M2BOSConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m benchmark",
        description="Dataset-driven m2bOS long-term memory benchmark",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    inspect = commands.add_parser("inspect", help="只解析并统计原始数据集")
    _dataset_arguments(inspect)

    coverage = commands.add_parser("coverage", help="审计实际数据是否覆盖复杂记忆场景")
    _dataset_arguments(coverage)
    coverage.add_argument("--strict", action="store_true", help="存在未覆盖 native 场景时返回非零")

    run = commands.add_parser("run", help="导入完整会话并执行真实记忆检索和回答")
    _dataset_arguments(run)
    run.add_argument("--config", type=Path, required=True, help="m2bOS 完整运行配置")
    run.add_argument("--output", type=Path, required=True, help="结果目录")
    run.add_argument("--work", type=Path, required=True, help="隔离的样本存储目录")
    run.add_argument("--top-k", type=int, default=OPENVIKING_PUBLIC_RETRIEVAL_LIMIT)
    run.add_argument(
        "--max-answer-context-chars",
        type=int,
        default=OPENVIKING_PUBLIC_ANSWER_CONTEXT_CHARS,
        help="回答模型可见的 Memory/关系/Summary 内容预算；默认与 OpenViking 公开基准一致",
    )
    run.add_argument("--question-concurrency", type=int, default=8)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--judge-config", type=Path, help="可选的独立 Judge 完整模型配置")
    run.add_argument("--judge-concurrency", type=int, default=16)
    run.add_argument("--judge-with-evidence", action="store_true")
    run.add_argument(
        "--judge-policy",
        choices=tuple(item.value for item in BenchmarkJudgePolicy),
        default=BenchmarkJudgePolicy.DATASET_DEFAULT.value,
    )

    judge = commands.add_parser("judge", help="使用独立模型评判已有 answers.jsonl")
    _dataset_arguments(judge)
    judge.add_argument("--answers", type=Path, required=True)
    judge.add_argument("--output", type=Path, required=True)
    judge.add_argument("--config", type=Path, required=True, help="Judge 的完整模型配置")
    judge.add_argument("--concurrency", type=int, default=16)
    judge.add_argument("--with-evidence", action="store_true")
    judge.add_argument(
        "--judge-policy",
        choices=tuple(item.value for item in BenchmarkJudgePolicy),
        default=BenchmarkJudgePolicy.DATASET_DEFAULT.value,
    )
    judge.add_argument("--resume", action="store_true")

    report = commands.add_parser("report", help="汇总准确率、召回行为、耗时和 Token")
    report.add_argument("--answers", type=Path, required=True)
    report.add_argument("--judge", type=Path)
    report.add_argument("--ingest", type=Path)
    report.add_argument("--output", type=Path, required=True)

    boundary_plan = commands.add_parser("boundary-plan", help="输出产品边界压力矩阵，不连接外部服务")
    _boundary_profile_arguments(boundary_plan)

    boundary_runtime = commands.add_parser("boundary-runtime", help="测量 Runtime、读写混合和 Job 队列边界")
    _dataset_arguments(boundary_runtime)
    _boundary_run_arguments(boundary_runtime, vector_capacity=False)
    boundary_runtime.add_argument("--config", type=Path, required=True)
    boundary_runtime.add_argument("--work", type=Path, required=True)
    boundary_runtime.add_argument("--top-k", type=int, default=10)
    boundary_runtime.add_argument("--drain-timeout", type=float, default=1_800.0)

    boundary_http = commands.add_parser("boundary-http", help="从真实远程调用方测量 HTTP 服务边界")
    _dataset_arguments(boundary_http)
    _boundary_run_arguments(boundary_http, conversation_scale=False, vector_capacity=False)
    boundary_http.add_argument("--server-url", required=True)
    boundary_http.add_argument(
        "--server-identity",
        required=True,
        help="不可变部署身份，例如代码 SHA 与配置版本组合；用于阻止误聚合不同服务",
    )
    boundary_http.add_argument("--api-key-env", default="M2BOS_HTTP_API_KEY")
    boundary_http.add_argument("--top-k", type=int, default=10)
    boundary_http.add_argument("--allow-writes", action="store_true")
    boundary_http.add_argument("--seed-dataset", action="store_true")
    boundary_http.add_argument("--request-timeout", type=float, default=120.0)
    boundary_http.add_argument("--drain-timeout", type=float, default=1_800.0)

    boundary_vector = commands.add_parser("boundary-vector", help="测量 VectorStore 规模与并发边界")
    _boundary_run_arguments(boundary_vector, conversation_scale=False, write_fraction=False)
    boundary_vector.add_argument("--input", type=Path, required=True, help="m2bos_vector_benchmark_v1 JSON")
    boundary_vector.add_argument("--config", type=Path, required=True)
    boundary_vector.add_argument("--work", type=Path, required=True)
    boundary_vector.add_argument("--top-k", type=int, default=10)
    boundary_vector.add_argument("--update-fraction", type=float, default=0.05)
    boundary_vector.add_argument("--delete-fraction", type=float, default=0.05)
    boundary_vector.add_argument("--maximum-documents", type=int)

    boundary_lifecycle = commands.add_parser("boundary-lifecycle", help="测量正式 Conversation 生命周期边界")
    _dataset_arguments(boundary_lifecycle)
    _boundary_run_arguments(
        boundary_lifecycle,
        vector_capacity=False,
        concurrency=False,
        write_fraction=False,
        timing=False,
    )
    boundary_lifecycle.add_argument("--config", type=Path, required=True)
    boundary_lifecycle.add_argument("--work", type=Path, required=True)
    boundary_lifecycle.add_argument("--age-days", type=int, default=400)
    boundary_lifecycle.add_argument("--max-cycles", type=int, default=10_000)

    boundary_recovery = commands.add_parser("boundary-recovery", help="测量连续故障与 Job 重试耗尽边界")
    _dataset_arguments(boundary_recovery)
    _boundary_run_arguments(
        boundary_recovery,
        conversation_scale=False,
        vector_capacity=False,
        concurrency=False,
        write_fraction=False,
        timing=False,
    )
    boundary_recovery.add_argument("--config", type=Path, required=True)
    boundary_recovery.add_argument("--work", type=Path, required=True)
    boundary_recovery.add_argument("--fault-count", type=int, action="append")

    boundary_aggregate = commands.add_parser("boundary-aggregate", help="聚合不同进程的独立边界结果")
    boundary_aggregate.add_argument("inputs", nargs="+", type=Path)
    boundary_aggregate.add_argument("--policy", type=Path, required=True)
    boundary_aggregate.add_argument("--output", type=Path, required=True)

    load = commands.add_parser("load", help="执行检索与 Conversation/MemoryJob 混合负载")
    _dataset_arguments(load)
    load.add_argument("--config", type=Path, required=True)
    load.add_argument("--output", type=Path, required=True)
    load.add_argument("--work", type=Path, required=True)
    load.add_argument("--search-operations", type=int, default=100)
    load.add_argument("--write-operations", type=int, default=20)
    load.add_argument("--concurrency", type=int, default=16)
    load.add_argument("--top-k", type=int, default=10)
    load.add_argument("--drain-timeout", type=float, default=1_800.0)

    lifecycle = commands.add_parser("lifecycle", help="执行 Summary 压缩和多层清理基准")
    _dataset_arguments(lifecycle)
    lifecycle.add_argument("--config", type=Path, required=True)
    lifecycle.add_argument("--output", type=Path, required=True)
    lifecycle.add_argument("--work", type=Path, required=True)
    lifecycle.add_argument("--age-days", type=int, default=400)
    lifecycle.add_argument("--max-cycles", type=int, default=100)
    lifecycle.add_argument("--stage-source-count", type=int, default=2)

    recovery = commands.add_parser("recovery", help="注入一次瞬态向量故障并测量耐久恢复")
    _dataset_arguments(recovery)
    recovery.add_argument("--config", type=Path, required=True)
    recovery.add_argument("--output", type=Path, required=True)
    recovery.add_argument("--work", type=Path, required=True)

    vector = commands.add_parser("vector", help="执行真实 VectorStore 质量与性能基准")
    vector.add_argument("--input", type=Path, required=True, help="m2bos_vector_benchmark_v1 JSON")
    vector.add_argument("--config", type=Path, required=True)
    vector.add_argument("--output", type=Path, required=True)
    vector.add_argument("--work", type=Path, required=True)
    vector.add_argument("--top-k", type=int, default=10)
    vector.add_argument("--concurrency", type=int, default=8)
    vector.add_argument("--repeats", type=int, default=1)
    vector.add_argument("--update-fraction", type=float, default=0.05)
    vector.add_argument("--delete-fraction", type=float, default=0.05)
    return parser


def _dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        choices=tuple(item.value for item in BenchmarkDatasetName),
        required=True,
    )
    parser.add_argument("--input", type=Path, required=True, help="官方或原生数据集 JSON")
    parser.add_argument("--sample", type=int, action="append", default=[])
    parser.add_argument("--question-limit", type=int)
    parser.add_argument(
        "--include-adversarial",
        action="store_true",
        help="LoCoMo 默认排除 category=5；显式开启后才纳入",
    )


def _boundary_profile_arguments(
    parser: argparse.ArgumentParser,
    *,
    conversation_scale: bool = True,
    vector_capacity: bool = True,
    concurrency: bool = True,
    write_fraction: bool = True,
    timing: bool = True,
) -> None:
    parser.add_argument(
        "--profile",
        choices=tuple(item.value for item in BoundaryProfileName),
        default=BoundaryProfileName.STANDARD.value,
    )
    if conversation_scale:
        parser.add_argument("--conversation-scale", type=int, action="append")
    if vector_capacity:
        parser.add_argument("--vector-capacity-fraction", type=float, action="append")
    if concurrency:
        parser.add_argument("--concurrency", type=int, action="append")
    if write_fraction:
        parser.add_argument("--write-fraction", type=float, action="append")
    if timing:
        parser.add_argument("--warmup-seconds", type=float)
        parser.add_argument("--phase-seconds", type=float)
    parser.add_argument("--repetitions", type=int)


def _boundary_run_arguments(
    parser: argparse.ArgumentParser,
    *,
    conversation_scale: bool = True,
    vector_capacity: bool = True,
    concurrency: bool = True,
    write_fraction: bool = True,
    timing: bool = True,
) -> None:
    _boundary_profile_arguments(
        parser,
        conversation_scale=conversation_scale,
        vector_capacity=vector_capacity,
        concurrency=concurrency,
        write_fraction=write_fraction,
        timing=timing,
    )
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result: Mapping[str, object]
    try:
        if args.command == "boundary-plan":
            print(json.dumps(_boundary_profile(args).to_dict(), ensure_ascii=False, indent=2))
            return 0
        if args.command == "boundary-aggregate":
            result = aggregate_boundary_runs(
                args.inputs,
                policy=BoundaryPolicy.from_file(args.policy),
                output_directory=args.output,
            )
            print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
            return 0
        if args.command == "report":
            summary = build_report(
                answers_path=args.answers,
                judge_path=args.judge,
                ingest_path=args.ingest,
                output_directory=args.output,
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            return 0
        if args.command == "boundary-vector":
            result = asyncio.run(
                VectorBoundaryBenchmark(
                    M2BOSConfig.from_file(args.config),
                    load_vector_dataset(args.input),
                    profile=_boundary_profile(args),
                    policy=BoundaryPolicy.from_file(args.policy),
                    output_directory=args.output,
                    work_directory=args.work,
                    top_k=args.top_k,
                    update_fraction=args.update_fraction,
                    delete_fraction=args.delete_fraction,
                    maximum_documents=args.maximum_documents,
                ).run()
            )
            print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
            return 0
        if args.command == "vector":
            result = asyncio.run(
                VectorBenchmarkRunner(
                    M2BOSConfig.from_file(args.config),
                    load_vector_dataset(args.input),
                    output_directory=args.output,
                    work_directory=args.work,
                    top_k=args.top_k,
                    concurrency=args.concurrency,
                    repeats=args.repeats,
                    update_fraction=args.update_fraction,
                    delete_fraction=args.delete_fraction,
                ).run()
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0

        dataset = _load_selected_dataset(args)
        if args.command == "boundary-runtime":
            result = asyncio.run(
                RuntimeBoundaryBenchmark(
                    M2BOSConfig.from_file(args.config),
                    dataset,
                    profile=_boundary_profile(args),
                    policy=BoundaryPolicy.from_file(args.policy),
                    output_directory=args.output,
                    work_directory=args.work,
                    top_k=args.top_k,
                    drain_timeout_seconds=args.drain_timeout,
                ).run()
            )
            print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
            return 0
        if args.command == "boundary-http":
            api_key = os.environ.get(args.api_key_env)
            if api_key is None or not api_key.strip():
                raise ValueError(f"HTTP benchmark API key environment is missing: {args.api_key_env}")
            result = asyncio.run(
                HTTPBoundaryBenchmark(
                    dataset,
                    profile=_boundary_profile(args),
                    policy=BoundaryPolicy.from_file(args.policy),
                    server_url=args.server_url,
                    server_identity=args.server_identity,
                    api_key=api_key,
                    output_directory=args.output,
                    top_k=args.top_k,
                    allow_writes=args.allow_writes,
                    seed_dataset=args.seed_dataset,
                    request_timeout_seconds=args.request_timeout,
                    drain_timeout_seconds=args.drain_timeout,
                ).run()
            )
            print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
            return 0
        if args.command == "boundary-lifecycle":
            result = asyncio.run(
                LifecycleBoundaryBenchmark(
                    M2BOSConfig.from_file(args.config),
                    dataset,
                    profile=_boundary_profile(args),
                    policy=BoundaryPolicy.from_file(args.policy),
                    output_directory=args.output,
                    work_directory=args.work,
                    age_days=args.age_days,
                    max_cycles=args.max_cycles,
                ).run()
            )
            print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
            return 0
        if args.command == "boundary-recovery":
            result = asyncio.run(
                RecoveryBoundaryBenchmark(
                    M2BOSConfig.from_file(args.config),
                    dataset,
                    profile=_boundary_profile(args),
                    policy=BoundaryPolicy.from_file(args.policy),
                    output_directory=args.output,
                    work_directory=args.work,
                    fault_counts=args.fault_count,
                ).run()
            )
            print(json.dumps(result["aggregate"], ensure_ascii=False, indent=2))
            return 0
        if args.command == "inspect":
            print(json.dumps(_dataset_summary(dataset), ensure_ascii=False, indent=2))
            return 0
        if args.command == "coverage":
            coverage_result = audit_dataset_coverage(dataset)
            print(json.dumps(coverage_result, ensure_ascii=False, indent=2))
            return 3 if args.strict and not coverage_result["native_scenarios_complete"] else 0
        if args.command == "judge":
            records = asyncio.run(
                judge_answers(
                    answers_path=args.answers,
                    output_path=args.output,
                    dataset=dataset,
                    judge_config=M2BOSConfig.from_file(args.config),
                    concurrency=args.concurrency,
                    include_evidence=args.with_evidence,
                    judge_policy=BenchmarkJudgePolicy(args.judge_policy),
                    resume=args.resume,
                )
            )
            print(json.dumps(_judge_summary(records), ensure_ascii=False, indent=2))
            return 0
        if args.command == "load":
            result = asyncio.run(
                RuntimeLoadBenchmark(
                    M2BOSConfig.from_file(args.config),
                    dataset,
                    output_directory=args.output,
                    work_directory=args.work,
                    search_operations=args.search_operations,
                    write_operations=args.write_operations,
                    concurrency=args.concurrency,
                    top_k=args.top_k,
                    drain_timeout_seconds=args.drain_timeout,
                ).run()
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "lifecycle":
            result = asyncio.run(
                LifecycleBenchmark(
                    M2BOSConfig.from_file(args.config),
                    dataset,
                    output_directory=args.output,
                    work_directory=args.work,
                    age_days=args.age_days,
                    max_cycles=args.max_cycles,
                    stage_source_count=args.stage_source_count,
                ).run()
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        if args.command == "recovery":
            result = asyncio.run(
                RecoveryBenchmark(
                    M2BOSConfig.from_file(args.config),
                    dataset,
                    output_directory=args.output,
                    work_directory=args.work,
                ).run()
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        return asyncio.run(_run(args, dataset))
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        print(f"benchmark error: {type(exc).__name__}: {exc}")
        return 2


async def _run(args: argparse.Namespace, dataset: BenchmarkDataset) -> int:
    output = args.output.expanduser().resolve()
    runner = BenchmarkRunner(
        M2BOSConfig.from_file(args.config),
        dataset,
        output_directory=output,
        work_directory=args.work,
        top_k=args.top_k,
        question_concurrency=args.question_concurrency,
        max_answer_context_chars=args.max_answer_context_chars,
        resume=args.resume,
    )
    answers = await runner.run()
    judge_path: Path | None = None
    if args.judge_config is not None:
        selected_judge_path = output / "judge.jsonl"
        await judge_answers(
            answers_path=runner.answers_path,
            output_path=selected_judge_path,
            dataset=dataset,
            judge_config=M2BOSConfig.from_file(args.judge_config),
            concurrency=args.judge_concurrency,
            include_evidence=args.judge_with_evidence,
            judge_policy=BenchmarkJudgePolicy(args.judge_policy),
            resume=args.resume,
        )
        judge_path = selected_judge_path
    summary = build_report(
        answers_path=runner.answers_path,
        judge_path=judge_path,
        ingest_path=runner.ingest_path,
        output_directory=output,
    )
    print(
        json.dumps(
            {
                "answer_count": len(answers),
                "answers": str(runner.answers_path),
                "judge": str(judge_path) if judge_path else None,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _load_selected_dataset(args: argparse.Namespace) -> BenchmarkDataset:
    return load_dataset(
        args.dataset,
        str(args.input),
        sample_indices=tuple(args.sample),
        question_limit=args.question_limit,
        include_adversarial=args.include_adversarial,
    )


def _boundary_profile(args: argparse.Namespace) -> BoundaryProfile:
    return BoundaryProfile.named(args.profile).override(
        conversation_scales=getattr(args, "conversation_scale", None),
        vector_capacity_fractions=getattr(args, "vector_capacity_fraction", None),
        concurrency_levels=getattr(args, "concurrency", None),
        write_fractions=getattr(args, "write_fraction", None),
        warmup_seconds=getattr(args, "warmup_seconds", None),
        phase_seconds=getattr(args, "phase_seconds", None),
        repetitions=args.repetitions,
    )


def _dataset_summary(dataset: BenchmarkDataset) -> dict[str, object]:
    by_type: dict[str, int] = {}
    for sample in dataset.samples:
        for question in sample.questions:
            by_type[question.question_type] = by_type.get(question.question_type, 0) + 1
    return {
        "dataset": dataset.name.value,
        "source": dataset.source_path,
        "sample_count": len(dataset.samples),
        "session_count": sum(len(sample.sessions) for sample in dataset.samples),
        "message_count": sum(len(session.messages) for sample in dataset.samples for session in sample.sessions),
        "question_count": sum(len(sample.questions) for sample in dataset.samples),
        "questions_by_type": dict(sorted(by_type.items())),
    }


def _judge_summary(records: Sequence[BenchmarkJudgeRecord]) -> dict[str, object]:
    verdicts = [record.verdict for record in records]
    graded = sum(value in {"correct", "wrong"} for value in verdicts)
    correct = verdicts.count("correct")
    return {
        "question_count": len(verdicts),
        "graded_count": graded,
        "correct": correct,
        "wrong": verdicts.count("wrong"),
        "judge_errors": verdicts.count("judge_error"),
        "accuracy": correct / graded if graded else None,
    }


__all__ = ["build_parser", "main"]
