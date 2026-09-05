"""汇总数据集基准的质量、检索行为、耗时和成本。"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from benchmark.evaluation import load_judge_records
from benchmark.metrics import latency_distribution
from benchmark.model import BenchmarkAnswerRecord
from benchmark.runner import BenchmarkRunError, load_answer_records


def build_report(
    *,
    answers_path: str | Path,
    output_directory: str | Path,
    judge_path: str | Path | None = None,
    ingest_path: str | Path | None = None,
) -> Mapping[str, object]:
    """生成机器可读 JSON 和便于比较的 Markdown 总结。"""

    answers = load_answer_records(answers_path)
    judged = load_judge_records(judge_path) if judge_path is not None else ()
    ingest = _load_jsonl(Path(ingest_path)) if ingest_path is not None else ()
    if judged:
        answer_keys = {(item.sample_id, item.question_id) for item in answers}
        judge_keys = {(str(item.get("sample_id")), str(item.get("question_id"))) for item in judged}
        if len(judge_keys) != len(judged):
            raise BenchmarkRunError("judge results contain duplicate question identities")
        if answer_keys != judge_keys:
            raise BenchmarkRunError("judge results do not cover exactly the answer records")

    summary: dict[str, object] = {
        "schema_version": "habitus_benchmark_report_v2",
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": answers[0].dataset if answers else "",
        "sample_count": len({item.sample_id for item in answers}),
        "question_count": len(answers),
        "answer_error_count": sum(bool(item.error) for item in answers),
        "quality": _quality(judged),
        "retrieval": _retrieval(answers),
        "latency_ms": _latency(answers, judged),
        "token_usage": _tokens(answers, judged),
        "ingestion": _ingestion(ingest),
    }
    output = Path(output_directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (output / "summary.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def _quality(records: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    if not records:
        return {"graded_count": 0, "correct": 0, "wrong": 0, "judge_errors": 0, "accuracy": None, "by_type": {}}
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"correct": 0, "wrong": 0, "judge_error": 0})
    totals = {"correct": 0, "wrong": 0, "judge_error": 0}
    for record in records:
        verdict = str(record.get("verdict", "judge_error"))
        if verdict not in totals:
            verdict = "judge_error"
        totals[verdict] += 1
        by_type[str(record.get("question_type", "unknown"))][verdict] += 1
    graded = totals["correct"] + totals["wrong"]
    type_summary = {
        name: {
            **values,
            "graded_count": values["correct"] + values["wrong"],
            "accuracy": (
                values["correct"] / (values["correct"] + values["wrong"])
                if values["correct"] + values["wrong"]
                else None
            ),
        }
        for name, values in sorted(by_type.items())
    }
    return {
        "graded_count": graded,
        "correct": totals["correct"],
        "wrong": totals["wrong"],
        "judge_errors": totals["judge_error"],
        "accuracy": totals["correct"] / graded if graded else None,
        "by_type": type_summary,
    }


def _retrieval(answers: Sequence[BenchmarkAnswerRecord]) -> Mapping[str, object]:
    if not answers:
        return {}
    direct = [len(item.retrieved_uris) for item in answers]
    related = [len(item.related_uris) for item in answers]
    summaries = [len(item.summary_fallback_ids) for item in answers]
    answer_direct = [len(item.answer_memory_uris) for item in answers]
    answer_related = [len(item.answer_related_uris) for item in answers]
    answer_summaries = [len(item.answer_summary_ids) for item in answers]
    skipped = [len(item.skipped_context_ids) for item in answers]
    sufficient = [item.retrieval_sufficient for item in answers if item.retrieval_sufficient is not None]
    evidence = [item.evidence_recall for item in answers if item.evidence_recall is not None]
    summary_route = [
        item.summary_fallback_route_correct for item in answers if item.summary_fallback_route_correct is not None
    ]
    related_route = [
        item.related_memory_route_correct for item in answers if item.related_memory_route_correct is not None
    ]
    return {
        "average_direct_memories": sum(direct) / len(direct),
        "average_related_memories": sum(related) / len(related),
        "average_summary_fallbacks": sum(summaries) / len(summaries),
        "average_answer_memories": sum(answer_direct) / len(answer_direct),
        "average_answer_related_memories": sum(answer_related) / len(answer_related),
        "average_answer_summaries": sum(answer_summaries) / len(answer_summaries),
        "average_context_items_skipped_by_budget": sum(skipped) / len(skipped),
        "context_budget_exhaustion_rate": sum(value > 0 for value in skipped) / len(skipped),
        "empty_memory_rate": sum(value == 0 for value in direct) / len(direct),
        "summary_fallback_attempt_rate": (sum(item.summary_fallback_attempted for item in answers) / len(answers)),
        "summary_fallback_hit_rate": sum(value > 0 for value in summaries) / len(summaries),
        "retrieval_sufficient_rate": (
            sum(value is True for value in sufficient) / len(sufficient) if sufficient else None
        ),
        "evidence_graded_count": len(evidence),
        "answer_context_evidence_recall": sum(evidence) / len(evidence) if evidence else None,
        "summary_fallback_route_accuracy": (sum(summary_route) / len(summary_route) if summary_route else None),
        "related_memory_route_accuracy": (sum(related_route) / len(related_route) if related_route else None),
        "average_context_chars": sum(item.context_chars for item in answers) / len(answers),
        "average_context_tokens_approx": (sum(item.context_tokens_approx for item in answers) / len(answers)),
    }


def _latency(
    answers: Sequence[BenchmarkAnswerRecord],
    judged: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    retrieval = [item.retrieval_latency_ms for item in answers]
    answer = [item.answer_latency_ms for item in answers]
    judge = [_number(item.get("judge_latency_ms", 0.0), "judge_latency_ms") for item in judged]
    return {
        "retrieval": _distribution(retrieval),
        "answer": _distribution(answer),
        "judge": _distribution(judge),
        "end_to_end": _distribution([left + right for left, right in zip(retrieval, answer, strict=True)]),
    }


def _tokens(
    answers: Sequence[BenchmarkAnswerRecord],
    judged: Sequence[Mapping[str, object]],
) -> Mapping[str, object]:
    answer_input = sum(item.answer_input_tokens for item in answers)
    answer_output = sum(item.answer_output_tokens for item in answers)
    answer_total = sum(item.answer_total_tokens for item in answers)
    judge_input = sum(_integer(item.get("judge_input_tokens", 0), "judge_input_tokens") for item in judged)
    judge_output = sum(_integer(item.get("judge_output_tokens", 0), "judge_output_tokens") for item in judged)
    judge_total = sum(_integer(item.get("judge_total_tokens", 0), "judge_total_tokens") for item in judged)
    return {
        "answer_input": answer_input,
        "answer_output": answer_output,
        "answer_total": answer_total,
        "judge_input": judge_input,
        "judge_output": judge_output,
        "judge_total": judge_total,
        "combined_total": answer_total + judge_total,
    }


def _ingestion(records: Sequence[Mapping[str, object]]) -> Mapping[str, object]:
    if not records:
        return {}
    kinds: dict[str, int] = defaultdict(int)
    for record in records:
        raw_kinds = record.get("memory_documents_by_kind", {})
        if isinstance(raw_kinds, Mapping):
            for name, count in raw_kinds.items():
                kinds[str(name)] += _integer(count, "memory document kind count")
    return {
        "sample_count": len(records),
        "session_count": sum(_integer(item.get("session_count", 0), "session_count") for item in records),
        "message_count": sum(_integer(item.get("message_count", 0), "message_count") for item in records),
        "job_count": sum(_integer(item.get("job_count", 0), "job_count") for item in records),
        "committed_receipt_count": sum(
            _integer(item.get("committed_receipt_count", 0), "committed_receipt_count") for item in records
        ),
        "memory_document_count": sum(
            _integer(item.get("memory_document_count", 0), "memory_document_count") for item in records
        ),
        "memory_documents_by_kind": dict(sorted(kinds.items())),
        "latency_ms": _distribution(
            [_number(item.get("ingest_latency_ms", 0.0), "ingest_latency_ms") for item in records]
        ),
    }


def _distribution(values: Sequence[float]) -> Mapping[str, float | int | None]:
    return latency_distribution(values)


def _markdown(summary: Mapping[str, object]) -> str:
    quality = summary["quality"]
    retrieval = summary["retrieval"]
    if not isinstance(quality, Mapping) or not isinstance(retrieval, Mapping):
        raise BenchmarkRunError("benchmark summary contains invalid quality or retrieval metrics")
    accuracy = quality.get("accuracy")
    accuracy_text = "not graded" if accuracy is None else f"{float(accuracy):.2%}"
    lines = [
        "# Habitus Memory Benchmark",
        "",
        f"- Dataset: `{summary['dataset']}`",
        f"- Samples: {summary['sample_count']}",
        f"- Questions: {summary['question_count']}",
        f"- Answer pipeline errors: {summary['answer_error_count']}",
        f"- Accuracy: {accuracy_text}",
        f"- Graded: {quality.get('graded_count', 0)}",
        f"- Judge errors: {quality.get('judge_errors', 0)}",
        "",
        "## Retrieval",
        "",
        f"- Average direct memories: {_format_number(retrieval.get('average_direct_memories'))}",
        f"- Average related memories: {_format_number(retrieval.get('average_related_memories'))}",
        f"- Average memories shown to answer model: {_format_number(retrieval.get('average_answer_memories'))}",
        f"- Average related memories shown to answer model: {_format_number(retrieval.get('average_answer_related_memories'))}",
        f"- Context budget exhaustion rate: {_format_percent(retrieval.get('context_budget_exhaustion_rate'))}",
        f"- Empty-memory rate: {_format_percent(retrieval.get('empty_memory_rate'))}",
        f"- Summary fallback attempt rate: {_format_percent(retrieval.get('summary_fallback_attempt_rate'))}",
        f"- Summary fallback hit rate: {_format_percent(retrieval.get('summary_fallback_hit_rate'))}",
        f"- Answer-context evidence recall: {_format_percent(retrieval.get('answer_context_evidence_recall'))}",
        f"- Summary route accuracy: {_format_percent(retrieval.get('summary_fallback_route_accuracy'))}",
        f"- Related-memory route accuracy: {_format_percent(retrieval.get('related_memory_route_accuracy'))}",
        "",
        "## Accuracy by question type",
        "",
        "| Type | Correct | Wrong | Judge errors | Accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    by_type = quality.get("by_type", {})
    if isinstance(by_type, Mapping):
        for name, raw in by_type.items():
            if not isinstance(raw, Mapping):
                continue
            lines.append(
                f"| {name} | {raw.get('correct', 0)} | {raw.get('wrong', 0)} | "
                f"{raw.get('judge_error', 0)} | {_format_percent(raw.get('accuracy'))} |"
            )
    return "\n".join(lines) + "\n"


def _format_number(value: object) -> str:
    return "n/a" if value is None else f"{_number(value, 'report number'):.2f}"


def _format_percent(value: object) -> str:
    return "n/a" if value is None else f"{_number(value, 'report percentage'):.2%}"


def _integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkRunError(f"{label} must be an integer")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BenchmarkRunError(f"{label} must be numeric")
    return float(value)


def _load_jsonl(path: Path) -> tuple[Mapping[str, object], ...]:
    source = path.expanduser().resolve(strict=True)
    result: list[Mapping[str, object]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BenchmarkRunError(f"JSONL is invalid at line {line_number}: {source}") from exc
        if not isinstance(value, Mapping):
            raise BenchmarkRunError(f"JSONL line {line_number} must be an object: {source}")
        result.append(value)
    return tuple(result)


__all__ = ["build_report"]
