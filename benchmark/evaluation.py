"""将 answers.jsonl 交给独立 Judge，并耐久保存评分记录。"""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, cast

from benchmark.judge import BenchmarkJudge
from benchmark.model import BenchmarkAnswerRecord, BenchmarkDataset, BenchmarkJudgeRecord
from benchmark.protocol import BenchmarkJudgePolicy, resolve_judge_policy
from benchmark.runner import BenchmarkRunError, load_answer_records
from Config import HabitusConfig
from ModelClient import ProviderFactory, StructuredChatClient
from ModelClient.adapters import register_builtin_adapters


async def judge_answers(
    *,
    answers_path: str | Path,
    output_path: str | Path,
    dataset: BenchmarkDataset,
    judge_config: HabitusConfig,
    concurrency: int = 16,
    include_evidence: bool = False,
    judge_policy: BenchmarkJudgePolicy = BenchmarkJudgePolicy.DATASET_DEFAULT,
    resume: bool = False,
) -> tuple[BenchmarkJudgeRecord, ...]:
    """独立评分全部回答；Judge 故障与回答错误分别记录。"""

    if not 1 <= concurrency <= 256:
        raise ValueError("judge concurrency must be between one and 256")
    answers = load_answer_records(answers_path)
    questions = {
        (sample.sample_id, question.question_id): question
        for sample in dataset.samples
        for question in sample.questions
    }
    if set((answer.sample_id, answer.question_id) for answer in answers) - set(questions):
        raise BenchmarkRunError("answers contain questions outside the selected dataset")
    destination = Path(output_path).expanduser().resolve()
    manifest_path = destination.with_suffix(f"{destination.suffix}.manifest.json")
    existing = _existing_judgements(destination) if resume else {}
    if destination.exists() and not resume:
        raise BenchmarkRunError("judge output exists; use --resume or a new output path")

    route = judge_config.models.chat.route
    effective_policy = resolve_judge_policy(dataset.name, judge_policy)
    identity: dict[str, object] = {
        "answers_sha256": _file_sha256(Path(answers_path)),
        "selected_sample_digests": [sample.digest for sample in dataset.samples],
        "include_evidence": include_evidence,
        "judge_policy": effective_policy.value,
        "concurrency": concurrency,
        "judge": {
            "provider": route.provider,
            "adapter": route.adapter,
            "model": route.model,
        },
        "config_fingerprint": hashlib.sha256(repr(judge_config).encode()).hexdigest(),
    }
    started_at = datetime.now(UTC).isoformat()
    if manifest_path.exists():
        previous = _read_object(manifest_path)
        if not resume or previous.get("identity") != identity:
            raise BenchmarkRunError("judge resume settings differ from the existing run")
        if isinstance(previous.get("started_at"), str):
            started_at = str(previous["started_at"])
    _write_json(
        manifest_path,
        {
            "schema_version": "habitus_benchmark_judge_run_v2",
            "status": "running",
            "started_at": started_at,
            "updated_at": datetime.now(UTC).isoformat(),
            "identity": identity,
        },
    )

    factory = ProviderFactory()
    register_builtin_adapters(factory)
    chat = factory.create_chat_client(judge_config.models.chat)
    structured = StructuredChatClient(
        chat,
        allow_json_repair=judge_config.models.structured_output.allow_json_repair,
        validation_retries=judge_config.models.structured_output.validation_retries,
    )
    judge = BenchmarkJudge(structured, policy=effective_policy)
    semaphore = asyncio.Semaphore(concurrency)

    async def grade(answer: BenchmarkAnswerRecord) -> BenchmarkJudgeRecord:
        key = (answer.sample_id, answer.question_id)
        if key in existing:
            return existing[key]
        if answer.error:
            return BenchmarkJudgeRecord(
                answer=answer,
                verdict="wrong",
                reasoning=f"Answer pipeline failed: {answer.error}",
            )
        evidence = questions[key].evidence_texts if include_evidence else ()
        async with semaphore:
            return await judge.grade(answer, evidence_texts=evidence)

    records = tuple(await asyncio.gather(*(grade(answer) for answer in answers)))
    for record in records:
        key = (record.answer.sample_id, record.answer.question_id)
        if key not in existing:
            _append_jsonl(destination, record.to_dict())
    _write_json(
        manifest_path,
        {
            "schema_version": "habitus_benchmark_judge_run_v2",
            "status": "completed",
            "started_at": started_at,
            "updated_at": datetime.now(UTC).isoformat(),
            "identity": identity,
            "record_count": len(records),
            "judge_error_count": sum(record.verdict == "judge_error" for record in records),
        },
    )
    return records


def load_judge_records(path: str | Path) -> tuple[Mapping[str, object], ...]:
    """读取 Judge JSONL 供统计；保留原始字段避免二次丢失。"""

    source = Path(path).expanduser().resolve(strict=True)
    result: list[Mapping[str, object]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise BenchmarkRunError(f"judge JSONL is invalid at line {line_number}") from exc
        if not isinstance(value, Mapping):
            raise BenchmarkRunError(f"judge JSONL line {line_number} must be an object")
        result.append(value)
    return tuple(result)


def _existing_judgements(path: Path) -> dict[tuple[str, str], BenchmarkJudgeRecord]:
    if not path.exists():
        return {}
    result: dict[tuple[str, str], BenchmarkJudgeRecord] = {}
    for value in load_judge_records(path):
        answer = _answer_from_judge(value)
        verdict = str(value.get("verdict", ""))
        if verdict not in {"correct", "wrong", "judge_error"}:
            raise BenchmarkRunError("judge record contains an invalid verdict")
        record = BenchmarkJudgeRecord(
            answer=answer,
            verdict=cast(Literal["correct", "wrong", "judge_error"], verdict),
            reasoning=str(value.get("judge_reasoning", "")),
            judge_input_tokens=_int_value(value.get("judge_input_tokens", 0), "judge_input_tokens"),
            judge_output_tokens=_int_value(value.get("judge_output_tokens", 0), "judge_output_tokens"),
            judge_total_tokens=_int_value(value.get("judge_total_tokens", 0), "judge_total_tokens"),
            judge_latency_ms=_float_value(value.get("judge_latency_ms", 0.0), "judge_latency_ms"),
        )
        key = (answer.sample_id, answer.question_id)
        if key in result:
            raise BenchmarkRunError("judge JSONL contains duplicate question identities")
        result[key] = record
    return result


def _answer_from_judge(value: Mapping[str, object]) -> BenchmarkAnswerRecord:
    from benchmark.runner import answer_record_from_mapping

    return answer_record_from_mapping(value)


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
        stream.flush()


def _read_object(path: Path) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BenchmarkRunError(f"invalid judge manifest: {path}") from exc
    if not isinstance(value, Mapping):
        raise BenchmarkRunError(f"judge manifest must be an object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _int_value(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise BenchmarkRunError(f"judge record {label} must be an integer")
    return value


def _float_value(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise BenchmarkRunError(f"judge record {label} must be numeric")
    return float(value)


__all__ = ["judge_answers", "load_judge_records"]
