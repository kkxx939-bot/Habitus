"""LoCoMo 官方 JSON 到 Habitus 记忆基准模型的 Adapter。"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from benchmark.datasets.common import (
    list_value,
    object_value,
    ordered_times,
    parse_time,
    read_json_array,
    text_value,
)
from benchmark.model import (
    BenchmarkDataset,
    BenchmarkDatasetName,
    BenchmarkMessage,
    BenchmarkMessageRole,
    BenchmarkQuestion,
    BenchmarkSample,
    BenchmarkSession,
)

_LOCOMO_TIME_FORMATS = (
    "%I:%M %p on %d %B, %Y",
    "%I:%M%p on %d %B, %Y",
    "%d %B, %Y",
)


def load_locomo(
    path: str | Path,
    *,
    sample_indices: tuple[int, ...] = (),
    question_limit: int | None = None,
    include_adversarial: bool = False,
) -> BenchmarkDataset:
    """直接读取 LoCoMo 原始数据；默认遵循 OpenViking 排除 category=5。"""

    raw_samples = read_json_array(path)
    selected = _selected_indices(len(raw_samples), sample_indices)
    samples = tuple(_sample(raw_samples[index], index, question_limit, include_adversarial) for index in selected)
    return BenchmarkDataset(
        name=BenchmarkDatasetName.LOCOMO,
        source_path=str(Path(path).expanduser().resolve()),
        samples=samples,
    )


def _sample(
    raw: object,
    index: int,
    question_limit: int | None,
    include_adversarial: bool,
) -> BenchmarkSample:
    item = object_value(raw, f"locomo[{index}]")
    conversation = object_value(item.get("conversation"), f"locomo[{index}].conversation")
    source_id = str(item.get("sample_id") or f"sample_{index}")
    sample_id = f"sample_{index}"
    speaker_a = text_value(conversation.get("speaker_a"), "speaker_a")
    speaker_b = text_value(conversation.get("speaker_b"), "speaker_b")
    session_keys = sorted(
        (key for key in conversation if key.startswith("session_") and not key.endswith("_date_time")),
        key=_session_number,
    )
    if not session_keys:
        raise ValueError(f"LoCoMo sample {index} contains no sessions")
    fallback = datetime(2000, 1, 1, tzinfo=timezone.utc)
    sessions: list[BenchmarkSession] = []
    evidence_lookup: dict[str, str] = {}
    for position, key in enumerate(session_keys, start=1):
        raw_messages = list_value(conversation[key], f"{sample_id}.{key}")
        started_at = parse_time(
            conversation.get(f"{key}_date_time"),
            _LOCOMO_TIME_FORMATS,
            fallback=fallback + timedelta(days=position - 1),
        )
        times = ordered_times(started_at, len(raw_messages))
        messages: list[BenchmarkMessage] = []
        for message_index, raw_message in enumerate(raw_messages):
            message = object_value(raw_message, f"{sample_id}.{key}[{message_index}]")
            speaker = text_value(message.get("speaker"), "LoCoMo speaker")
            text = text_value(message.get("text"), "LoCoMo message")
            caption = message.get("blip_caption")
            if isinstance(caption, str) and caption.strip():
                text = f"{text}\n[图片语义：{caption.strip()}]"
            role: BenchmarkMessageRole = "prompt" if speaker == speaker_a else "completion"
            if speaker not in {speaker_a, speaker_b}:
                role = "prompt"
            message_id = f"s{position:03d}-m{message_index + 1:04d}"
            messages.append(
                BenchmarkMessage(
                    message_id=message_id,
                    role=role,
                    content=f"[{speaker}]: {text}",
                    occurred_at=times[message_index],
                )
            )
            evidence_lookup[f"D{position}:{message_index + 1}"] = f"{speaker}: {text}"
        sessions.append(
            BenchmarkSession(
                session_id=f"session-{position:03d}",
                started_at=started_at,
                messages=tuple(messages),
                source_label=str(conversation.get(f"{key}_date_time") or key),
            )
        )

    raw_questions = list_value(item.get("qa"), f"{sample_id}.qa")
    questions: list[BenchmarkQuestion] = []
    for question_index, raw_question in enumerate(raw_questions):
        question = object_value(raw_question, f"{sample_id}.qa[{question_index}]")
        category = question.get("category", "unknown")
        if str(category) == "5" and not include_adversarial:
            continue
        evidence_raw = question.get("evidence", [])
        evidence = tuple(str(value) for value in list_value(evidence_raw, "LoCoMo evidence"))
        questions.append(
            BenchmarkQuestion(
                question_id=f"q{question_index:04d}",
                question=text_value(question.get("question"), "LoCoMo question"),
                reference_answer=text_value(question.get("answer"), "LoCoMo answer"),
                question_type=f"category_{category}",
                question_time=sessions[-1].started_at,
                evidence_refs=evidence,
                evidence_texts=tuple(evidence_lookup[item] for item in evidence if item in evidence_lookup),
                metadata={"category": category},
            )
        )
        if question_limit is not None and len(questions) >= question_limit:
            break
    return BenchmarkSample(
        sample_id=sample_id,
        source_id=source_id,
        sessions=tuple(sessions),
        questions=tuple(questions),
    )


def _selected_indices(size: int, requested: tuple[int, ...]) -> tuple[int, ...]:
    if not requested:
        return tuple(range(size))
    if len(set(requested)) != len(requested) or any(index < 0 or index >= size for index in requested):
        raise ValueError("LoCoMo sample indices must be unique and in range")
    return requested


def _session_number(value: str) -> int:
    try:
        return int(value.removeprefix("session_"))
    except ValueError as exc:
        raise ValueError(f"invalid LoCoMo session key: {value}") from exc


__all__ = ["load_locomo"]
