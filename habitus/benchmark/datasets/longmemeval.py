"""LongMemEval 官方 JSON 到 Habitus 记忆基准模型的 Adapter。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from habitus.benchmark.datasets.common import (
    list_value,
    object_value,
    ordered_times,
    parse_time,
    read_json_array,
    safe_identifier,
    text_value,
)
from habitus.benchmark.model import (
    BenchmarkDataset,
    BenchmarkDatasetName,
    BenchmarkMessage,
    BenchmarkQuestion,
    BenchmarkSample,
    BenchmarkSession,
)

_LONGMEMEVAL_TIME_FORMATS = ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d %H:%M", "%Y-%m-%d")


def load_longmemeval(
    path: str | Path,
    *,
    sample_indices: tuple[int, ...] = (),
    question_limit: int | None = None,
) -> BenchmarkDataset:
    """直接读取 LongMemEval 原始数据，每个问题保持自己的 haystack。"""

    raw_samples = read_json_array(path)
    selected = _selected_indices(len(raw_samples), sample_indices)
    if question_limit is not None:
        if question_limit <= 0:
            raise ValueError("question_limit must be positive")
        selected = selected[:question_limit]
    samples = tuple(_sample(raw_samples[index], index) for index in selected)
    return BenchmarkDataset(
        name=BenchmarkDatasetName.LONGMEMEVAL,
        source_path=str(Path(path).expanduser().resolve()),
        samples=samples,
    )


def _sample(raw: object, index: int) -> BenchmarkSample:
    item = object_value(raw, f"longmemeval[{index}]")
    source_id = text_value(item.get("question_id"), "LongMemEval question_id")
    sample_id = f"sample-{index:05d}-{safe_identifier(source_id, fallback='question')[:80]}"
    raw_sessions = list_value(item.get("haystack_sessions"), f"{sample_id}.haystack_sessions")
    raw_dates = list_value(item.get("haystack_dates", []), f"{sample_id}.haystack_dates")
    raw_ids = list_value(item.get("haystack_session_ids", []), f"{sample_id}.haystack_session_ids")
    fallback = datetime(2000, 1, 1, tzinfo=UTC)
    sessions: list[BenchmarkSession] = []
    session_evidence: dict[str, str] = {}
    for session_index, raw_session in enumerate(raw_sessions):
        messages_raw = list_value(raw_session, f"{sample_id}.session[{session_index}]")
        date_value = raw_dates[session_index] if session_index < len(raw_dates) else ""
        started_at = parse_time(
            date_value,
            _LONGMEMEVAL_TIME_FORMATS,
            fallback=fallback + timedelta(days=session_index),
        )
        times = ordered_times(started_at, len(messages_raw))
        source_session_id = (
            str(raw_ids[session_index]) if session_index < len(raw_ids) else f"session_{session_index + 1}"
        )
        messages: list[BenchmarkMessage] = []
        for message_index, raw_message in enumerate(messages_raw):
            message = object_value(raw_message, f"{sample_id}.session[{session_index}][{message_index}]")
            source_role = text_value(message.get("role", "user"), "LongMemEval message role")
            if source_role not in {"user", "assistant"}:
                raise ValueError("LongMemEval supports only user and assistant messages")
            messages.append(
                BenchmarkMessage(
                    message_id=f"s{session_index + 1:04d}-m{message_index + 1:04d}",
                    role="prompt" if source_role == "user" else "completion",
                    content=text_value(message.get("content"), "LongMemEval message content"),
                    occurred_at=times[message_index],
                )
            )
        sessions.append(
            BenchmarkSession(
                session_id=f"session-{session_index + 1:04d}",
                started_at=started_at,
                messages=tuple(messages),
                source_label=source_session_id,
            )
        )
        session_evidence[source_session_id] = "\n".join(str(message.content) for message in messages)
    question_time = parse_time(
        item.get("question_date"),
        _LONGMEMEVAL_TIME_FORMATS,
        fallback=sessions[-1].started_at if sessions else fallback,
    )
    answer_session_ids = tuple(
        str(value) for value in list_value(item.get("answer_session_ids", []), "LongMemEval answer_session_ids")
    )
    question = BenchmarkQuestion(
        question_id="q0000",
        question=text_value(item.get("question"), "LongMemEval question"),
        reference_answer=text_value(item.get("answer"), "LongMemEval answer"),
        question_type=text_value(item.get("question_type", "unknown"), "LongMemEval question_type"),
        question_time=question_time,
        evidence_refs=answer_session_ids,
        evidence_texts=tuple(
            session_evidence[session_id] for session_id in answer_session_ids if session_id in session_evidence
        ),
        metadata={"question_id": source_id},
    )
    return BenchmarkSample(
        sample_id=sample_id,
        source_id=source_id,
        sessions=tuple(sessions),
        questions=(question,),
    )


def _selected_indices(size: int, requested: tuple[int, ...]) -> tuple[int, ...]:
    if not requested:
        return tuple(range(size))
    if len(set(requested)) != len(requested) or any(index < 0 or index >= size for index in requested):
        raise ValueError("LongMemEval sample indices must be unique and in range")
    return requested


__all__ = ["load_longmemeval"]
