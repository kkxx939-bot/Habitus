"""Habitus 原生数据集协议，支持真实工具调用和任意记忆场景。"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

from habitus.benchmark.datasets.common import (
    list_value,
    object_value,
    parse_time,
    read_json_array,
    safe_identifier,
    text_value,
)
from habitus.benchmark.model import (
    BenchmarkDataset,
    BenchmarkDatasetName,
    BenchmarkMessage,
    BenchmarkMessageRole,
    BenchmarkQuestion,
    BenchmarkSample,
    BenchmarkSession,
    BenchmarkToolStatus,
)

_TIME_FORMATS = ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S%z", "%Y-%m-%d")


def load_native(
    path: str | Path,
    *,
    sample_indices: tuple[int, ...] = (),
    question_limit: int | None = None,
) -> BenchmarkDataset:
    """读取不含内部预期操作的原生 session/question 数据。"""

    raw_samples = read_json_array(path)
    selected = _selected_indices(len(raw_samples), sample_indices)
    samples = tuple(_sample(raw_samples[index], index, question_limit) for index in selected)
    return BenchmarkDataset(
        name=BenchmarkDatasetName.HABITUS,
        source_path=str(Path(path).expanduser().resolve()),
        samples=samples,
    )


def _sample(raw: object, index: int, question_limit: int | None) -> BenchmarkSample:
    item = object_value(raw, f"habitus[{index}]")
    source_id = text_value(item.get("sample_id", f"sample_{index}"), "Habitus sample_id")
    sample_id = f"sample-{index:05d}-{safe_identifier(source_id, fallback='sample')[:80]}"
    sessions_raw = list_value(item.get("sessions"), f"{sample_id}.sessions")
    fallback = datetime(2000, 1, 1, tzinfo=UTC)
    sessions: list[BenchmarkSession] = []
    for session_index, raw_session in enumerate(sessions_raw):
        session = object_value(raw_session, f"{sample_id}.sessions[{session_index}]")
        started_at = parse_time(
            session.get("started_at"),
            _TIME_FORMATS,
            fallback=fallback + timedelta(days=session_index),
        )
        messages_raw = list_value(session.get("messages"), f"{sample_id}.sessions[{session_index}].messages")
        messages: list[BenchmarkMessage] = []
        for message_index, raw_message in enumerate(messages_raw):
            message = object_value(raw_message, "Habitus benchmark message")
            role = _message_role(message.get("role"))
            occurred_at = parse_time(
                message.get("occurred_at"),
                _TIME_FORMATS,
                fallback=started_at + timedelta(microseconds=message_index),
            )
            messages.append(
                BenchmarkMessage(
                    message_id=f"s{session_index + 1:04d}-m{message_index + 1:04d}",
                    role=role,
                    content=message.get("content"),
                    occurred_at=occurred_at,
                    tool_call_id=_optional_text(message.get("tool_call_id")),
                    tool_name=_optional_text(message.get("tool_name")),
                    tool_status=_tool_status(message.get("tool_status")),
                    source_ref=_optional_text(message.get("source_ref")),
                )
            )
        sessions.append(
            BenchmarkSession(
                session_id=f"session-{session_index + 1:04d}",
                started_at=started_at,
                messages=tuple(messages),
                source_label=str(session.get("session_id") or session_index + 1),
            )
        )
    questions_raw = list_value(item.get("questions"), f"{sample_id}.questions")
    questions: list[BenchmarkQuestion] = []
    for question_index, raw_question in enumerate(questions_raw):
        question = object_value(raw_question, "Habitus benchmark question")
        questions.append(
            BenchmarkQuestion(
                question_id=f"q{question_index:04d}",
                question=text_value(question.get("question"), "Habitus question"),
                reference_answer=text_value(question.get("answer"), "Habitus answer"),
                question_type=text_value(question.get("question_type", "custom"), "Habitus question_type"),
                question_time=(
                    parse_time(question["question_time"], _TIME_FORMATS, fallback=sessions[-1].started_at)
                    if "question_time" in question
                    else sessions[-1].started_at
                ),
                evidence_refs=tuple(
                    str(item) for item in list_value(question.get("evidence_refs", []), "evidence_refs")
                ),
                evidence_texts=tuple(
                    text_value(item, "evidence_text")
                    for item in list_value(question.get("evidence_texts", []), "evidence_texts")
                ),
                metadata=object_value(question.get("metadata", {}), "question metadata"),
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


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    return text_value(value, "optional message field")


def _message_role(value: object) -> BenchmarkMessageRole:
    role = text_value(value, "Habitus message role")
    if role not in {"prompt", "completion", "tool_call", "tool_result"}:
        raise ValueError("unsupported Habitus benchmark message role")
    return cast(BenchmarkMessageRole, role)


def _tool_status(value: object) -> BenchmarkToolStatus | None:
    status = _optional_text(value)
    if status is None:
        return None
    if status not in {"completed", "error", "cancelled"}:
        raise ValueError("unsupported Habitus benchmark tool status")
    return cast(BenchmarkToolStatus, status)


def _selected_indices(size: int, requested: tuple[int, ...]) -> tuple[int, ...]:
    if not requested:
        return tuple(range(size))
    if len(set(requested)) != len(requested) or any(index < 0 or index >= size for index in requested):
        raise ValueError("Habitus sample indices must be unique and in range")
    return requested


__all__ = ["load_native"]
