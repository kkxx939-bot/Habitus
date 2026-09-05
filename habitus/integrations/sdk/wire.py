"""HTTP 服务端与 SDK 客户端共享的无框架 JSON 编解码。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

from habitus.integrations.sdk.contracts import (
    AgentFlushResult,
    AgentMemoryConsistency,
    AgentMemoryJob,
    AgentRecallDegradation,
    AgentRecallMemory,
    AgentRecallResult,
    AgentRecallSummary,
    AgentRememberResult,
    ServiceCapabilities,
)


def encode_remember(value: AgentRememberResult) -> dict[str, object]:
    return {
        "ignored_items": value.ignored_items,
        "after_turn": value.after_turn,
        "next_sequence": value.next_sequence,
        "jobs": [_encode_job(item) for item in value.jobs],
        "consistency": [_encode_consistency(item) for item in value.consistency],
    }


def decode_remember(value: Mapping[str, Any]) -> AgentRememberResult:
    return AgentRememberResult(
        ignored_items=_integer(value.get("ignored_items"), "ignored_items", minimum=0),
        after_turn=_boolean(value.get("after_turn"), "after_turn"),
        next_sequence=_integer(value.get("next_sequence"), "next_sequence", minimum=0),
        jobs=_decode_jobs(value.get("jobs")),
        consistency=_decode_consistency(value.get("consistency")),
    )


def encode_flush(value: AgentFlushResult) -> dict[str, object]:
    return {
        "jobs": [_encode_job(item) for item in value.jobs],
        "consistency": [_encode_consistency(item) for item in value.consistency],
    }


def decode_flush(value: Mapping[str, Any]) -> AgentFlushResult:
    return AgentFlushResult(
        jobs=_decode_jobs(value.get("jobs")),
        consistency=_decode_consistency(value.get("consistency")),
    )


def encode_recall(value: AgentRecallResult) -> dict[str, object]:
    return {
        "query": value.query,
        "queries": list(value.queries),
        "context": value.context,
        "memories": [
            {
                "uri": item.uri,
                "score": item.score,
                "matched_queries": list(item.matched_queries),
            }
            for item in value.memories
        ],
        "summaries": [
            {"reference": item.reference, "score": item.score}
            for item in value.summaries
        ],
        "degradations": [
            {"stage": item.stage, "error_type": item.error_type}
            for item in value.degradations
        ],
        "budget_exhausted": value.budget_exhausted,
    }


def decode_recall(value: Mapping[str, Any]) -> AgentRecallResult:
    return AgentRecallResult(
        query=_text(value.get("query"), "query"),
        queries=tuple(_text(item, "queries item") for item in _sequence(value.get("queries"), "queries")),
        context=_text(value.get("context"), "context", allow_empty=True),
        memories=tuple(
            AgentRecallMemory(
                uri=_text(item.get("uri"), "memory.uri"),
                score=_number(item.get("score"), "memory.score"),
                matched_queries=tuple(
                    _text(query, "memory.matched_query")
                    for query in _sequence(item.get("matched_queries"), "memory.matched_queries")
                ),
            )
            for item in _mappings(value.get("memories"), "memories")
        ),
        summaries=tuple(
            AgentRecallSummary(
                reference=_text(item.get("reference"), "summary.reference"),
                score=_number(item.get("score"), "summary.score"),
            )
            for item in _mappings(value.get("summaries"), "summaries")
        ),
        degradations=tuple(
            AgentRecallDegradation(
                stage=_text(item.get("stage"), "degradation.stage"),
                error_type=_text(item.get("error_type"), "degradation.error_type"),
            )
            for item in _mappings(value.get("degradations"), "degradations")
        ),
        budget_exhausted=_boolean(value.get("budget_exhausted"), "budget_exhausted"),
    )


def decode_capabilities(value: Mapping[str, Any]) -> ServiceCapabilities:
    return ServiceCapabilities(
        api_version=_text(value.get("api_version"), "api_version"),
        service_version=_text(value.get("service_version"), "service_version"),
        protocols=tuple(_text(item, "protocol") for item in _sequence(value.get("protocols"), "protocols")),
        features=tuple(_text(item, "feature") for item in _sequence(value.get("features"), "features")),
    )


def decode_cursor(value: Mapping[str, Any]) -> int:
    return _integer(value.get("next_sequence"), "next_sequence", minimum=0)


def require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{label} must be an object with text keys")
    return value


def _encode_job(value: AgentMemoryJob) -> dict[str, object]:
    return {
        "memory_sequence": value.memory_sequence,
        "conversation_id": value.conversation_id,
        "started_on": value.started_on.isoformat(),
        "status": value.status,
    }


def _encode_consistency(value: AgentMemoryConsistency) -> dict[str, object]:
    return {"memory_sequence": value.memory_sequence, "state": value.state}


def _decode_jobs(value: object) -> tuple[AgentMemoryJob, ...]:
    return tuple(
        AgentMemoryJob(
            memory_sequence=_integer(item.get("memory_sequence"), "job.memory_sequence", minimum=1),
            conversation_id=_text(item.get("conversation_id"), "job.conversation_id"),
            started_on=_date(item.get("started_on"), "job.started_on"),
            status=_text(item.get("status"), "job.status"),
        )
        for item in _mappings(value, "jobs")
    )


def _decode_consistency(value: object) -> tuple[AgentMemoryConsistency, ...]:
    return tuple(
        AgentMemoryConsistency(
            memory_sequence=_integer(
                item.get("memory_sequence"),
                "consistency.memory_sequence",
                minimum=1,
            ),
            state=_text(item.get("state"), "consistency.state"),
        )
        for item in _mappings(value, "consistency")
    )


def _sequence(value: object, label: str) -> Sequence[object]:
    if not isinstance(value, list | tuple):
        raise ValueError(f"{label} must be an array")
    return value


def _mappings(value: object, label: str) -> tuple[Mapping[str, Any], ...]:
    return tuple(require_mapping(item, f"{label} item") for item in _sequence(value, label))


def _text(value: object, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise ValueError(f"{label} must be text")
    return value


def _integer(value: object, label: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} must be an integer greater than or equal to {minimum}")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be a number")
    return float(value)


def _boolean(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be boolean")
    return value


def _date(value: object, label: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO-8601 date")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO-8601 date") from exc


__all__ = [
    "decode_capabilities",
    "decode_cursor",
    "decode_flush",
    "decode_recall",
    "decode_remember",
    "encode_flush",
    "encode_recall",
    "encode_remember",
    "require_mapping",
]
