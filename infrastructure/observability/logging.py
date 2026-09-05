"""安全结构化日志后端与独立 HTTP 进程日志配置。"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

from foundation.observability import (
    ObservationEvent,
    ObservationStatus,
    current_observation_context,
)
from foundation.redaction import redact_sensitive_text

_SAFE_LOG_FIELDS = frozenset(
    {
        "request_id",
        "memory_sequence",
        "transaction_id",
        "worker_id",
        "attempt",
        "error_code",
        "error_type",
        "retryable",
        "http_method",
        "http_route",
        "http_status_code",
        "duration_ms",
    }
)
_STANDARD_LOG_FIELDS = frozenset(logging.makeLogRecord({}).__dict__)
class StructuredLogObserver:
    """把有界观察事件写成结构化日志，不展开业务输入或输出。"""

    def __init__(self, *, enabled: bool = True, logger_name: str = "habitus.observability") -> None:
        if not isinstance(enabled, bool):
            raise TypeError("enabled must be boolean")
        self.enabled = enabled
        self.logger = logging.getLogger(logger_name)

    def record(self, event: ObservationEvent) -> None:
        if not isinstance(event, ObservationEvent):
            raise TypeError("event must be ObservationEvent")
        if not self.enabled:
            return
        level = logging.DEBUG
        if event.category == "http" and event.operation == "request":
            level = logging.INFO
        if event.status is ObservationStatus.DEGRADED:
            level = logging.WARNING
        elif event.status is ObservationStatus.FAILURE:
            level = logging.ERROR
        payload: dict[str, object] = {
            "event": "observation",
            "category": event.category,
            "operation": event.operation,
            "status": event.status.value,
            "duration_ms": round(event.duration_seconds * 1000, 3),
        }
        payload.update(_safe_attributes(event.attributes))
        payload.update(_context_fields(event))
        self.logger.log(level, "habitus observation", extra={"observation": payload})


class JSONLogFormatter(logging.Formatter):
    """把标准日志也绑定到同一请求和 Job 上下文。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": _safe_message(record.getMessage()),
        }
        observation = getattr(record, "observation", None)
        if isinstance(observation, dict):
            payload.update(observation)
        context = current_observation_context()
        for name in ("request_id", "memory_sequence", "transaction_id", "worker_id", "attempt"):
            value = getattr(context, name)
            if value is not None:
                payload.setdefault(name, value)
        for name in _SAFE_LOG_FIELDS:
            value = getattr(record, name, None)
            if isinstance(value, str | int | float | bool):
                payload[name] = value
        payload.update(_trace_fields())
        if record.exc_info is not None and record.exc_info[0] is not None:
            payload["exception_type"] = record.exc_info[0].__name__
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _STANDARD_LOG_FIELDS and key in _SAFE_LOG_FIELDS
        }
        payload.update(extras)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def configure_json_logging(*, level: str = "INFO") -> None:
    """仅由独立进程入口显式调用；嵌入式 Runtime 不修改宿主日志。"""

    numeric_level = getattr(logging, level, None)
    if not isinstance(numeric_level, int):
        raise ValueError("level must be a standard logging level")
    root = logging.getLogger()
    handler = next(
        (item for item in root.handlers if getattr(item, "_habitus_json_handler", False)),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler(sys.stderr)
        handler._habitus_json_handler = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    handler.setFormatter(JSONLogFormatter())
    root.setLevel(numeric_level)


def _safe_attributes(attributes: object) -> dict[str, str | int | float | bool]:
    if not isinstance(attributes, dict):
        return {}
    allowed = {
        "provider",
        "model",
        "http_method",
        "http_route",
        "http_status_code",
        "http_status_class",
        "error_code",
        "error_type",
        "retryable",
        "retry_count",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "result_count",
        "degradation_count",
        "stage",
        "worker_state",
        "job_status",
        "queue_staged",
        "queue_queued",
        "queue_running",
        "queue_failed",
        "queue_committed",
        "queue_oldest_age_seconds",
        "queue_high_watermark",
        "active_locks",
        "hanging_locks",
        "max_active_lock_age_seconds",
    }
    return {
        key: value
        for key, value in attributes.items()
        if key in allowed and isinstance(value, str | int | float | bool)
    }


def _context_fields(event: ObservationEvent) -> dict[str, object]:
    result: dict[str, object] = {}
    for name in ("request_id", "memory_sequence", "transaction_id", "worker_id", "attempt"):
        value = getattr(event.context, name)
        if value is not None:
            result[name] = value
    return result


def _trace_fields() -> dict[str, str]:
    try:
        from opentelemetry import trace
    except ImportError:
        return {}
    span = trace.get_current_span()
    context = span.get_span_context()
    if not context.is_valid:
        return {}
    return {
        "trace_id": f"{context.trace_id:032x}",
        "span_id": f"{context.span_id:016x}",
    }


def _safe_message(value: str) -> str:
    return redact_sensitive_text(value)[:2_000]


__all__ = ["JSONLogFormatter", "StructuredLogObserver", "configure_json_logging"]
