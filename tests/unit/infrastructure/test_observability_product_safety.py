"""验证可观测旁路的失败隔离、最小审计和敏感信息边界。"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import pytest

from Config.observability import ObservabilityConfig, ObservabilityTracingConfig
from foundation.observability import (
    ObservationEvent,
    ObservationStatus,
    bind_observation_context,
)
from infrastructure.observability import AuditStore, JSONLogFormatter, ManagedObservability

UTC = timezone.utc


def _event(
    category: str,
    operation: str,
    *,
    occurred_at: datetime,
    attributes: dict[str, str | int | float | bool] | None = None,
) -> ObservationEvent:
    return ObservationEvent(
        category=category,
        operation=operation,
        status=ObservationStatus.FAILURE,
        duration_seconds=0.01,
        occurred_at=occurred_at,
        attributes=attributes or {},
    )


def test_audit_store_records_only_security_and_operations_with_safe_fields(tmp_path) -> None:
    store = AuditStore(tmp_path / "audit" / "events.sqlite3", retention_days=14, max_records=2)
    now = datetime(2026, 7, 30, tzinfo=UTC)
    with bind_observation_context(request_id="request-audit", memory_sequence=7, transaction_id="tx-audit"):
        store.record(
            _event(
                "retrieval",
                "search",
                occurred_at=now,
                attributes={"error_type": "ignored"},
            )
        )
        for index in range(3):
            store.record(
                _event(
                    "security",
                    "authentication",
                    occurred_at=now + timedelta(seconds=index),
                    attributes={
                        "error_code": f"invalid-{index}",
                        "provider": "must-not-be-audited",
                    },
                )
            )

    records = store.recent(limit=10)

    assert len(records) == 2
    assert [item.attributes["error_code"] for item in records] == ["invalid-2", "invalid-1"]
    assert all("provider" not in item.attributes for item in records)
    assert all(item.request_id == "request-audit" for item in records)
    assert all(item.memory_sequence == 7 for item in records)
    assert store.path.stat().st_mode & 0o777 == 0o600
    assert store.path.parent.stat().st_mode & 0o777 == 0o700


def test_json_logging_redacts_secrets_and_posix_paths_but_keeps_correlation() -> None:
    formatter = JSONLogFormatter()
    record = logging.LogRecord(
        name="m2bos.test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="Bearer secret-token api_key=sk-private failed at /Users/alice/private/file.txt",
        args=(),
        exc_info=None,
    )
    with bind_observation_context(request_id="request-log", memory_sequence=9, transaction_id="tx-log"):
        payload = json.loads(formatter.format(record))

    assert "secret-token" not in payload["message"]
    assert "sk-private" not in payload["message"]
    assert "/Users/alice" not in payload["message"]
    assert payload["request_id"] == "request-log"
    assert payload["memory_sequence"] == 9
    assert payload["transaction_id"] == "tx-log"


@pytest.mark.parametrize(
    "path",
    [r"C:\Users\alice\private\secret.txt", r"\\server\private\secret.txt"],
)
@pytest.mark.xfail(
    strict=True,
    reason="M2BOS-BUG-OBS-001: structured logs do not redact Windows or UNC absolute paths",
)
def test_json_logging_redacts_cross_platform_absolute_paths(path: str) -> None:
    formatter = JSONLogFormatter()
    record = logging.LogRecord(
        name="m2bos.test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg=f"storage failure at {path}",
        args=(),
        exc_info=None,
    )

    payload = json.loads(formatter.format(record))

    assert "alice" not in payload["message"]
    assert "server" not in payload["message"]
    assert "[PATH]" in payload["message"]


def test_managed_observability_reports_otel_failure_without_breaking_business(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ObservabilityConfig(
        tracing=ObservabilityTracingConfig(enabled=True),
    )
    managed = ManagedObservability(config, workflow_root=tmp_path)
    assert managed.otel is not None
    monkeypatch.setattr(managed.otel, "initialize", lambda: (_ for _ in ()).throw(ImportError("sdk missing")))

    managed.initialize()
    managed.record(
        _event(
            "runtime",
            "initialization",
            occurred_at=datetime(2026, 7, 30, tzinfo=UTC),
            attributes={"error_type": "ImportError"},
        )
    )

    status, detail = managed.health()
    assert status == "degraded"
    assert detail == "otel:ImportError"
    assert managed.recent_audit(limit=10)[0].operation == "initialization"
