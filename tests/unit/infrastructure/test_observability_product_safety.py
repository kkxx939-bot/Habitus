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
        name="habitus.test",
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
    [
        r"C:\Users\Alice Smith\private folder\secret.txt",
        "C:/Users/Alice Smith/private folder/secret.txt",
        r"\\server\private share\Alice Smith\secret.txt",
        "/Users/Alice Smith/private folder/secret.txt",
        "/Users Name/private/secret.txt",
        "/Top Folder/private/secret.txt",
        "/Users/Alice, Smith/private/secret.txt",
        r"C:\Users\Alice, Smith\private\secret.txt",
        r"\\server\Alice, Smith\private\secret.txt",
        "//server/share/private folder/secret.txt",
        "file://server/share/private/secret.txt",
        "smb://server/private/Alice/secret.txt",
        "nfs://server/home/alice/secret",
        "afp://server/private/Alice/secret.txt",
        "path:/Users/Alice Smith/private/secret.txt",
        r"\Users\Alice Smith\private\secret.txt",
        r"\ProgramData\Alice Smith\secret.txt",
        r"C:Users\Alice Smith\private\secret.txt",
        r"\Top Folder\private\secret.txt",
        r"C:Top Folder\private\secret.txt",
        "/secret",
        r"\secret",
    ],
)
def test_json_logging_redacts_cross_platform_absolute_paths(path: str) -> None:
    formatter = JSONLogFormatter()
    record = logging.LogRecord(
        name="habitus.test",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg=f"storage failure at {path}",
        args=(),
        exc_info=None,
    )

    payload = json.loads(formatter.format(record))

    assert "Alice Smith" not in payload["message"]
    assert "server" not in payload["message"]
    assert path not in payload["message"]
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


@pytest.mark.parametrize("failure_phase", ["create", "enter", "exit"])
def test_managed_observability_isolates_span_lifecycle_failures(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    failure_phase: str,
) -> None:
    managed = ManagedObservability(
        ObservabilityConfig(tracing=ObservabilityTracingConfig(enabled=True)),
        workflow_root=tmp_path,
    )
    assert managed.otel is not None
    managed._initialized = True
    managed.otel._initialized = True

    class FaultingManager:
        def __enter__(self) -> None:
            if failure_phase == "enter":
                raise RuntimeError("enter failed")

        def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
            if failure_phase == "exit":
                raise RuntimeError("exit failed")

    def start_span(*args: object, **kwargs: object) -> FaultingManager:
        if failure_phase == "create":
            raise RuntimeError("creation failed")
        return FaultingManager()

    monkeypatch.setattr(managed.otel, "start_span", start_span)
    business_completed = False
    with managed.start_span("http", "request"):
        business_completed = True

    assert business_completed is True
    status, detail = managed.health()
    assert status == "degraded"
    expected_stage = "otel_span_exit" if failure_phase == "exit" else "otel_span_enter"
    assert detail == f"{expected_stage}:RuntimeError"


def test_managed_observability_preserves_business_error_when_span_exit_fails(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    managed = ManagedObservability(
        ObservabilityConfig(tracing=ObservabilityTracingConfig(enabled=True)),
        workflow_root=tmp_path,
    )
    assert managed.otel is not None
    managed._initialized = True
    managed.otel._initialized = True

    class FaultingExitManager:
        def __enter__(self) -> None:
            return None

        def __exit__(self, _type: object, _value: object, _traceback: object) -> None:
            raise RuntimeError("span exit failed")

    monkeypatch.setattr(managed.otel, "start_span", lambda *args, **kwargs: FaultingExitManager())

    with pytest.raises(ValueError, match="business failed"):
        with managed.start_span("memory", "commit"):
            raise ValueError("business failed")

    assert managed.health() == ("degraded", "otel_span_exit:RuntimeError")
