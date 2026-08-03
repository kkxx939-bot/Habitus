"""单用户本地服务的启动前诊断与进程级所有权契约。"""

from __future__ import annotations

import socket
import stat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml
from fastapi.testclient import TestClient

from Config import M2BOSConfig
from integrations.http_api.app import create_http_app
from integrations.local_service import (
    DoctorStatus,
    ServiceInstanceLock,
    ServiceInstanceLockError,
    run_doctor,
    run_doctor_from_env,
)
from integrations.local_service import doctor as doctor_module
from Runtime import Runtime

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _config(
    tmp_path: Path,
    *,
    port: int = 8787,
    credentials: bool = False,
    filled: bool = False,
) -> M2BOSConfig:
    payload = yaml.safe_load((REPOSITORY_ROOT / "Config" / "example.yaml").read_text(encoding="utf-8"))
    payload["storage"]["root"] = str(tmp_path / "data")
    payload["http"]["port"] = port
    if not credentials:
        for name in ("chat", "embedding", "rerank"):
            payload["models"][name]["route"]["credential_ref"] = ""
        payload["memory"]["vector_store"]["route"]["credential_ref"] = ""
        payload["conversation"]["summary_vector_store"]["route"]["credential_ref"] = ""
    elif filled:
        payload["credentials"]["deepseek"]["api_key"] = "deepseek-secret"
        payload["credentials"]["ark"]["api_key"] = "ark-secret"
        payload["credentials"]["dashscope"]["api_key"] = "dashscope-secret"
        payload["credentials"]["vikingdb"]["access_key"] = "viking-access"
        payload["credentials"]["vikingdb"]["secret_key"] = "viking-secret"
    return M2BOSConfig.from_mapping(payload)


def _runtime() -> Runtime:
    runtime = object.__new__(Runtime)
    runtime.start = AsyncMock()  # type: ignore[method-assign]
    runtime.close = AsyncMock()  # type: ignore[method-assign]
    runtime.components = SimpleNamespace(  # type: ignore[assignment]
        infrastructure=SimpleNamespace(observer=None, managed_observability=None)
    )
    return runtime


def test_instance_lock_is_private_exclusive_and_reusable(tmp_path: Path) -> None:
    path = tmp_path / "service" / "http.lock"
    first = ServiceInstanceLock(path)
    second = ServiceInstanceLock(path)

    first.acquire()
    try:
        assert first.acquired is True
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert '"schema_version":"m2bos_local_service_lock_v1"' in path.read_text(encoding="utf-8")
        with pytest.raises(ServiceInstanceLockError, match="already owns"):
            second.acquire()
    finally:
        first.release()

    second.acquire()
    assert second.acquired is True
    second.release()


def test_http_lifespan_holds_and_releases_the_same_instance_lock(tmp_path: Path) -> None:
    runtime = _runtime()
    lock = ServiceInstanceLock(tmp_path / "service" / "http.lock")
    contender = ServiceInstanceLock(lock.path)
    app = create_http_app(runtime, instance_lock=lock)

    with TestClient(app, base_url="http://127.0.0.1:8787"):
        assert lock.acquired is True
        with pytest.raises(ServiceInstanceLockError):
            contender.acquire()

    assert lock.acquired is False
    contender.acquire()
    contender.release()
    runtime.start.assert_awaited_once()
    runtime.close.assert_awaited_once()


def test_doctor_reports_empty_yaml_credentials_without_a_service_api_key(tmp_path: Path) -> None:
    report = run_doctor(_config(tmp_path, credentials=True), check_port=False)
    credentials = next(check for check in report.checks if check.name == "credentials")

    assert credentials.status is DoctorStatus.FAIL
    assert "credentials.deepseek.api_key" in credentials.detail
    assert "credentials.ark.api_key" in credentials.detail
    assert "credentials.dashscope.api_key" in credentials.detail
    assert "credentials.vikingdb.access_key" in credentials.detail
    assert "M2BOS_HTTP" not in credentials.detail


def test_doctor_detects_a_listener_collision_without_starting_runtime(tmp_path: Path) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        port = occupied.getsockname()[1]
        report = run_doctor(_config(tmp_path, port=port), check_port=True)

    port_check = next(check for check in report.checks if check.name == "port")
    assert port_check.status is DoctorStatus.FAIL
    assert report.ok is False


def test_doctor_accepts_multiple_filled_named_yaml_credentials(tmp_path: Path) -> None:
    report = run_doctor(_config(tmp_path, credentials=True, filled=True), check_port=False)
    credentials = next(check for check in report.checks if check.name == "credentials")

    assert credentials.status is DoctorStatus.PASS
    assert "4 named YAML credentials" in credentials.detail


def test_doctor_requires_private_permissions_for_the_secret_bearing_yaml(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_bytes((REPOSITORY_ROOT / "Config" / "example.yaml").read_bytes())
    path.chmod(0o644)

    exposed = run_doctor_from_env(
        environ={"M2BOS_CONFIG_FILE": str(path)},
        check_port=False,
    )
    permissions = next(check for check in exposed.checks if check.name == "config_permissions")
    assert permissions.status is DoctorStatus.FAIL

    path.chmod(0o600)
    private = run_doctor_from_env(
        environ={"M2BOS_CONFIG_FILE": str(path)},
        check_port=False,
    )
    permissions = next(check for check in private.checks if check.name == "config_permissions")
    assert permissions.status is DoctorStatus.PASS


def test_deep_doctor_is_opt_in_and_appends_provider_probes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    called = 0

    def deep_checks(*_args: object, **_kwargs: object) -> list[doctor_module.DoctorCheck]:
        nonlocal called
        called += 1
        return [doctor_module.DoctorCheck("embedding_probe", DoctorStatus.PASS, "dimension=1024")]

    monkeypatch.setattr(doctor_module, "_deep_checks", deep_checks)
    shallow = run_doctor(_config(tmp_path), check_port=False)
    deep = run_doctor(_config(tmp_path), check_port=False, deep=True, probe_timeout_seconds=1)

    assert called == 1
    assert all(check.name != "embedding_probe" for check in shallow.checks)
    assert any(check.name == "embedding_probe" for check in deep.checks)
