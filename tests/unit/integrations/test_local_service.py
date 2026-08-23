"""单用户本地服务的启动前诊断与进程级所有权契约。"""

from __future__ import annotations

import asyncio
import socket
import stat
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import yaml
from fastapi.testclient import TestClient

from Config import HabitusConfig
from infrastructure.vector import VectorStoreState
from integrations.http_api.app import create_http_app
from integrations.local_service import (
    DoctorStatus,
    ServiceInstanceLock,
    ServiceInstanceLockError,
    run_doctor,
    run_doctor_from_env,
)
from integrations.local_service import doctor as doctor_module
from ModelClient import EmbeddingVector
from Runtime import Runtime

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _config(
    tmp_path: Path,
    *,
    port: int = 8787,
    credentials: bool = False,
    filled: bool = False,
) -> HabitusConfig:
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
    return HabitusConfig.from_mapping(payload)


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
        assert '"schema_version":"habitus_local_service_lock_v1"' in path.read_text(encoding="utf-8")
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
    assert "HABITUS_HTTP" not in credentials.detail


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


def test_doctor_resolves_conditional_adapter_dependencies_from_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checked: list[str] = []

    def dependency(module: str) -> doctor_module.DoctorCheck:
        checked.append(module)
        return doctor_module.DoctorCheck(module, DoctorStatus.PASS, "available")

    monkeypatch.setattr(doctor_module, "_python_dependency_check", dependency)
    managed = _config(tmp_path)
    run_doctor(managed, check_port=False)
    assert "volcengine" in checked

    payload = yaml.safe_load((REPOSITORY_ROOT / "Config" / "example.yaml").read_text(encoding="utf-8"))
    payload["storage"]["root"] = str(tmp_path / "api-key")
    for group, name in (("memory", "vector_store"), ("conversation", "summary_vector_store")):
        target = payload[group][name]
        target["route"]["credential_ref"] = ""
        target["options"].update({"auth_mode": "api_key", "schema_mode": "precreated"})
    checked.clear()
    run_doctor(HabitusConfig.from_mapping(payload), check_port=False)

    assert "volcengine" not in checked


def test_doctor_reports_unregistered_adapter_without_raising(tmp_path: Path) -> None:
    payload = yaml.safe_load((REPOSITORY_ROOT / "Config" / "example.yaml").read_text(encoding="utf-8"))
    payload["storage"]["root"] = str(tmp_path / "data")
    payload["models"]["chat"]["route"].update(
        {"adapter": "future_chat", "credential_ref": ""}
    )
    config = HabitusConfig.from_mapping(payload)

    report = run_doctor(config, check_port=False)

    adapter = next(check for check in report.checks if check.name == "adapter_configuration")
    dependencies = next(check for check in report.checks if check.name == "dependencies")
    assert adapter.status is DoctorStatus.FAIL
    assert dependencies.status is DoctorStatus.FAIL


def test_doctor_fails_adapter_capacity_before_remote_probe(tmp_path: Path) -> None:
    payload = yaml.safe_load((REPOSITORY_ROOT / "Config" / "example.yaml").read_text(encoding="utf-8"))
    payload["storage"]["root"] = str(tmp_path / "data")
    payload["models"]["embedding"]["dimension"] = 65_536
    config = HabitusConfig.from_mapping(payload)

    report = run_doctor(config, check_port=False)

    adapter = next(check for check in report.checks if check.name == "adapter_configuration")
    assert adapter.status is DoctorStatus.FAIL
    assert "read page sizes" in adapter.detail


def test_doctor_requires_private_permissions_for_the_secret_bearing_yaml(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_bytes((REPOSITORY_ROOT / "Config" / "example.yaml").read_bytes())
    path.chmod(0o644)

    exposed = run_doctor_from_env(
        environ={"HABITUS_CONFIG_FILE": str(path)},
        check_port=False,
    )
    permissions = next(check for check in exposed.checks if check.name == "config_permissions")
    assert permissions.status is DoctorStatus.FAIL

    path.chmod(0o600)
    private = run_doctor_from_env(
        environ={"HABITUS_CONFIG_FILE": str(path)},
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


def test_deep_doctor_does_not_report_an_unready_vector_publication_as_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Chat:
        def health_check(self) -> dict[str, bool]:
            return {"ok": True}

    class Embedder:
        async def embed_query(self, _text: str) -> EmbeddingVector:
            return EmbeddingVector(tuple(0.0 for _ in range(1024)))

    state = VectorStoreState("schema", "fingerprint", 1024, 0, 1, 0, ready=False)
    store = SimpleNamespace(state=AsyncMock(return_value=state))
    runtime = SimpleNamespace(
        components=SimpleNamespace(
            models=SimpleNamespace(chat=Chat(), embedder=Embedder()),
            memory=SimpleNamespace(vector_index=SimpleNamespace(store=store)),
            conversation=SimpleNamespace(summary_vector_index=SimpleNamespace(store=store)),
        ),
        close=AsyncMock(),
    )
    monkeypatch.setattr("Runtime.build_runtime", Mock(return_value=runtime))

    checks = doctor_module._deep_checks(_config(tmp_path), timeout_seconds=0.5)

    vector_checks = [check for check in checks if check.name.endswith("vector_probe")]
    assert len(vector_checks) == 2
    assert all(check.status is not DoctorStatus.PASS for check in vector_checks)


def test_deep_doctor_chat_timeout_is_a_real_wall_clock_bound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class Chat:
        def health_check(self) -> dict[str, bool]:
            time.sleep(0.3)
            return {"ok": True}

        async def health_check_async(self) -> dict[str, bool]:
            await asyncio.sleep(0.3)
            return {"ok": True}

    class Embedder:
        async def embed_query(self, _text: str) -> EmbeddingVector:
            return EmbeddingVector(tuple(0.0 for _ in range(1024)))

    store = SimpleNamespace(state=AsyncMock(return_value=None))
    runtime = SimpleNamespace(
        components=SimpleNamespace(
            models=SimpleNamespace(chat=Chat(), embedder=Embedder()),
            memory=SimpleNamespace(vector_index=SimpleNamespace(store=store)),
            conversation=SimpleNamespace(summary_vector_index=SimpleNamespace(store=store)),
        ),
        close=AsyncMock(),
    )
    monkeypatch.setattr("Runtime.build_runtime", Mock(return_value=runtime))
    started = time.monotonic()

    checks = doctor_module._deep_checks(_config(tmp_path), timeout_seconds=0.01)

    assert time.monotonic() - started < 0.15
    assert next(check for check in checks if check.name == "chat_probe").status is DoctorStatus.FAIL


def test_repair_source_outputs_refuses_while_a_local_service_owns_the_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """修复会删除耐久文件，因此服务在跑时必须拒绝，而不是并发去动存储。"""

    from integrations.local_service import cli
    from integrations.local_service.instance_lock import ServiceInstanceLockError

    class StubConfig:
        storage_root = tmp_path

        @classmethod
        def from_env(cls, *, environ):
            return cls()

    class RefusingLock:
        def __init__(self, path) -> None:
            self.released = False

        def acquire(self) -> None:
            raise ServiceInstanceLockError("already owned")

        def release(self) -> None:  # pragma: no cover - 未取得锁时不应被调用
            self.released = True

    import Config
    import integrations.local_service.instance_lock as instance_lock

    monkeypatch.setattr(Config, "HabitusConfig", StubConfig)
    monkeypatch.setattr(instance_lock, "ServiceInstanceLock", RefusingLock)

    args = cli._parser().parse_args(
        ["repair-source-outputs", "a" * 64, "--consumer", "memory"]
    )
    assert cli._repair_source_outputs(args, {}) == 3
    assert "stop it before repairing" in capsys.readouterr().err


def test_repair_source_outputs_rejects_an_unknown_consumer(tmp_path: Path) -> None:
    from integrations.local_service import cli

    with pytest.raises(SystemExit):
        cli._parser().parse_args(
            ["repair-source-outputs", "a" * 64, "--consumer", "not_a_consumer"]
        )
