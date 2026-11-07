"""单用户本地服务的启动前诊断与进程级所有权契约。"""

from __future__ import annotations

import asyncio
import socket
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import yaml
from fastapi.testclient import TestClient

from habitus.config import HabitusConfig
from habitus.infrastructure.store.filesystem import restrict_file_to_owner
from habitus.infrastructure.vector import VectorStoreState
from habitus.integrations.http_api.app import create_http_app
from habitus.integrations.local_service import (
    DoctorStatus,
    ServiceInstanceLock,
    ServiceInstanceLockError,
    run_doctor,
    run_doctor_from_env,
)
from habitus.integrations.local_service import doctor as doctor_module
from habitus.model_client import EmbeddingVector
from habitus.runtime import Runtime
from tests.platform_helpers import grant_everyone_read, grant_group_read, is_private_file

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _config(
    tmp_path: Path,
    *,
    port: int = 8787,
    credentials: bool = False,
    filled: bool = False,
) -> HabitusConfig:
    payload = yaml.safe_load((REPOSITORY_ROOT / "habitus" / "config" / "example.yaml").read_text(encoding="utf-8"))
    payload["storage"]["root"] = str(tmp_path / "data")
    payload["http"]["port"] = port
    if not credentials:
        for name in ("chat", "embedding", "rerank"):
            payload["models"][name]["route"]["credential_ref"] = ""
        payload["memory"]["vector_store"]["route"]["credential_ref"] = ""
        payload["conversation"]["summary_vector_store"]["route"]["credential_ref"] = ""
    elif filled:
        payload["credentials"]["chat"]["api_key"] = "chat-secret"
        payload["credentials"]["embedding"]["api_key"] = "embedding-secret"
        payload["credentials"]["dashscope"]["api_key"] = "dashscope-secret"
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
        assert is_private_file(path)
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
    assert "credentials.chat.api_key" in credentials.detail
    assert "credentials.embedding.api_key" in credentials.detail
    assert "credentials.dashscope.api_key" in credentials.detail
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
    assert "3 named YAML credentials" in credentials.detail


def test_doctor_checks_only_the_base_dependencies_for_the_default_local_vector_store(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checked: list[str] = []

    def dependency(module: str) -> doctor_module.DoctorCheck:
        checked.append(module)
        return doctor_module.DoctorCheck(module, DoctorStatus.PASS, "available")

    monkeypatch.setattr(doctor_module, "_python_dependency_check", dependency)
    config = _config(tmp_path)
    assert config.memory.vector_store.adapter == "local_sqlite"
    run_doctor(config, check_port=False)

    assert sorted(checked) == ["fastapi", "json_repair", "jsonschema", "uvicorn"]


def test_doctor_reports_unregistered_adapter_without_raising(tmp_path: Path) -> None:
    payload = yaml.safe_load((REPOSITORY_ROOT / "habitus" / "config" / "example.yaml").read_text(encoding="utf-8"))
    payload["storage"]["root"] = str(tmp_path / "data")
    payload["models"]["chat"]["route"].update({"adapter": "future_chat", "credential_ref": ""})
    config = HabitusConfig.from_mapping(payload)

    report = run_doctor(config, check_port=False)

    adapter = next(check for check in report.checks if check.name == "adapter_configuration")
    dependencies = next(check for check in report.checks if check.name == "dependencies")
    assert adapter.status is DoctorStatus.FAIL
    assert dependencies.status is DoctorStatus.FAIL


@pytest.mark.parametrize(
    ("store", "change", "expected"),
    [
        ("vector_store", {"options": {"busy_timeout_seconds": 0}}, "busy_timeout_seconds"),
        ("summary_vector_store", {"options": {"legacy_option": True}}, "unknown fields"),
        ("vector_store", {"route": {"base_url": "https://vector.example.com"}}, "base_url"),
    ],
)
def test_doctor_fails_local_vector_misconfiguration_before_any_probe(
    tmp_path: Path, store: str, change: dict[str, dict[str, object]], expected: str
) -> None:
    payload = yaml.safe_load((REPOSITORY_ROOT / "habitus" / "config" / "example.yaml").read_text(encoding="utf-8"))
    payload["storage"]["root"] = str(tmp_path / "data")
    target = payload["memory" if store == "vector_store" else "conversation"][store]
    for section, values in change.items():
        target[section] = {**target[section], **values} if section == "route" else values
    config = HabitusConfig.from_mapping(payload)

    report = run_doctor(config, check_port=False)

    adapter = next(check for check in report.checks if check.name == "adapter_configuration")
    assert adapter.status is DoctorStatus.FAIL
    assert expected in adapter.detail
    assert not (tmp_path / "data" / "vector").exists()


def test_doctor_reports_an_out_of_range_override_as_a_config_failure_naming_the_key(tmp_path: Path) -> None:
    """kinds_* 的值域归领域配置（配置层只查类型）：非 deep 的 doctor 照旧在 config 那一项报失败，点得出配置键。"""

    payload = yaml.safe_load((REPOSITORY_ROOT / "habitus" / "config" / "example.yaml").read_text(encoding="utf-8"))
    payload["behavior"] = {"kinds_revision_weekday": 7}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    restrict_file_to_owner(path)

    report = run_doctor_from_env(environ={"HABITUS_CONFIG_FILE": str(path)}, check_port=False)
    assert [(check.name, check.status) for check in report.checks] == [("config", DoctorStatus.FAIL)]
    assert "invalid 'config.behavior.kinds_revision_weekday'" in report.checks[0].detail


def test_doctor_requires_private_permissions_for_the_secret_bearing_yaml(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_bytes((REPOSITORY_ROOT / "habitus" / "config" / "example.yaml").read_bytes())
    grant_everyone_read(path)

    exposed = run_doctor_from_env(
        environ={"HABITUS_CONFIG_FILE": str(path)},
        check_port=False,
    )
    permissions = next(check for check in exposed.checks if check.name == "config_permissions")
    assert permissions.status is DoctorStatus.FAIL

    restrict_file_to_owner(path)
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
            models=SimpleNamespace(chat=Chat(), embedder=Embedder(), reranker=None),
            memory=SimpleNamespace(vector_index=SimpleNamespace(store=store)),
            conversation=SimpleNamespace(summary_vector_index=SimpleNamespace(store=store)),
        ),
        close=AsyncMock(),
    )
    monkeypatch.setattr("habitus.runtime.build_runtime", Mock(return_value=runtime))

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
            models=SimpleNamespace(chat=Chat(), embedder=Embedder(), reranker=None),
            memory=SimpleNamespace(vector_index=SimpleNamespace(store=store)),
            conversation=SimpleNamespace(summary_vector_index=SimpleNamespace(store=store)),
        ),
        close=AsyncMock(),
    )
    monkeypatch.setattr("habitus.runtime.build_runtime", Mock(return_value=runtime))
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

    from habitus.integrations.local_service import cli
    from habitus.integrations.local_service.instance_lock import ServiceInstanceLockError

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

    import habitus.config as Config
    import habitus.integrations.local_service.instance_lock as instance_lock

    monkeypatch.setattr(Config, "HabitusConfig", StubConfig)
    monkeypatch.setattr(instance_lock, "ServiceInstanceLock", RefusingLock)

    args = cli._parser().parse_args(["repair-source-outputs", "a" * 64, "--consumer", "memory"])
    assert cli._repair_source_outputs(args, {}) == 3
    assert "stop it before repairing" in capsys.readouterr().err


def test_repair_source_outputs_rejects_an_unknown_consumer(tmp_path: Path) -> None:
    from habitus.integrations.local_service import cli

    with pytest.raises(SystemExit):
        cli._parser().parse_args(["repair-source-outputs", "a" * 64, "--consumer", "not_a_consumer"])


def _write_config(tmp_path: Path, *, port: int = 8787) -> Path:
    payload = yaml.safe_load((REPOSITORY_ROOT / "habitus" / "config" / "example.yaml").read_text(encoding="utf-8"))
    payload["storage"]["root"] = str(tmp_path / "data")
    payload["http"]["port"] = port
    path = tmp_path / "config" / "config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    restrict_file_to_owner(path)
    return path


def test_serve_without_http_extra_exits_with_the_extra_to_install(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """没装 http extra 时 serve 先检查依赖再导入 HTTP 应用：一行退出提示，点名要装的 extra。"""

    import sys

    from habitus.integrations.local_service import cli

    for module in ("fastapi", "uvicorn", "habitus.integrations.http_api.app"):
        monkeypatch.setitem(sys.modules, module, None)

    with pytest.raises(SystemExit) as raised:
        cli.main(["serve", "--config", str(_write_config(tmp_path))])

    message = str(raised.value.code)
    assert "'http' extra" in message
    assert doctor_module.pip_install_command("server") in message
    assert doctor_module.pip_install_command("http") in message
    assert "\n" not in message


def test_doctor_names_the_extra_that_provides_a_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    real_find_spec = doctor_module.importlib.util.find_spec
    monkeypatch.setattr(
        doctor_module.importlib.util,
        "find_spec",
        lambda name, *args: None if name == "fastapi" else real_find_spec(name, *args),
    )

    fastapi_check = doctor_module._python_dependency_check("fastapi")

    assert fastapi_check.status is DoctorStatus.FAIL
    assert "'http' extra" in fastapi_check.detail
    assert fastapi_check.detail.endswith(f'pip install "habitus[server] @ {doctor_module.HABITUS_SOURCE}"')
    assert "git+https://github.com/kkxx939-bot/Habitus" in doctor_module.HABITUS_SOURCE


def test_serve_turns_a_failed_startup_preflight_into_one_line_and_never_starts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """预检失败（凭据为空、端口被占）：serve 以一行 SystemExit 退出，不打栈、不构造 Runtime、不起 uvicorn。"""

    import uvicorn

    import habitus.infrastructure.observability as observability
    from habitus.integrations.local_service import cli

    build = Mock()
    run = Mock()
    monkeypatch.setattr("habitus.runtime.build_runtime", build)
    monkeypatch.setattr(uvicorn, "run", run)
    monkeypatch.setattr(observability, "configure_json_logging", Mock())
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        path = _write_config(tmp_path, port=occupied.getsockname()[1])
        with pytest.raises(SystemExit) as raised:
            cli.main(["serve", "--config", str(path)])

    message = str(raised.value.code)
    assert message.startswith("Habitus startup preflight failed: ")
    assert "\n" not in message and "Traceback" not in message
    assert "already in use" in message
    build.assert_not_called()
    run.assert_not_called()


def test_serve_turns_a_failed_runtime_start_into_one_line_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
) -> None:
    """预检过了、``runtime.start`` 才失败（凭据非空但填错）：真实 uvicorn 起停，一行原因、非零退出，日志里没有栈。"""

    import habitus.infrastructure.observability as observability
    from habitus.infrastructure.vector import VectorStoreError
    from habitus.integrations.local_service import cli

    runtime = _runtime()
    runtime.start = AsyncMock(  # type: ignore[method-assign]
        side_effect=VectorStoreError(
            "vector store request failed: 401 InvalidAccessKey:\n  The security token[AKLTleaked0001] included is invalid"
        )
    )
    monkeypatch.setattr("habitus.runtime.build_runtime", Mock(return_value=runtime))
    monkeypatch.setattr(doctor_module, "run_startup_preflight", Mock())
    monkeypatch.setattr(observability, "configure_json_logging", Mock())
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    path = _write_config(tmp_path, port=port)

    with pytest.raises(SystemExit) as raised:
        cli.main(["serve", "--config", str(path)])

    message = str(raised.value.code)
    # 回包带出的凭据原文不进这一行，也不进服务日志
    assert message == (
        "Habitus startup failed: vector store request failed: 401 InvalidAccessKey: "
        "The security token[REDACTED] included is invalid"
    )
    runtime.close.assert_awaited_once()
    captured = capfd.readouterr()
    assert "Traceback" not in captured.err + captured.out
    assert "AKLTleaked0001" not in captured.err + captured.out


@pytest.fixture(autouse=True)
def plugin_default_connection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """插件默认读的那份连接文件换到 tmp 下（serve 启动时写它、doctor 读它），本文件的测试都不碰本机的 ``~/.habitus``。"""

    from habitus.integrations.local_service import initialization

    default = tmp_path / "default-home" / ".habitus" / "agent-plugin" / "connection.json"
    monkeypatch.setattr(initialization, "DEFAULT_PLUGIN_CONNECTION_PATH", default)
    monkeypatch.delenv("HABITUS_PLUGIN_STATE_DIR", raising=False)
    return default


def test_serve_rewrites_the_plugin_connection_files_to_the_port_it_listens_on(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    plugin_default_connection: Path,
) -> None:
    """T11（裁定 75 第 4 条）：init 之后改了 ``http.port``，serve 启动时按这次的监听地址把连接文件刷新（配置旁边那份与默认那份，
    与 init 写的同几份），插件不用重跑 init 就连得上；之后 doctor 那一项通过。"""

    import uvicorn

    import habitus.infrastructure.observability as observability
    from habitus.integrations.local_service import cli

    path = _write_config(tmp_path, port=18791)
    adjacent = path.parent / "agent-plugin" / "connection.json"
    for stale in (adjacent, plugin_default_connection):
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text('{"base_url": "http://127.0.0.1:8787", "schema_version": 1}\n', encoding="utf-8")
    environ = {"HABITUS_CONFIG_FILE": str(path)}
    stale_check = next(
        check
        for check in run_doctor_from_env(environ=environ, check_port=False).checks
        if check.name == "plugin_connection"
    )
    assert stale_check.status is DoctorStatus.WARN

    run = Mock()
    monkeypatch.setattr("habitus.runtime.build_runtime", Mock(return_value=_runtime()))
    monkeypatch.setattr(doctor_module, "run_startup_preflight", Mock())
    monkeypatch.setattr(observability, "configure_json_logging", Mock())
    monkeypatch.setattr(uvicorn, "run", run)
    cli.main(["serve", "--config", str(path)])

    run.assert_called_once()
    for connection in (adjacent, plugin_default_connection):
        assert '"base_url": "http://127.0.0.1:18791"' in connection.read_text(encoding="utf-8"), connection
    fresh = next(
        check
        for check in run_doctor_from_env(environ=environ, check_port=False).checks
        if check.name == "plugin_connection"
    )
    assert fresh.status is DoctorStatus.PASS


def test_doctor_names_a_plugin_connection_file_that_points_elsewhere(
    tmp_path: Path,
    plugin_default_connection: Path,
) -> None:
    """T11：连接文件指的端口与配置的不一样（改了端口、服务还没重启）、或者缺了：doctor 报 WARN，写出哪份文件指向哪里、
    配置监听在哪、怎么办；不算失败。"""

    path = _write_config(tmp_path, port=18791)
    adjacent = path.parent / "agent-plugin" / "connection.json"
    adjacent.parent.mkdir(parents=True)
    adjacent.write_text('{"base_url": "http://127.0.0.1:8787", "schema_version": 1}\n', encoding="utf-8")

    report = run_doctor_from_env(environ={"HABITUS_CONFIG_FILE": str(path)}, check_port=False)

    check = next(check for check in report.checks if check.name == "plugin_connection")
    assert check.status is DoctorStatus.WARN
    assert f"{adjacent} points to http://127.0.0.1:8787" in check.detail
    assert f"{plugin_default_connection} is missing" in check.detail
    assert "listens on http://127.0.0.1:18791" in check.detail
    assert "restart habitus-server" in check.detail and "habitus-server init" in check.detail


def test_port_check_says_the_port_is_in_use_and_how_to_skip_it(tmp_path: Path) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen()
        port = occupied.getsockname()[1]
        check = doctor_module._port_check("127.0.0.1", port)

    assert check.status is DoctorStatus.FAIL
    assert f"127.0.0.1:{port} is already in use" in check.detail
    assert "--skip-port" in check.detail


def test_doctor_rejects_a_group_readable_config(tmp_path: Path) -> None:
    """只给同组读权限（0640）也要判失败：组和其他人任何一位都不许有。"""

    path = _write_config(tmp_path)
    grant_group_read(path)

    report = run_doctor_from_env(environ={"HABITUS_CONFIG_FILE": str(path)}, check_port=False)

    permissions = next(check for check in report.checks if check.name == "config_permissions")
    assert permissions.status is DoctorStatus.FAIL


def test_doctor_command_exit_code_follows_the_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from habitus.integrations.local_service import cli

    path = _write_config(tmp_path)
    failing = doctor_module.DoctorReport((doctor_module.DoctorCheck("config", DoctorStatus.FAIL, "broken"),))
    passing = doctor_module.DoctorReport((doctor_module.DoctorCheck("config", DoctorStatus.PASS, "ok"),))

    monkeypatch.setattr(doctor_module, "run_doctor_from_env", Mock(return_value=failing))
    with pytest.raises(SystemExit) as raised:
        cli.main(["doctor", "--config", str(path), "--skip-port"])
    assert raised.value.code == 1

    monkeypatch.setattr(doctor_module, "run_doctor_from_env", Mock(return_value=passing))
    cli.main(["doctor", "--config", str(path), "--skip-port"])


def test_repair_source_outputs_dry_run_only_reports_and_never_repairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import habitus.config as Config
    import habitus.integrations.local_service.adapter_catalog as adapter_catalog
    import habitus.integrations.local_service.instance_lock as instance_lock
    from habitus.integrations.local_service import cli

    class StubConfig:
        storage_root = tmp_path

        @classmethod
        def from_env(cls, *, environ):  # noqa: ANN001, ANN206
            return cls()

    class Lock:
        def __init__(self, path) -> None:  # noqa: ANN001
            self.path = path

        def acquire(self) -> None:
            pass

        def release(self) -> None:
            pass

    runtime = SimpleNamespace(
        initialize=Mock(),
        close=AsyncMock(),
        repair_conversation_source_outputs=Mock(),
    )
    report = Mock(return_value=0)
    monkeypatch.setattr(Config, "HabitusConfig", StubConfig)
    monkeypatch.setattr(instance_lock, "ServiceInstanceLock", Lock)
    monkeypatch.setattr(
        adapter_catalog,
        "load_adapter_catalog",
        Mock(return_value=SimpleNamespace(providers=None, vector_stores=None)),
    )
    monkeypatch.setattr("habitus.runtime.build_runtime", Mock(return_value=runtime))
    monkeypatch.setattr(cli, "_report_repair_plan", report)

    args = cli._parser().parse_args(["repair-source-outputs", "a" * 64, "--consumer", "memory", "--dry-run"])

    assert cli._repair_source_outputs(args, {}) == 0
    report.assert_called_once()
    runtime.repair_conversation_source_outputs.assert_not_called()


def _ingress_check(tmp_path: Path, **sizes: int) -> doctor_module.DoctorCheck:
    payload = yaml.safe_load((REPOSITORY_ROOT / "habitus" / "config" / "example.yaml").read_text(encoding="utf-8"))
    payload["storage"]["root"] = str(tmp_path / "data")
    for dotted, value in sizes.items():
        section, *middle, key = dotted.split("__")
        target = payload[section]
        for name in middle:
            target = target[name]
        target[key] = value
    return doctor_module._ingress_capacity_check(HabitusConfig.from_mapping(payload))


def test_doctor_ingress_capacity_passes_when_every_ceiling_holds_the_largest_ingress(tmp_path: Path) -> None:
    check = _ingress_check(tmp_path)

    assert check.status is DoctorStatus.PASS
    assert "conversation.source.max_envelope_bytes=67108864" in check.detail


def test_doctor_ingress_capacity_fails_on_a_conversation_source_envelope_below_the_parts_limit(tmp_path: Path) -> None:
    """旧模板写死 16 MiB 的会话源单条上限：分块拼回的大轮必被会话源拒收，doctor 要报 FAIL 并说改哪个键。"""

    check = _ingress_check(tmp_path, conversation__source__max_envelope_bytes=16 * 1024 * 1024)

    assert check.status is DoctorStatus.FAIL
    assert check.detail.startswith(
        "conversation.source.max_envelope_bytes=16777216 is below http.remember_parts_max_bytes=67108864: "
        "set conversation.source.max_envelope_bytes to at least 67108864"
    )
    assert "conversation.journal.max_file_bytes" not in check.detail.split(" (")[0]


def test_doctor_ingress_capacity_measures_the_parts_limit_not_only_the_request_limit(tmp_path: Path) -> None:
    check = _ingress_check(tmp_path, http__remember_parts_max_bytes=128 * 1024 * 1024)

    assert check.status is DoctorStatus.FAIL
    fixes = check.detail.split(" (")[0]
    assert "is below http.remember_parts_max_bytes=134217728" in fixes
    assert "set conversation.source.max_envelope_bytes to at least 134217728" in fixes
    assert "set conversation.journal.max_file_bytes to at least 134217728" in fixes


def test_install_hints_use_the_readme_install_from_the_repository() -> None:
    """还没上 PyPI：提示照 README 从仓库装（pip install 'habitus[server]' 会失败，或装上同名的第三方包）。"""

    assert (
        doctor_module.pip_install_command("server") == f'pip install "habitus[server] @ {doctor_module.HABITUS_SOURCE}"'
    )
    for readme in ("README.md", "README.zh-CN.md"):
        assert doctor_module.pip_install_command("server") in (REPOSITORY_ROOT / readme).read_text(encoding="utf-8")


def test_chinese_readme_calls_the_service_local() -> None:
    """Habitus 没有服务端，只有本机的本地服务（裁定：「没有服务端，我们是本地的」）。"""

    assert "服务端" not in (REPOSITORY_ROOT / "README.zh-CN.md").read_text(encoding="utf-8")


def test_readmes_say_what_init_and_the_part_staging_actually_do() -> None:
    """裁定 75 第 2、6 条只改 README：自定义配置路径的 init 同时改写默认的 ``connection.json``（T8，行为由
    ``test_custom_config_init_persists_plugin_connection_in_the_default_state_root`` 钉住）；分块暂存在下一块到来时才清掉过期的
    （T20，行为由 ``test_staged_parts_expire_after_their_ttl_and_free_their_bytes`` 钉住）。"""

    english = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    chinese = (REPOSITORY_ROOT / "README.zh-CN.md").read_text(encoding="utf-8")
    assert "also rewrites the default `~/.habitus/agent-plugin/connection.json`" in english
    assert "同时改写默认的 `~/.habitus/agent-plugin/connection.json`" in chinese
    assert "is dropped when the next part of any upload arrives" in english
    assert "在下一次有任何上传的块到来时清掉" in chinese
    assert "receives no new part for `http.remember_parts_ttl_seconds` (600 seconds by default) is dropped;" not in english


def test_doctor_deep_writes_only_structured_logs_to_stderr(tmp_path: Path) -> None:
    """--deep 探测失败时运行时记观测日志：CLI 先配好日志，stderr 上只有 JSON 结构化日志，没有裸的 "habitus observation"。"""

    import json
    import os
    import subprocess
    import sys

    text = (REPOSITORY_ROOT / "habitus" / "config" / "example.yaml").read_text(encoding="utf-8")
    payload = yaml.safe_load(text)
    payload["storage"]["root"] = str(tmp_path / "data")
    # 模型指向本机关着的端口：探测立刻失败、不出网；向量库是默认的本地文件
    for name in ("chat", "embedding", "rerank"):
        payload["models"][name]["route"]["base_url"] = "http://127.0.0.1:9"
    payload["credentials"]["chat"]["api_key"] = "fake"
    payload["credentials"]["embedding"]["api_key"] = "fake"
    payload["credentials"]["dashscope"]["api_key"] = "fake"
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    restrict_file_to_owner(config)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "habitus.integrations.local_service.cli",
            "doctor",
            "--config",
            str(config),
            "--deep",
            "--skip-port",
            # 探测超时留够客户端把重试走完、记下 ERROR：POSIX 上连关着的端口立刻被拒；Windows 上每次要等约 2 秒的
            # SYN 重传才判失败，3 秒会先超时，日志里就没有那条 ERROR
            "--probe-timeout",
            "60",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        cwd=REPOSITORY_ROOT,
        env={**os.environ, "HOME": str(tmp_path)},
    )

    assert "[FAIL] embedding_probe" in completed.stdout
    records = [json.loads(line) for line in completed.stderr.splitlines()]
    for record in records:
        assert {"level", "logger", "message", "timestamp"} <= record.keys()
    assert any(record["level"] == "ERROR" for record in records)


@pytest.mark.parametrize(
    ("returncode", "stdout", "status"),
    [(0, "v16.20.2\n", DoctorStatus.WARN), (0, "v18.0.0\n", DoctorStatus.PASS), (1, "v20.0.0\n", DoctorStatus.WARN)],
)
def test_node_check_warns_below_node_18_or_on_a_failed_version_call(
    monkeypatch: pytest.MonkeyPatch,
    returncode: int,
    stdout: str,
    status: DoctorStatus,
) -> None:
    monkeypatch.setattr(doctor_module.shutil, "which", Mock(return_value="/usr/bin/node"))
    monkeypatch.setattr(doctor_module.subprocess, "run", Mock(return_value=Mock(returncode=returncode, stdout=stdout)))

    check = doctor_module._node_check()

    assert check.status is status
    if status is DoctorStatus.WARN:
        assert "Node.js >=18 is required" in check.detail


def test_storage_check_fails_with_less_than_one_gibibyte_free(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    usage = SimpleNamespace(total=0, used=0, free=1024**3 - 1)
    monkeypatch.setattr(doctor_module.shutil, "disk_usage", Mock(return_value=usage))

    check = doctor_module._storage_check(tmp_path / "data")

    assert check.status is DoctorStatus.FAIL
    assert f"free_bytes={1024**3 - 1}" in check.detail


@pytest.mark.parametrize("timeout", [0, 120.5, 121])
def test_doctor_refuses_a_probe_timeout_outside_zero_to_120_seconds(tmp_path: Path, timeout: float) -> None:
    with pytest.raises(ValueError, match="at most 120"):
        run_doctor(_config(tmp_path), check_port=False, probe_timeout_seconds=timeout)


@pytest.mark.parametrize("raw_path", ["", "   "])
def test_doctor_treats_a_blank_config_path_as_missing(raw_path: str) -> None:
    """环境变量里的配置路径空白：公开入口在读配置那一步就失败；权限检查自己也按"没给路径"失败（W16）。"""

    environ = {"HABITUS_CONFIG_FILE": raw_path}
    (config_check,) = run_doctor_from_env(environ=environ, check_port=False).checks
    assert (config_check.name, config_check.status) == ("config", DoctorStatus.FAIL)
    assert doctor_module._config_file_security(environ) == doctor_module.DoctorCheck(
        "config_permissions", DoctorStatus.FAIL, "HABITUS_CONFIG_FILE is missing"
    )
