"""验证本地 HTTP 进程只监听 loopback，并复用统一配置、预检和实例锁。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY, Mock

import pytest

import habitus.config as Config
import habitus.runtime as Runtime
from habitus.config import HTTPAPIConfig
from habitus.integrations.http_api import app as app_module
from habitus.integrations.local_service import ServiceInstanceLock
from habitus.integrations.local_service import cli as local_cli
from habitus.integrations.local_service import doctor as doctor_module


class _DoctorReport:
    ok = True
    checks = (SimpleNamespace(name="config", status=SimpleNamespace(value="pass"), detail="loaded"),)

    @staticmethod
    def to_dict() -> dict[str, object]:
        return {"ok": True, "checks": []}


def test_http_config_rejects_remote_binding_and_removed_auth_fields() -> None:
    assert HTTPAPIConfig(host="127.0.0.9").host == "127.0.0.9"
    assert HTTPAPIConfig(host="::1").host == "::1"
    assert HTTPAPIConfig(host="localhost").host == "localhost"
    for host in ("0.0.0.0", "192.168.1.10", "habitus.example.com"):
        with pytest.raises(ValueError, match="loopback"):
            HTTPAPIConfig(host=host)
    with pytest.raises(Exception, match="api_key_env"):
        HTTPAPIConfig.from_mapping({"api_key_env": "HABITUS_HTTP_API_KEY"})


def test_process_entrypoint_preflights_then_binds_configured_listener(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = SimpleNamespace(
        http=HTTPAPIConfig(host="127.0.0.9", port=9876),
        storage_root=tmp_path,
        observability=SimpleNamespace(logging=SimpleNamespace(enabled=False, level="INFO")),
    )
    runtime = object()
    app = object()
    run = Mock()
    preflight = Mock()
    monkeypatch.setattr(Config.HabitusConfig, "from_env", Mock(return_value=config))
    monkeypatch.setattr(doctor_module, "run_startup_preflight", preflight)
    monkeypatch.setattr(Runtime, "build_runtime", Mock(return_value=runtime))
    create = Mock(return_value=app)
    monkeypatch.setattr(app_module, "create_http_app", create)
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=run))
    monkeypatch.setenv("HABITUS_CONFIG_FILE", str(tmp_path / "config" / "config.yaml"))
    monkeypatch.setenv("HABITUS_PLUGIN_STATE_DIR", str(tmp_path / "plugin-state"))
    refresh = Mock()
    monkeypatch.setattr(local_cli, "_write_plugin_connections", refresh)

    local_cli.main([])

    preflight.assert_called_once()
    # 起 uvicorn 之前按这次的配置刷新插件的连接文件：配置旁边那份与插件默认读的那份
    refresh.assert_called_once_with(
        config,
        (tmp_path / "config" / "agent-plugin" / "connection.json", tmp_path / "plugin-state" / "connection.json"),
        "refresh",
    )
    run.assert_called_once_with(ANY, host="127.0.0.9", port=9876, log_config=None)
    served = run.call_args.args[0]
    # 交给 uvicorn 的是包了启动失败截获的同一个应用
    assert isinstance(served, local_cli._StartupFailureGuard) and served._app is app
    call = create.call_args
    assert call.args == (runtime,)
    assert call.kwargs["config"] is config.http
    lock = call.kwargs["instance_lock"]
    assert isinstance(lock, ServiceInstanceLock)
    assert lock.path == tmp_path / "service" / "http.lock"


def test_doctor_subcommand_uses_the_same_config_and_never_builds_runtime(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    doctor = Mock(return_value=_DoctorReport())
    monkeypatch.setattr(doctor_module, "run_doctor_from_env", doctor)

    local_cli.main(["doctor", "--config", "/tmp/doctor.yaml", "--json", "--skip-port"])

    values = doctor.call_args.kwargs["environ"]
    assert values["HABITUS_CONFIG_FILE"] == "/tmp/doctor.yaml"
    doctor.assert_called_once_with(
        environ=values,
        check_port=False,
        deep=False,
        probe_timeout_seconds=15.0,
        catalog=ANY,
    )
    assert capsys.readouterr().out == '{"checks": [], "ok": true}\n'
