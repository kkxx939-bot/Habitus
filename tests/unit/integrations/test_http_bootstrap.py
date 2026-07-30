"""验证 HTTP 进程装配只读取统一配置并安全解析密钥。"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from Config import HTTPAPIConfig
from integrations.http_api import bootstrap


def test_api_keys_are_normalized_bounded_and_operations_are_opt_in() -> None:
    config = HTTPAPIConfig()
    main = "m" * 32
    operations = "o" * 32

    assert bootstrap.resolve_api_key(config, environ={config.api_key_env: main}) == main
    assert bootstrap.resolve_operations_api_key(config, environ={}) is None
    assert bootstrap.resolve_operations_api_key(
        config,
        environ={config.operations_api_key_env: operations},
    ) == operations

    for environ in ({}, {config.api_key_env: "short"}, {config.api_key_env: f" {main}"}):
        with pytest.raises(bootstrap.HTTPAPIBootstrapError):
            bootstrap.resolve_api_key(config, environ=environ)
    for value in ("short", f"{operations} "):
        with pytest.raises(bootstrap.HTTPAPIBootstrapError):
            bootstrap.resolve_operations_api_key(
                config,
                environ={config.operations_api_key_env: value},
            )


def test_factory_uses_one_config_and_passes_same_environment_to_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    environ = {
        "M2BOS_CONFIG_FILE": "/tmp/m2bos.yaml",
        "M2BOS_HTTP_API_KEY": "m" * 32,
        "M2BOS_HTTP_OPERATIONS_API_KEY": "o" * 32,
    }
    config = SimpleNamespace(
        http=HTTPAPIConfig(),
        observability=SimpleNamespace(logging=SimpleNamespace(enabled=True, level="WARNING")),
    )
    runtime = object()
    app = object()
    from_env = Mock(return_value=config)
    build = Mock(return_value=runtime)
    create = Mock(return_value=app)
    configure = Mock()
    monkeypatch.setattr(bootstrap.M2BOSConfig, "from_env", from_env)
    monkeypatch.setattr(bootstrap, "build_runtime", build)
    monkeypatch.setattr(bootstrap, "create_http_app", create)
    monkeypatch.setattr(bootstrap, "configure_json_logging", configure)

    result = bootstrap.create_app_from_env(environ=environ)

    assert result is app
    from_env.assert_called_once_with(environ=environ)
    build.assert_called_once_with(config, environ=environ)
    configure.assert_called_once_with(level="WARNING")
    create.assert_called_once_with(
        runtime,
        api_key="m" * 32,
        operations_api_key="o" * 32,
        config=config.http,
    )


def test_process_entrypoint_binds_configured_listener_without_rebuilding_logic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = SimpleNamespace(
        http=HTTPAPIConfig(host="127.0.0.9", port=9876),
        observability=SimpleNamespace(logging=SimpleNamespace(enabled=False, level="INFO")),
    )
    runtime = object()
    app = object()
    run = Mock()
    monkeypatch.setattr(bootstrap.M2BOSConfig, "from_env", Mock(return_value=config))
    monkeypatch.setattr(bootstrap, "build_runtime", Mock(return_value=runtime))
    monkeypatch.setattr(bootstrap, "create_http_app", Mock(return_value=app))
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=run))
    monkeypatch.setenv(config.http.api_key_env, "m" * 32)
    monkeypatch.delenv(config.http.operations_api_key_env, raising=False)

    bootstrap.main()

    run.assert_called_once_with(app, host="127.0.0.9", port=9876, log_config=None)

