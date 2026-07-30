"""从唯一 m2bOS YAML 配置启动可选 HTTP API。"""

from __future__ import annotations

import os
from collections.abc import Mapping

from Config import HTTPAPIConfig, M2BOSConfig
from infrastructure.observability import configure_json_logging
from integrations.http_api.app import create_http_app
from Runtime import build_runtime


class HTTPAPIBootstrapError(RuntimeError):
    """HTTP API 启动配置不完整或不安全。"""


def resolve_api_key(
    config: HTTPAPIConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> str:
    if not isinstance(config, HTTPAPIConfig):
        raise TypeError("config must be HTTPAPIConfig")
    values = os.environ if environ is None else environ
    if not isinstance(values, Mapping):
        raise TypeError("environ must be a string mapping")
    api_key = values.get(config.api_key_env)
    if not isinstance(api_key, str) or not api_key or api_key != api_key.strip():
        raise HTTPAPIBootstrapError(f"HTTP API key is missing or invalid: {config.api_key_env}")
    if len(api_key) < 32:
        raise HTTPAPIBootstrapError("HTTP API key must contain at least 32 characters")
    return api_key


def resolve_operations_api_key(
    config: HTTPAPIConfig,
    *,
    environ: Mapping[str, str] | None = None,
) -> str | None:
    """只在显式配置独立密钥时启用有状态运维接口。"""

    if not isinstance(config, HTTPAPIConfig):
        raise TypeError("config must be HTTPAPIConfig")
    values = os.environ if environ is None else environ
    if not isinstance(values, Mapping):
        raise TypeError("environ must be a string mapping")
    api_key = values.get(config.operations_api_key_env)
    if api_key is None:
        return None
    if not isinstance(api_key, str) or not api_key or api_key != api_key.strip():
        raise HTTPAPIBootstrapError(f"HTTP operations API key is invalid: {config.operations_api_key_env}")
    if len(api_key) < 32:
        raise HTTPAPIBootstrapError("HTTP operations API key must contain at least 32 characters")
    return api_key


def create_app_from_env(*, environ: Mapping[str, str] | None = None):  # noqa: ANN201
    """供进程入口或 Uvicorn factory 使用的显式装配函数。"""

    values = os.environ if environ is None else environ
    config = M2BOSConfig.from_env(environ=values)
    if config.observability.logging.enabled:
        configure_json_logging(level=config.observability.logging.level)
    api_key = resolve_api_key(config.http, environ=values)
    operations_api_key = resolve_operations_api_key(config.http, environ=values)
    runtime = build_runtime(config, environ=values)
    return create_http_app(
        runtime,
        api_key=api_key,
        operations_api_key=operations_api_key,
        config=config.http,
    )


def main() -> None:
    """启动单进程 HTTP Adapter；并发与 Job 执行仍由 Runtime 管理。"""

    import uvicorn

    config = M2BOSConfig.from_env()
    if config.observability.logging.enabled:
        configure_json_logging(level=config.observability.logging.level)
    api_key = resolve_api_key(config.http)
    operations_api_key = resolve_operations_api_key(config.http)
    runtime = build_runtime(config)
    app = create_http_app(
        runtime,
        api_key=api_key,
        operations_api_key=operations_api_key,
        config=config.http,
    )
    uvicorn.run(
        app,
        host=config.http.host,
        port=config.http.port,
        log_config=None,
    )


__all__ = [
    "HTTPAPIBootstrapError",
    "create_app_from_env",
    "main",
    "resolve_api_key",
    "resolve_operations_api_key",
]
