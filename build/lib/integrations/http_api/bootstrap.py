"""HTTP Factory 兼容入口；进程命令由 local_service.cli 承载。"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence


def create_app_from_env(*, environ: Mapping[str, str] | None = None):  # noqa: ANN201
    """从唯一配置构造应用，并执行与命令入口相同的启动预检。"""

    from Config import M2BOSConfig
    from infrastructure.observability import configure_json_logging
    from integrations.http_api.app import create_http_app
    from integrations.local_service.doctor import run_startup_preflight
    from integrations.local_service.instance_lock import ServiceInstanceLock
    from Runtime import build_runtime

    values = os.environ if environ is None else environ
    config = M2BOSConfig.from_env(environ=values)
    if config.observability.logging.enabled:
        configure_json_logging(level=config.observability.logging.level)
    run_startup_preflight(config, environ=values)
    runtime = build_runtime(config, environ=values)
    return create_http_app(
        runtime,
        config=config.http,
        instance_lock=ServiceInstanceLock(config.storage_root / "service" / "http.lock"),
    )


def main(argv: Sequence[str] | None = None) -> None:
    """保留旧模块调用路径，但不在导入时加载 FastAPI。"""

    from integrations.local_service.cli import main as local_service_main

    local_service_main(argv)


__all__ = ["create_app_from_env", "main"]
