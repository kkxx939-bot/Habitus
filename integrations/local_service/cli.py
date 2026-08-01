"""单用户本地服务的轻量命令入口。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from integrations.local_service.doctor import run_doctor_from_env


def main(argv: Sequence[str] | None = None) -> None:
    """先分派 doctor/serve，再按需导入 HTTP 与 Runtime 依赖。"""

    args = _parser().parse_args(list(argv) if argv is not None else None)
    values = _environment(args.config)
    if args.command == "doctor":
        report = run_doctor_from_env(
            environ=values,
            check_port=not args.skip_port,
            deep=args.deep,
            probe_timeout_seconds=args.probe_timeout,
        )
        if args.json:
            sys.stdout.write(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        else:
            for check in report.checks:
                sys.stdout.write(f"[{check.status.value.upper()}] {check.name}: {check.detail}\n")
        if not report.ok:
            raise SystemExit(1)
        return
    _serve(values)


def _serve(values: Mapping[str, str]) -> None:
    """服务模式才加载可选 HTTP 依赖和重量级组合根。"""

    from Config import M2BOSConfig
    from infrastructure.observability import configure_json_logging
    from integrations.http_api.app import create_http_app
    from integrations.local_service.doctor import run_startup_preflight
    from integrations.local_service.instance_lock import ServiceInstanceLock
    from Runtime import build_runtime

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("m2bos-http serve requires the 'http' optional dependencies") from exc

    config = M2BOSConfig.from_env(environ=values)
    if config.observability.logging.enabled:
        configure_json_logging(level=config.observability.logging.level)
    run_startup_preflight(config, environ=values)
    runtime = build_runtime(config, environ=values)
    app = create_http_app(
        runtime,
        config=config.http,
        instance_lock=ServiceInstanceLock(config.storage_root / "service" / "http.lock"),
    )
    uvicorn.run(app, host=config.http.host, port=config.http.port, log_config=None)


def _environment(config_path: str | None) -> dict[str, str]:
    values = dict(os.environ)
    if config_path is not None:
        values["M2BOS_CONFIG_FILE"] = str(Path(config_path).expanduser().absolute())
    return values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="m2bos-http")
    parser.add_argument("--config", help="覆盖 M2BOS_CONFIG_FILE 指向的配置文件")
    subparsers = parser.add_subparsers(dest="command")
    serve = subparsers.add_parser("serve", help="启动本地 HTTP 服务")
    serve.add_argument("--config", default=argparse.SUPPRESS, help="覆盖 M2BOS_CONFIG_FILE 指向的配置文件")
    doctor = subparsers.add_parser("doctor", help="检查本地服务配置和依赖")
    doctor.add_argument("--config", default=argparse.SUPPRESS, help="覆盖 M2BOS_CONFIG_FILE 指向的配置文件")
    doctor.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    doctor.add_argument("--skip-port", action="store_true", help="不检查监听端口是否可用")
    doctor.add_argument("--deep", action="store_true", help="有界探测 embedding 与向量服务")
    doctor.add_argument("--probe-timeout", type=float, default=15.0, help="每个深度探测的超时秒数")
    parser.set_defaults(command="serve", json=False, skip_port=False, deep=False, probe_timeout=15.0)
    return parser


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover - 由 console script 与人工诊断共用
    main()
