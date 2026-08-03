"""OpenViking 式的单用户 m2bOS 轻量启动与初始化入口。"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TextIO


def main(argv: Sequence[str] | None = None) -> None:
    """先处理 init/doctor/plugin，再延迟导入 HTTP 与 Runtime。"""

    args = _parser().parse_args(list(argv) if argv is not None else None)
    values = _environment(args.config)
    if args.command == "init":
        code = _initialize(args, values)
        if code:
            raise SystemExit(code)
        return
    if args.command == "doctor":
        code = _doctor(args, values)
        if code:
            raise SystemExit(code)
        return
    if args.command == "plugin":
        arguments = list(args.plugin_arguments)
        if args.plugin_help:
            arguments.insert(0, "--help")
        _delegate_plugin(arguments)
        return
    if args.command == "harnesses":
        _delegate_plugin(["harnesses", "--json"])
        return
    if _maybe_offer_init(args.config, values):
        return
    _serve(values)


def _doctor(args: argparse.Namespace, values: Mapping[str, str]) -> int:
    from integrations.local_service.doctor import run_doctor_from_env

    report = run_doctor_from_env(
        environ=values,
        check_port=not args.skip_port,
        deep=args.deep,
        probe_timeout_seconds=args.probe_timeout,
    )
    if args.json:
        sys.stdout.write(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    else:
        _write_doctor(report, sys.stdout)
    return 0 if report.ok else 1


def _initialize(args: argparse.Namespace, values: dict[str, str]) -> int:
    from integrations.local_service.initialization import initialize_config, resolve_config_path

    destination = resolve_config_path(args.config, environ=values)
    source = Path(args.from_config).expanduser().absolute() if args.from_config else None
    try:
        result = initialize_config(destination, source=source, force=args.force)
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"m2bOS initialization failed: {exc}") from exc
    values["M2BOS_CONFIG_FILE"] = str(result.path)
    state = "created" if result.created else "using existing"
    sys.stdout.write(f"m2bOS config {state}: {result.path}\n")
    if result.backup_path is not None:
        sys.stdout.write(f"Previous config backup: {result.backup_path}\n")

    interactive = not args.non_interactive and _interactive(sys.stdin, sys.stdout)
    if interactive:
        _configure_missing_credentials(result.path)
    run_doctor = args.run_doctor
    if run_doctor is None and interactive:
        run_doctor = _confirm("Validate the setup now?", default=True)
    doctor_ok: bool | None = None
    if run_doctor:
        doctor_args = argparse.Namespace(
            skip_port=False,
            deep=args.deep,
            probe_timeout=args.probe_timeout,
            json=False,
        )
        doctor_ok = _doctor(doctor_args, values) == 0

    harnesses = list(args.harnesses)
    if args.all_harnesses:
        harnesses = ["all"]
    elif not harnesses and interactive and _confirm(
        "Install m2bOS memory plugins for detected Agent Harnesses?",
        default=False,
    ):
        harnesses = ["all"]
    if harnesses:
        delegated = ["install"]
        for harness in harnesses:
            delegated.extend(("--harness", harness))
        _delegate_plugin(delegated)

    start = args.start
    if start is None and interactive and doctor_ok is not False:
        start = _confirm("Start the m2bOS server now?", default=False)
    if start:
        if doctor_ok is False:
            sys.stderr.write("m2bOS server was not started because Doctor reported failures.\n")
            return 1
        _exec_serve(result.path)
    if doctor_ok is False:
        return 1
    sys.stdout.write(
        f"Next: m2bos-server --config {result.path}\n"
        "Agent plugins: m2bos plugin install --harness <id>\n"
    )
    return 0


def _configure_missing_credentials(path: Path) -> None:
    from integrations.local_service.initialization import (
        configure_credentials,
        missing_credential_fields,
    )

    missing = missing_credential_fields(path)
    if not missing or not _confirm("Configure missing provider credentials now?", default=True):
        return
    updates: dict[str, dict[str, str]] = {}
    for item in missing:
        secret = _prompt_secret(f"{item.reference}.{item.field}")
        if secret:
            updates.setdefault(item.reference, {})[item.field] = secret
    if not updates:
        return
    try:
        configure_credentials(path, updates)
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"m2bOS credential configuration failed: {exc}") from exc
    count = sum(len(fields) for fields in updates.values())
    sys.stdout.write(f"Configured {count} credential fields in {path}\n")


def _maybe_offer_init(explicit: str | None, values: dict[str, str]) -> bool:
    from integrations.local_service.initialization import resolve_config_path

    path = resolve_config_path(explicit, environ=values)
    if path.exists() or not _interactive(sys.stdin, sys.stdout):
        return False
    sys.stdout.write(f"No m2bOS configuration found at {path}.\n")
    if not _confirm("Run interactive setup now?", default=True):
        return False
    init_args = argparse.Namespace(
        config=str(path),
        from_config=None,
        force=False,
        non_interactive=False,
        run_doctor=None,
        deep=False,
        probe_timeout=15.0,
        harnesses=[],
        all_harnesses=False,
        start=None,
    )
    code = _initialize(init_args, values)
    if code:
        raise SystemExit(code)
    return True


def _delegate_plugin(arguments: Sequence[str]) -> None:
    from integrations.local_service.plugin_cli import run

    try:
        code = run(arguments)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    if code:
        raise SystemExit(code)


def _serve(values: Mapping[str, str]) -> None:
    """服务模式才加载可选 HTTP 依赖和重量级组合根。"""

    from Config import ConfigError, M2BOSConfig
    from infrastructure.observability import configure_json_logging
    from integrations.http_api.app import create_http_app
    from integrations.local_service.doctor import run_startup_preflight
    from integrations.local_service.instance_lock import ServiceInstanceLock
    from Runtime import build_runtime

    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("m2bos-server requires the 'http' optional dependencies") from exc

    try:
        config = M2BOSConfig.from_env(environ=values)
    except (ConfigError, OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"m2bOS configuration failed: {exc}") from exc
    if config.observability.logging.enabled:
        configure_json_logging(level=config.observability.logging.level)
    run_startup_preflight(config, environ=values)
    runtime = build_runtime(config)
    app = create_http_app(
        runtime,
        config=config.http,
        instance_lock=ServiceInstanceLock(config.storage_root / "service" / "http.lock"),
    )
    uvicorn.run(app, host=config.http.host, port=config.http.port, log_config=None)


def _exec_serve(path: Path) -> None:
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "integrations.local_service.cli",
            "serve",
            "--config",
            str(path),
        ],
    )


def _environment(config_path: str | None) -> dict[str, str]:
    values = dict(os.environ)
    if config_path is not None:
        values["M2BOS_CONFIG_FILE"] = str(Path(config_path).expanduser().absolute())
    elif not values.get("M2BOS_CONFIG_FILE", "").strip():
        values["M2BOS_CONFIG_FILE"] = str(Path("~/.m2bos/config.yaml").expanduser().absolute())
    return values


def _interactive(stdin: TextIO, stdout: TextIO) -> bool:
    try:
        return stdin.isatty() and stdout.isatty()
    except (AttributeError, ValueError):
        return False


def _confirm(prompt: str, *, default: bool) -> bool:
    suffix = "[Y/n]" if default else "[y/N]"
    try:
        value = input(f"{prompt} {suffix}: ").strip().lower()
    except (EOFError, OSError):
        return default
    if not value:
        return default
    return value in {"y", "yes"}


def _prompt_secret(label: str) -> str:
    try:
        return getpass.getpass(f"{label}: ")
    except (EOFError, OSError):
        return ""


def _write_doctor(report: object, output: TextIO) -> None:
    for check in report.checks:  # type: ignore[attr-defined]
        output.write(f"[{check.status.value.upper()}] {check.name}: {check.detail}\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="m2bos")
    parser.add_argument("--config", help="覆盖 M2BOS_CONFIG_FILE 指向的配置文件")
    subparsers = parser.add_subparsers(dest="command")

    serve = subparsers.add_parser("serve", help="启动本地 HTTP 服务")
    serve.add_argument("--config", default=argparse.SUPPRESS, help="配置文件")

    doctor = subparsers.add_parser("doctor", help="检查本地服务配置和依赖")
    doctor.add_argument("--config", default=argparse.SUPPRESS, help="配置文件")
    doctor.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    doctor.add_argument("--skip-port", action="store_true", help="不检查监听端口")
    doctor.add_argument("--deep", action="store_true", help="探测 embedding 与向量服务")
    doctor.add_argument("--probe-timeout", type=float, default=15.0, help="单项探测超时秒数")

    init = subparsers.add_parser("init", help="安全初始化配置、Doctor 和 Agent Harness")
    init.add_argument("--config", default=argparse.SUPPRESS, help="目标配置文件")
    init.add_argument("--from-config", help="从一个已经通过严格校验的 YAML 初始化")
    init.add_argument("--force", action="store_true", help="覆盖配置并保留 .bak")
    init.add_argument("--non-interactive", action="store_true", help="禁用全部提示")
    doctor_group = init.add_mutually_exclusive_group()
    doctor_group.add_argument("--doctor", dest="run_doctor", action="store_true", help="初始化后运行 Doctor")
    doctor_group.add_argument("--skip-doctor", dest="run_doctor", action="store_false", help="不运行 Doctor")
    init.set_defaults(run_doctor=None)
    init.add_argument("--deep", action="store_true", help="Doctor 使用深度远端探测")
    init.add_argument("--probe-timeout", type=float, default=15.0, help="单项探测超时秒数")
    init.add_argument("--harness", dest="harnesses", action="append", default=[], help="安装指定 Harness，可重复")
    init.add_argument("--all-harnesses", action="store_true", help="安装所有已检测到的 Harness")
    start_group = init.add_mutually_exclusive_group()
    start_group.add_argument("--start", dest="start", action="store_true", help="初始化后启动服务")
    start_group.add_argument("--no-start", dest="start", action="store_false", help="初始化后不启动服务")
    init.set_defaults(start=None)

    plugin = subparsers.add_parser("plugin", add_help=False, help="委托 Agent Harness 插件生命周期")
    plugin.add_argument("-h", "--help", dest="plugin_help", action="store_true")
    plugin.add_argument("plugin_arguments", nargs=argparse.REMAINDER)
    subparsers.add_parser("harnesses", help="列出已注册和可用的 Agent Harness")

    parser.set_defaults(
        command="serve",
        json=False,
        skip_port=False,
        deep=False,
        probe_timeout=15.0,
        plugin_help=False,
    )
    return parser


__all__ = ["main"]


if __name__ == "__main__":  # pragma: no cover
    main()
