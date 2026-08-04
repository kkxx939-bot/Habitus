"""单用户 m2bOS 的轻量初始化、本地服务和插件启动入口。"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

if TYPE_CHECKING:
    from integrations.local_service.cloud_setup import (
        CloudSetupSelection,
        ProfileSelection,
    )
    from integrations.local_service.setup_registry import SetupField, SetupRegistry


class WizardCancelled(RuntimeError):
    """交互输入流结束或用户取消，且尚未授权写入。"""


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
    from integrations.local_service.adapter_catalog import load_adapter_catalog
    from integrations.local_service.doctor import run_doctor_from_env

    catalog = load_adapter_catalog()
    report = run_doctor_from_env(
        environ=values,
        check_port=not args.skip_port,
        deep=args.deep,
        probe_timeout_seconds=args.probe_timeout,
        catalog=catalog,
    )
    if args.json:
        sys.stdout.write(json.dumps(report.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
    else:
        _write_doctor(report, sys.stdout)
    return 0 if report.ok else 1


def _initialize(args: argparse.Namespace, values: dict[str, str]) -> int:
    from Config import M2BOSConfig
    from integrations.local_service.adapter_catalog import load_adapter_catalog
    from integrations.local_service.cloud_setup import (
        apply_cloud_selection,
        default_cloud_selection,
        selection_from_mapping,
    )
    from integrations.local_service.initialization import (
        initialize_config,
        initialize_config_from_mapping,
        load_initialization_mapping,
        resolve_config_path,
        resolve_plugin_connection_path,
        write_plugin_connection,
    )

    catalog = load_adapter_catalog()
    destination = resolve_config_path(args.config, environ=values)
    source = Path(args.from_config).expanduser().absolute() if args.from_config else None
    interactive = not args.non_interactive and _interactive(sys.stdin, sys.stdout)
    existed = destination.exists()
    try:
        configure_cloud = interactive and source is None and (
            not existed
            or args.force
            or _confirm("Update cloud provider routes now?", default=False)
        )
        if configure_cloud:
            payload = load_initialization_mapping(destination if existed else None)
            defaults = (
                selection_from_mapping(payload, catalog.setup)
                if existed
                else default_cloud_selection(catalog.setup)
            )
            selection = _prompt_cloud_setup(defaults, catalog.setup)
            if selection is None:
                sys.stdout.write("m2bOS cloud setup cancelled.\n")
                return 0
            configured = apply_cloud_selection(payload, selection, catalog.setup)
            result = initialize_config_from_mapping(
                destination,
                configured,
                force=existed or args.force,
            )
        elif source is None and (not existed or args.force):
            payload = apply_cloud_selection(
                load_initialization_mapping(),
                default_cloud_selection(catalog.setup),
                catalog.setup,
            )
            result = initialize_config_from_mapping(
                destination,
                payload,
                force=args.force,
            )
        else:
            result = initialize_config(destination, source=source, force=args.force)
    except WizardCancelled:
        sys.stdout.write("m2bOS setup cancelled before configuration was written.\n")
        return 0
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"m2bOS initialization failed: {exc}") from exc
    values["M2BOS_CONFIG_FILE"] = str(result.path)
    try:
        config = M2BOSConfig.from_file(result.path)
        adjacent_connection = result.path.parent / "agent-plugin" / "connection.json"
        write_plugin_connection(config, adjacent_connection)
        default_connection = resolve_plugin_connection_path(environ=values)
        if default_connection != adjacent_connection:
            write_plugin_connection(config, default_connection)
    except (OSError, TypeError, ValueError) as exc:
        raise SystemExit(f"m2bOS plugin connection initialization failed: {exc}") from exc
    state = (
        "updated"
        if result.backup_path is not None
        else "created"
        if result.created
        else "using existing"
    )
    sys.stdout.write(f"m2bOS config {state}: {result.path}\n")
    if result.backup_path is not None:
        sys.stdout.write(f"Previous config backup: {result.backup_path}\n")

    if interactive:
        _configure_missing_credentials(result.path)
    run_doctor = args.run_doctor
    if run_doctor is None and interactive:
        run_doctor = _confirm_after_write("Validate the setup now?", default=True)
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
    elif not harnesses and interactive and _confirm_after_write(
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
        start = _confirm_after_write("Start the m2bOS server now?", default=False)
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


def _prompt_cloud_setup(
    defaults: object,
    registry: SetupRegistry | None = None,
) -> CloudSetupSelection | None:
    from integrations.local_service.cloud_setup import CloudSetupSelection
    from integrations.local_service.setup_registry import build_builtin_setup_registry

    if not isinstance(defaults, CloudSetupSelection):
        raise TypeError("cloud setup defaults must be CloudSetupSelection")
    resolved_registry = registry or build_builtin_setup_registry()
    sys.stdout.write(
        "\nCloud provider setup\n"
        "Local Chat, Embedding and Rerank are intentionally not included in this setup.\n"
    )
    selections = {
        capability: _prompt_registered_profile(
            resolved_registry,
            capability,
            getattr(defaults, capability),
        )
        for capability in ("chat", "embedding", "rerank", "vector")
    }
    selection = CloudSetupSelection(**selections)
    sys.stdout.write("\nCloud setup summary\n")
    for capability in ("chat", "embedding", "rerank", "vector"):
        selected = getattr(selection, capability)
        profile = resolved_registry.profile(capability, selected.profile_id)
        sys.stdout.write(f"  {capability.title()}: {profile.display_name}\n")
    return selection if _confirm("Save this cloud configuration?", default=True) else None


def _prompt_registered_profile(
    registry: SetupRegistry,
    capability: str,
    default_selection: ProfileSelection,
) -> ProfileSelection:
    from integrations.local_service.cloud_setup import profile_selection

    default_profile = registry.profile(capability, default_selection.profile_id)  # type: ignore[arg-type]
    profiles = list(registry.profiles(capability))  # type: ignore[arg-type]
    if default_profile not in profiles:
        profiles.insert(0, default_profile)
    default_index = profiles.index(default_profile) + 1
    selected = profiles[
        _prompt_choice(
            f"{capability.title()} profile",
            tuple(profile.display_name for profile in profiles),
            default=default_index,
        )
        - 1
    ]
    preserve_existing = (
        selected.profile_id == default_selection.profile_id
        and default_selection.preserve_existing
    )
    defaults = (
        dict(default_selection.values)
        if selected.profile_id == default_selection.profile_id
        else {field.key: field.default for field in selected.fields}
    )
    values = {
        field.key: _prompt_setup_field(field, defaults.get(field.key, field.default))
        for field in selected.fields
    }
    return profile_selection(
        selected,
        values,
        preserve_existing=preserve_existing,
    )


def _prompt_setup_field(field: SetupField, default: object) -> object:
    if field.kind == "text":
        if not isinstance(default, str) or not default:
            raise ValueError(f"setup field default must be non-empty: {field.key}")
        return _prompt_text(field.label, default)
    if field.kind == "required_text":
        resolved = (
            default
            if isinstance(default, str)
            and default.strip()
            and not any(fragment in default for fragment in field.forbidden_fragments)
            else None
        )
        return _prompt_required_text(field.label, resolved)
    if field.kind == "positive_int":
        if isinstance(default, bool) or not isinstance(default, int):
            raise ValueError(f"setup field default must be an integer: {field.key}")
        return _prompt_positive_int(field.label, default)
    default_index = next(
        (
            index
            for index, choice in enumerate(field.choices, start=1)
            if choice.value == default
        ),
        1,
    )
    return field.choices[
        _prompt_choice(
            field.label,
            tuple(choice.label for choice in field.choices),
            default=default_index,
        )
        - 1
    ].value


def _configure_missing_credentials(path: Path) -> None:
    from integrations.local_service.initialization import (
        configure_credentials,
        missing_credential_fields,
    )

    try:
        missing = missing_credential_fields(path)
        if not missing or not _confirm("Configure missing provider credentials now?", default=True):
            return
        updates: dict[str, dict[str, str]] = {}
        for item in missing:
            secret = _prompt_secret(f"{item.reference}.{item.field}")
            if secret:
                updates.setdefault(item.reference, {})[item.field] = secret
    except WizardCancelled:
        sys.stdout.write("Credential setup stopped; the configuration file remains saved.\n")
        return
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
    try:
        if not _confirm("Run interactive setup now?", default=True):
            return False
    except WizardCancelled:
        sys.stdout.write("m2bOS setup offer cancelled.\n")
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
    from integrations.local_service.adapter_catalog import load_adapter_catalog
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
    catalog = load_adapter_catalog()
    run_startup_preflight(config, environ=values, catalog=catalog)
    runtime = build_runtime(
        config,
        providers=catalog.providers,
        vector_stores=catalog.vector_stores,
    )
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
    except (EOFError, OSError) as exc:
        raise WizardCancelled("interactive confirmation input ended") from exc
    if not value:
        return default
    return value in {"y", "yes"}


def _confirm_after_write(prompt: str, *, default: bool) -> bool:
    try:
        return _confirm(prompt, default=default)
    except WizardCancelled:
        sys.stdout.write("Interactive setup stopped; the configuration file remains saved.\n")
        return False


def _prompt_choice(
    prompt: str,
    choices: Sequence[str],
    *,
    default: int,
) -> int:
    if not choices or not 1 <= default <= len(choices):
        raise ValueError("prompt choices and default must form a valid menu")
    sys.stdout.write(f"\n{prompt}:\n")
    for index, choice in enumerate(choices, start=1):
        marker = " (default)" if index == default else ""
        sys.stdout.write(f"  {index}. {choice}{marker}\n")
    while True:
        try:
            raw = input(f"Select [1-{len(choices)}]: ").strip()
        except (EOFError, OSError) as exc:
            raise WizardCancelled("interactive menu input ended") from exc
        if not raw:
            return default
        if raw.isdigit() and 1 <= int(raw) <= len(choices):
            return int(raw)
        sys.stdout.write("Invalid selection.\n")


def _prompt_text(prompt: str, default: str) -> str:
    if not isinstance(default, str) or not default.strip():
        raise ValueError("prompt text default must be non-empty")
    try:
        value = input(f"{prompt} [{default}]: ").strip()
    except (EOFError, OSError) as exc:
        raise WizardCancelled("interactive text input ended") from exc
    return value or default


def _prompt_required_text(prompt: str, default: str | None = None) -> str:
    if default is not None and (not isinstance(default, str) or not default.strip()):
        raise ValueError("prompt text default must be non-empty")
    suffix = f" [{default}]" if default is not None else ""
    while True:
        try:
            value = input(f"{prompt}{suffix}: ").strip()
        except (EOFError, OSError) as exc:
            raise WizardCancelled("interactive required input ended") from exc
        if value:
            return value
        if default is not None:
            return default
        sys.stdout.write("Value is required.\n")


def _prompt_positive_int(prompt: str, default: int) -> int:
    if isinstance(default, bool) or not isinstance(default, int) or default <= 0:
        raise ValueError("prompt integer default must be positive")
    while True:
        raw = _prompt_text(prompt, str(default))
        try:
            value = int(raw)
        except ValueError:
            sys.stdout.write("Value must be a positive integer.\n")
            continue
        if value > 0:
            return value
        sys.stdout.write("Value must be a positive integer.\n")


def _prompt_secret(label: str) -> str:
    try:
        return getpass.getpass(f"{label}: ")
    except (EOFError, OSError) as exc:
        raise WizardCancelled("interactive secret input ended") from exc


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
