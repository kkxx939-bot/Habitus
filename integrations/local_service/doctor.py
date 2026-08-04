"""无需运行 HTTP 服务的配置与本机依赖诊断。"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import shutil
import socket
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from Config import ConfigError, M2BOSConfig

if TYPE_CHECKING:
    from integrations.local_service.adapter_catalog import AdapterCatalog

_MINIMUM_FREE_BYTES = 1024**3


class DoctorStatus(str, Enum):
    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: DoctorStatus
    detail: str


@dataclass(frozen=True)
class DoctorReport:
    checks: tuple[DoctorCheck, ...]

    @property
    def ok(self) -> bool:
        return all(check.status is not DoctorStatus.FAIL for check in self.checks)

    def to_dict(self) -> dict[str, object]:
        return {
            "ok": self.ok,
            "checks": [
                {"name": check.name, "status": check.status.value, "detail": check.detail}
                for check in self.checks
            ],
        }


class StartupPreflightError(RuntimeError):
    """启动前的本机硬性条件不满足。"""


def run_doctor(
    config: M2BOSConfig,
    *,
    check_port: bool = True,
    deep: bool = False,
    probe_timeout_seconds: float = 15.0,
    catalog: AdapterCatalog | None = None,
) -> DoctorReport:
    if not isinstance(config, M2BOSConfig):
        raise TypeError("config must be M2BOSConfig")
    if not isinstance(deep, bool):
        raise TypeError("deep must be boolean")
    if (
        isinstance(probe_timeout_seconds, bool)
        or not isinstance(probe_timeout_seconds, int | float)
        or not 0 < float(probe_timeout_seconds) <= 120
    ):
        raise ValueError("probe_timeout_seconds must be greater than zero and at most 120")
    if catalog is None:
        from integrations.local_service.adapter_catalog import load_adapter_catalog

        catalog = load_adapter_catalog()
    checks = [
        DoctorCheck("config", DoctorStatus.PASS, "strict configuration loaded"),
        DoctorCheck("listener", DoctorStatus.PASS, f"loopback {config.http.host}:{config.http.port}"),
        _storage_check(config.storage_root),
        _adapter_configuration_check(config, catalog),
        *_dependency_checks(config, catalog),
        _node_check(),
        _credential_check(config, catalog),
        _ingress_capacity_check(config),
    ]
    if check_port:
        checks.append(_port_check(config.http.host, config.http.port))
    if deep:
        checks.extend(
            _deep_checks(
                config,
                timeout_seconds=float(probe_timeout_seconds),
                catalog=catalog,
            )
        )
    return DoctorReport(tuple(checks))


def run_doctor_from_env(
    *,
    environ: Mapping[str, str] | None = None,
    check_port: bool = True,
    deep: bool = False,
    probe_timeout_seconds: float = 15.0,
    catalog: AdapterCatalog | None = None,
) -> DoctorReport:
    """即使配置无效也返回结构化诊断，不把解析异常泄露到 CLI。"""

    values = os.environ if environ is None else environ
    if not isinstance(values, Mapping):
        raise TypeError("environ must be a string mapping")
    try:
        config = M2BOSConfig.from_env(environ=values)
    except (ConfigError, OSError, TypeError, ValueError) as exc:
        detail = str(exc).strip() or "configuration could not be loaded"
        return DoctorReport((DoctorCheck("config", DoctorStatus.FAIL, detail),))
    report = run_doctor(
        config,
        check_port=check_port,
        deep=deep,
        probe_timeout_seconds=probe_timeout_seconds,
        catalog=catalog,
    )
    return DoctorReport((_config_file_security(values), *report.checks))


def run_startup_preflight(
    config: M2BOSConfig,
    *,
    environ: Mapping[str, str] | None = None,
    catalog: AdapterCatalog | None = None,
) -> None:
    report = run_doctor(config, check_port=True, catalog=catalog)
    if environ is not None:
        report = DoctorReport((_config_file_security(environ), *report.checks))
    failures = [check for check in report.checks if check.status is DoctorStatus.FAIL]
    if failures:
        detail = "; ".join(f"{check.name}: {check.detail}" for check in failures)
        raise StartupPreflightError(detail)


def _storage_check(root: Path) -> DoctorCheck:
    current = Path(root)
    while not current.exists():
        if current.is_symlink():
            return DoctorCheck("storage", DoctorStatus.FAIL, "storage path traverses a symbolic link")
        parent = current.parent
        if parent == current:
            return DoctorCheck("storage", DoctorStatus.FAIL, "storage path has no existing ancestor")
        current = parent
    if current.is_symlink() or not current.is_dir():
        return DoctorCheck("storage", DoctorStatus.FAIL, "storage ancestor is not a real directory")
    if not os.access(current, os.W_OK | os.X_OK):
        return DoctorCheck("storage", DoctorStatus.FAIL, "storage ancestor is not writable")
    free = shutil.disk_usage(current).free
    if free < _MINIMUM_FREE_BYTES:
        return DoctorCheck("storage", DoctorStatus.FAIL, f"free_bytes={free}; at least 1 GiB is required")
    return DoctorCheck("storage", DoctorStatus.PASS, f"writable; free_bytes={free}")


def _python_dependency_check(module: str) -> DoctorCheck:
    if importlib.util.find_spec(module) is None:
        return DoctorCheck(module, DoctorStatus.FAIL, "Python dependency is not installed")
    return DoctorCheck(module, DoctorStatus.PASS, "available")


def _dependency_checks(
    config: M2BOSConfig,
    catalog: AdapterCatalog,
) -> tuple[DoctorCheck, ...]:
    try:
        modules = catalog.setup.dependency_modules(config)
    except (TypeError, ValueError) as exc:
        detail = str(exc).strip() or type(exc).__name__
        return (DoctorCheck("dependencies", DoctorStatus.FAIL, detail),)
    return tuple(_python_dependency_check(module) for module in modules)


def _adapter_configuration_check(
    config: M2BOSConfig,
    catalog: AdapterCatalog,
) -> DoctorCheck:
    try:
        catalog.validate(config)
    except (OSError, TypeError, ValueError) as exc:
        detail = str(exc).strip() or type(exc).__name__
        return DoctorCheck("adapter_configuration", DoctorStatus.FAIL, detail)
    return DoctorCheck(
        "adapter_configuration",
        DoctorStatus.PASS,
        "registered model and vector adapters accept the configured limits",
    )


def _node_check() -> DoctorCheck:
    executable = shutil.which("node")
    if executable is None:
        return DoctorCheck("node", DoctorStatus.WARN, "not installed; Agent plugins cannot run")
    try:
        completed = subprocess.run(
            [executable, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return DoctorCheck("node", DoctorStatus.WARN, f"version check failed: {type(exc).__name__}")
    version = completed.stdout.strip().removeprefix("v")
    try:
        major = int(version.split(".", maxsplit=1)[0])
    except ValueError:
        return DoctorCheck("node", DoctorStatus.WARN, f"unrecognized version: {version or 'empty'}")
    if completed.returncode != 0 or major < 18:
        return DoctorCheck("node", DoctorStatus.WARN, f"Node.js >=18 is required for Agent plugins; found {version or 'unknown'}")
    return DoctorCheck("node", DoctorStatus.PASS, f"{executable}; version={version}")


def _ingress_capacity_check(config: M2BOSConfig) -> DoctorCheck:
    request_bytes = config.http.max_request_bytes
    journal_bytes = config.conversation.journal.max_file_bytes
    if request_bytes > journal_bytes:
        return DoctorCheck(
            "ingress_capacity",
            DoctorStatus.FAIL,
            f"http_max_request_bytes={request_bytes} exceeds conversation_journal_max_file_bytes={journal_bytes}",
        )
    return DoctorCheck(
        "ingress_capacity",
        DoctorStatus.PASS,
        f"http_max_request_bytes={request_bytes}; conversation_journal_max_file_bytes={journal_bytes}",
    )


def _deep_checks(
    config: M2BOSConfig,
    *,
    timeout_seconds: float,
    catalog: AdapterCatalog | None = None,
) -> list[DoctorCheck]:
    """只在显式 deep 模式构造真实 Adapter，并执行有界远端探测。"""

    try:
        from Runtime import build_runtime

        if catalog is None:
            from integrations.local_service.adapter_catalog import load_adapter_catalog

            catalog = load_adapter_catalog()
        runtime = build_runtime(
            config,
            providers=catalog.providers,
            vector_stores=catalog.vector_stores,
        )
    except Exception as exc:
        return [DoctorCheck("component_construction", DoctorStatus.FAIL, type(exc).__name__)]

    checks = [
        DoctorCheck(
            "component_construction",
            DoctorStatus.PASS,
            "chat, embedding, optional rerank, and both vector stores constructed",
        )
    ]

    async def probe() -> list[DoctorCheck]:
        results: list[DoctorCheck] = []
        try:
            try:
                chat_result = await asyncio.wait_for(
                    runtime.components.models.chat.health_check_async(),
                    timeout=timeout_seconds,
                )
                chat_ok = chat_result.get("ok") is True
                results.append(
                    DoctorCheck(
                        "chat_probe",
                        DoctorStatus.PASS if chat_ok else DoctorStatus.FAIL,
                        "provider health endpoint responded"
                        if chat_ok
                        else str(chat_result.get("error_code", "health check failed")),
                    )
                )
            except Exception as exc:
                results.append(DoctorCheck("chat_probe", DoctorStatus.FAIL, type(exc).__name__))
            try:
                vector = await asyncio.wait_for(
                    runtime.components.models.embedder.embed_query("m2bOS doctor probe"),
                    timeout=timeout_seconds,
                )
                results.append(
                    DoctorCheck(
                        "embedding_probe",
                        DoctorStatus.PASS,
                        f"dimension={len(vector.values)}; expected={config.models.embedding.dimension}",
                    )
                )
            except Exception as exc:
                results.append(DoctorCheck("embedding_probe", DoctorStatus.FAIL, type(exc).__name__))
            for name, store in (
                ("memory_vector_probe", runtime.components.memory.vector_index.store),
                ("summary_vector_probe", runtime.components.conversation.summary_vector_index.store),
            ):
                try:
                    state = await asyncio.wait_for(store.state(), timeout=timeout_seconds)
                    if state is None:
                        results.append(
                            DoctorCheck(
                                name,
                                DoctorStatus.WARN,
                                "control plane is reachable; collection has no m2bOS publication yet",
                            )
                        )
                    elif not state.ready:
                        results.append(
                            DoctorCheck(
                                name,
                                DoctorStatus.WARN,
                                "publication exists but is not ready",
                            )
                        )
                    else:
                        results.append(
                            DoctorCheck(
                                name,
                                DoctorStatus.PASS,
                                f"publication is readable; ready={state.ready}",
                            )
                        )
                except Exception as exc:
                    results.append(DoctorCheck(name, DoctorStatus.FAIL, type(exc).__name__))
            if config.models.rerank is not None:
                results.append(
                    DoctorCheck(
                        "rerank_probe",
                        DoctorStatus.WARN,
                        "configured adapter has no side-effect-free health endpoint; construction passed",
                    )
                )
        finally:
            try:
                await asyncio.wait_for(runtime.close(), timeout=timeout_seconds)
            except Exception as exc:
                results.append(DoctorCheck("deep_probe_cleanup", DoctorStatus.WARN, type(exc).__name__))
        return results

    try:
        checks.extend(asyncio.run(probe()))
    except RuntimeError as exc:
        checks.append(DoctorCheck("deep_probe", DoctorStatus.FAIL, type(exc).__name__))
    return checks


def _credential_check(config: M2BOSConfig, catalog: AdapterCatalog) -> DoctorCheck:
    try:
        required = {
            (reference, field)
            for reference, fields in catalog.setup.required_credentials(config).items()
            for field in fields
        }
    except (TypeError, ValueError) as exc:
        detail = str(exc).strip() or type(exc).__name__
        return DoctorCheck("credentials", DoctorStatus.FAIL, detail)
    tracing_reference = config.observability.tracing.credential_ref
    if config.observability.tracing.enabled and tracing_reference:
        required.update(
            (tracing_reference, field)
            for field in config.credentials.resolve(tracing_reference)
        )
    missing = sorted(
        f"credentials.{reference}.{field}"
        for reference, field in required
        if not config.credentials.resolve(reference).get(field, "").strip()
    )
    if missing:
        return DoctorCheck("credentials", DoctorStatus.FAIL, "empty YAML credential fields: " + ", ".join(missing))
    references = {reference for reference, _field in required}
    return DoctorCheck("credentials", DoctorStatus.PASS, f"resolved {len(references)} named YAML credentials")


def _config_file_security(environ: Mapping[str, str]) -> DoctorCheck:
    raw_path = environ.get("M2BOS_CONFIG_FILE")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return DoctorCheck("config_permissions", DoctorStatus.FAIL, "M2BOS_CONFIG_FILE is missing")
    path = Path(raw_path).expanduser().absolute()
    try:
        mode = path.stat().st_mode & 0o777
    except OSError as exc:
        return DoctorCheck("config_permissions", DoctorStatus.FAIL, type(exc).__name__)
    if mode & 0o077:
        return DoctorCheck(
            "config_permissions",
            DoctorStatus.FAIL,
            f"{path} mode={mode:03o}; expected no group or other permissions",
        )
    return DoctorCheck("config_permissions", DoctorStatus.PASS, f"{path} mode={mode:03o}")


def _port_check(host: str, port: int) -> DoctorCheck:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as listener:
        try:
            listener.bind((host, port))
        except OSError as exc:
            return DoctorCheck("port", DoctorStatus.FAIL, f"listener unavailable: {type(exc).__name__}")
    return DoctorCheck("port", DoctorStatus.PASS, "available")


__all__ = [
    "DoctorCheck",
    "DoctorReport",
    "DoctorStatus",
    "StartupPreflightError",
    "run_doctor",
    "run_doctor_from_env",
    "run_startup_preflight",
]
