"""单用户 Habitus 配置的安全初始化原语。"""

from __future__ import annotations

import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml

from Config import HabitusConfig
from Config.loader import load_config_object, strict_object
from infrastructure.store.filesystem.durable_io.atomic_file import atomic_replace_bytes

DEFAULT_CONFIG_PATH = Path("~/.habitus/config.yaml")
DEFAULT_PLUGIN_CONNECTION_PATH = Path("~/.habitus/agent-plugin/connection.json")
_MAX_INITIALIZED_CONFIG_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ConfigInitializationResult:
    path: Path
    created: bool
    backup_path: Path | None


@dataclass(frozen=True, order=True)
class CredentialField:
    reference: str
    field: str


def resolve_config_path(
    explicit: str | Path | None,
    *,
    environ: dict[str, str] | None = None,
) -> Path:
    """按显式参数、统一环境变量、单用户默认路径解析配置位置。"""

    values = os.environ if environ is None else environ
    selected: str | Path
    if explicit is not None:
        selected = explicit
    else:
        configured = values.get("HABITUS_CONFIG_FILE")
        selected = configured if isinstance(configured, str) and configured.strip() else DEFAULT_CONFIG_PATH
    path = Path(selected).expanduser().absolute()
    if path.suffix.casefold() != ".yaml":
        raise ValueError("Habitus config path must use the .yaml suffix")
    if path.parent == Path(path.anchor):
        raise ValueError("Habitus config must be stored in a dedicated child directory")
    return path


def resolve_plugin_connection_path(
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """解析 Hook 的持久连接投影；显式 state root 优先于单用户默认目录。"""

    values = os.environ if environ is None else environ
    configured = values.get("HABITUS_PLUGIN_STATE_DIR")
    if isinstance(configured, str) and configured.strip():
        state_root = Path(configured).expanduser().absolute()
        if state_root == Path(state_root.anchor):
            raise ValueError("plugin state root must be a dedicated child directory")
        return state_root / "connection.json"
    return DEFAULT_PLUGIN_CONNECTION_PATH.expanduser().absolute()


def initialize_config(
    destination: Path,
    *,
    source: Path | None = None,
    force: bool = False,
) -> ConfigInitializationResult:
    """验证模板后以 0600 原子写入；覆盖时保留最后一个安全备份。"""

    path = resolve_config_path(destination)
    if not isinstance(force, bool):
        raise TypeError("force must be boolean")
    existing = _read_existing(path)
    if existing is not None and not force:
        if stat.S_IMODE(path.stat().st_mode) & 0o077:
            atomic_replace_bytes(path, existing, artifact_root=path.parent)
        HabitusConfig.from_file(path)
        return ConfigInitializationResult(path=path, created=False, backup_path=None)

    encoded = _validated_template(source)
    backup_path = None
    if existing is not None:
        backup_path = path.with_name(f"{path.stem}.bak{path.suffix}")
        atomic_replace_bytes(backup_path, existing, artifact_root=path.parent)
    atomic_replace_bytes(path, encoded, artifact_root=path.parent)
    HabitusConfig.from_file(path)
    return ConfigInitializationResult(path=path, created=True, backup_path=backup_path)


def initialize_config_from_mapping(
    destination: Path,
    payload: Mapping[str, object],
    *,
    force: bool = False,
) -> ConfigInitializationResult:
    """把已经完成云端规划的完整严格配置安全写入目标位置。"""

    if not isinstance(payload, Mapping):
        raise TypeError("config payload must be an object")
    HabitusConfig.from_mapping(payload)
    encoded = yaml.safe_dump(dict(payload), allow_unicode=True, sort_keys=False).encode("utf-8")
    if len(encoded) > _MAX_INITIALIZED_CONFIG_BYTES:
        raise ValueError("Habitus config exceeds the one-megabyte limit")
    return _initialize_encoded(resolve_config_path(destination), encoded, force=force)


def load_initialization_mapping(source: Path | None = None) -> dict[str, object]:
    """读取现有配置或包内模板，并返回供初始化规划使用的普通对象。"""

    if source is not None:
        path = resolve_config_path(source)
        encoded = _read_required_regular_file(path)
        payload = load_config_object(path)
    else:
        resource = resources.files("Config").joinpath("example.yaml")
        with resources.as_file(resource) as template:
            encoded = template.read_bytes()
            payload = load_config_object(template)
    if not encoded:
        raise ValueError("Habitus config cannot be empty")
    HabitusConfig.from_mapping(payload)
    return payload


def missing_credential_fields(path: Path) -> tuple[CredentialField, ...]:
    """返回当前运行链实际引用但尚未填写的秘密字段。"""

    config = HabitusConfig.from_file(path)
    from integrations.local_service.adapter_catalog import load_adapter_catalog

    required = {
        CredentialField(reference, field)
        for reference, fields in load_adapter_catalog().setup.required_credentials(config).items()
        for field in fields
    }
    tracing_reference = config.observability.tracing.credential_ref
    if config.observability.tracing.enabled and tracing_reference:
        required.update(
            CredentialField(tracing_reference, field)
            for field in config.credentials.resolve(tracing_reference)
        )
    return tuple(
        item
        for item in sorted(required)
        if not config.credentials.resolve(item.reference).get(item.field, "")
    )


def configure_credentials(
    path: Path,
    values: Mapping[str, Mapping[str, str]],
) -> HabitusConfig:
    """只更新已经声明的秘密字段，并以 0600 原子替换统一 YAML。"""

    if not isinstance(values, Mapping):
        raise TypeError("credential updates must be an object")
    payload = load_config_object(path)
    registry = strict_object(payload.get("credentials"), path="config.credentials")
    for reference, fields in values.items():
        if not isinstance(reference, str):
            raise ValueError(f"unknown credential reference: {reference}")
        raw_reference = _normalized_key(registry, reference)
        if raw_reference is None:
            raise ValueError(f"unknown credential reference: {reference}")
        if not isinstance(fields, Mapping):
            raise TypeError(f"credential update '{reference}' must be an object")
        current = strict_object(
            registry[raw_reference],
            path=f"config.credentials.{reference}",
        )
        for field_name, secret in fields.items():
            if not isinstance(field_name, str):
                raise ValueError(f"unknown credential field: {reference}.{field_name}")
            raw_field_name = _normalized_key(current, field_name)
            if raw_field_name is None:
                raise ValueError(f"unknown credential field: {reference}.{field_name}")
            if not isinstance(secret, str) or not secret:
                raise ValueError(f"credential value is missing: {reference}.{field_name}")
            current[raw_field_name] = secret
        registry[raw_reference] = current
    payload["credentials"] = registry
    config = HabitusConfig.from_mapping(payload)
    encoded = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False).encode("utf-8")
    atomic_replace_bytes(path, encoded, artifact_root=path.parent)
    HabitusConfig.from_file(path)
    return config


def write_plugin_connection(
    config: HabitusConfig,
    destination: Path | None = None,
) -> Path:
    """把 YAML 中的本地监听地址投影成插件可读取的无秘密派生配置。"""

    if not isinstance(config, HabitusConfig):
        raise TypeError("config must be HabitusConfig")
    path = (
        Path(destination).expanduser().absolute()
        if destination is not None
        else DEFAULT_PLUGIN_CONNECTION_PATH.expanduser().absolute()
    )
    if path.parent == Path(path.anchor):
        raise ValueError("plugin connection must be stored in a dedicated child directory")
    host = config.http.host
    authority = f"[{host}]" if ":" in host else host
    payload = {
        "schema_version": 1,
        "base_url": f"http://{authority}:{config.http.port}",
    }
    encoded = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    atomic_replace_bytes(path, encoded, artifact_root=path.parent)
    return path


def _validated_template(source: Path | None) -> bytes:
    if source is not None:
        template = Path(source).expanduser().absolute()
        encoded = _read_required_regular_file(template)
        HabitusConfig.from_file(template)
        return encoded
    resource = resources.files("Config").joinpath("example.yaml")
    with resources.as_file(resource) as template:
        HabitusConfig.from_file(template)
        return template.read_bytes()


def _initialize_encoded(
    path: Path,
    encoded: bytes,
    *,
    force: bool,
) -> ConfigInitializationResult:
    if not isinstance(force, bool):
        raise TypeError("force must be boolean")
    existing = _read_existing(path)
    if existing is not None and not force:
        if stat.S_IMODE(path.stat().st_mode) & 0o077:
            atomic_replace_bytes(path, existing, artifact_root=path.parent)
        HabitusConfig.from_file(path)
        return ConfigInitializationResult(path=path, created=False, backup_path=None)
    backup_path = None
    if existing is not None:
        backup_path = path.with_name(f"{path.stem}.bak{path.suffix}")
        atomic_replace_bytes(backup_path, existing, artifact_root=path.parent)
    atomic_replace_bytes(path, encoded, artifact_root=path.parent)
    HabitusConfig.from_file(path)
    return ConfigInitializationResult(path=path, created=True, backup_path=backup_path)


def _read_existing(path: Path) -> bytes | None:
    try:
        return _read_required_regular_file(path)
    except FileNotFoundError:
        return None


def _normalized_key(mapping: Mapping[str, object], requested: str) -> str | None:
    normalized = requested.strip().lower()
    return next(
        (
            key
            for key in mapping
            if isinstance(key, str) and key.strip().lower() == normalized
        ),
        None,
    )


def _read_required_regular_file(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("Habitus config must be a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(descriptor)


__all__ = [
    "ConfigInitializationResult",
    "CredentialField",
    "DEFAULT_CONFIG_PATH",
    "DEFAULT_PLUGIN_CONNECTION_PATH",
    "configure_credentials",
    "initialize_config",
    "initialize_config_from_mapping",
    "load_initialization_mapping",
    "missing_credential_fields",
    "resolve_config_path",
    "resolve_plugin_connection_path",
    "write_plugin_connection",
]
