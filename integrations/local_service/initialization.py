"""单用户 m2bOS 配置的安全初始化原语。"""

from __future__ import annotations

import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import yaml

from Config import M2BOSConfig
from Config.loader import load_config_object, strict_object
from infrastructure.store.filesystem.durable_io.atomic_file import atomic_replace_bytes

DEFAULT_CONFIG_PATH = Path("~/.m2bos/config.yaml")


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
        configured = values.get("M2BOS_CONFIG_FILE")
        selected = configured if isinstance(configured, str) and configured.strip() else DEFAULT_CONFIG_PATH
    path = Path(selected).expanduser().absolute()
    if path.suffix.casefold() != ".yaml":
        raise ValueError("m2bOS config path must use the .yaml suffix")
    if path.parent == Path(path.anchor):
        raise ValueError("m2bOS config must be stored in a dedicated child directory")
    return path


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
        M2BOSConfig.from_file(path)
        return ConfigInitializationResult(path=path, created=False, backup_path=None)

    encoded = _validated_template(source)
    backup_path = None
    if existing is not None:
        backup_path = path.with_name(f"{path.stem}.bak{path.suffix}")
        atomic_replace_bytes(backup_path, existing, artifact_root=path.parent)
    atomic_replace_bytes(path, encoded, artifact_root=path.parent)
    M2BOSConfig.from_file(path)
    return ConfigInitializationResult(path=path, created=True, backup_path=backup_path)


def missing_credential_fields(path: Path) -> tuple[CredentialField, ...]:
    """返回当前运行链实际引用但尚未填写的秘密字段。"""

    config = M2BOSConfig.from_file(path)
    required: set[CredentialField] = set()
    model_routes = [config.models.chat.route, config.models.embedding.route]
    if config.models.rerank is not None:
        model_routes.append(config.models.rerank.route)
    for route in model_routes:
        if route.credential_ref:
            required.add(CredentialField(route.credential_ref, "api_key"))
    for vector in (config.memory.vector_store, config.conversation.summary_vector_store):
        reference = vector.route.credential_ref
        if reference:
            required.update(
                CredentialField(reference, field)
                for field in config.credentials.resolve(reference)
            )
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
) -> M2BOSConfig:
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
    config = M2BOSConfig.from_mapping(payload)
    encoded = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False).encode("utf-8")
    atomic_replace_bytes(path, encoded, artifact_root=path.parent)
    M2BOSConfig.from_file(path)
    return config


def _validated_template(source: Path | None) -> bytes:
    if source is not None:
        template = Path(source).expanduser().absolute()
        encoded = _read_required_regular_file(template)
        M2BOSConfig.from_file(template)
        return encoded
    resource = resources.files("Config").joinpath("example.yaml")
    with resources.as_file(resource) as template:
        M2BOSConfig.from_file(template)
        return template.read_bytes()


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
            raise ValueError("m2bOS config must be a regular file")
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
    "configure_credentials",
    "initialize_config",
    "missing_credential_fields",
    "resolve_config_path",
]
