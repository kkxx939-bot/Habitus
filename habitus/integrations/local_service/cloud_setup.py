"""由声明式 Profile Registry 驱动的云端配置规划。"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType

from habitus.config import HabitusConfig
from habitus.integrations.local_service.setup_registry import (
    SetupCapability,
    SetupProfile,
    SetupRegistry,
    build_builtin_setup_registry,
)


@dataclass(frozen=True)
class ProfileSelection:
    """一次 Profile 选择及其用户可编辑值；不包含任何秘密。"""

    profile_id: str
    values: Mapping[str, object] = field(default_factory=dict)
    preserve_existing: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise ValueError("setup profile selection id must be non-empty")
        if not isinstance(self.values, Mapping):
            raise TypeError("setup profile selection values must be an object")
        if not isinstance(self.preserve_existing, bool):
            raise TypeError("preserve_existing must be boolean")
        object.__setattr__(self, "values", MappingProxyType(deepcopy(dict(self.values))))


@dataclass(frozen=True)
class CloudSetupSelection:
    """云端能力选择；厂商细节只通过已注册 Profile 进入。"""

    chat: ProfileSelection
    embedding: ProfileSelection
    rerank: ProfileSelection
    vector: ProfileSelection

    def __post_init__(self) -> None:
        for name in ("chat", "embedding", "rerank", "vector"):
            if not isinstance(getattr(self, name), ProfileSelection):
                raise TypeError(f"{name} must be ProfileSelection")


def default_cloud_selection(
    registry: SetupRegistry | None = None,
) -> CloudSetupSelection:
    """返回不含占位 Rerank 端点的发行包默认 Profile 组合。"""

    resolved = registry or build_builtin_setup_registry()
    return CloudSetupSelection(
        chat=_default_selection(resolved.profile("chat", "chat.deepseek")),
        embedding=_default_selection(
            resolved.profile("embedding", "embedding.volcengine.ark")
        ),
        rerank=_default_selection(resolved.profile("rerank", "rerank.disabled")),
        vector=_default_selection(
            resolved.profile("vector", "vector.vikingdb.managed")
        ),
    )


def selection_from_mapping(
    payload: Mapping[str, object],
    registry: SetupRegistry | None = None,
) -> CloudSetupSelection:
    """从完整 YAML 对象提取无损向导默认值，不读取或复制秘密。"""

    if not isinstance(payload, Mapping):
        raise TypeError("setup payload must be an object")
    config = HabitusConfig.from_mapping(payload)
    resolved = registry or build_builtin_setup_registry()
    models = _mapping(payload, "models")
    rerank_value = models.get("rerank")
    rerank_document: Mapping[str, object]
    if rerank_value is None:
        rerank_document = {"enabled": False}
    elif isinstance(rerank_value, Mapping):
        rerank_document = rerank_value
    else:
        raise TypeError("config.models.rerank must be an object or null")
    vector_document = {
        "memory": _normalized_route_document(
            _mapping(_mapping(payload, "memory"), "vector_store"),
            config.memory.vector_store.route,
        ),
        "summary": deepcopy(
            _normalized_route_document(
                _mapping(_mapping(payload, "conversation"), "summary_vector_store"),
                config.conversation.summary_vector_store.route,
            )
        ),
    }
    chat_document = _normalized_route_document(
        _mapping(models, "chat"),
        config.models.chat.route,
    )
    embedding_document = _normalized_route_document(
        _mapping(models, "embedding"),
        config.models.embedding.route,
    )
    if config.models.rerank is not None:
        rerank_document = _normalized_route_document(
            rerank_document,
            config.models.rerank.route,
        )
    return CloudSetupSelection(
        chat=_current_selection(resolved, "chat", chat_document),
        embedding=_current_selection(
            resolved,
            "embedding",
            embedding_document,
        ),
        rerank=_current_selection(resolved, "rerank", rerank_document),
        vector=_current_selection(resolved, "vector", vector_document),
    )


def selection_from_config(
    config: HabitusConfig,
    registry: SetupRegistry | None = None,
) -> CloudSetupSelection:
    """兼容类型化调用者；向导应优先使用无损 YAML 映射入口。"""

    if not isinstance(config, HabitusConfig):
        raise TypeError("config must be HabitusConfig")
    return selection_from_mapping(_config_mapping(config), registry)


def apply_cloud_selection(
    payload: Mapping[str, object],
    selection: CloudSetupSelection,
    registry: SetupRegistry | None = None,
) -> dict[str, object]:
    """应用 Profile 选择、增量补齐凭据，并执行根配置与 Adapter 校验。"""

    if not isinstance(payload, Mapping):
        raise TypeError("cloud setup payload must be an object")
    if not isinstance(selection, CloudSetupSelection):
        raise TypeError("selection must be CloudSetupSelection")
    current_config = HabitusConfig.from_mapping(payload)
    resolved = registry or build_builtin_setup_registry()
    result = deepcopy(dict(payload))
    models = _section(result, "models")

    chat = _apply_profile(
        resolved,
        "chat",
        selection.chat,
        _normalized_route_document(_section(models, "chat"), current_config.models.chat.route),
    )
    models["chat"] = chat
    embedding = _apply_profile(
        resolved,
        "embedding",
        selection.embedding,
        _normalized_route_document(
            _section(models, "embedding"),
            current_config.models.embedding.route,
        ),
    )
    models["embedding"] = embedding
    rerank_profile = resolved.profile("rerank", selection.rerank.profile_id)
    if rerank_profile.target_state == "disabled":
        models["rerank"] = None
        rerank = None
    else:
        existing_rerank = models.get("rerank")
        base = dict(existing_rerank) if isinstance(existing_rerank, Mapping) else {}
        if current_config.models.rerank is not None:
            base = _normalized_route_document(base, current_config.models.rerank.route)
        rerank = _apply_profile(
            resolved,
            "rerank",
            selection.rerank,
            base,
        )
        models["rerank"] = rerank

    vector_base = {
        "memory": _normalized_route_document(
            _section(_section(result, "memory"), "vector_store"),
            current_config.memory.vector_store.route,
        ),
        "summary": deepcopy(
            _normalized_route_document(
                _section(_section(result, "conversation"), "summary_vector_store"),
                current_config.conversation.summary_vector_store.route,
            )
        ),
    }
    vector = _apply_profile(
        resolved,
        "vector",
        selection.vector,
        vector_base,
    )
    memory_vector = _mapping(vector, "memory")
    summary_vector = _mapping(vector, "summary")
    _section(result, "memory")["vector_store"] = deepcopy(dict(memory_vector))
    _section(result, "conversation")["summary_vector_store"] = deepcopy(
        dict(summary_vector)
    )

    documents: list[tuple[SetupCapability, Mapping[str, object]]] = [
        ("chat", chat),
        ("embedding", embedding),
        ("vector", memory_vector),
        ("vector", summary_vector),
    ]
    if rerank is not None:
        documents.append(("rerank", rerank))
    required = resolved.required_credentials_for_documents(tuple(documents))
    result["credentials"] = _additive_credentials(result, required)
    config = HabitusConfig.from_mapping(result)
    resolved.validate(config)
    return result


def profile_selection(
    profile: SetupProfile,
    values: Mapping[str, object],
    *,
    preserve_existing: bool,
) -> ProfileSelection:
    """供通用 CLI 构造经过 Profile 字段校验的选择。"""

    profile.materialize(values)
    return ProfileSelection(
        profile_id=profile.profile_id,
        values=values,
        preserve_existing=preserve_existing,
    )


def _default_selection(profile: SetupProfile) -> ProfileSelection:
    return ProfileSelection(
        profile_id=profile.profile_id,
        values=profile.values_from(profile.patch),
    )


def _current_selection(
    registry: SetupRegistry,
    capability: SetupCapability,
    document: Mapping[str, object],
) -> ProfileSelection:
    profile = registry.identify(capability, document)
    return ProfileSelection(
        profile_id=profile.profile_id,
        values=profile.values_from(document),
        preserve_existing=True,
    )


def _apply_profile(
    registry: SetupRegistry,
    capability: SetupCapability,
    selection: ProfileSelection,
    existing: Mapping[str, object],
) -> dict[str, object]:
    profile = registry.profile(capability, selection.profile_id)
    if profile.capability != capability:
        raise ValueError("setup profile capability does not match its target")
    field_keys: frozenset[str] | None = None
    if selection.preserve_existing:
        current_profile = registry.identify(capability, existing)
        if current_profile.profile_id != profile.profile_id:
            raise ValueError(
                "preserve_existing requires the selected profile to match the current configuration"
            )
        current_values = profile.values_from(existing)
        field_keys = frozenset(
            key
            for key, value in selection.values.items()
            if current_values.get(key) != value
        )
    return profile.materialize(
        selection.values,
        base=existing,
        apply_defaults=not selection.preserve_existing,
        field_keys=field_keys,
    )


def _additive_credentials(
    result: Mapping[str, object],
    required: Mapping[str, set[str]],
) -> dict[str, dict[str, str]]:
    existing_value = result.get("credentials")
    if not isinstance(existing_value, Mapping):
        raise TypeError("config.credentials must be an object")
    credentials: dict[str, dict[str, str]] = {}
    for raw_name, raw_fields in existing_value.items():
        if not isinstance(raw_name, str) or not isinstance(raw_fields, Mapping):
            raise TypeError("config.credentials must contain named objects")
        fields: dict[str, str] = {}
        for raw_field, raw_value in raw_fields.items():
            if not isinstance(raw_field, str) or not isinstance(raw_value, str):
                raise TypeError("config.credentials fields must contain strings")
            fields[raw_field] = raw_value
        credentials[raw_name.strip().lower()] = {
            field_name.strip().lower(): value for field_name, value in fields.items()
        }
    for reference, field_names in required.items():
        target = credentials.setdefault(reference, {})
        for field_name in sorted(field_names):
            target.setdefault(field_name, "")
    return credentials


def _normalized_route_document(
    document: Mapping[str, object],
    route: object,
) -> dict[str, object]:
    """用严格配置的规范身份覆盖原始 YAML，同时保留所有高级字段。"""

    result = deepcopy(dict(document))
    route_value = result.get("route")
    if not isinstance(route_value, Mapping):
        raise TypeError("setup capability route must be an object")
    normalized = deepcopy(dict(route_value))
    for name in ("provider", "adapter", "model", "base_url", "credential_ref"):
        if hasattr(route, name):
            normalized[name] = getattr(route, name)
    result["route"] = normalized
    return result


def _config_mapping(config: HabitusConfig) -> dict[str, object]:
    """只服务类型化旧调用者；保留全部公开运行字段。"""

    from dataclasses import fields, is_dataclass
    from pathlib import Path

    def plain(value: object) -> object:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, Mapping):
            return {str(key): plain(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return [plain(item) for item in value]
        if is_dataclass(value):
            return {
                item.name: plain(getattr(value, item.name))
                for item in fields(value)
            }
        return value

    result = plain(config)
    if not isinstance(result, dict):
        raise TypeError("config serialization must produce an object")
    credentials = result.get("credentials")
    if not isinstance(credentials, dict):
        raise TypeError("config credentials serialization must produce an object")
    entries = credentials.get("entries")
    if not isinstance(entries, dict):
        raise TypeError("config credential entries serialization must produce an object")
    result["credentials"] = entries
    return result


def _mapping(parent: Mapping[str, object], name: str) -> Mapping[str, object]:
    value = parent.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"config.{name} must be an object")
    return value


def _section(parent: dict[str, object], name: str) -> dict[str, object]:
    value = parent.get(name)
    if not isinstance(value, Mapping):
        raise TypeError(f"config.{name} must be an object")
    section = deepcopy(dict(value))
    parent[name] = section
    return section


__all__ = [
    "CloudSetupSelection",
    "ProfileSelection",
    "apply_cloud_selection",
    "default_cloud_selection",
    "profile_selection",
    "selection_from_config",
    "selection_from_mapping",
]
