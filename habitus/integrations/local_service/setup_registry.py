"""本地产品外壳使用的声明式 Provider、模型与向量配置注册表。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from habitus.config import HabitusConfig
from habitus.infrastructure.vector import VectorStoreConfig
from habitus.infrastructure.vector.adapters import LocalSQLiteVectorStoreConfig
from habitus.model_client import CapabilityConfig

SetupCapability = Literal["chat", "embedding", "rerank", "vector"]
SetupInputKind = Literal["text", "required_text", "positive_int", "choice"]
SetupTargetState = Literal["configured", "disabled"]


@dataclass(frozen=True)
class SetupChoice:
    """一个由 Registry 声明、由通用向导渲染的枚举值。"""

    value: object
    label: str

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("setup choice label must be non-empty")


@dataclass(frozen=True)
class SetupField:
    """一个 Profile 可编辑字段；paths 允许一次写入共享向量配置。"""

    key: str
    label: str
    paths: tuple[str, ...]
    kind: SetupInputKind
    default: object
    choices: tuple[SetupChoice, ...] = ()
    placeholder_values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key:
            raise ValueError("setup field key must be non-empty")
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("setup field label must be non-empty")
        if not self.paths or any(not isinstance(path, str) or not path for path in self.paths):
            raise ValueError("setup field paths must be non-empty")
        if self.kind not in {"text", "required_text", "positive_int", "choice"}:
            raise ValueError("setup field kind is invalid")
        if self.kind == "choice" and not self.choices:
            raise ValueError("choice setup field must declare choices")
        if self.kind != "choice" and self.choices:
            raise ValueError("only choice setup fields may declare choices")
        if any(not isinstance(item, str) or not item for item in self.placeholder_values):
            raise ValueError("setup field placeholder values must be non-empty strings")
        if self.kind not in {"text", "required_text"} and self.placeholder_values:
            raise ValueError("only text setup fields may declare placeholder values")
        if self.kind == "positive_int" and (
            isinstance(self.default, bool) or not isinstance(self.default, int) or self.default <= 0
        ):
            raise ValueError("positive integer setup field requires a positive default")


@dataclass(frozen=True)
class SetupProfile:
    """一个可注册的模型或向量产品配置模板。"""

    profile_id: str
    capability: SetupCapability
    display_name: str
    patch: Mapping[str, object]
    match: Mapping[str, object]
    fields: tuple[SetupField, ...] = ()
    replace_paths: tuple[str, ...] = ()
    selectable: bool = True
    target_state: SetupTargetState = "configured"

    def __post_init__(self) -> None:
        if not isinstance(self.profile_id, str) or not self.profile_id:
            raise ValueError("setup profile id must be non-empty")
        if self.capability not in {"chat", "embedding", "rerank", "vector"}:
            raise ValueError("setup profile capability is invalid")
        if not isinstance(self.display_name, str) or not self.display_name.strip():
            raise ValueError("setup profile display_name must be non-empty")
        if not isinstance(self.patch, Mapping) or not isinstance(self.match, Mapping):
            raise TypeError("setup profile patch and match must be objects")
        if self.target_state not in {"configured", "disabled"}:
            raise ValueError("setup profile target_state is invalid")
        if self.target_state == "disabled" and self.capability != "rerank":
            raise ValueError("only an optional rerank profile may be disabled")
        if any(not isinstance(path, str) or not path for path in self.replace_paths):
            raise ValueError("setup profile replace_paths must be non-empty paths")
        field_keys = [item.key for item in self.fields]
        if len(field_keys) != len(set(field_keys)):
            raise ValueError("setup profile fields must have unique keys")
        object.__setattr__(self, "patch", MappingProxyType(deepcopy(dict(self.patch))))
        object.__setattr__(self, "match", MappingProxyType(deepcopy(dict(self.match))))

    def matches(self, document: Mapping[str, object]) -> bool:
        return all(_read_path(document, path, missing=_MISSING) == value for path, value in self.match.items())

    def values_from(self, document: Mapping[str, object]) -> dict[str, object]:
        values: dict[str, object] = {}
        for item in self.fields:
            current = _read_path(document, item.paths[0], missing=_MISSING)
            values[item.key] = item.default if current is _MISSING else deepcopy(current)
        return values

    def materialize(
        self,
        values: Mapping[str, object],
        *,
        base: Mapping[str, object] | None = None,
        apply_defaults: bool = True,
        field_keys: frozenset[str] | None = None,
    ) -> dict[str, object]:
        if not isinstance(values, Mapping):
            raise TypeError("setup profile values must be an object")
        unknown = sorted(set(values) - {item.key for item in self.fields})
        if unknown:
            raise ValueError(f"setup profile contains unknown values: {unknown}")
        if field_keys is not None and not field_keys <= {item.key for item in self.fields}:
            raise ValueError("setup profile field_keys contain unknown fields")
        document = deepcopy(dict(base)) if base is not None else {}
        if apply_defaults:
            for path in self.replace_paths:
                _delete_path(document, path)
            _merge(document, self.patch)
        for item in self.fields:
            if field_keys is not None and item.key not in field_keys:
                continue
            value = deepcopy(values.get(item.key, item.default))
            _validate_field_value(item, value)
            for path in item.paths:
                _write_path(document, path, value)
        return document


CredentialFieldsResolver = Callable[[Mapping[str, object]], tuple[str, ...]]
DependencyResolver = Callable[[Mapping[str, object]], tuple[str, ...]]
ConfigurationValidator = Callable[[HabitusConfig], None]


@dataclass(frozen=True)
class AdapterProductRegistration:
    """Adapter 在产品层需要公开的凭据、依赖和只读校验元数据。"""

    capability: SetupCapability
    adapter: str
    credential_fields: CredentialFieldsResolver
    dependency_modules: DependencyResolver = field(default=lambda _config: ())
    validate: ConfigurationValidator | None = None

    def __post_init__(self) -> None:
        if self.capability not in {"chat", "embedding", "rerank", "vector"}:
            raise ValueError("adapter product capability is invalid")
        if not isinstance(self.adapter, str) or not self.adapter:
            raise ValueError("adapter product name must be non-empty")
        if not callable(self.credential_fields) or not callable(self.dependency_modules):
            raise TypeError("adapter product resolvers must be callable")
        if self.validate is not None and not callable(self.validate):
            raise TypeError("adapter product validator must be callable")


class SetupRegistry:
    """集中注册向导 Profile 与运行 Adapter 的产品元数据。"""

    def __init__(self) -> None:
        self._profiles: dict[tuple[SetupCapability, str], SetupProfile] = {}
        self._adapters: dict[tuple[SetupCapability, str], AdapterProductRegistration] = {}

    def register_profile(self, profile: SetupProfile) -> None:
        if not isinstance(profile, SetupProfile):
            raise TypeError("profile must be SetupProfile")
        key = (profile.capability, profile.profile_id)
        if key in self._profiles:
            raise ValueError(f"setup profile is already registered: {profile.profile_id}")
        self._profiles[key] = profile

    def register_adapter(self, registration: AdapterProductRegistration) -> None:
        if not isinstance(registration, AdapterProductRegistration):
            raise TypeError("registration must be AdapterProductRegistration")
        key = (registration.capability, registration.adapter)
        if key in self._adapters:
            raise ValueError(f"setup adapter is already registered: {registration.capability}/{registration.adapter}")
        self._adapters[key] = registration

    def profiles(
        self,
        capability: SetupCapability,
        *,
        include_unselectable: bool = False,
    ) -> tuple[SetupProfile, ...]:
        entries = tuple(
            profile
            for (item_capability, _profile_id), profile in self._profiles.items()
            if item_capability == capability and (include_unselectable or profile.selectable)
        )
        return entries

    def profile(self, capability: SetupCapability, profile_id: str) -> SetupProfile:
        try:
            return self._profiles[(capability, profile_id)]
        except KeyError as exc:
            raise ValueError(f"setup profile is not registered: {capability}/{profile_id}") from exc

    def registered_adapters(self, capability: SetupCapability) -> tuple[str, ...]:
        return tuple(sorted(adapter for (item_capability, adapter) in self._adapters if item_capability == capability))

    def configured_adapters(
        self,
        config: HabitusConfig,
    ) -> tuple[tuple[SetupCapability, str], ...]:
        if not isinstance(config, HabitusConfig):
            raise TypeError("config must be HabitusConfig")
        return tuple(
            (capability, str(_mapping_at(document, "route").get("adapter", "")))
            for capability, document in _active_adapter_documents(config)
        )

    def identify(self, capability: SetupCapability, document: Mapping[str, object]) -> SetupProfile:
        matches = tuple(
            profile for profile in self.profiles(capability, include_unselectable=True) if profile.matches(document)
        )
        if not matches:
            adapter = _read_path(document, "route.adapter", missing="")
            raise ValueError(f"no setup profile is registered for {capability} adapter/configuration: {adapter}")
        specificity = max(len(profile.match) for profile in matches)
        preferred = tuple(profile for profile in matches if len(profile.match) == specificity)
        if len(preferred) != 1:
            identifiers = ", ".join(sorted(profile.profile_id for profile in preferred))
            raise ValueError(f"ambiguous setup profiles for {capability}: {identifiers}")
        return preferred[0]

    def required_credentials(self, config: HabitusConfig) -> dict[str, set[str]]:
        required: dict[str, set[str]] = {}
        for capability, document in _active_adapter_documents(config):
            self._add_required_credentials(required, capability, document)
        return required

    def required_credentials_for_documents(
        self,
        documents: tuple[tuple[SetupCapability, Mapping[str, object]], ...],
    ) -> dict[str, set[str]]:
        """在根配置构造前为 Planner 补齐已注册 Adapter 的凭据字段。"""

        required: dict[str, set[str]] = {}
        for capability, document in documents:
            self._add_required_credentials(required, capability, document)
        return required

    def _add_required_credentials(
        self,
        required: dict[str, set[str]],
        capability: SetupCapability,
        document: Mapping[str, object],
    ) -> None:
        route = _mapping_at(document, "route")
        reference = str(route.get("credential_ref", ""))
        adapter = str(route.get("adapter", ""))
        registration = self._adapter(capability, adapter)
        if reference:
            required.setdefault(reference, set()).update(registration.credential_fields(document))

    def dependency_modules(self, config: HabitusConfig) -> tuple[str, ...]:
        modules = {"fastapi", "uvicorn", "json_repair", "jsonschema"}
        for capability, document in _active_adapter_documents(config):
            route = _mapping_at(document, "route")
            registration = self._adapter(capability, str(route.get("adapter", "")))
            modules.update(registration.dependency_modules(document))
        return tuple(sorted(modules))

    def validate(self, config: HabitusConfig) -> None:
        """确认配置引用的 Adapter 已注册，并运行其无网络、无写入校验。"""

        seen: set[tuple[SetupCapability, str]] = set()
        for capability, document in _active_adapter_documents(config):
            route = _mapping_at(document, "route")
            adapter = str(route.get("adapter", ""))
            registration = self._adapter(capability, adapter)
            key = (capability, adapter)
            if key not in seen and registration.validate is not None:
                registration.validate(config)
            seen.add(key)

    def _adapter(
        self,
        capability: SetupCapability,
        adapter: str,
    ) -> AdapterProductRegistration:
        try:
            return self._adapters[(capability, adapter)]
        except KeyError as exc:
            raise ValueError(f"product adapter is not registered: {capability}/{adapter}") from exc


_MISSING = object()


def is_setup_placeholder(value: object, placeholder_values: tuple[str, ...]) -> bool:
    """仍是向导占位：等于某个出厂模板原值，或留着未替换的 ``{var}``。精确匹配，不按子串判，免得误伤真实值。"""
    return isinstance(value, str) and (value in placeholder_values or "{" in value)


def build_builtin_setup_registry() -> SetupRegistry:
    """注册当前发行包支持的云端 Profile；这里不读取秘密或访问网络。"""

    from habitus.integrations.local_service.setup_chat_profiles import chat_adapter_registrations

    registry = SetupRegistry()
    for registration in (
        *chat_adapter_registrations(),
        AdapterProductRegistration(
            "embedding",
            "openai_compatible_embedding",
            _api_key_fields,
        ),
        AdapterProductRegistration(
            "rerank",
            "openai_compatible_rerank",
            _api_key_fields,
        ),
        AdapterProductRegistration(
            "vector",
            "local_sqlite",
            _no_credential_fields,
            validate=_validate_local_sqlite,
        ),
    ):
        registry.register_adapter(registration)
    for profile in _builtin_profiles():
        registry.register_profile(profile)
    return registry


def _builtin_profiles() -> tuple[SetupProfile, ...]:
    # chat 的 Profile 在单独模块里；它要用本模块的字段构造函数，所以在这里（本模块加载完之后）再导入
    from habitus.integrations.local_service.setup_chat_profiles import chat_profiles

    qwen_instruction = "Given a web search query, retrieve relevant passages that answer the query."
    return (
        *chat_profiles(),
        SetupProfile(
            "embedding.openai_compatible.custom",
            "embedding",
            "OpenAI-compatible embedding endpoint (cloud or self-hosted)",
            patch={
                "route": {
                    "provider": "custom",
                    "adapter": "openai_compatible_embedding",
                    "model": "example.embedding-model",
                    "base_url": "https://embedding.example.com/v1",
                    "credential_ref": "embedding",
                    "extra_body": {},
                },
                "dimension": 1024,
                "query_parameters": {},
                "document_parameters": {},
                "query_prefix": "",
                "document_prefix": "",
            },
            match={"route.adapter": "openai_compatible_embedding"},
            fields=(
                _text("provider", "Embedding provider identifier", "route.provider", "custom"),
                _required(
                    "model", "Embedding model", "route.model", "", placeholder_values=("example.embedding-model",)
                ),
                _required(
                    "base_url",
                    "Embedding API base URL",
                    "route.base_url",
                    "",
                    placeholder_values=("https://embedding.example.com/v1",),
                ),
                _text("credential_ref", "Embedding credential reference", "route.credential_ref", "embedding"),
                _positive("dimension", "Embedding dimension (must match the model output)", "dimension", 1024),
            ),
            replace_paths=(
                "route.extra_body",
                "query_parameters",
                "document_parameters",
                "query_prefix",
                "document_prefix",
            ),
        ),
        SetupProfile(
            "rerank.disabled",
            "rerank",
            "Disabled (vector relevance fallback)",
            patch={"enabled": False},
            match={"enabled": False},
            target_state="disabled",
        ),
        SetupProfile(
            "rerank.aliyun.qwen3",
            "rerank",
            "Alibaba Cloud Qwen3 Rerank",
            patch={
                "route": {
                    "provider": "aliyun",
                    "adapter": "openai_compatible_rerank",
                    "model": "qwen3-rerank",
                    "base_url": "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-api/v1",
                    "credential_ref": "dashscope",
                    "extra_body": {"instruct": qwen_instruction},
                },
                "max_documents": 100,
                "max_query_chars": 8_000,
                "max_document_chars": 16_000,
            },
            match={
                "route.provider": "aliyun",
                "route.adapter": "openai_compatible_rerank",
                "route.model": "qwen3-rerank",
            },
            fields=(
                # 没有 example 模板原值，占位只由 is_setup_placeholder 的 {var} 规则拦（README 里的 {WorkspaceId}）
                _required("base_url", "Alibaba Workspace API base URL", "route.base_url", ""),
            ),
            replace_paths=("route.extra_body",),
        ),
        SetupProfile(
            "rerank.openai_compatible.custom",
            "rerank",
            "Custom OpenAI-compatible rerank endpoint",
            patch={
                "route": {
                    "provider": "custom",
                    "adapter": "openai_compatible_rerank",
                    "model": "rerank-model",
                    "base_url": "https://rerank.example.com/v1",
                    "credential_ref": "custom_rerank",
                    "extra_body": {},
                },
                "max_documents": 100,
                "max_query_chars": 8_000,
                "max_document_chars": 16_000,
            },
            match={"route.adapter": "openai_compatible_rerank"},
            fields=(
                _text("provider", "Rerank provider identifier", "route.provider", "custom"),
                _text("model", "Rerank model", "route.model", "rerank-model"),
                _required("base_url", "Rerank API base URL", "route.base_url", ""),
                _text("credential_ref", "Rerank credential reference", "route.credential_ref", "custom_rerank"),
            ),
            replace_paths=("route.extra_body",),
        ),
        _local_sqlite_profile(),
    )


def _local_sqlite_profile() -> SetupProfile:
    """进程内本地向量库：存储根下每个 Collection 一个 SQLite 文件，不要账号、不走网络。"""

    route = {
        "provider": "local",
        "adapter": "local_sqlite",
        "base_url": "",
        "credential_ref": "",
        "extra_headers": {},
    }
    options = {"busy_timeout_seconds": 5.0}
    return SetupProfile(
        "vector.local_sqlite",
        "vector",
        "Local (embedded SQLite + numpy, no account)",
        patch={
            "memory": {"route": route, "collection": "memory", "options": options},
            "summary": {
                "route": deepcopy(route),
                "collection": "conversation_summaries",
                "options": deepcopy(options),
            },
        },
        match={"memory.route.adapter": "local_sqlite"},
        replace_paths=(
            "memory.options",
            "summary.options",
            "memory.route.extra_headers",
            "summary.route.extra_headers",
        ),
    )


def _api_key_fields(_document: Mapping[str, object]) -> tuple[str, ...]:
    return ("api_key",)


def _no_credential_fields(_document: Mapping[str, object]) -> tuple[str, ...]:
    return ()


def _validate_local_sqlite(config: HabitusConfig) -> None:
    for vector in (config.memory.vector_store, config.conversation.summary_vector_store):
        LocalSQLiteVectorStoreConfig.from_mapping(vector.options).validate_store(vector)


def _active_adapter_documents(
    config: HabitusConfig,
) -> tuple[tuple[SetupCapability, Mapping[str, object]], ...]:
    documents: list[tuple[SetupCapability, Mapping[str, object]]] = [
        ("chat", model_document(config.models.chat)),
        ("embedding", model_document(config.models.embedding)),
    ]
    if config.models.rerank is not None:
        documents.append(("rerank", model_document(config.models.rerank)))
    documents.extend(
        (
            ("vector", _vector_document(config.memory.vector_store)),
            ("vector", _vector_document(config.conversation.summary_vector_store)),
        )
    )
    return tuple(documents)


def model_document(model: CapabilityConfig) -> Mapping[str, object]:
    route = model.route
    return {
        "route": {
            "provider": route.provider,
            "adapter": route.adapter,
            "model": route.model,
            "base_url": route.base_url,
            "credential_ref": route.credential_ref,
        }
    }


def _vector_document(vector: VectorStoreConfig) -> Mapping[str, object]:
    route = vector.route
    return {
        "route": {
            "provider": route.provider,
            "adapter": route.adapter,
            "base_url": route.base_url,
            "credential_ref": route.credential_ref,
        },
        "options": dict(vector.options),
    }


def _text(
    key: str,
    label: str,
    paths: str | tuple[str, ...],
    default: str,
) -> SetupField:
    return SetupField(key, label, _paths(paths), "text", default)


def _required(
    key: str,
    label: str,
    paths: str | tuple[str, ...],
    default: str,
    *,
    placeholder_values: tuple[str, ...] = (),
) -> SetupField:
    return SetupField(
        key,
        label,
        _paths(paths),
        "required_text",
        default,
        placeholder_values=placeholder_values,
    )


def _positive(
    key: str,
    label: str,
    paths: str | tuple[str, ...],
    default: int,
) -> SetupField:
    return SetupField(key, label, _paths(paths), "positive_int", default)


def _choice(
    key: str,
    label: str,
    paths: str | tuple[str, ...],
    default: object,
    choices: tuple[tuple[object, str], ...],
) -> SetupField:
    return SetupField(
        key,
        label,
        _paths(paths),
        "choice",
        default,
        tuple(SetupChoice(value, choice_label) for value, choice_label in choices),
    )


def _paths(value: str | tuple[str, ...]) -> tuple[str, ...]:
    return (value,) if isinstance(value, str) else value


def _validate_field_value(field: SetupField, value: object) -> None:
    if field.kind in {"text", "required_text"}:
        if not isinstance(value, str) or (field.kind == "required_text" and not value.strip()):
            raise ValueError(f"setup field is invalid: {field.key}")
        if is_setup_placeholder(value, field.placeholder_values):
            raise ValueError(f"setup field contains a placeholder value: {field.key}")
    elif field.kind == "positive_int":
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"setup field must be a positive integer: {field.key}")
    elif not any(value == choice.value for choice in field.choices):
        raise ValueError(f"setup field choice is invalid: {field.key}")


def _mapping_at(document: Mapping[str, object], path: str) -> Mapping[str, object]:
    value = _read_path(document, path, missing=_MISSING)
    if not isinstance(value, Mapping):
        raise TypeError(f"setup document path must be an object: {path}")
    return value


def _read_path(document: Mapping[str, object], path: str, *, missing: object) -> object:
    current: object = document
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return missing
        current = current[part]
    return current


def _write_path(document: dict[str, object], path: str, value: object) -> None:
    parts = path.split(".")
    current = document
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, Mapping):
            child_mapping: dict[str, object] = {}
        else:
            child_mapping = dict(child)
        current[part] = child_mapping
        current = child_mapping
    current[parts[-1]] = deepcopy(value)


def _delete_path(document: dict[str, object], path: str) -> None:
    parts = path.split(".")
    current: dict[str, object] = document
    for part in parts[:-1]:
        child = current.get(part)
        if not isinstance(child, dict):
            return
        current = child
    current.pop(parts[-1], None)


def _merge(target: dict[str, object], patch: Mapping[str, object]) -> None:
    for key, value in patch.items():
        current = target.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            child = dict(current)
            _merge(child, value)
            target[key] = child
        else:
            target[key] = deepcopy(value)


__all__ = [
    "AdapterProductRegistration",
    "SetupChoice",
    "SetupField",
    "SetupProfile",
    "SetupRegistry",
    "build_builtin_setup_registry",
    "model_document",
]
