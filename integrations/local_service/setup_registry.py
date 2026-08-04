"""本地产品外壳使用的声明式 Provider、模型与向量配置注册表。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from Config import M2BOSConfig
from infrastructure.vector import VectorStoreConfig, VectorStoreRequirements
from infrastructure.vector.adapters.vikingdb_config import VikingDBVectorStoreConfig
from ModelClient import CapabilityConfig

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
    forbidden_fragments: tuple[str, ...] = ()

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
        if any(not isinstance(item, str) or not item for item in self.forbidden_fragments):
            raise ValueError("setup field forbidden fragments must be non-empty strings")
        if self.kind not in {"text", "required_text"} and self.forbidden_fragments:
            raise ValueError("only text setup fields may reject placeholder fragments")
        if self.kind == "positive_int" and (
            isinstance(self.default, bool)
            or not isinstance(self.default, int)
            or self.default <= 0
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
ConfigurationValidator = Callable[[M2BOSConfig], None]


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
            raise ValueError(
                f"setup adapter is already registered: {registration.capability}/{registration.adapter}"
            )
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
        return tuple(
            sorted(
                adapter
                for (item_capability, adapter) in self._adapters
                if item_capability == capability
            )
        )

    def configured_adapters(
        self,
        config: M2BOSConfig,
    ) -> tuple[tuple[SetupCapability, str], ...]:
        if not isinstance(config, M2BOSConfig):
            raise TypeError("config must be M2BOSConfig")
        return tuple(
            (capability, str(_mapping_at(document, "route").get("adapter", "")))
            for capability, document in _active_adapter_documents(config)
        )

    def identify(self, capability: SetupCapability, document: Mapping[str, object]) -> SetupProfile:
        matches = tuple(
            profile
            for profile in self.profiles(capability, include_unselectable=True)
            if profile.matches(document)
        )
        if not matches:
            adapter = _read_path(document, "route.adapter", missing="")
            raise ValueError(
                f"no setup profile is registered for {capability} adapter/configuration: {adapter}"
            )
        specificity = max(len(profile.match) for profile in matches)
        preferred = tuple(profile for profile in matches if len(profile.match) == specificity)
        if len(preferred) != 1:
            identifiers = ", ".join(sorted(profile.profile_id for profile in preferred))
            raise ValueError(
                f"ambiguous setup profiles for {capability}: {identifiers}"
            )
        return preferred[0]

    def required_credentials(self, config: M2BOSConfig) -> dict[str, set[str]]:
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
            required.setdefault(reference, set()).update(
                registration.credential_fields(document)
            )

    def dependency_modules(self, config: M2BOSConfig) -> tuple[str, ...]:
        modules = {"fastapi", "uvicorn", "json_repair", "jsonschema"}
        for capability, document in _active_adapter_documents(config):
            route = _mapping_at(document, "route")
            registration = self._adapter(capability, str(route.get("adapter", "")))
            modules.update(registration.dependency_modules(document))
        return tuple(sorted(modules))

    def validate(self, config: M2BOSConfig) -> None:
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
            raise ValueError(
                f"product adapter is not registered: {capability}/{adapter}"
            ) from exc


_MISSING = object()


def build_builtin_setup_registry() -> SetupRegistry:
    """注册当前发行包支持的云端 Profile；这里不读取秘密或访问网络。"""

    registry = SetupRegistry()
    for registration in (
        AdapterProductRegistration(
            "chat",
            "openai_compatible_chat",
            _api_key_fields,
        ),
        AdapterProductRegistration(
            "embedding",
            "ark_multimodal",
            _api_key_fields,
        ),
        AdapterProductRegistration(
            "rerank",
            "openai_compatible_rerank",
            _api_key_fields,
        ),
        AdapterProductRegistration(
            "vector",
            "vikingdb",
            _vikingdb_credential_fields,
            _vikingdb_dependencies,
            _validate_vikingdb,
        ),
    ):
        registry.register_adapter(registration)
    for profile in _builtin_profiles():
        registry.register_profile(profile)
    return registry


def _builtin_profiles() -> tuple[SetupProfile, ...]:
    qwen_instruction = (
        "Given a web search query, retrieve relevant passages that answer the query."
    )
    return (
        SetupProfile(
            "chat.deepseek",
            "chat",
            "DeepSeek (OpenAI-compatible)",
            patch={
                "route": {
                    "provider": "deepseek",
                    "adapter": "openai_compatible_chat",
                    "model": "deepseek-chat",
                    "base_url": "https://api.deepseek.com",
                    "credential_ref": "deepseek",
                    "extra_body": {},
                },
                "context_window_tokens": 64_000,
                "structured_output_mode": "json_object",
                "reasoning": False,
            },
            match={
                "route.provider": "deepseek",
                "route.adapter": "openai_compatible_chat",
                "route.model": "deepseek-chat",
                "route.base_url": "https://api.deepseek.com",
            },
            replace_paths=("route.extra_body", "max_output_tokens"),
        ),
        SetupProfile(
            "chat.openai_compatible.custom",
            "chat",
            "Custom OpenAI-compatible cloud endpoint",
            patch={
                "route": {
                    "provider": "custom",
                    "adapter": "openai_compatible_chat",
                    "model": "chat-model",
                    "base_url": "https://chat.example.com/v1",
                    "credential_ref": "custom_chat",
                    "extra_body": {},
                },
                "context_window_tokens": 64_000,
                "structured_output_mode": "json_object",
                "reasoning": False,
            },
            match={"route.adapter": "openai_compatible_chat"},
            fields=(
                _text("provider", "Chat provider identifier", "route.provider", "custom"),
                _text("model", "Chat model", "route.model", "chat-model"),
                _required("base_url", "Chat API base URL", "route.base_url", ""),
                _text("credential_ref", "Chat credential reference", "route.credential_ref", "custom_chat"),
                _positive("context_window_tokens", "Chat context window tokens", "context_window_tokens", 64_000),
                _choice(
                    "structured_output_mode",
                    "Chat structured output",
                    "structured_output_mode",
                    "json_object",
                    (("json_object", "JSON object"), ("json_schema", "JSON schema"), ("none", "Prompt-only / none")),
                ),
                _choice(
                    "reasoning",
                    "Chat reasoning capability",
                    "reasoning",
                    False,
                    ((False, "Disabled"), (True, "Enabled")),
                ),
            ),
            replace_paths=("route.extra_body",),
        ),
        SetupProfile(
            "embedding.volcengine.ark",
            "embedding",
            "VolcEngine Ark multimodal",
            patch={
                "route": {
                    "provider": "volcengine",
                    "adapter": "ark_multimodal",
                    "model": "doubao-embedding-vision-251215",
                    "base_url": "https://ark.cn-beijing.volces.com/api/v3",
                    "credential_ref": "ark",
                    "extra_body": {},
                },
                "dimension": 1024,
                "input_mode": "multimodal",
                "query_parameters": {},
                "document_parameters": {},
            },
            match={
                "route.provider": "volcengine",
                "route.adapter": "ark_multimodal",
                "route.model": "doubao-embedding-vision-251215",
                "route.base_url": "https://ark.cn-beijing.volces.com/api/v3",
            },
            replace_paths=("route.extra_body", "query_parameters", "document_parameters"),
        ),
        SetupProfile(
            "embedding.ark_multimodal.custom",
            "embedding",
            "Custom Ark multimodal-compatible cloud endpoint",
            patch={
                "route": {
                    "provider": "custom",
                    "adapter": "ark_multimodal",
                    "model": "embedding-model",
                    "base_url": "https://embedding.example.com/api/v3",
                    "credential_ref": "custom_embedding",
                    "extra_body": {},
                },
                "dimension": 1024,
                "input_mode": "multimodal",
                "query_parameters": {},
                "document_parameters": {},
            },
            match={"route.adapter": "ark_multimodal"},
            fields=(
                _text("provider", "Embedding provider identifier", "route.provider", "custom"),
                _text("model", "Embedding model", "route.model", "embedding-model"),
                _required("base_url", "Embedding API base URL", "route.base_url", ""),
                _text(
                    "credential_ref",
                    "Embedding credential reference",
                    "route.credential_ref",
                    "custom_embedding",
                ),
                _positive("dimension", "Embedding dimension", "dimension", 1024),
            ),
            replace_paths=("route.extra_body", "query_parameters", "document_parameters"),
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
                _required(
                    "base_url",
                    "Alibaba Workspace API base URL",
                    "route.base_url",
                    "",
                    forbidden_fragments=("{", "example."),
                ),
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
        _vikingdb_profile(
            "vector.vikingdb.managed",
            "VikingDB AK/SK with managed collections and indexes",
            auth_mode="ak_sk",
            schema_mode="managed",
        ),
        _vikingdb_profile(
            "vector.vikingdb.precreated",
            "VikingDB API Key with precreated collections and indexes",
            auth_mode="api_key",
            schema_mode="precreated",
        ),
        SetupProfile(
            "vector.vikingdb.private_current",
            "vector",
            "Keep current private VikingDB configuration",
            patch={},
            match={
                "memory.route.adapter": "vikingdb",
                "memory.options.auth_mode": "private_headers",
            },
            selectable=False,
        ),
    )


def _vikingdb_profile(
    profile_id: str,
    display_name: str,
    *,
    auth_mode: str,
    schema_mode: str,
) -> SetupProfile:
    options = {
        "auth_mode": auth_mode,
        "schema_mode": schema_mode,
        "project_name": "default",
        "index_name": "default",
        "region": "cn-beijing",
        "console_url": "",
        "credential_headers": {},
        "upsert_batch_size": 100,
        "fetch_batch_size": 64,
        "delete_batch_size": 100,
        "search_page_size": 64,
        "scan_page_size": 64,
        "max_search_hits": 10_000,
        "max_records": 1_000_000,
        "index_sync_timeout_seconds": 60.0,
        "index_sync_poll_interval_seconds": 1.0,
    }
    credential_ref = "vikingdb_api_key" if auth_mode == "api_key" else "vikingdb"
    route = {
        "provider": "volcengine",
        "adapter": "vikingdb",
        "base_url": "",
        "credential_ref": credential_ref,
        "extra_headers": {},
    }
    return SetupProfile(
        profile_id,
        "vector",
        display_name,
        patch={
            "memory": {"route": route, "collection": "memory", "options": options},
            "summary": {
                "route": deepcopy(route),
                "collection": "conversation_summaries",
                "options": deepcopy(options),
            },
        },
        match={
            "memory.route.adapter": "vikingdb",
            "memory.options.auth_mode": auth_mode,
            "memory.options.schema_mode": schema_mode,
        },
        fields=(
            _choice(
                "region",
                "VikingDB region",
                ("memory.options.region", "summary.options.region"),
                "cn-beijing",
                tuple(
                    (region, region)
                    for region in (
                        "cn-beijing",
                        "cn-shanghai",
                        "cn-guangzhou",
                        "ap-southeast-1",
                    )
                ),
            ),
            _text(
                "project_name",
                "VikingDB project name",
                ("memory.options.project_name", "summary.options.project_name"),
                "default",
            ),
            _text(
                "index_name",
                "VikingDB index name",
                ("memory.options.index_name", "summary.options.index_name"),
                "default",
            ),
            _text("memory_collection", "Memory collection", "memory.collection", "memory"),
            _text(
                "summary_collection",
                "Conversation summary collection",
                "summary.collection",
                "conversation_summaries",
            ),
        ),
        replace_paths=("memory.options", "summary.options", "memory.route.extra_headers", "summary.route.extra_headers"),
    )


def _api_key_fields(_document: Mapping[str, object]) -> tuple[str, ...]:
    return ("api_key",)


def _vikingdb_credential_fields(document: Mapping[str, object]) -> tuple[str, ...]:
    options = _mapping_at(document, "options")
    auth_mode = options.get("auth_mode")
    if auth_mode == "api_key":
        return ("api_key",)
    if auth_mode == "ak_sk":
        return ("access_key", "secret_key")
    if auth_mode == "private_headers":
        headers = options.get("credential_headers")
        if not isinstance(headers, Mapping):
            raise ValueError("private VikingDB credential_headers must be an object")
        from infrastructure.vector.adapters.vikingdb_config import credential_template_names

        return tuple(
            sorted(
                {
                    name
                    for template in headers.values()
                    if isinstance(template, str)
                    for name in credential_template_names(template)
                }
            )
        )
    raise ValueError("VikingDB auth_mode is not registered")


def _vikingdb_dependencies(document: Mapping[str, object]) -> tuple[str, ...]:
    return ("volcengine",) if _mapping_at(document, "options").get("auth_mode") == "ak_sk" else ()


def _validate_vikingdb(config: M2BOSConfig) -> None:
    pairs = (
        (
            config.memory.vector_store,
            VectorStoreRequirements(
                dimension=config.models.embedding.dimension,
                max_records=config.memory.vector_index.max_records,
                max_search_hits=config.memory.vector_index.max_search_hits,
                max_record_chars=config.memory.vector_index.max_record_chars,
            ),
        ),
        (
            config.conversation.summary_vector_store,
            VectorStoreRequirements(
                dimension=config.models.embedding.dimension,
                max_records=config.conversation.summary_vector_index.max_records,
                max_search_hits=config.conversation.summary_vector_index.max_search_hits,
                max_record_chars=config.conversation.summary_vector_index.max_record_chars,
            ),
        ),
    )
    for vector, requirements in pairs:
        adapter_config = VikingDBVectorStoreConfig.from_mapping(vector.options)
        adapter_config.validate_requirements(requirements, vector.route)

def _active_adapter_documents(
    config: M2BOSConfig,
) -> tuple[tuple[SetupCapability, Mapping[str, object]], ...]:
    documents: list[tuple[SetupCapability, Mapping[str, object]]] = [
        ("chat", _model_document(config.models.chat)),
        ("embedding", _model_document(config.models.embedding)),
    ]
    if config.models.rerank is not None:
        documents.append(("rerank", _model_document(config.models.rerank)))
    documents.extend(
        (
            ("vector", _vector_document(config.memory.vector_store)),
            ("vector", _vector_document(config.conversation.summary_vector_store)),
        )
    )
    return tuple(documents)


def _model_document(model: CapabilityConfig) -> Mapping[str, object]:
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
    forbidden_fragments: tuple[str, ...] = (),
) -> SetupField:
    return SetupField(
        key,
        label,
        _paths(paths),
        "required_text",
        default,
        forbidden_fragments=forbidden_fragments,
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
        if isinstance(value, str) and any(fragment in value for fragment in field.forbidden_fragments):
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
]
