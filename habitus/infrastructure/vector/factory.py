"""按协议显式注册并构造向量数据库 Adapter。"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

from habitus.foundation.integrity import canonical_digest
from habitus.infrastructure.store.contracts import PathLock
from habitus.infrastructure.vector.config import VectorStoreConfig, VectorStoreRequirements
from habitus.infrastructure.vector.contracts import RawVectorBackend, VectorStore
from habitus.infrastructure.vector.model import VectorStoreError
from habitus.infrastructure.vector.publication.store import PublishedVectorStore

_ADAPTER_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")


@dataclass(frozen=True)
class VectorStoreBuildContext:
    """向 builder 传递路由、厂商参数、已经解析的秘密凭据，以及本地落盘型后端可用的数据目录。"""

    config: VectorStoreConfig
    requirements: VectorStoreRequirements
    credentials: Mapping[str, str] = field(default_factory=dict, repr=False)
    data_root: Path | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.config, VectorStoreConfig):
            raise TypeError("vector build config must be VectorStoreConfig")
        if not isinstance(self.requirements, VectorStoreRequirements):
            raise TypeError("vector build requirements must be VectorStoreRequirements")
        if not isinstance(self.credentials, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str) for key, value in self.credentials.items()
        ):
            raise TypeError("vector build credentials must be a string mapping")
        object.__setattr__(self, "credentials", MappingProxyType(dict(self.credentials)))


VectorBackendBuilder = Callable[[VectorStoreBuildContext], RawVectorBackend]


class VectorStoreFactory:
    """工厂只按 Adapter 解析协议；provider 只代表实际服务来源。"""

    def __init__(self) -> None:
        self._builders: dict[str, VectorBackendBuilder] = {}

    def register_adapter(self, adapter: str, builder: VectorBackendBuilder) -> None:
        normalized = self._adapter(adapter)
        if normalized in self._builders:
            raise ValueError(f"vector store adapter is already registered: {normalized}")
        if not callable(builder):
            raise TypeError("vector store builder must be callable")
        self._builders[normalized] = builder

    def registered_adapters(self) -> tuple[str, ...]:
        return tuple(sorted(self._builders))

    def create(
        self,
        config: VectorStoreConfig,
        *,
        requirements: VectorStoreRequirements,
        credentials: Mapping[str, str] | None = None,
        path_lock: PathLock | None = None,
        data_root: Path | None = None,
    ) -> VectorStore:
        if not isinstance(config, VectorStoreConfig):
            raise TypeError("config must be VectorStoreConfig")
        if not isinstance(requirements, VectorStoreRequirements):
            raise TypeError("requirements must be VectorStoreRequirements")
        if path_lock is None:
            raise VectorStoreError("vector store creation requires an explicit publication PathLock")
        if not isinstance(path_lock, PathLock):
            raise TypeError("path_lock must be PathLock")
        builder = self._builders.get(config.adapter)
        if builder is None:
            raise VectorStoreError(f"vector store adapter is not registered: {config.adapter}")
        resolved_credentials = self._credentials(config, credentials)
        backend = builder(
            VectorStoreBuildContext(
                config=config,
                requirements=requirements,
                credentials=resolved_credentials,
                data_root=data_root,
            )
        )
        for name in (
            "initialize",
            "read_metadata",
            "write_metadata",
            "ensure_schema",
            "read",
            "delete_all",
            "upsert",
            "delete",
            "validate_records",
            "wait_visible",
            "search",
            "scan",
            "close",
        ):
            if not callable(getattr(backend, name, None)):
                raise VectorStoreError(f"raw vector backend is missing method: {name}")
        if str(getattr(backend, "adapter_name", "")) != config.adapter:
            raise VectorStoreError("vector store adapter identity does not match its config")
        if str(getattr(backend, "provider_name", "")) != config.provider:
            raise VectorStoreError("vector store provider identity does not match its config")
        if str(getattr(backend, "collection", "")) != config.collection:
            raise VectorStoreError("vector store collection does not match its config")
        publication_identity = canonical_digest(
            {
                "adapter": config.adapter,
                "provider": config.provider,
                "base_url": config.route.base_url,
                "collection": config.collection,
            }
        )
        return PublishedVectorStore(
            backend,
            path_lock=path_lock,
            publication_lock_key=f"vector-publication:{publication_identity}",
        )

    @staticmethod
    def _credentials(
        config: VectorStoreConfig,
        credentials: Mapping[str, str] | None,
    ) -> dict[str, str]:
        if credentials is None:
            resolved: dict[str, str] = {}
        elif not isinstance(credentials, Mapping):
            raise TypeError("vector store credentials must be a string mapping")
        else:
            resolved = {}
            for name, value in credentials.items():
                if not isinstance(name, str) or not isinstance(value, str):
                    raise TypeError("vector store credentials must be a string mapping")
                if not name or not value.strip():
                    raise VectorStoreError("vector store credentials cannot contain empty values")
                resolved[name] = value.strip()
        if config.route.credential_ref and not resolved:
            raise VectorStoreError(f"vector store credential is missing for reference: {config.route.credential_ref}")
        if not config.route.credential_ref and resolved:
            raise VectorStoreError("vector store route received undeclared credentials")
        return resolved

    @staticmethod
    def _adapter(value: object) -> str:
        normalized = str(value or "").strip().lower()
        if _ADAPTER_NAME.fullmatch(normalized) is None:
            raise ValueError("vector store adapter must be a normalized name")
        return normalized


__all__ = [
    "VectorBackendBuilder",
    "VectorStoreBuildContext",
    "VectorStoreFactory",
]
