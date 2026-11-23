"""产品外壳唯一的 Adapter 组合目录与已安装扩展发现入口。"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import lru_cache
from importlib import metadata
from typing import cast

from habitus.config import HabitusConfig
from habitus.model_client import ModelCapability, ProviderBuilder, ProviderFactory
from habitus.model_client.adapters import register_builtin_adapters

from .setup_registry import (
    AdapterProductRegistration,
    SetupProfile,
    SetupRegistry,
    build_builtin_setup_registry,
)

_ENTRY_POINT_GROUP = "habitus.adapter_packages"


@dataclass(frozen=True)
class AdapterCatalog:
    """同一份注册结果同时服务向导、Doctor 与 Runtime 组合根。"""

    setup: SetupRegistry
    providers: ProviderFactory

    def register_model_adapter(
        self,
        capability: ModelCapability,
        adapter: str,
        builder: ProviderBuilder,
        *,
        product: AdapterProductRegistration,
        profiles: Sequence[SetupProfile] = (),
    ) -> None:
        if product.capability != capability or product.adapter != adapter:
            raise ValueError("model product metadata must match its runtime adapter key")
        if any(profile.capability != capability for profile in profiles):
            raise ValueError("model setup profiles must match their adapter capability")
        self.providers.register_adapter(capability, adapter, builder)
        self.setup.register_adapter(product)
        for profile in profiles:
            self.setup.register_profile(profile)

    def validate(self, config: HabitusConfig) -> None:
        self.setup.validate(config)
        for capability, adapter in self.setup.configured_adapters(config):
            if adapter not in self.providers.registered_adapters(capability):
                raise ValueError(f"runtime adapter is not registered: {capability}/{adapter}")

    def assert_consistent(self) -> None:
        product = set(self.setup.registered_adapters("chat"))
        runtime = set(self.providers.registered_adapters("chat"))
        if product != runtime:
            raise ValueError(
                f"model adapter registry mismatch for chat: product={sorted(product)}, runtime={sorted(runtime)}"
            )


AdapterRegistrar = Callable[[AdapterCatalog], None]


def build_adapter_catalog(
    *,
    registrars: Sequence[AdapterRegistrar] = (),
) -> AdapterCatalog:
    """构造内置目录，并让每个已安装包只执行一个显式注册入口。"""

    providers = ProviderFactory()
    register_builtin_adapters(providers)
    catalog = AdapterCatalog(
        setup=build_builtin_setup_registry(),
        providers=providers,
    )
    for registrar in registrars:
        if not callable(registrar):
            raise TypeError("adapter package registrar must be callable")
        registrar(catalog)
    catalog.assert_consistent()
    return catalog


@lru_cache(maxsize=1)
def load_adapter_catalog() -> AdapterCatalog:
    """发现已安装发行包的 entry point，并在当前进程只组合一次。"""

    entries = tuple(metadata.entry_points().select(group=_ENTRY_POINT_GROUP))
    registrars: list[AdapterRegistrar] = []
    for entry in sorted(entries, key=lambda item: (item.name, item.value)):
        loaded = entry.load()
        if not callable(loaded):
            raise TypeError(f"adapter package entry point is not callable: {entry.name}")
        registrars.append(cast(AdapterRegistrar, loaded))
    return build_adapter_catalog(registrars=tuple(registrars))


__all__ = [
    "AdapterCatalog",
    "AdapterRegistrar",
    "build_adapter_catalog",
    "load_adapter_catalog",
]
