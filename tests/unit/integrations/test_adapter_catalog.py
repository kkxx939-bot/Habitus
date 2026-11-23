"""外层 Adapter Catalog 的单次注册、Doctor 与 Runtime 组合测试。"""

from __future__ import annotations

from pathlib import Path

import yaml

from habitus.config import HabitusConfig
from habitus.integrations.local_service import adapter_catalog as catalog_module
from habitus.integrations.local_service.adapter_catalog import (
    build_adapter_catalog,
    load_adapter_catalog,
)
from habitus.integrations.local_service.doctor import DoctorStatus, run_doctor
from habitus.integrations.local_service.setup_registry import (
    AdapterProductRegistration,
    SetupProfile,
)
from habitus.model_client import ProviderBuildContext
from habitus.runtime import build_runtime

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class FutureChat:
    """外部扩展包提供的假 Chat Provider：只满足工厂的组件校验，不发请求。"""

    provider_name = "future"
    model = "future-chat"
    is_remote = True

    def prepare(self, *_args: object, **_kwargs: object) -> object:
        raise NotImplementedError

    def complete(self, *_args: object, **_kwargs: object) -> object:
        raise NotImplementedError

    async def complete_async(self, *_args: object, **_kwargs: object) -> object:
        raise NotImplementedError

    def stream(self, *_args: object, **_kwargs: object) -> object:
        raise NotImplementedError

    async def stream_async(self, *_args: object, **_kwargs: object) -> object:
        raise NotImplementedError

    def health_check(self) -> dict[str, object]:
        return {"ok": True}

    async def aclose(self) -> None:
        return None


def test_one_adapter_package_registration_is_shared_by_setup_doctor_and_runtime(
    tmp_path: Path,
) -> None:
    registrations = 0
    built: list[FutureChat] = []

    def register(package) -> None:  # type: ignore[no-untyped-def]
        nonlocal registrations
        registrations += 1

        def build(context: ProviderBuildContext) -> FutureChat:
            assert dict(context.credentials) == {"token": "future-secret"}
            provider = FutureChat()
            built.append(provider)
            return provider

        package.register_model_adapter(
            "chat",
            "future_chat",
            build,
            product=AdapterProductRegistration(
                "chat",
                "future_chat",
                lambda _document: ("token",),
            ),
            profiles=(
                SetupProfile(
                    "chat.future",
                    "chat",
                    "Future chat",
                    patch={},
                    match={"route.adapter": "future_chat"},
                ),
            ),
        )

    catalog = build_adapter_catalog(registrars=(register,))
    payload = yaml.safe_load((REPOSITORY_ROOT / "habitus" / "config" / "example.yaml").read_text(encoding="utf-8"))
    payload["storage"]["root"] = str(tmp_path / "data")
    payload["credentials"]["future"] = {"token": "future-secret"}
    payload["models"]["chat"]["route"].update(
        {"provider": "future", "adapter": "future_chat", "model": "future-chat", "credential_ref": "future"}
    )
    config = HabitusConfig.from_mapping(payload)

    catalog.validate(config)
    report = run_doctor(config, check_port=False, catalog=catalog)
    adapter_check = next(check for check in report.checks if check.name == "adapter_configuration")
    assert adapter_check.status is DoctorStatus.PASS
    assert catalog.setup.identify("chat", payload["models"]["chat"]).profile_id == "chat.future"
    runtime = build_runtime(config, providers=catalog.providers)

    assert registrations == 1
    assert runtime.components.models.chat.provider is built[-1]


def test_installed_adapter_entry_points_are_discovered_only_once_per_process(
    monkeypatch,
) -> None:  # type: ignore[no-untyped-def]
    calls = 0

    def register(_catalog) -> None:  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1

    class EntryPoint:
        name = "future"
        value = "future_package:register"

        @staticmethod
        def load():
            return register

    class EntryPoints(tuple):
        def select(self, *, group: str):
            assert group == "habitus.adapter_packages"
            return self

    load_adapter_catalog.cache_clear()
    monkeypatch.setattr(
        catalog_module.metadata,
        "entry_points",
        lambda: EntryPoints((EntryPoint(),)),
    )

    first = load_adapter_catalog()
    second = load_adapter_catalog()

    assert first is second
    assert calls == 1
    load_adapter_catalog.cache_clear()
