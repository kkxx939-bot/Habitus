"""外层 Adapter Catalog 的单次注册、Doctor 与 Runtime 组合测试。"""

from __future__ import annotations

from pathlib import Path

import yaml

from Config import HabitusConfig
from integrations.local_service import adapter_catalog as catalog_module
from integrations.local_service.adapter_catalog import (
    build_adapter_catalog,
    load_adapter_catalog,
)
from integrations.local_service.doctor import DoctorStatus, run_doctor
from integrations.local_service.setup_registry import (
    AdapterProductRegistration,
    SetupProfile,
)
from ModelClient import ProviderBuildContext
from Runtime import build_runtime

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class FutureReranker:
    provider_name = "future"
    model = "future-rerank"
    is_remote = True

    async def rerank(self, _query: str, documents: tuple[str, ...]) -> tuple[float, ...]:
        return tuple(1.0 for _ in documents)

    async def aclose(self) -> None:
        return None


def test_one_adapter_package_registration_is_shared_by_setup_doctor_and_runtime(
    tmp_path: Path,
) -> None:
    registrations = 0

    def register(package) -> None:  # type: ignore[no-untyped-def]
        nonlocal registrations
        registrations += 1

        def build(context: ProviderBuildContext) -> FutureReranker:
            assert dict(context.credentials) == {"token": "future-secret"}
            return FutureReranker()

        package.register_model_adapter(
            "rerank",
            "future_rerank",
            build,
            product=AdapterProductRegistration(
                "rerank",
                "future_rerank",
                lambda _document: ("token",),
            ),
            profiles=(
                SetupProfile(
                    "rerank.future",
                    "rerank",
                    "Future rerank",
                    patch={},
                    match={"route.adapter": "future_rerank"},
                ),
            ),
        )

    catalog = build_adapter_catalog(registrars=(register,))
    payload = yaml.safe_load(
        (REPOSITORY_ROOT / "Config" / "example.yaml").read_text(encoding="utf-8")
    )
    payload["storage"]["root"] = str(tmp_path / "data")
    payload["models"]["rerank"]["route"].update(
        {
            "provider": "future",
            "adapter": "future_rerank",
            "model": "future-rerank",
            "credential_ref": "future",
        }
    )
    payload["credentials"]["future"] = {"token": "future-secret"}
    payload["credentials"]["deepseek"]["api_key"] = "chat-secret"
    payload["credentials"]["ark"]["api_key"] = "embedding-secret"
    payload["credentials"]["vikingdb"]["access_key"] = "vector-access"
    payload["credentials"]["vikingdb"]["secret_key"] = "vector-secret"
    config = HabitusConfig.from_mapping(payload)

    catalog.validate(config)
    report = run_doctor(config, check_port=False, catalog=catalog)
    adapter_check = next(
        check for check in report.checks if check.name == "adapter_configuration"
    )
    assert adapter_check.status is DoctorStatus.PASS
    assert catalog.setup.identify("rerank", payload["models"]["rerank"]).profile_id == "rerank.future"
    runtime = build_runtime(
        config,
        providers=catalog.providers,
        vector_stores=catalog.vector_stores,
    )

    assert registrations == 1
    assert runtime.components.models.reranker is not None


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
