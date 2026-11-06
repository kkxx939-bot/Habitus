"""W10a：Chat 默认不绑厂商。非交互 init 写出占位模板，doctor 与启动前检查报"还是占位值"；向导可在
OpenAI 兼容与 Anthropic Messages 两种协议间选，选 Anthropic Messages 后得到能直接组装运行时的严格配置。"""

from __future__ import annotations

import builtins
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from habitus.config import HabitusConfig
from habitus.integrations.local_service import cli
from habitus.integrations.local_service import initialization as initialization_module
from habitus.integrations.local_service.cloud_setup import (
    ProfileSelection,
    apply_cloud_selection,
    default_cloud_selection,
)
from habitus.integrations.local_service.doctor import (
    DoctorStatus,
    StartupPreflightError,
    run_doctor,
    run_startup_preflight,
)
from habitus.integrations.local_service.initialization import load_initialization_mapping
from habitus.integrations.local_service.setup_registry import build_builtin_setup_registry
from habitus.model_client.adapters import AnthropicMessagesProvider
from habitus.runtime import build_runtime

_ANTHROPIC_VALUES = {
    "provider": "anthropic",
    "model": "claude-model-1",
    "base_url": "https://api.anthropic.com",
    "credential_ref": "chat",
    "context_window_tokens": 200_000,
    "max_output_tokens": 16_000,
    "structured_output_mode": "json_schema",
    "reasoning": False,
}


@pytest.fixture(autouse=True)
def _isolate_default_plugin_connection(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        initialization_module,
        "DEFAULT_PLUGIN_CONNECTION_PATH",
        tmp_path / "default-home" / ".habitus" / "agent-plugin" / "connection.json",
    )


def _noninteractive_init(tmp_path: Path) -> Path:
    destination = tmp_path / "config" / "config.yaml"
    cli.main(["init", "--config", str(destination), "--non-interactive", "--skip-doctor", "--no-start"])
    return destination


def _with_real_embedding(destination: Path, **chat_route: str) -> HabitusConfig:
    payload = yaml.safe_load(destination.read_text(encoding="utf-8"))
    payload["storage"]["root"] = str(destination.parent / "data")
    payload["models"]["embedding"]["route"].update(
        {"model": "text-embedding-3-small", "base_url": "https://api.openai.com/v1"}
    )
    payload["models"]["chat"]["route"].update(chat_route)
    return HabitusConfig.from_mapping(payload)


def _placeholder_check(config: HabitusConfig):  # type: ignore[no-untyped-def]
    return next(item for item in run_doctor(config, check_port=False).checks if item.name == "setup_placeholders")


def test_default_selection_writes_a_vendor_neutral_chat_template() -> None:
    assert default_cloud_selection().chat.profile_id == "chat.openai_compatible.custom"
    config = HabitusConfig.from_mapping(apply_cloud_selection(load_initialization_mapping(), default_cloud_selection()))

    route = config.models.chat.route
    assert (route.provider, route.adapter, route.model, route.base_url, route.credential_ref) == (
        "custom",
        "openai_compatible_chat",
        "example.chat-model",
        "https://chat.example.com/v1",
        "chat",
    )


def test_noninteractive_init_lists_every_chat_item_to_fill(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination = _noninteractive_init(tmp_path)

    output = capsys.readouterr().out
    hint = next(line for line in output.splitlines() if line.startswith("Not configured yet: models.chat"))
    for item in ("models.chat.route.model", "models.chat.route.base_url", "models.chat.context_window_tokens"):
        assert item in hint
    assert str(destination) in hint
    assert "credentials.chat.api_key" in output


def test_doctor_and_preflight_name_the_chat_placeholder_instead_of_a_connection_error(tmp_path: Path) -> None:
    config = _with_real_embedding(_noninteractive_init(tmp_path))

    check = _placeholder_check(config)
    assert check.status is DoctorStatus.FAIL
    assert "models.chat.route.model" in check.detail
    assert "models.chat.route.base_url" in check.detail
    assert "models.embedding" not in check.detail
    with pytest.raises(StartupPreflightError, match=r"setup_placeholders: models\.chat still holds"):
        run_startup_preflight(config)


@pytest.mark.parametrize(
    ("route", "remaining"),
    [
        ({"model": "chat-model-1"}, "models.chat.route.base_url"),
        ({"base_url": "https://api.openai.com/v1"}, "models.chat.route.model"),
    ],
)
def test_one_remaining_chat_placeholder_is_still_reported(
    tmp_path: Path, route: dict[str, str], remaining: str
) -> None:
    check = _placeholder_check(_with_real_embedding(_noninteractive_init(tmp_path), **route))

    assert check.status is DoctorStatus.FAIL
    assert remaining in check.detail.split(";")[0]


def test_real_chat_values_clear_the_placeholder_check(tmp_path: Path) -> None:
    config = _with_real_embedding(
        _noninteractive_init(tmp_path), model="chat-model-1", base_url="https://api.openai.com/v1"
    )

    assert _placeholder_check(config).status is DoctorStatus.PASS


def test_anthropic_messages_profile_template_is_reported_until_filled() -> None:
    selection = replace(default_cloud_selection(), chat=ProfileSelection("chat.anthropic_messages.custom"))
    payload = apply_cloud_selection(load_initialization_mapping(), selection)
    config = HabitusConfig.from_mapping(payload)

    assert config.models.chat.route.adapter == "anthropic_messages"
    assert config.models.chat.max_output_tokens == 16_000
    assert config.models.chat.structured_output_mode == "json_schema"
    check = _placeholder_check(config)
    assert check.status is DoctorStatus.FAIL
    assert "models.chat.max_output_tokens" in check.detail


def test_anthropic_messages_selection_builds_the_runtime_provider_without_network(tmp_path: Path) -> None:
    selection = replace(
        default_cloud_selection(),
        chat=ProfileSelection("chat.anthropic_messages.custom", _ANTHROPIC_VALUES),
    )
    payload = apply_cloud_selection(load_initialization_mapping(), selection)
    payload["storage"]["root"] = str(tmp_path / "data")  # type: ignore[index]
    payload["models"]["embedding"]["route"].update(  # type: ignore[index]
        {"model": "text-embedding-3-small", "base_url": "https://api.openai.com/v1"}
    )
    payload["credentials"]["chat"]["api_key"] = "chat-secret"  # type: ignore[index]
    payload["credentials"]["embedding"]["api_key"] = "embedding-secret"  # type: ignore[index]
    config = HabitusConfig.from_mapping(payload)

    report = run_doctor(config, check_port=False)
    for name in ("setup_placeholders", "adapter_configuration", "credentials"):
        assert next(check for check in report.checks if check.name == name).status is DoctorStatus.PASS
    runtime = build_runtime(config)
    provider = runtime.components.models.chat.provider
    assert isinstance(provider, AnthropicMessagesProvider)
    assert provider.model == "claude-model-1"


def test_empty_chat_credential_fails_startup_preflight(tmp_path: Path) -> None:
    config = _with_real_embedding(
        _noninteractive_init(tmp_path), model="chat-model-1", base_url="https://api.openai.com/v1"
    )

    with pytest.raises(StartupPreflightError, match=r"credentials\.chat\.api_key"):
        run_startup_preflight(config)


def test_chat_wizard_offers_both_protocols_and_rejects_anthropic_placeholders(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    registry = build_builtin_setup_registry()
    adapters = {profile.patch["route"]["adapter"] for profile in registry.profiles("chat")}  # type: ignore[index]
    assert adapters == {"openai_compatible_chat", "anthropic_messages"}
    profile = registry.profile("chat", "chat.anthropic_messages.custom")
    base_url = next(field for field in profile.fields if field.key == "base_url")
    answers = iter(("", "https://messages.example.com", "https://api.anthropic.com"))
    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    monkeypatch.setattr(builtins, "input", fake_input)

    value = cli._prompt_setup_field(base_url, "https://messages.example.com")

    assert value == "https://api.anthropic.com"
    assert len(prompts) == 3
    assert all("[" not in prompt for prompt in prompts), "the placeholder must not be offered as the default"
    output = capsys.readouterr().out
    assert output.count("is a setup placeholder") == 1
    assert output.count("Value is required") == 1


@pytest.mark.parametrize(
    ("chat_overrides", "message"),
    [
        ({"structured_output_mode": "json_object"}, "json_object"),
        ({"max_output_tokens": None}, "max_output_tokens"),
        ({"route": {"extra_body": {"temperature": 0.2}}}, "temperature"),
    ],
)
def test_doctor_and_preflight_reject_an_anthropic_route_the_protocol_cannot_honor(
    tmp_path: Path, chat_overrides: dict[str, object], message: str
) -> None:
    """把模板里的 adapter 改成 anthropic_messages、却留着它不支持的设置：doctor 与启动前检查当场报原因，不拖到组装运行时。"""

    selection = replace(
        default_cloud_selection(),
        chat=ProfileSelection("chat.anthropic_messages.custom", _ANTHROPIC_VALUES),
    )
    payload = apply_cloud_selection(load_initialization_mapping(), selection)
    payload["storage"]["root"] = str(tmp_path / "data")  # type: ignore[index]
    payload["models"]["embedding"]["route"].update(  # type: ignore[index]
        {"model": "text-embedding-3-small", "base_url": "https://api.openai.com/v1"}
    )
    payload["credentials"]["chat"]["api_key"] = "chat-secret"  # type: ignore[index]
    payload["credentials"]["embedding"]["api_key"] = "embedding-secret"  # type: ignore[index]
    chat = payload["models"]["chat"]  # type: ignore[index]
    for key, value in chat_overrides.items():
        if key == "route":
            chat["route"].update(value)  # type: ignore[index,union-attr]
        else:
            chat[key] = value  # type: ignore[index]
    config = HabitusConfig.from_mapping(payload)

    check = next(item for item in run_doctor(config, check_port=False).checks if item.name == "adapter_configuration")
    assert check.status is DoctorStatus.FAIL
    assert message in check.detail
    with pytest.raises(StartupPreflightError, match=message):
        run_startup_preflight(config)
