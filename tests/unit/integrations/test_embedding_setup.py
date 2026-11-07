"""W7：Embedding 不预置厂商。非交互 init 写出占位模板并列出要填的项，doctor 与启动前检查直接报"还是占位值"，
交互式 init 不接受占位值。"""

from __future__ import annotations

import builtins
from pathlib import Path

import pytest
import yaml

from habitus.config import HabitusConfig
from habitus.integrations.local_service import cli
from habitus.integrations.local_service import initialization as initialization_module
from habitus.integrations.local_service.doctor import (
    DoctorStatus,
    StartupPreflightError,
    run_doctor,
    run_startup_preflight,
)
from habitus.integrations.local_service.setup_registry import build_builtin_setup_registry

_TO_FILL = ("models.embedding.route.model", "models.embedding.route.base_url", "models.embedding.dimension")


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


def _filled(destination: Path, **embedding_route: str) -> HabitusConfig:
    payload = yaml.safe_load(destination.read_text(encoding="utf-8"))
    payload["storage"]["root"] = str(destination.parent / "data")
    # W10a 起 Chat 同样是占位模板；这里填上真实值，让本文件只看 Embedding 的占位
    payload["models"]["chat"]["route"].update({"model": "chat-model-1", "base_url": "https://api.openai.com/v1"})
    payload["models"]["embedding"]["route"].update(embedding_route)
    return HabitusConfig.from_mapping(payload)


def test_noninteractive_init_writes_the_template_and_lists_every_embedding_item_to_fill(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination = _noninteractive_init(tmp_path)

    config = HabitusConfig.from_file(destination)
    assert config.models.embedding.route.adapter == "openai_compatible_embedding"
    assert config.models.embedding.route.base_url == "https://embedding.example.com/v1"
    output = capsys.readouterr().out
    hint = next(line for line in output.splitlines() if line.startswith("Not configured yet: models.embedding"))
    for item in _TO_FILL:
        assert item in hint
    assert str(destination) in hint
    assert "credentials.embedding.api_key" in output


def test_doctor_names_the_placeholder_instead_of_a_connection_error(tmp_path: Path) -> None:
    config = _filled(_noninteractive_init(tmp_path))

    report = run_doctor(config, check_port=False)

    check = next(item for item in report.checks if item.name == "setup_placeholders")
    assert check.status is DoctorStatus.FAIL
    assert "placeholder" in check.detail
    assert "models.embedding.route.base_url" in check.detail
    assert "models.embedding.route.model" in check.detail
    for item in _TO_FILL:
        assert item in check.detail
    assert report.ok is False


def test_startup_preflight_refuses_a_placeholder_embedding_endpoint(tmp_path: Path) -> None:
    config = _filled(_noninteractive_init(tmp_path))

    with pytest.raises(StartupPreflightError, match=r"setup_placeholders: models\.embedding still holds"):
        run_startup_preflight(config)


@pytest.mark.parametrize(
    ("model", "base_url"),
    [
        ("text-embedding-3-small", "https://api.openai.com/v1"),
        ("nomic-embed-text", "http://127.0.0.1:11434/v1"),
    ],
)
def test_real_embedding_values_clear_the_placeholder_check(tmp_path: Path, model: str, base_url: str) -> None:
    config = _filled(_noninteractive_init(tmp_path), model=model, base_url=base_url)

    check = next(item for item in run_doctor(config, check_port=False).checks if item.name == "setup_placeholders")
    assert check.status is DoctorStatus.PASS


@pytest.mark.parametrize(
    ("route", "placeholders"),
    [
        ({"model": "text-embedding-3-small"}, ("route.base_url",)),
        ({"base_url": "https://api.openai.com/v1"}, ("route.model",)),
    ],
)
def test_one_remaining_placeholder_is_still_reported(
    tmp_path: Path,
    route: dict[str, str],
    placeholders: tuple[str, ...],
) -> None:
    config = _filled(_noninteractive_init(tmp_path), **route)

    check = next(item for item in run_doctor(config, check_port=False).checks if item.name == "setup_placeholders")
    assert check.status is DoctorStatus.FAIL
    assert all(f"models.embedding.{path}" in check.detail.split(";")[0].split("(")[1] for path in placeholders)


def test_interactive_wizard_rejects_a_placeholder_and_asks_again(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profile = build_builtin_setup_registry().profile("embedding", "embedding.openai_compatible.custom")
    base_url = next(field for field in profile.fields if field.key == "base_url")
    answers = iter(
        ("https://embedding.example.com/v1", "https://{WorkspaceId}.vendor.test/v1", "https://api.openai.com/v1")
    )
    prompts: list[str] = []

    def fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return next(answers)

    monkeypatch.setattr(builtins, "input", fake_input)

    value = cli._prompt_setup_field(base_url, "https://embedding.example.com/v1")

    assert value == "https://api.openai.com/v1"
    assert len(prompts) == 3
    assert all("[" not in prompt for prompt in prompts), "the placeholder must not be offered as the default"
    assert capsys.readouterr().out.count("is a setup placeholder") == 2


@pytest.mark.parametrize(
    ("model", "base_url"),
    [
        ("text-embedding-3-small", "https://api.counterexample.ai/v1"),
        ("text-embedding-3-small", "https://embed.myexample.com/v1"),
        # example.com 作为中间标签、真实域名其实是 evil.vendor：子串匹配会误判，精确匹配不会
        ("text-embedding-3-small", "https://embedding.example.com.evil.vendor/v1"),
        # 模型名里含 .example.com：同理不能误判
        ("acme.example.com-v1", "https://api.openai.com/v1"),
    ],
)
def test_real_values_that_merely_contain_example_are_not_placeholders(
    tmp_path: Path, model: str, base_url: str
) -> None:
    """占位判据只认“等于出厂模板原值或留着 {var}”，不做子串匹配：真实的 URL 与模型名哪怕含 example.com 也不算占位。"""

    config = _filled(_noninteractive_init(tmp_path), model=model, base_url=base_url)

    check = next(item for item in run_doctor(config, check_port=False).checks if item.name == "setup_placeholders")
    assert check.status is DoctorStatus.PASS


def test_blank_credential_ref_survives_interactive_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    """本机无密钥服务把 credential_ref 置空后，交互式 init 重跑不能崩，回车要保留空值。"""

    profile = build_builtin_setup_registry().profile("embedding", "embedding.openai_compatible.custom")
    credential_ref = next(field for field in profile.fields if field.key == "credential_ref")
    monkeypatch.setattr(builtins, "input", lambda _prompt: "")

    assert cli._prompt_setup_field(credential_ref, "") == ""


@pytest.mark.parametrize("dimension", [True, "1024", 1024.0, 0])
def test_setup_values_reject_a_dimension_that_is_not_a_positive_integer(dimension: object) -> None:
    """向导或已有配置交来的 dimension 是 bool、文本、浮点或 0 时，物化前就拒（W16：判定收进公共函数后行为不变）。"""

    profile = build_builtin_setup_registry().profile("embedding", "embedding.openai_compatible.custom")
    with pytest.raises(ValueError, match="setup field must be a positive integer: dimension"):
        profile.materialize({"dimension": dimension}, field_keys=frozenset({"dimension"}))
