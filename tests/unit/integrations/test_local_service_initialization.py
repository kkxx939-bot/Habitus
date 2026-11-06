"""统一启动入口、配置初始化和 Harness 委托的产品边界测试。"""

from __future__ import annotations

import os
import stat
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from habitus.config import HabitusConfig
from habitus.integrations.local_service import cli
from habitus.integrations.local_service import initialization as initialization_module
from habitus.integrations.local_service.cloud_setup import (
    ProfileSelection,
    apply_cloud_selection,
    default_cloud_selection,
    selection_from_mapping,
)
from habitus.integrations.local_service.initialization import (
    configure_credentials,
    initialize_config,
    initialize_config_from_mapping,
    load_initialization_mapping,
    missing_credential_fields,
    resolve_config_path,
    resolve_plugin_connection_path,
    write_plugin_connection,
)
from habitus.integrations.local_service.setup_registry import (
    SetupProfile,
    build_builtin_setup_registry,
)
from habitus.runtime import build_runtime

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_CONFIG = REPOSITORY_ROOT / "habitus" / "config" / "example.yaml"


@pytest.fixture(autouse=True)
def _isolate_default_plugin_connection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        initialization_module,
        "DEFAULT_PLUGIN_CONNECTION_PATH",
        tmp_path / "default-home" / ".habitus" / "agent-plugin" / "connection.json",
    )


def test_config_path_resolution_has_one_explicit_environment_default_chain(tmp_path: Path) -> None:
    configured = tmp_path / "environment" / "config.yaml"
    explicit = tmp_path / "explicit" / "config.yaml"

    assert resolve_config_path(explicit, environ={"HABITUS_CONFIG_FILE": str(configured)}) == explicit
    assert resolve_config_path(None, environ={"HABITUS_CONFIG_FILE": str(configured)}) == configured
    assert resolve_config_path(None, environ={}).name == "config.yaml"

    with pytest.raises(ValueError, match="yaml"):
        resolve_config_path(tmp_path / "config.json", environ={})


def test_plugin_connection_path_honors_an_explicit_isolated_state_root(
    tmp_path: Path,
) -> None:
    state_root = tmp_path / "plugin-state"

    assert (
        resolve_plugin_connection_path(environ={"HABITUS_PLUGIN_STATE_DIR": str(state_root)})
        == state_root / "connection.json"
    )


def test_cli_always_binds_the_default_yaml_without_requiring_an_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HABITUS_CONFIG_FILE", raising=False)

    values = cli._environment(None)

    assert values["HABITUS_CONFIG_FILE"] == str(Path("~/.habitus/config.yaml").expanduser().absolute())


def test_initializer_writes_private_valid_config_without_overwriting_by_default(tmp_path: Path) -> None:
    destination = tmp_path / "config" / "habitus.yaml"

    created = initialize_config(destination, source=EXAMPLE_CONFIG)
    destination.chmod(0o644)
    replayed = initialize_config(destination, source=EXAMPLE_CONFIG)

    assert created.created is True
    assert replayed.created is False
    assert created.backup_path is None
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert HabitusConfig.from_file(destination).http.port == 8787


def test_force_initialization_preserves_previous_config_in_private_backup(tmp_path: Path) -> None:
    destination = tmp_path / "config" / "habitus.yaml"
    original = EXAMPLE_CONFIG.read_bytes()
    initialize_config(destination, source=EXAMPLE_CONFIG)
    destination.write_bytes(original.replace(b"port: 8787", b"port: 8799", 1))

    result = initialize_config(destination, source=EXAMPLE_CONFIG, force=True)

    assert result.backup_path == destination.with_name("habitus.bak.yaml")
    assert result.backup_path.read_bytes() != destination.read_bytes()
    assert HabitusConfig.from_file(result.backup_path).http.port == 8799
    assert stat.S_IMODE(result.backup_path.stat().st_mode) == 0o600


def test_initializer_refuses_to_write_an_out_of_range_override(tmp_path: Path) -> None:
    """值域归领域配置（配置层只查类型）：init 写配置前照旧拒越界值，报错带配置键，什么都不写。"""

    payload = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    payload["scene"] = {"relations": {"fdr": 1.0}}
    source = tmp_path / "source.yaml"
    source.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    destination = tmp_path / "habitus" / "config.yaml"

    with pytest.raises(ValueError, match="invalid 'config.scene.relations.fdr'"):
        initialize_config(destination, source=source)
    with pytest.raises(ValueError, match="invalid 'config.scene.relations.fdr'"):
        initialize_config_from_mapping(destination, payload)
    assert not destination.exists()


def test_initializer_rejects_a_symbolic_link_destination(tmp_path: Path) -> None:
    real = tmp_path / "real.yaml"
    real.write_bytes(EXAMPLE_CONFIG.read_bytes())
    linked = tmp_path / "linked.yaml"
    linked.symlink_to(real)

    with pytest.raises(OSError):
        initialize_config(linked, source=EXAMPLE_CONFIG, force=True)


def test_init_through_a_symbolic_link_directory_exits_with_one_line_naming_the_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """配置目录途经符号链接：拒绝写入，一行原因带被拒的目录，不打栈。"""

    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    monkeypatch.setattr(
        initialization_module,
        "DEFAULT_PLUGIN_CONNECTION_PATH",
        tmp_path / "home" / ".habitus" / "agent-plugin" / "connection.json",
    )

    with pytest.raises(SystemExit) as raised:
        cli.main(["init", "--config", str(link / "config.yaml"), "--non-interactive", "--skip-doctor", "--no-start"])

    message = str(raised.value.code)
    assert message.startswith("Habitus initialization failed: ")
    assert "\n" not in message and "Traceback" not in message
    assert "symbolic link" in message
    assert message.endswith(f": {link}")
    assert not (real / "config.yaml").exists()
    assert not (real / "agent-plugin").exists()


def test_init_with_plugin_state_through_a_symbolic_link_exits_with_one_line_naming_the_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """默认插件状态目录途经符号链接：连接文件那一步同样一行原因带目录。"""

    real = tmp_path / "real-home"
    real.mkdir()
    link = tmp_path / "home"
    link.symlink_to(real, target_is_directory=True)
    default_connection = link / ".habitus" / "agent-plugin" / "connection.json"
    monkeypatch.setattr(initialization_module, "DEFAULT_PLUGIN_CONNECTION_PATH", default_connection)
    destination = tmp_path / "custom" / "config.yaml"

    with pytest.raises(SystemExit) as raised:
        cli.main(["init", "--config", str(destination), "--non-interactive", "--skip-doctor", "--no-start"])

    message = str(raised.value.code)
    assert message.startswith("Habitus plugin connection initialization failed: ")
    assert "\n" not in message and "Traceback" not in message
    assert "symbolic link" in message
    assert message.endswith(f": {default_connection.parent}")
    assert not (real / ".habitus" / "agent-plugin" / "connection.json").exists()


def test_default_cloud_setup_generates_a_strict_config_without_placeholder_rerank() -> None:
    payload = apply_cloud_selection(
        load_initialization_mapping(),
        default_cloud_selection(),
    )

    config = HabitusConfig.from_mapping(payload)

    # Chat 与 Embedding 都不预置厂商：写出的是通用协议加占位值，等用户填
    assert config.models.chat.route.adapter == "openai_compatible_chat"
    assert config.models.chat.route.provider == "custom"
    assert config.models.chat.route.model == "example.chat-model"
    assert config.models.chat.route.base_url == "https://chat.example.com/v1"
    assert config.models.chat.route.credential_ref == "chat"
    assert config.models.embedding.route.adapter == "openai_compatible_embedding"
    assert config.models.embedding.route.provider == "custom"
    assert config.models.embedding.route.model == "example.embedding-model"
    assert config.models.embedding.route.base_url == "https://embedding.example.com/v1"
    assert config.models.embedding.route.credential_ref == "embedding"
    assert config.models.rerank is None
    # 向量库默认是本地文件：两段都走 local_sqlite，不引用任何凭据
    for vector in (config.memory.vector_store, config.conversation.summary_vector_store):
        assert (vector.provider, vector.adapter, vector.route.credential_ref) == ("local", "local_sqlite", "")
    assert config.credentials.names == ("chat", "dashscope", "embedding")


def test_cloud_setup_applies_custom_routes_with_the_local_vector_store(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "cloud" / "config.yaml"
    defaults = default_cloud_selection()
    selection = replace(
        defaults,
        chat=ProfileSelection(
            "chat.openai_compatible.custom",
            {
                "provider": "custom-chat",
                "model": "chat-model",
                "base_url": "https://chat.vendor.test/v1",
                "credential_ref": "custom_chat",
                "context_window_tokens": 128_000,
                "structured_output_mode": "json_schema",
                "reasoning": False,
            },
        ),
        embedding=ProfileSelection(
            "embedding.openai_compatible.custom",
            {
                "provider": "custom-embedding",
                "model": "embedding-model",
                "base_url": "https://embedding.vendor.test/v1",
                "credential_ref": "custom_embedding",
                "dimension": 2048,
            },
        ),
        rerank=ProfileSelection(
            "rerank.aliyun.qwen3",
            {"base_url": "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-api/v1"},
        ),
        vector=ProfileSelection("vector.local_sqlite"),
    )
    payload = apply_cloud_selection(load_initialization_mapping(), selection)

    result = initialize_config_from_mapping(destination, payload)
    config = HabitusConfig.from_file(destination)

    assert result.created is True
    assert config.models.chat.route.provider == "custom-chat"
    assert config.models.chat.context_window_tokens == 128_000
    assert config.models.chat.structured_output_mode == "json_schema"
    assert config.models.embedding.dimension == 2048
    assert config.models.rerank is not None
    assert config.models.rerank.route.base_url.startswith("https://workspace.")
    assert config.memory.vector_store.adapter == "local_sqlite"
    assert config.memory.vector_store.collection == "memory"
    assert config.conversation.summary_vector_store.collection == "conversation_summaries"
    assert config.models.embedding.route.base_url == "https://embedding.vendor.test/v1"
    assert config.credentials.names == (
        "chat",
        "custom_chat",
        "custom_embedding",
        "dashscope",
        "embedding",
    )
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600


def test_cloud_setup_preserves_an_unselected_nonempty_credential() -> None:
    payload = load_initialization_mapping()
    payload["credentials"]["dashscope"]["api_key"] = "existing-rerank-secret"  # type: ignore[index]

    configured = apply_cloud_selection(payload, default_cloud_selection())
    config = HabitusConfig.from_mapping(configured)

    assert config.models.rerank is None
    assert config.credentials.resolve("dashscope")["api_key"] == "existing-rerank-secret"


def test_existing_registered_profiles_round_trip_without_losing_advanced_fields() -> None:
    payload = load_initialization_mapping()
    payload["models"]["chat"]["route"]["extra_headers"] = {"X-Trace": "chat"}  # type: ignore[index]
    payload["models"]["chat"]["route"]["extra_body"] = {"temperature": 0.2}  # type: ignore[index]
    payload["models"]["chat"]["max_output_tokens"] = 2048  # type: ignore[index]
    payload["models"]["chat"]["reasoning"] = True  # type: ignore[index]
    payload["models"]["embedding"]["route"]["extra_headers"] = {"X-Trace": "embedding"}  # type: ignore[index]
    payload["models"]["embedding"]["route"]["extra_body"] = {"encoding": "dense"}  # type: ignore[index]
    payload["models"]["embedding"]["query_parameters"] = {"instruction": "query"}  # type: ignore[index]
    payload["models"]["embedding"]["document_parameters"] = {"instruction": "document"}  # type: ignore[index]
    payload["models"]["rerank"]["route"]["extra_headers"] = {"X-Trace": "rerank"}  # type: ignore[index]
    payload["conversation"]["summary_vector_store"]["options"]["busy_timeout_seconds"] = 9.0  # type: ignore[index]
    original_models = yaml.safe_load(yaml.safe_dump(payload["models"]))
    original_memory_vector = yaml.safe_load(yaml.safe_dump(payload["memory"]["vector_store"]))  # type: ignore[index]
    original_summary_vector = yaml.safe_load(
        yaml.safe_dump(payload["conversation"]["summary_vector_store"])  # type: ignore[index]
    )

    configured = apply_cloud_selection(payload, selection_from_mapping(payload))

    assert configured["models"] == original_models
    assert configured["memory"]["vector_store"] == original_memory_vector  # type: ignore[index]
    assert configured["conversation"]["summary_vector_store"] == original_summary_vector  # type: ignore[index]


def test_cloud_setup_adds_required_fields_without_deleting_optional_credentials() -> None:
    payload = load_initialization_mapping()
    payload["credentials"]["embedding"]["session_token"] = "temporary-token"  # type: ignore[index]
    payload["credentials"]["tracing"] = {"authorization": ""}  # type: ignore[index]
    payload["observability"]["tracing"].update(  # type: ignore[index]
        {"enabled": True, "credential_ref": "tracing"}
    )

    configured = apply_cloud_selection(payload, default_cloud_selection())

    assert configured["credentials"]["embedding"]["session_token"] == "temporary-token"  # type: ignore[index]
    assert configured["credentials"]["tracing"] == {"authorization": ""}  # type: ignore[index]


def test_switching_vector_profile_replaces_adapter_options_instead_of_retaining_unknowns() -> None:
    payload = load_initialization_mapping()
    payload["memory"]["vector_store"]["options"]["legacy_option"] = True  # type: ignore[index]
    payload["conversation"]["summary_vector_store"]["options"]["legacy_option"] = True  # type: ignore[index]
    defaults = default_cloud_selection()
    selection = replace(
        defaults,
        vector=ProfileSelection("vector.local_sqlite"),
    )

    configured = apply_cloud_selection(payload, selection)

    assert "legacy_option" not in configured["memory"]["vector_store"]["options"]  # type: ignore[index]
    assert "legacy_option" not in configured["conversation"]["summary_vector_store"]["options"]  # type: ignore[index]


def test_default_local_vector_configuration_builds_the_runtime_without_any_vector_account(
    tmp_path: Path,
) -> None:
    payload = apply_cloud_selection(load_initialization_mapping(), default_cloud_selection())
    payload["storage"]["root"] = str(tmp_path / "data")  # type: ignore[index]
    payload["credentials"]["chat"]["api_key"] = "chat"  # type: ignore[index]
    payload["credentials"]["embedding"]["api_key"] = "embedding"  # type: ignore[index]

    runtime = build_runtime(HabitusConfig.from_mapping(payload))

    store = runtime.components.memory.vector_index.store
    assert (store.adapter_name, store.collection) == ("local_sqlite", "memory")
    assert runtime.components.conversation.summary_vector_index.store.collection == "conversation_summaries"
    assert not (tmp_path / "data" / "vector").exists(), "组装阶段不得落盘"


def test_setup_round_trip_uses_normalized_route_identity_and_credential_reference() -> None:
    payload = load_initialization_mapping()
    payload["models"]["chat"]["route"]["adapter"] = " OPENAI_COMPATIBLE_CHAT "  # type: ignore[index]
    payload["models"]["chat"]["route"]["credential_ref"] = " Chat "  # type: ignore[index]

    configured = apply_cloud_selection(payload, selection_from_mapping(payload))
    config = HabitusConfig.from_mapping(configured)

    assert config.models.chat.route.adapter == "openai_compatible_chat"
    assert config.models.chat.route.credential_ref == "chat"
    assert " Chat " not in configured["credentials"]  # type: ignore[operator]


def test_switching_from_large_custom_chat_to_deepseek_clears_incompatible_output_limit() -> None:
    payload = load_initialization_mapping()
    payload["models"]["chat"]["route"].update(  # type: ignore[index]
        {
            "provider": "custom",
            "model": "big-chat",
            "base_url": "https://large-chat.example.com/v1",
            "credential_ref": "custom_chat",
        }
    )
    payload["models"]["chat"]["context_window_tokens"] = 200_000  # type: ignore[index]
    payload["models"]["chat"]["max_output_tokens"] = 100_000  # type: ignore[index]
    payload["credentials"]["custom_chat"] = {"api_key": ""}  # type: ignore[index]
    current = selection_from_mapping(payload)

    configured = apply_cloud_selection(
        payload,
        replace(current, chat=ProfileSelection("chat.deepseek")),
    )

    assert configured["models"]["chat"].get("max_output_tokens") is None  # type: ignore[index,union-attr]


def test_profile_identification_prefers_the_most_specific_match_independent_of_order() -> None:
    registry = build_builtin_setup_registry()
    registry.register_profile(
        SetupProfile(
            "chat.deepseek.reasoner",
            "chat",
            "DeepSeek reasoner",
            patch={},
            match={
                "route.adapter": "openai_compatible_chat",
                "route.provider": "deepseek",
                "route.model": "deepseek-reasoner",
            },
        )
    )
    document = {
        "route": {
            "adapter": "openai_compatible_chat",
            "provider": "deepseek",
            "model": "deepseek-reasoner",
        }
    }

    assert registry.identify("chat", document).profile_id == "chat.deepseek.reasoner"


def test_any_registry_profile_can_declare_the_rerank_capability_disabled() -> None:
    registry = build_builtin_setup_registry()
    registry.register_profile(
        SetupProfile(
            "rerank.offline",
            "rerank",
            "Disabled by policy",
            patch={"enabled": False},
            match={"enabled": False},
            target_state="disabled",
        )
    )
    selection = replace(
        default_cloud_selection(registry),
        rerank=ProfileSelection("rerank.offline"),
    )

    configured = apply_cloud_selection(
        load_initialization_mapping(),
        selection,
        registry,
    )

    assert configured["models"]["rerank"] is None  # type: ignore[index]


def test_mapping_initializer_rejects_an_oversized_config_before_writing(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "oversized" / "config.yaml"
    payload = load_initialization_mapping()
    payload["models"]["chat"]["route"]["extra_body"] = {  # type: ignore[index]
        "large": "x" * (1024 * 1024)
    }

    with pytest.raises(ValueError, match="one-megabyte"):
        initialize_config_from_mapping(destination, payload)

    assert not destination.exists()


def _real_embedding_answers(label: str, default: str | None, **_kwargs: object) -> str:
    """交互式 init 里 Chat 与 Embedding 的必填项：用户填的是真实端点与模型（向导模板只有占位值）。"""

    assert default is None, f"placeholder must not be offered as the default for {label}"
    return {
        "Chat model": "chat-model-1",
        "Chat API base URL": "https://api.openai.com/v1",
        "Embedding model": "text-embedding-3-small",
        "Embedding API base URL": "https://api.openai.com/v1",
    }[label]


def test_cloud_wizard_default_path_selects_cloud_only_routes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # 每个菜单都直接回车：取向导标出的默认项
    monkeypatch.setattr(cli, "_prompt_choice", lambda *_args, default, **_kwargs: default)
    monkeypatch.setattr(cli, "_prompt_text", lambda _prompt, default: default)
    monkeypatch.setattr(cli, "_prompt_positive_int", lambda _prompt, default: default)
    monkeypatch.setattr(cli, "_prompt_required_text", _real_embedding_answers)
    monkeypatch.setattr(cli, "_confirm", Mock(return_value=True))

    selection = cli._prompt_cloud_setup(default_cloud_selection())

    assert selection is not None
    assert selection.chat.profile_id == "chat.openai_compatible.custom"
    assert selection.chat.values["model"] == "chat-model-1"
    assert selection.chat.values["base_url"] == "https://api.openai.com/v1"
    assert selection.chat.values["credential_ref"] == "chat"
    assert selection.embedding.profile_id == "embedding.openai_compatible.custom"
    assert selection.embedding.values["model"] == "text-embedding-3-small"
    assert selection.embedding.values["base_url"] == "https://api.openai.com/v1"
    assert selection.embedding.values["credential_ref"] == "embedding"
    assert selection.embedding.values["dimension"] == 1024
    assert selection.rerank.profile_id == "rerank.disabled"
    assert selection.vector.profile_id == "vector.local_sqlite"
    output = capsys.readouterr().out
    assert "Local Chat, Embedding and Rerank are intentionally not included" in output
    assert "Rerank: Disabled (vector relevance fallback)" in output


def test_cloud_wizard_requires_a_real_alibaba_workspace_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    choices = iter((1, 1, 2, 1, 1))
    workspace = "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-api/v1"
    required_text = Mock(
        side_effect=lambda label, default, **_kwargs: (
            workspace if label.startswith("Alibaba") else _real_embedding_answers(label, default)
        )
    )
    monkeypatch.setattr(
        cli,
        "_prompt_choice",
        lambda *_args, **_kwargs: next(choices),
    )
    monkeypatch.setattr(cli, "_prompt_text", lambda _prompt, default: default)
    monkeypatch.setattr(cli, "_prompt_required_text", required_text)
    monkeypatch.setattr(cli, "_confirm", Mock(return_value=True))

    selection = cli._prompt_cloud_setup(default_cloud_selection())

    assert selection is not None
    assert selection.rerank.profile_id == "rerank.aliyun.qwen3"
    assert str(selection.rerank.values["base_url"]).startswith("https://workspace.")
    required_text.assert_any_call("Alibaba Workspace API base URL", None, placeholder_values=())


def test_cloud_wizard_cancellation_returns_no_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    choices = iter((1, 1, 1, 1, 1))
    monkeypatch.setattr(
        cli,
        "_prompt_choice",
        lambda *_args, **_kwargs: next(choices),
    )
    monkeypatch.setattr(cli, "_prompt_text", lambda _prompt, default: default)
    monkeypatch.setattr(cli, "_prompt_required_text", _real_embedding_answers)
    monkeypatch.setattr(cli, "_confirm", Mock(return_value=False))

    assert cli._prompt_cloud_setup(default_cloud_selection()) is None


def test_initializer_updates_multiple_named_credentials_in_the_same_private_yaml(tmp_path: Path) -> None:
    destination = tmp_path / "config" / "habitus.yaml"
    initialize_config(destination, source=EXAMPLE_CONFIG)

    assert {(item.reference, item.field) for item in missing_credential_fields(destination)} == {
        ("chat", "api_key"),
        ("dashscope", "api_key"),
        ("embedding", "api_key"),
    }
    config = configure_credentials(
        destination,
        {
            "chat": {"api_key": "chat-secret"},
            "embedding": {"api_key": "embedding-secret"},
            "dashscope": {"api_key": "dashscope-secret"},
        },
    )

    assert missing_credential_fields(destination) == ()
    assert config.credentials.resolve("chat")["api_key"] == "chat-secret"
    assert config.credentials.resolve("dashscope")["api_key"] == "dashscope-secret"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600


def test_credential_updates_follow_the_same_normalized_name_rules_as_config_loading(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "config" / "habitus.yaml"
    payload = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    payload["credentials"][" Chat "] = payload["credentials"].pop("chat")
    payload["credentials"][" Chat "][" API_KEY "] = payload["credentials"][" Chat "].pop("api_key")
    destination.parent.mkdir(parents=True)
    destination.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    destination.chmod(0o600)

    config = configure_credentials(
        destination,
        {"chat": {"api_key": "chat-secret"}},
    )

    assert config.credentials.resolve("chat")["api_key"] == "chat-secret"


def test_interactive_initializer_prompts_each_missing_credential_without_echoing_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "config" / "habitus.yaml"
    initialize_config(destination, source=EXAMPLE_CONFIG)
    secrets = {
        "embedding.api_key": "embedding-secret",
        "dashscope.api_key": "dashscope-secret",
        "chat.api_key": "chat-secret",
    }
    monkeypatch.setattr(cli, "_confirm", Mock(return_value=True))
    monkeypatch.setattr(cli, "_prompt_secret", lambda label: secrets[label])

    cli._configure_missing_credentials(destination)

    config = HabitusConfig.from_file(destination)
    assert config.credentials.resolve("embedding")["api_key"] == "embedding-secret"
    assert missing_credential_fields(destination) == ()
    output = capsys.readouterr().out
    assert "embedding-secret" not in output
    assert "chat-secret" not in output


def test_noninteractive_init_hands_harness_installs_to_the_npm_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """插件装在 npm 包里（方案 36）：装了 ``habitus`` 命令就替用户跑 ``habitus plugin install``，宿主名原样转手；没装、
    或 PATH 上的 ``habitus`` 是早先 pip 留下的 Python 入口，就说怎么装，安装提示只打一遍。"""

    destination = tmp_path / "config" / "habitus.yaml"
    npm_command = tmp_path / "bin" / "habitus.mjs"
    npm_command.parent.mkdir()
    npm_command.write_text("#!/usr/bin/env node\n")
    stale_command = tmp_path / "bin" / "habitus"
    stale_command.write_text("#!/opt/anaconda3/bin/python\nfrom integrations.local_service.cli import main\n")
    run = Mock(return_value=Mock(returncode=0))
    monkeypatch.setattr(cli.shutil, "which", Mock(return_value=str(npm_command)))
    monkeypatch.setattr(cli.subprocess, "run", run)
    arguments = [
        "init",
        "--config",
        str(destination),
        "--from-config",
        str(EXAMPLE_CONFIG),
        "--non-interactive",
        "--skip-doctor",
        "--no-start",
        "--harness",
        "codex",
        "--harness",
        "future-harness",
    ]

    cli.main(arguments)

    run.assert_called_once_with(
        [str(npm_command), "plugin", "install", "--harness", "codex", "--harness", "future-harness"],
        check=False,
    )
    assert destination.is_file()
    assert "npm install" not in capsys.readouterr().out
    for which in (None, str(stale_command)):
        monkeypatch.setattr(cli.shutil, "which", Mock(return_value=which))
        cli.main([*arguments, "--force"])
        assert capsys.readouterr().out.count("npm install -g habitus") == 1
    assert run.call_count == 1


def test_noninteractive_default_init_uses_valid_cloud_defaults_without_placeholder_rerank(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "default" / "config.yaml"

    cli.main(
        [
            "init",
            "--config",
            str(destination),
            "--non-interactive",
            "--skip-doctor",
            "--no-start",
        ]
    )

    assert HabitusConfig.from_file(destination).models.rerank is None
    connection = destination.parent / "agent-plugin" / "connection.json"
    assert '"base_url": "http://127.0.0.1:8787"' in connection.read_text(encoding="utf-8")


def test_custom_config_init_persists_plugin_connection_in_the_default_state_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "custom" / "config.yaml"
    default_connection = tmp_path / "home" / ".habitus" / "agent-plugin" / "connection.json"
    monkeypatch.setattr(
        initialization_module,
        "DEFAULT_PLUGIN_CONNECTION_PATH",
        default_connection,
    )

    cli.main(
        [
            "init",
            "--config",
            str(destination),
            "--non-interactive",
            "--skip-doctor",
            "--no-start",
        ]
    )

    assert default_connection.is_file()
    assert '"base_url": "http://127.0.0.1:8787"' in default_connection.read_text(encoding="utf-8")


def test_interactive_eof_cancels_before_writing_configuration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "cancelled" / "config.yaml"
    monkeypatch.setattr(cli, "_interactive", Mock(return_value=True))
    monkeypatch.setattr("builtins.input", Mock(side_effect=EOFError))

    cli.main(["init", "--config", str(destination), "--skip-doctor", "--no-start"])

    assert not destination.exists()


def test_missing_config_offer_handles_eof_without_leaking_wizard_exception(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "missing" / "config.yaml"
    monkeypatch.setattr(cli, "_interactive", Mock(return_value=True))
    monkeypatch.setattr("builtins.input", Mock(side_effect=EOFError))

    handled = cli._maybe_offer_init(str(destination), {})

    assert handled is False
    assert not destination.exists()


def test_plugin_connection_projection_uses_the_configured_loopback_port(tmp_path: Path) -> None:
    payload = load_initialization_mapping()
    payload["http"]["port"] = 8899  # type: ignore[index]
    config = HabitusConfig.from_mapping(payload)

    path = write_plugin_connection(config, tmp_path / "agent-plugin" / "connection.json")

    assert path.stat().st_mode & 0o777 == 0o600
    assert '"base_url": "http://127.0.0.1:8899"' in path.read_text(encoding="utf-8")


def test_builtin_setup_registry_rejects_duplicate_profile_registration() -> None:
    registry = build_builtin_setup_registry()
    profile = registry.profile("chat", "chat.deepseek")

    with pytest.raises(ValueError, match="already registered"):
        registry.register_profile(profile)


def test_server_release_extra_closes_the_first_use_dependency_set() -> None:
    source = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    server = source.split("server = [", maxsplit=1)[1].split("]", maxsplit=1)[0]

    for dependency in ("fastapi", "uvicorn", "json-repair", "jsonschema"):
        assert dependency in server
    # 分组 11-06 由 cloud 改名为 server：向量库已不在云上，旧名不留别名
    assert "\ncloud = [" not in source
    # 默认向量库是本地后端，它要的 numpy 在基础依赖里；云上向量库的 SDK 不再是任何分组的依赖
    base = source.split("dependencies = [", maxsplit=1)[1].split("]", maxsplit=1)[0]
    assert '"numpy' in base
    assert "volcengine" not in source


def test_interactive_init_uses_cloud_wizard_before_prompting_selected_credentials(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "interactive" / "config.yaml"
    setup = default_cloud_selection()
    prompt = Mock(return_value=setup)
    credentials = Mock()
    monkeypatch.setattr(cli, "_interactive", Mock(return_value=True))
    monkeypatch.setattr(cli, "_prompt_cloud_setup", prompt)
    monkeypatch.setattr(cli, "_configure_missing_credentials", credentials)
    monkeypatch.setattr(cli, "_confirm", Mock(return_value=False))
    monkeypatch.setattr(cli, "_prompt_text", lambda _prompt, default: default)  # W3：新建配置时多一问时区，回车接受

    cli.main(
        [
            "init",
            "--config",
            str(destination),
            "--skip-doctor",
            "--no-start",
        ]
    )

    config = HabitusConfig.from_file(destination)
    assert config.models.rerank is None
    prompt.assert_called_once()
    credentials.assert_called_once_with(destination)


def test_interactive_cloud_update_preserves_existing_config_in_backup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "interactive" / "config.yaml"
    initialize_config(destination, source=EXAMPLE_CONFIG)
    original = destination.read_bytes().replace(b"port: 8787", b"port: 8799", 1)
    destination.write_bytes(original)
    destination.chmod(0o600)
    confirmations = iter((True, False))
    monkeypatch.setattr(cli, "_interactive", Mock(return_value=True))
    monkeypatch.setattr(cli, "_prompt_cloud_setup", Mock(return_value=default_cloud_selection()))
    monkeypatch.setattr(cli, "_configure_missing_credentials", Mock())
    monkeypatch.setattr(cli, "_confirm", lambda *_args, **_kwargs: next(confirmations))

    cli.main(
        [
            "init",
            "--config",
            str(destination),
            "--skip-doctor",
            "--no-start",
        ]
    )

    backup = destination.with_name("config.bak.yaml")
    assert HabitusConfig.from_file(destination).http.port == 8799
    assert HabitusConfig.from_file(destination).models.rerank is None
    assert HabitusConfig.from_file(backup).models.rerank is not None


def test_init_execs_the_single_server_entrypoint_when_start_is_requested(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "config" / "habitus.yaml"
    execute = Mock()
    monkeypatch.setattr(cli, "_exec_serve", execute)

    cli.main(
        [
            "init",
            "--config",
            str(destination),
            "--from-config",
            str(EXAMPLE_CONFIG),
            "--non-interactive",
            "--skip-doctor",
            "--start",
        ]
    )

    execute.assert_called_once_with(destination)


def test_missing_config_offer_delegates_to_init_only_in_an_interactive_terminal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "missing" / "habitus.yaml"
    initialize = Mock(return_value=0)
    monkeypatch.setattr(cli, "_interactive", Mock(return_value=True))
    monkeypatch.setattr(cli, "_confirm", Mock(return_value=True))
    monkeypatch.setattr(cli, "_initialize", initialize)

    handled = cli._maybe_offer_init(str(destination), {})

    assert handled is True
    initialize.assert_called_once()
    assert initialize.call_args.args[0].config == str(destination)


def test_exec_serve_replaces_process_with_module_bootstrap(monkeypatch: pytest.MonkeyPatch) -> None:
    replace = Mock()
    monkeypatch.setattr(os, "execv", replace)
    config = Path("/tmp/habitus-test/config.yaml")

    cli._exec_serve(config)

    executable, arguments = replace.call_args.args
    assert executable == cli.sys.executable
    assert arguments == [
        cli.sys.executable,
        "-m",
        "habitus.integrations.local_service.cli",
        "serve",
        "--config",
        str(config),
    ]


def test_init_keeps_the_permissions_of_an_existing_config_directory(tmp_path: Path) -> None:
    """配置放进用户已有的目录（如 ~/Documents，0755）：init 不改它的权限；只有自己新建的 agent-plugin/ 是 0700。"""

    documents = tmp_path / "Documents"
    documents.mkdir()
    documents.chmod(0o755)
    (documents / "notes.txt").write_text("mine\n", encoding="utf-8")
    destination = documents / "habitus.yaml"

    cli.main(["init", "--config", str(destination), "--non-interactive", "--skip-doctor", "--no-start"])

    assert stat.S_IMODE(documents.stat().st_mode) == 0o755
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert stat.S_IMODE((documents / "agent-plugin").stat().st_mode) == 0o700


def test_noninteractive_init_with_empty_credentials_says_where_to_fill_them(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "config" / "config.yaml"

    cli.main(["init", "--config", str(destination), "--non-interactive", "--skip-doctor", "--no-start"])

    output = capsys.readouterr().out
    assert "Credentials are empty" in output
    assert f"under 'credentials' in {destination}" in output
    assert f"habitus-server doctor --config {destination}" in output


def test_init_after_a_failed_doctor_gives_the_next_step_and_does_not_offer_plugins(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "config" / "habitus.yaml"
    confirm = Mock(return_value=False)
    install = Mock()
    monkeypatch.setattr(cli, "_interactive", Mock(return_value=True))
    monkeypatch.setattr(cli, "_configure_missing_credentials", Mock())
    monkeypatch.setattr(cli, "_doctor", Mock(return_value=1))
    monkeypatch.setattr(cli, "_confirm_after_write", confirm)
    monkeypatch.setattr(cli, "_install_plugins", install)

    with pytest.raises(SystemExit) as raised:
        cli.main(["init", "--config", str(destination), "--from-config", str(EXAMPLE_CONFIG), "--doctor"])

    assert raised.value.code == 1
    output = capsys.readouterr().out
    assert (
        f"Next: fix the [FAIL] items above (config file: {destination}), "
        f"then run: habitus-server doctor --config {destination}"
    ) in output
    assert confirm.call_count == 0
    install.assert_not_called()


def test_init_all_harnesses_hands_all_to_the_plugin_installer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    install = Mock()
    monkeypatch.setattr(cli, "_install_plugins", install)

    cli.main(
        [
            "init",
            "--config",
            str(tmp_path / "config" / "habitus.yaml"),
            "--from-config",
            str(EXAMPLE_CONFIG),
            "--non-interactive",
            "--skip-doctor",
            "--no-start",
            "--all-harnesses",
        ]
    )

    install.assert_called_once_with(["all"])


def test_plugin_connection_brackets_an_ipv6_loopback_host(tmp_path: Path) -> None:
    payload = load_initialization_mapping()
    payload["http"]["host"] = "::1"  # type: ignore[index]
    config = HabitusConfig.from_mapping(payload)

    path = write_plugin_connection(config, tmp_path / "agent-plugin" / "connection.json")

    assert '"base_url": "http://[::1]:8787"' in path.read_text(encoding="utf-8")


def test_plugin_connection_keeps_the_permissions_of_an_existing_directory(tmp_path: Path) -> None:
    """连接文件写进用户已有的目录（如 0755 的 ~/.config/habitus）：只写文件本身 0600，不收紧那个目录。"""

    existing = tmp_path / "habitus-config"
    existing.mkdir()
    existing.chmod(0o755)
    config = HabitusConfig.from_mapping(load_initialization_mapping())

    path = write_plugin_connection(config, existing / "connection.json")

    assert stat.S_IMODE(existing.stat().st_mode) == 0o755
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_init_exits_with_the_plugin_installer_exit_code_when_it_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """``habitus plugin install`` 失败（非零退出码）时 init 以同一个退出码结束，不当作装好了。"""

    npm_command = tmp_path / "bin" / "habitus.mjs"
    npm_command.parent.mkdir()
    npm_command.write_text("#!/usr/bin/env node\n")
    monkeypatch.setattr(cli.shutil, "which", Mock(return_value=str(npm_command)))
    monkeypatch.setattr(cli.subprocess, "run", Mock(return_value=Mock(returncode=3)))
    execute = Mock()
    monkeypatch.setattr(cli, "_exec_serve", execute)

    with pytest.raises(SystemExit) as raised:
        cli.main(
            [
                "init",
                "--config",
                str(tmp_path / "config" / "habitus.yaml"),
                "--from-config",
                str(EXAMPLE_CONFIG),
                "--non-interactive",
                "--skip-doctor",
                "--start",
                "--harness",
                "codex",
            ]
        )

    assert raised.value.code == 3
    execute.assert_not_called()


def test_init_does_not_start_the_server_after_a_failed_doctor_even_when_start_is_requested(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    execute = Mock()
    monkeypatch.setattr(cli, "_doctor", Mock(return_value=1))
    monkeypatch.setattr(cli, "_exec_serve", execute)

    with pytest.raises(SystemExit) as raised:
        cli.main(
            [
                "init",
                "--config",
                str(tmp_path / "config" / "habitus.yaml"),
                "--from-config",
                str(EXAMPLE_CONFIG),
                "--non-interactive",
                "--doctor",
                "--start",
            ]
        )

    assert raised.value.code == 1
    execute.assert_not_called()
    assert "Habitus server was not started because Doctor reported failures." in capsys.readouterr().err


def test_config_and_plugin_state_paths_directly_under_the_filesystem_root_are_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dedicated child directory"):
        resolve_config_path(Path("/habitus.yaml"))
    with pytest.raises(ValueError, match="dedicated child directory"):
        resolve_plugin_connection_path(environ={"HABITUS_PLUGIN_STATE_DIR": "/"})
    assert resolve_plugin_connection_path(environ={"HABITUS_PLUGIN_STATE_DIR": str(tmp_path)}) == (
        tmp_path / "connection.json"
    )


def test_mapping_initializer_repairs_the_permissions_of_an_existing_config_it_keeps(tmp_path: Path) -> None:
    """不加 --force 时沿用已有配置，但它若被放宽到组或其他人可读，就收回成 0600（配置里有密钥），内容一字不动。"""

    destination = tmp_path / "config" / "habitus.yaml"
    payload = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    initialize_config_from_mapping(destination, payload)
    original = destination.read_bytes()
    destination.chmod(0o644)

    result = initialize_config_from_mapping(destination, {**payload, "http": {**payload["http"], "port": 9999}})

    assert result.created is False
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert destination.read_bytes() == original
