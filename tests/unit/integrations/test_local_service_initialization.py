"""统一启动入口、配置初始化和 Harness 委托的产品边界测试。"""

from __future__ import annotations

import os
import stat
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from Config import HabitusConfig
from integrations.local_service import cli, plugin_cli
from integrations.local_service import initialization as initialization_module
from integrations.local_service.cloud_setup import (
    ProfileSelection,
    apply_cloud_selection,
    default_cloud_selection,
    selection_from_mapping,
)
from integrations.local_service.initialization import (
    configure_credentials,
    initialize_config,
    initialize_config_from_mapping,
    load_initialization_mapping,
    missing_credential_fields,
    resolve_config_path,
    resolve_plugin_connection_path,
    write_plugin_connection,
)
from integrations.local_service.setup_registry import (
    SetupProfile,
    build_builtin_setup_registry,
)
from Runtime import build_runtime

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_CONFIG = REPOSITORY_ROOT / "Config" / "example.yaml"


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

    assert resolve_plugin_connection_path(
        environ={"HABITUS_PLUGIN_STATE_DIR": str(state_root)}
    ) == state_root / "connection.json"


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


def test_initializer_rejects_a_symbolic_link_destination(tmp_path: Path) -> None:
    real = tmp_path / "real.yaml"
    real.write_bytes(EXAMPLE_CONFIG.read_bytes())
    linked = tmp_path / "linked.yaml"
    linked.symlink_to(real)

    with pytest.raises(OSError):
        initialize_config(linked, source=EXAMPLE_CONFIG, force=True)


def test_default_cloud_setup_generates_a_strict_config_without_placeholder_rerank() -> None:
    payload = apply_cloud_selection(
        load_initialization_mapping(),
        default_cloud_selection(),
    )

    config = HabitusConfig.from_mapping(payload)

    assert config.models.chat.route.provider == "deepseek"
    assert config.models.embedding.route.adapter == "ark_multimodal"
    assert config.models.rerank is None
    assert config.credentials.names == ("ark", "dashscope", "deepseek", "vikingdb")


def test_cloud_setup_applies_custom_routes_and_shared_vikingdb_resources(
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
                "base_url": "https://chat.example.com/v1",
                "credential_ref": "custom_chat",
                "context_window_tokens": 128_000,
                "structured_output_mode": "json_schema",
                "reasoning": False,
            },
        ),
        embedding=ProfileSelection(
            "embedding.ark_multimodal.custom",
            {
                "provider": "custom-ark",
                "model": "embedding-model",
                "base_url": "https://embedding.example.com/api/v3",
                "credential_ref": "custom_embedding",
                "dimension": 2048,
            },
        ),
        rerank=ProfileSelection(
            "rerank.aliyun.qwen3",
            {
                "base_url": "https://workspace.cn-beijing.maas.aliyuncs.com/compatible-api/v1"
            },
        ),
        vector=ProfileSelection(
            "vector.vikingdb.managed",
            {
                "region": "cn-shanghai",
                "project_name": "habitus-project",
                "index_name": "habitus-index",
                "memory_collection": "memory-main",
                "summary_collection": "summary-main",
            },
        ),
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
    assert config.memory.vector_store.collection == "memory-main"
    assert config.conversation.summary_vector_store.collection == "summary-main"
    assert config.memory.vector_store.options["region"] == "cn-shanghai"
    assert config.credentials.names == (
        "ark",
        "custom_chat",
        "custom_embedding",
        "dashscope",
        "deepseek",
        "vikingdb",
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
    payload["conversation"]["summary_vector_store"]["options"]["region"] = "cn-shanghai"  # type: ignore[index]
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
    payload["credentials"]["vikingdb"]["session_token"] = "temporary-token"  # type: ignore[index]
    payload["credentials"]["tracing"] = {"authorization": ""}  # type: ignore[index]
    payload["observability"]["tracing"].update(  # type: ignore[index]
        {"enabled": True, "credential_ref": "tracing"}
    )

    configured = apply_cloud_selection(payload, default_cloud_selection())

    assert configured["credentials"]["vikingdb"]["session_token"] == "temporary-token"  # type: ignore[index]
    assert configured["credentials"]["tracing"] == {"authorization": ""}  # type: ignore[index]


def test_setup_planner_rejects_embedding_and_vector_capacity_mismatch() -> None:
    defaults = default_cloud_selection()
    selection = replace(
        defaults,
        embedding=ProfileSelection(
            "embedding.ark_multimodal.custom",
            {
                "provider": "custom-ark",
                "model": "large-embedding",
                "base_url": "https://embedding.example.com/api/v3",
                "credential_ref": "custom_embedding",
                "dimension": 65_536,
            },
        ),
    )

    with pytest.raises(ValueError, match="read page sizes"):
        apply_cloud_selection(load_initialization_mapping(), selection)


def test_switching_vector_profile_replaces_adapter_options_instead_of_retaining_unknowns() -> None:
    payload = load_initialization_mapping()
    payload["memory"]["vector_store"]["options"]["legacy_option"] = True  # type: ignore[index]
    payload["conversation"]["summary_vector_store"]["options"]["legacy_option"] = True  # type: ignore[index]
    defaults = default_cloud_selection()
    selection = replace(
        defaults,
        vector=ProfileSelection(
            "vector.vikingdb.precreated",
            {
                "region": "cn-beijing",
                "project_name": "default",
                "index_name": "default",
                "memory_collection": "memory",
                "summary_collection": "conversation_summaries",
            },
        ),
    )

    configured = apply_cloud_selection(payload, selection)

    assert "legacy_option" not in configured["memory"]["vector_store"]["options"]  # type: ignore[index]
    assert "legacy_option" not in configured["conversation"]["summary_vector_store"]["options"]  # type: ignore[index]


def test_existing_private_vikingdb_profile_can_be_kept_without_public_cloud_migration() -> None:
    payload = load_initialization_mapping()
    payload["credentials"]["vikingdb"]["token"] = "private-token"  # type: ignore[index]
    for group, name in (("memory", "vector_store"), ("conversation", "summary_vector_store")):
        target = payload[group][name]  # type: ignore[index]
        target["route"]["provider"] = "private-viking"
        target["route"]["base_url"] = "https://viking.example.com"
        target["options"].update(
            {
                "auth_mode": "private_headers",
                "schema_mode": "precreated",
                "credential_headers": {"X-API-Key": "{token}"},
                "console_url": "",
            }
        )
    selection = selection_from_mapping(payload)

    configured = apply_cloud_selection(payload, selection)

    assert selection.vector.profile_id == "vector.vikingdb.private_current"
    assert configured["memory"]["vector_store"] == payload["memory"]["vector_store"]  # type: ignore[index]


def test_cloud_setup_supports_precreated_vikingdb_with_api_key() -> None:
    defaults = default_cloud_selection()
    selection = replace(
        defaults,
        vector=ProfileSelection(
            "vector.vikingdb.precreated",
            {
                "region": "cn-beijing",
                "project_name": "default",
                "index_name": "default",
                "memory_collection": "memory",
                "summary_collection": "conversation_summaries",
            },
        ),
    )

    payload = apply_cloud_selection(load_initialization_mapping(), selection)
    config = HabitusConfig.from_mapping(payload)

    assert config.memory.vector_store.options["auth_mode"] == "api_key"
    assert config.conversation.summary_vector_store.options["schema_mode"] == "precreated"
    assert dict(config.credentials.resolve("vikingdb_api_key")) == {"api_key": ""}


def test_precreated_vikingdb_configuration_constructs_the_runtime_without_mixed_auth(
    tmp_path: Path,
) -> None:
    defaults = default_cloud_selection()
    selection = replace(
        defaults,
        vector=ProfileSelection(
            "vector.vikingdb.precreated",
            {
                "region": "cn-beijing",
                "project_name": "default",
                "index_name": "default",
                "memory_collection": "memory",
                "summary_collection": "conversation_summaries",
            },
        ),
    )
    payload = apply_cloud_selection(load_initialization_mapping(), selection)
    payload["storage"]["root"] = str(tmp_path / "data")  # type: ignore[index]
    payload["credentials"]["deepseek"]["api_key"] = "chat"  # type: ignore[index]
    payload["credentials"]["ark"]["api_key"] = "embedding"  # type: ignore[index]
    payload["credentials"]["vikingdb_api_key"]["api_key"] = "vector"  # type: ignore[index]

    runtime = build_runtime(HabitusConfig.from_mapping(payload))

    assert runtime.components.memory.vector_index.store is not None


def test_setup_round_trip_uses_normalized_route_identity_and_credential_reference() -> None:
    payload = load_initialization_mapping()
    payload["models"]["chat"]["route"]["adapter"] = " OPENAI_COMPATIBLE_CHAT "  # type: ignore[index]
    payload["models"]["chat"]["route"]["credential_ref"] = " DeepSeek "  # type: ignore[index]

    configured = apply_cloud_selection(payload, selection_from_mapping(payload))
    config = HabitusConfig.from_mapping(configured)

    assert config.models.chat.route.adapter == "openai_compatible_chat"
    assert config.models.chat.route.credential_ref == "deepseek"
    assert " DeepSeek " not in configured["credentials"]  # type: ignore[operator]


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


def test_cloud_wizard_default_path_selects_cloud_only_routes(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    choices = iter((1, 1, 1, 1, 1))
    monkeypatch.setattr(
        cli,
        "_prompt_choice",
        lambda *_args, **_kwargs: next(choices),
    )
    monkeypatch.setattr(cli, "_prompt_text", lambda _prompt, default: default)
    monkeypatch.setattr(cli, "_confirm", Mock(return_value=True))

    selection = cli._prompt_cloud_setup(default_cloud_selection())

    assert selection is not None
    assert selection.chat.profile_id == "chat.deepseek"
    assert selection.embedding.profile_id == "embedding.volcengine.ark"
    assert selection.rerank.profile_id == "rerank.disabled"
    assert selection.vector.profile_id == "vector.vikingdb.managed"
    assert selection.vector.values["region"] == "cn-beijing"
    output = capsys.readouterr().out
    assert "Local Chat, Embedding and Rerank are intentionally not included" in output
    assert "Rerank: Disabled (vector relevance fallback)" in output


def test_cloud_wizard_requires_a_real_alibaba_workspace_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    choices = iter((1, 1, 2, 1, 1))
    required_text = Mock(
        return_value="https://workspace.cn-beijing.maas.aliyuncs.com/compatible-api/v1"
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
    required_text.assert_called_once_with("Alibaba Workspace API base URL", None)


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
    monkeypatch.setattr(cli, "_confirm", Mock(return_value=False))

    assert cli._prompt_cloud_setup(default_cloud_selection()) is None


def test_initializer_updates_multiple_named_credentials_in_the_same_private_yaml(tmp_path: Path) -> None:
    destination = tmp_path / "config" / "habitus.yaml"
    initialize_config(destination, source=EXAMPLE_CONFIG)

    assert {(item.reference, item.field) for item in missing_credential_fields(destination)} == {
        ("ark", "api_key"),
        ("dashscope", "api_key"),
        ("deepseek", "api_key"),
        ("vikingdb", "access_key"),
        ("vikingdb", "secret_key"),
    }
    config = configure_credentials(
        destination,
        {
            "deepseek": {"api_key": "deepseek-secret"},
            "ark": {"api_key": "ark-secret"},
            "dashscope": {"api_key": "dashscope-secret"},
            "vikingdb": {
                "access_key": "viking-access",
                "secret_key": "viking-secret",
            },
        },
    )

    assert missing_credential_fields(destination) == ()
    assert config.credentials.resolve("deepseek")["api_key"] == "deepseek-secret"
    assert config.credentials.resolve("vikingdb")["secret_key"] == "viking-secret"
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600


def test_credential_updates_follow_the_same_normalized_name_rules_as_config_loading(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "config" / "habitus.yaml"
    payload = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    payload["credentials"][" DeepSeek "] = payload["credentials"].pop("deepseek")
    payload["credentials"][" DeepSeek "][" API_KEY "] = payload["credentials"][
        " DeepSeek "
    ].pop("api_key")
    destination.parent.mkdir(parents=True)
    destination.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    destination.chmod(0o600)

    config = configure_credentials(
        destination,
        {"deepseek": {"api_key": "deepseek-secret"}},
    )

    assert config.credentials.resolve("deepseek")["api_key"] == "deepseek-secret"


def test_interactive_initializer_prompts_each_missing_credential_without_echoing_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    destination = tmp_path / "config" / "habitus.yaml"
    initialize_config(destination, source=EXAMPLE_CONFIG)
    secrets = {
        "ark.api_key": "ark-secret",
        "dashscope.api_key": "dashscope-secret",
        "deepseek.api_key": "deepseek-secret",
        "vikingdb.access_key": "viking-access",
        "vikingdb.secret_key": "viking-secret",
    }
    monkeypatch.setattr(cli, "_confirm", Mock(return_value=True))
    monkeypatch.setattr(cli, "_prompt_secret", lambda label: secrets[label])

    cli._configure_missing_credentials(destination)

    config = HabitusConfig.from_file(destination)
    assert config.credentials.resolve("ark")["api_key"] == "ark-secret"
    assert missing_credential_fields(destination) == ()
    output = capsys.readouterr().out
    assert "ark-secret" not in output
    assert "viking-secret" not in output


def test_noninteractive_init_can_install_multiple_harnesses_without_knowing_their_types(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "config" / "habitus.yaml"
    delegated = Mock()
    monkeypatch.setattr(cli, "_delegate_plugin", delegated)

    cli.main(
        [
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
    )

    delegated.assert_called_once_with(
        ["install", "--harness", "codex", "--harness", "future-harness"]
    )
    assert destination.is_file()


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
    assert '"base_url": "http://127.0.0.1:8787"' in default_connection.read_text(
        encoding="utf-8"
    )


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


def test_plugin_assets_can_be_found_in_a_pip_target_scheme(tmp_path: Path) -> None:
    target = tmp_path / "target"
    plugin_root = target / "share" / "habitus" / "plugins"
    plugin_root.mkdir(parents=True)
    (plugin_root / "install-memory-plugin.mjs").write_text("// installed\n", encoding="utf-8")

    resolved = plugin_cli._plugin_root(
        source=tmp_path / "source-without-assets",
        search_paths=(target,),
    )

    assert resolved == plugin_root


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


def test_cloud_release_extra_closes_the_first_use_dependency_set() -> None:
    source = (REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    cloud = source.split("cloud = [", maxsplit=1)[1].split("]", maxsplit=1)[0]

    for dependency in ("fastapi", "uvicorn", "json-repair", "jsonschema", "volcengine"):
        assert dependency in cloud


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
        "integrations.local_service.cli",
        "serve",
        "--config",
        str(config),
    ]
