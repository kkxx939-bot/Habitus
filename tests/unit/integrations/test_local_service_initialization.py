"""统一启动入口、配置初始化和 Harness 委托的产品边界测试。"""

from __future__ import annotations

import os
import stat
from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from Config import M2BOSConfig
from integrations.local_service import cli
from integrations.local_service.initialization import (
    configure_credentials,
    initialize_config,
    missing_credential_fields,
    resolve_config_path,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_CONFIG = REPOSITORY_ROOT / "Config" / "example.yaml"


def test_config_path_resolution_has_one_explicit_environment_default_chain(tmp_path: Path) -> None:
    configured = tmp_path / "environment" / "config.yaml"
    explicit = tmp_path / "explicit" / "config.yaml"

    assert resolve_config_path(explicit, environ={"M2BOS_CONFIG_FILE": str(configured)}) == explicit
    assert resolve_config_path(None, environ={"M2BOS_CONFIG_FILE": str(configured)}) == configured
    assert resolve_config_path(None, environ={}).name == "config.yaml"

    with pytest.raises(ValueError, match="yaml"):
        resolve_config_path(tmp_path / "config.json", environ={})


def test_cli_always_binds_the_default_yaml_without_requiring_an_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("M2BOS_CONFIG_FILE", raising=False)

    values = cli._environment(None)

    assert values["M2BOS_CONFIG_FILE"] == str(Path("~/.m2bos/config.yaml").expanduser().absolute())


def test_initializer_writes_private_valid_config_without_overwriting_by_default(tmp_path: Path) -> None:
    destination = tmp_path / "config" / "m2bos.yaml"

    created = initialize_config(destination, source=EXAMPLE_CONFIG)
    destination.chmod(0o644)
    replayed = initialize_config(destination, source=EXAMPLE_CONFIG)

    assert created.created is True
    assert replayed.created is False
    assert created.backup_path is None
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert M2BOSConfig.from_file(destination).http.port == 8787


def test_force_initialization_preserves_previous_config_in_private_backup(tmp_path: Path) -> None:
    destination = tmp_path / "config" / "m2bos.yaml"
    original = EXAMPLE_CONFIG.read_bytes()
    initialize_config(destination, source=EXAMPLE_CONFIG)
    destination.write_bytes(original.replace(b"port: 8787", b"port: 8799", 1))

    result = initialize_config(destination, source=EXAMPLE_CONFIG, force=True)

    assert result.backup_path == destination.with_name("m2bos.bak.yaml")
    assert result.backup_path.read_bytes() != destination.read_bytes()
    assert M2BOSConfig.from_file(result.backup_path).http.port == 8799
    assert stat.S_IMODE(result.backup_path.stat().st_mode) == 0o600


def test_initializer_rejects_a_symbolic_link_destination(tmp_path: Path) -> None:
    real = tmp_path / "real.yaml"
    real.write_bytes(EXAMPLE_CONFIG.read_bytes())
    linked = tmp_path / "linked.yaml"
    linked.symlink_to(real)

    with pytest.raises(OSError):
        initialize_config(linked, source=EXAMPLE_CONFIG, force=True)


def test_initializer_updates_multiple_named_credentials_in_the_same_private_yaml(tmp_path: Path) -> None:
    destination = tmp_path / "config" / "m2bos.yaml"
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
    destination = tmp_path / "config" / "m2bos.yaml"
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
    destination = tmp_path / "config" / "m2bos.yaml"
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

    config = M2BOSConfig.from_file(destination)
    assert config.credentials.resolve("ark")["api_key"] == "ark-secret"
    assert missing_credential_fields(destination) == ()
    output = capsys.readouterr().out
    assert "ark-secret" not in output
    assert "viking-secret" not in output


def test_noninteractive_init_can_install_multiple_harnesses_without_knowing_their_types(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "config" / "m2bos.yaml"
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


def test_init_execs_the_single_server_entrypoint_when_start_is_requested(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "config" / "m2bos.yaml"
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
    destination = tmp_path / "missing" / "m2bos.yaml"
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
    config = Path("/tmp/m2bos-test/config.yaml")

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
