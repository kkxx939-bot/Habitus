"""唯一 YAML 配置入口、严格类型和跨领域容量约束测试。"""

from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from Config import ConfigError, M2BOSConfig
from Config.loader import load_config_object, required_field, strict_fields, strict_object

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_CONFIG = REPOSITORY_ROOT / "Config" / "example.yaml"


def valid_mapping(tmp_path: Path) -> dict[str, object]:
    payload = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    payload["storage"]["root"] = str(tmp_path / "data")
    return payload


def test_example_yaml_declares_a_complete_cross_domain_configuration(tmp_path) -> None:
    config = M2BOSConfig.from_mapping(valid_mapping(tmp_path))
    assert config.storage_root == (tmp_path / "data").resolve()
    assert config.memory_root == config.storage_root / "memory"
    assert config.conversation_root == config.storage_root / "conversation"
    assert config.workflow_root == config.storage_root / "workflow"
    assert config.memory.recall_lifecycle.enabled
    assert config.memory.recall_lifecycle.ranking_alpha == 0.2
    assert config.memory.recall_lifecycle.profile_half_life_days == 180.0
    assert config.memory.recall_lifecycle.event_half_life_days == 14.0
    assert config.memory.semantic_search.vector_score_threshold == 0.0
    assert config.memory.semantic_search.rerank_score_threshold == 0.2
    assert config.workflow.jobs.max_attempts == 5
    assert config.models.rerank is not None
    assert config.models.rerank.route.provider == "aliyun"
    assert config.models.rerank.route.adapter == "openai_compatible_rerank"
    assert config.models.rerank.route.model == "qwen3-rerank"


def test_from_file_and_from_env_use_only_the_single_yaml_entrypoint(tmp_path) -> None:
    payload = valid_mapping(tmp_path)
    path = tmp_path / "m2bos.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")

    direct = M2BOSConfig.from_file(path)
    from_env = M2BOSConfig.from_env(environ={"M2BOS_CONFIG_FILE": str(path)})
    assert from_env == direct
    with pytest.raises(ConfigError, match="missing"):
        M2BOSConfig.from_env(environ={})


def test_named_credentials_support_multiple_vendors_without_leaking_repr(tmp_path) -> None:
    payload = valid_mapping(tmp_path)
    payload["credentials"]["deepseek"]["api_key"] = "deepseek-secret"
    payload["credentials"]["ark"]["api_key"] = "ark-secret"
    payload["credentials"]["dashscope"]["api_key"] = "dashscope-secret"
    payload["credentials"]["vikingdb"]["access_key"] = "viking-access"
    payload["credentials"]["vikingdb"]["secret_key"] = "viking-secret"

    config = M2BOSConfig.from_mapping(payload)

    assert config.models.chat.route.credential_ref == "deepseek"
    assert config.models.embedding.route.credential_ref == "ark"
    assert config.models.rerank is not None
    assert config.models.rerank.route.credential_ref == "dashscope"
    assert dict(config.credentials.resolve("vikingdb")) == {
        "access_key": "viking-access",
        "secret_key": "viking-secret",
    }
    rendered = repr(config)
    assert "deepseek-secret" not in rendered
    assert "viking-secret" not in rendered


def test_secret_bearing_yaml_requires_private_file_permissions(tmp_path) -> None:
    payload = valid_mapping(tmp_path)
    payload["credentials"]["deepseek"]["api_key"] = "private-secret"
    path = tmp_path / "m2bos.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    path.chmod(0o644)

    with pytest.raises(ConfigError, match="group or other"):
        M2BOSConfig.from_file(path)

    path.chmod(0o600)
    assert M2BOSConfig.from_file(path).credentials.resolve("deepseek")["api_key"] == "private-secret"


def test_credential_registry_rejects_missing_references_and_unsafe_values_without_owning_adapter_fields(tmp_path) -> None:
    payload = valid_mapping(tmp_path)
    payload["models"]["chat"]["route"]["credential_ref"] = "missing-provider"
    with pytest.raises(ConfigError, match="does not exist"):
        M2BOSConfig.from_mapping(payload)

    payload = valid_mapping(tmp_path)
    payload["credentials"]["deepseek"].pop("api_key")
    payload["credentials"]["deepseek"]["token"] = "secret"
    config = M2BOSConfig.from_mapping(payload)
    assert dict(config.credentials.resolve("deepseek")) == {"token": "secret"}

    payload = valid_mapping(tmp_path)
    payload["credentials"]["deepseek"]["api_key"] = " secret "
    with pytest.raises(ValueError, match="surrounding whitespace"):
        M2BOSConfig.from_mapping(payload)


def test_yaml_loader_rejects_duplicate_keys_unknown_fields_and_typo_with_suggestion(tmp_path) -> None:
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("storage: {}\nstorage: {}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate"):
        load_config_object(duplicate)

    payload = valid_mapping(tmp_path)
    payload["memroy"] = payload.pop("memory")
    with pytest.raises(ConfigError, match="did you mean 'config.memory'"):
        M2BOSConfig.from_mapping(payload)


def test_retired_behavior_yaml_group_is_rejected_as_unknown(tmp_path) -> None:
    payload = valid_mapping(tmp_path)
    payload["behavior"] = {}

    with pytest.raises(ConfigError, match=r"unknown config field 'config\.behavior'"):
        M2BOSConfig.from_mapping(payload)


def test_yaml_parse_errors_never_echo_secret_source_lines(tmp_path) -> None:
    path = tmp_path / "malformed.yaml"
    path.write_text('credentials: ["do-not-echo-secret"\n', encoding="utf-8")

    with pytest.raises(ConfigError) as captured:
        load_config_object(path)

    assert "do-not-echo-secret" not in str(captured.value)


@pytest.mark.parametrize(
    "source",
    [
        "value: .nan\n",
        "value: .inf\n",
        "- not\n- an\n- object\n",
        "null\n",
    ],
)
def test_yaml_loader_rejects_non_finite_and_non_object_roots(tmp_path, source: str) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config_object(path)


def test_yaml_loader_rejects_non_yaml_suffix_symlink_and_oversized_file(tmp_path) -> None:
    json_path = tmp_path / "config.json"
    json_path.write_text("{}", encoding="utf-8")
    with pytest.raises(ConfigError, match=".yaml"):
        load_config_object(json_path)

    real = tmp_path / "real.yaml"
    real.write_text("{}", encoding="utf-8")
    symlink = tmp_path / "link.yaml"
    symlink.symlink_to(real)
    with pytest.raises(ConfigError, match="symbolic"):
        load_config_object(symlink)

    huge = tmp_path / "huge.yaml"
    huge.write_bytes(b"a" * (1024 * 1024 + 1))
    with pytest.raises(ConfigError, match="one-megabyte"):
        load_config_object(huge)


def test_loader_does_not_expand_environment_placeholders(tmp_path) -> None:
    path = tmp_path / "literal.yaml"
    path.write_text("value: ${SECRET}\n", encoding="utf-8")
    assert load_config_object(path) == {"value": "${SECRET}"}


def test_strict_helpers_reject_loose_types_and_missing_required_fields() -> None:
    with pytest.raises(ConfigError, match="must be an object"):
        strict_object([], path="config")
    with pytest.raises(ConfigError, match="unknown"):
        strict_fields({"extra": 1}, path="config", allowed={"known"})
    with pytest.raises(ConfigError, match="missing required"):
        required_field({}, "root", path="config.storage")


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda p: p["memory"]["document"].update(max_encoded_bytes=300000),
            "snapshot.max_item_bytes",
        ),
        (
            lambda p: p["conversation"]["summary_vector_store"].update(collection=p["memory"]["vector_store"]["collection"]),
            "different vector collections",
        ),
        (
            lambda p: p["workflow"]["worker"].update(heartbeat_interval_seconds=60.0),
            "one third",
        ),
        (
            lambda p: p["memory"]["search_service"].update(
                max_recent_messages=201,
                max_planner_context_chars=1_000_000,
            ),
            "live message bound",
        ),
        (
            lambda p: p["memory"]["extraction"].update(max_input_tokens=60_000),
            "context_window_tokens",
        ),
        (
            lambda p: p["memory"]["recall_lifecycle"].update(max_batch_size=10),
            "maximum search candidate batch",
        ),
    ],
)
def test_cross_domain_capacity_mismatches_fail_before_runtime_assembly(tmp_path, mutator, message: str) -> None:
    payload = deepcopy(valid_mapping(tmp_path))
    mutator(payload)
    with pytest.raises(ConfigError, match=message):
        M2BOSConfig.from_mapping(payload)


def test_rerank_limits_are_checked_only_when_a_real_route_is_configured(tmp_path) -> None:
    payload = valid_mapping(tmp_path)
    payload["models"]["rerank"] = {
        "route": {
            "provider": "future-provider",
            "adapter": "future-rerank",
            "model": "rerank-model",
            "base_url": "https://example.com/v1",
            "credential_ref": "dashscope",
        },
        "max_documents": 1,
        "max_query_chars": 8000,
        "max_document_chars": 16000,
    }
    with pytest.raises(ConfigError, match="max_rerank_candidates"):
        M2BOSConfig.from_mapping(payload)
