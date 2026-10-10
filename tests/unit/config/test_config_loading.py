"""唯一 YAML 配置入口、严格类型和跨领域容量约束测试。"""

import dataclasses
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from habitus.config import ConfigError, HabitusConfig
from habitus.config.loader import load_config_object, required_field, strict_fields, strict_object
from tests.unit.runtime.fixtures import STARTUP_PARAMETERS

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_CONFIG = REPOSITORY_ROOT / "habitus" / "config" / "example.yaml"


def valid_mapping(tmp_path: Path) -> dict[str, object]:
    payload = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    payload["storage"]["root"] = str(tmp_path / "data")
    return payload


def test_example_yaml_declares_a_complete_cross_domain_configuration(tmp_path) -> None:
    config = HabitusConfig.from_mapping(valid_mapping(tmp_path))
    assert config.storage_root == (tmp_path / "data").resolve()
    assert config.memory_root == config.storage_root / "memory"
    assert config.conversation_root == config.storage_root / "conversation"
    assert config.workflow_root == config.storage_root / "workflow"
    assert config.foresight_root == config.storage_root / "foresight"
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
    path = tmp_path / "habitus.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")

    direct = HabitusConfig.from_file(path)
    from_env = HabitusConfig.from_env(environ={"HABITUS_CONFIG_FILE": str(path)})
    assert from_env == direct
    with pytest.raises(ConfigError, match="missing"):
        HabitusConfig.from_env(environ={})


def test_named_credentials_support_multiple_vendors_without_leaking_repr(tmp_path) -> None:
    payload = valid_mapping(tmp_path)
    payload["credentials"]["deepseek"]["api_key"] = "deepseek-secret"
    payload["credentials"]["ark"]["api_key"] = "ark-secret"
    payload["credentials"]["dashscope"]["api_key"] = "dashscope-secret"
    payload["credentials"]["vikingdb"]["access_key"] = "viking-access"
    payload["credentials"]["vikingdb"]["secret_key"] = "viking-secret"

    config = HabitusConfig.from_mapping(payload)

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
    path = tmp_path / "habitus.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    path.chmod(0o644)

    with pytest.raises(ConfigError, match="group or other"):
        HabitusConfig.from_file(path)

    path.chmod(0o600)
    assert HabitusConfig.from_file(path).credentials.resolve("deepseek")["api_key"] == "private-secret"


def test_credential_registry_rejects_missing_references_and_unsafe_values_without_owning_adapter_fields(tmp_path) -> None:
    payload = valid_mapping(tmp_path)
    payload["models"]["chat"]["route"]["credential_ref"] = "missing-provider"
    with pytest.raises(ConfigError, match="does not exist"):
        HabitusConfig.from_mapping(payload)

    payload = valid_mapping(tmp_path)
    payload["credentials"]["deepseek"].pop("api_key")
    payload["credentials"]["deepseek"]["token"] = "secret"
    config = HabitusConfig.from_mapping(payload)
    assert dict(config.credentials.resolve("deepseek")) == {"token": "secret"}

    payload = valid_mapping(tmp_path)
    payload["credentials"]["deepseek"]["api_key"] = " secret "
    with pytest.raises(ValueError, match="surrounding whitespace"):
        HabitusConfig.from_mapping(payload)


def test_yaml_loader_rejects_duplicate_keys_unknown_fields_and_typo_with_suggestion(tmp_path) -> None:
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text("storage: {}\nstorage: {}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="duplicate"):
        load_config_object(duplicate)

    payload = valid_mapping(tmp_path)
    payload["memroy"] = payload.pop("memory")
    with pytest.raises(ConfigError, match="did you mean 'config.memory'"):
        HabitusConfig.from_mapping(payload)


def test_behavior_group_is_optional_and_disabled_until_a_subject_is_named(tmp_path) -> None:
    """behavior 组随 BHV-RUNTIME-001 回归为纯标量配置：缺省=行为侧未启用（上游感知未接入的
    合法状态），填入 primary_subject 才启用；未知字段照旧拒绝。"""

    payload = valid_mapping(tmp_path)
    config = HabitusConfig.from_mapping(payload)
    assert config.behavior.enabled is False

    payload["behavior"] = {"primary_subject": "家庭成员A", "reduction_sweep_interval_seconds": 300}
    enabled = HabitusConfig.from_mapping(payload)
    assert enabled.behavior.enabled is True
    assert enabled.behavior.primary_subject == "家庭成员A"
    assert enabled.behavior_root == enabled.storage_root / "behavior"

    payload["behavior"] = {"unknown_knob": 1}
    with pytest.raises(ConfigError, match="unknown config field"):
        HabitusConfig.from_mapping(payload)


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
        HabitusConfig.from_mapping(payload)


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
        HabitusConfig.from_mapping(payload)


def test_behavior_config_enforces_its_scalar_bounds() -> None:
    """行为组的手写边界校验逐条钉死：bool 拒收、上下界、5 分钟默认值（用户裁定的落点）。"""

    from habitus.config.behavior import BehaviorConfig

    assert BehaviorConfig().reduction_sweep_interval_seconds == 300.0  # 用户裁定的 5 分钟
    assert BehaviorConfig().context_lookback_seconds is None  # 不复制唯一出处的数值

    for kwargs, match in (
        ({"context_limit": 0}, "context_limit"),
        ({"context_limit": True}, "context_limit"),
        ({"context_lookback_seconds": 59}, "context_lookback_seconds"),
        ({"context_lookback_seconds": 90_000}, "context_lookback_seconds"),
        ({"fusion_poll_interval_seconds": 0.5}, "fusion_poll_interval_seconds"),
        ({"reduction_sweep_interval_seconds": 0.5}, "reduction_sweep_interval_seconds"),
        ({"worker_shutdown_timeout_seconds": 0}, "worker_shutdown_timeout_seconds"),
        ({"reduction_sweep_lock_ttl_seconds": 59}, "reduction_sweep_lock_ttl_seconds"),
        ({"reduction_sweep_lock_ttl_seconds": 86_401}, "reduction_sweep_lock_ttl_seconds"),
        ({"reduction_sweep_lock_ttl_seconds": True}, "reduction_sweep_lock_ttl_seconds"),
    ):
        with pytest.raises(ValueError, match=match):
            BehaviorConfig(**kwargs)
    with pytest.raises(TypeError):
        BehaviorConfig(primary_subject=123)  # type: ignore[arg-type]


def test_behavior_kinds_fields_are_bounded_and_derived_by_prefix() -> None:
    """kinds_* 与 behavior/kinds/config.py 的字段一一对应、下界一致；覆盖项按前缀派生，不双份维护。"""

    from dataclasses import fields

    from habitus.behavior.kinds.config import BehaviorKindConfig
    from habitus.config.behavior import BehaviorConfig

    assert BehaviorConfig().kinds_overrides() == {}
    domain = {field.name for field in fields(BehaviorKindConfig)}
    exposed = {field.name.removeprefix("kinds_") for field in fields(BehaviorConfig) if field.name.startswith("kinds_")}
    assert exposed == domain  # 每个门槛都能从配置调（裁定 6），而且不多出领域里没有的字段
    config = BehaviorConfig(kinds_batch_size=5, kinds_nightly_hour=4, kinds_max_anchor_pull=0.05, kinds_default_lane="physical")
    overrides = config.kinds_overrides()
    assert overrides == {"batch_size": 5, "nightly_hour": 4, "max_anchor_pull": 0.05, "default_lane": "physical"}
    assert BehaviorKindConfig(**overrides).default_lane.value == "physical"
    for kwargs, match in (
        ({"kinds_batch_size": 0}, "kinds_batch_size"),
        ({"kinds_revision_weekday": 7}, "kinds_revision_weekday"),
        ({"kinds_recurrence_min_count": True}, "kinds_recurrence_min_count"),
        ({"kinds_transient_retry_delay_seconds": 601}, "kinds_transient_retry_delay_seconds"),
        ({"kinds_max_anchor_pull": 1.5}, "kinds_max_anchor_pull"),
        ({"kinds_default_lane": "chat"}, "kinds_default_lane"),
    ):
        with pytest.raises(ValueError, match=match):
            BehaviorConfig(**kwargs)


def test_scene_config_is_a_switch_plus_the_relation_thresholds(tmp_path) -> None:
    """旧语义树的那组旋钮随它删掉（2026-09-26）；新树只有开关与关系检验的统计门槛（裁定 26）。"""

    from habitus.config.scene import SceneConfig

    config = HabitusConfig.from_mapping(valid_mapping(tmp_path))
    assert config.scene.enabled is False
    assert config.scene_root == config.storage_root / "scene"
    assert SceneConfig.from_mapping({"enabled": True}).enabled is True
    for gone in (
        "lookback_days",
        "pending_expiry_days",
        "max_occurrences_per_call",
        "retained_generations",
        "refresh_interval_seconds",
        "max_prompt_chars",
        "max_targets_per_call",
        "association_per_candidate",
        "max_cause_rows",
    ):
        with pytest.raises(ConfigError, match="unknown"):
            SceneConfig.from_mapping({gone: 3})
    with pytest.raises(ConfigError, match="scene.enabled"):
        SceneConfig.from_mapping({"enabled": "yes"})


def test_relation_thresholds_come_from_config_and_the_clock_from_the_prediction_tree(tmp_path) -> None:
    """写了的门槛覆盖默认值、没写的用默认值；槽宽、转移窗口、久别重来只认预测树那一组（语义树不另配）。"""

    from habitus.config.scene import SceneConfig
    from habitus.runtime.scene_night import relation_config
    from habitus.scene.relations import RelationThresholds

    scene = SceneConfig.from_mapping({"enabled": True, "relations": {"fdr": 0.2, "forward_min_antecedents": 3}})
    assert scene.relations.overrides() == {"fdr": 0.2, "forward_min_antecedents": 3}
    mapping = valid_mapping(tmp_path)
    mapping["prediction"] = dict(STARTUP_PARAMETERS)
    mapping["behavior"] = {"primary_subject": "用户"}
    mapping["scene"] = {"enabled": True, "relations": {"fdr": 0.2, "forward_min_antecedents": 3}}
    config = HabitusConfig.from_mapping(mapping)
    built = relation_config(config)
    assert built.thresholds == RelationThresholds(fdr=0.2, forward_min_antecedents=3)
    assert (built.slot_minutes, built.transition_window_slots, built.recurrence_window_days) == (
        config.prediction.slot_minutes,
        config.prediction.transition_window_slots,
        config.prediction.recurrence_window_days,
    )
    for bad, match in (
        ({"fdr": 1.0}, "fdr"),
        ({"min_relative_lift": -0.1}, "min_relative_lift"),
        ({"maintenance_blocks": 0}, "maintenance_blocks"),
        ({"slot_minutes": 15}, "unknown"),  # 时间的数不在这一组
    ):
        with pytest.raises(ConfigError, match=match):
            SceneConfig.from_mapping({"relations": bad})


def test_locale_is_one_group_shared_by_the_scene_and_foresight_layers(tmp_path) -> None:
    """时区与当地日历只有一份：情景读侧标日型、预测层造此刻时钟，两处分别配会对不上。"""

    from habitus.config.locale import LocaleConfig

    config = HabitusConfig.from_mapping(valid_mapping(tmp_path))
    assert config.locale.timezone == "Asia/Shanghai" and config.locale.region == "CN"
    assert str(config.locale.zone()) == "Asia/Shanghai"
    assert config.locale.calendar_path is None  # 没有日历数据 = 日型恒为空，显式的零修正
    for field_name, bad in (("timezone", "Nowhere/Nothing"), ("timezone", " "), ("region", "cn"), ("region", "CHN")):
        with pytest.raises(ConfigError, match=f"locale.{field_name}"):
            LocaleConfig.from_mapping({field_name: bad})
    # 数据格式还没定：给了路径直接拒，不静默忽略——静默忽略会让人以为调休已经生效。
    with pytest.raises(ConfigError, match="not supported yet"):
        LocaleConfig.from_mapping({"calendar_path": "/tmp/cn-2026.json"})
    with pytest.raises(ConfigError, match="unknown"):
        LocaleConfig.from_mapping({"zone": "UTC"})


def test_foresight_only_carries_protective_limits_and_needs_both_derived_trees(tmp_path) -> None:
    """预测层这一组只有保护闸——邻域与转移窗住在 prediction 那边，两处不各配一份。

    ``window_days`` 是卡上"上一次"往回看多少天，原先借的是 ``scene.lookback_days``；归组删掉之后
    它回到真正的使用者这边，不再跨组借。取历史卡不按日历窗，那道闸是 ``max_days_per_layer``。

    启用它却没启用两棵派生树是**配置自相矛盾**：数字取自预测树、与之对应的历史取自情景树，
    缺一边就装配不出证据。放过去的下场是启动一切正常、什么都没发生、无处可查。
    """

    from habitus.config.foresight import ForesightConfig

    config = HabitusConfig.from_mapping(valid_mapping(tmp_path))
    assert config.foresight.enabled is False
    assert [field.name for field in dataclasses.fields(ForesightConfig)] == [
        "enabled",
        "window_days",
        "max_days_per_layer",
        "judge_transient_retries",
        "judge_transient_retry_delay_seconds",
        "worker_shutdown_timeout_seconds",
    ]
    for field_name, bad in (
        ("window_days", 0),
        ("max_days_per_layer", 0),
        ("max_days_per_layer", True),
        ("judge_transient_retries", -1),
        ("judge_transient_retry_delay_seconds", -1.0),
        ("worker_shutdown_timeout_seconds", 0),
    ):
        with pytest.raises(ConfigError, match=f"foresight.{field_name}"):
            ForesightConfig.from_mapping({field_name: bad})
    with pytest.raises(ConfigError, match="unknown"):
        ForesightConfig.from_mapping({"pool_half_width": 3})

    mapping = valid_mapping(tmp_path)
    mapping["foresight"] = {"enabled": True}
    with pytest.raises(ConfigError, match="config.foresight is enabled"):
        HabitusConfig.from_mapping(mapping)


def test_behavior_kinds_bounds_match_the_domain_bounds() -> None:
    """配置层不 import behavior，上下界只能各写一份——用测试钉住两份一致。"""

    from habitus.behavior.kinds.config import _INTEGER_BOUNDS
    from habitus.config.behavior import _KINDS_INT_BOUNDS

    assert {(f"kinds_{name}", low, high) for name, low, high in _INTEGER_BOUNDS} == set(_KINDS_INT_BOUNDS)


def test_the_scene_relations_group_lists_every_relation_threshold() -> None:
    """配置组与 ``RelationThresholds`` 两边各列一份字段：钉住一致，给门槛加一项而忘了加进配置就报红（第四轮评审 E17）。"""

    from habitus.config.scene import SceneRelationsConfig
    from habitus.scene.relations import RelationThresholds

    assert {item.name for item in dataclasses.fields(SceneRelationsConfig)} == {
        item.name for item in dataclasses.fields(RelationThresholds)
    }
