"""唯一 YAML 配置入口、严格类型和跨领域容量约束测试。"""

import dataclasses
import re
from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from habitus.config import ConfigError, HabitusConfig
from habitus.config.loader import load_config_object, required_field, strict_fields, strict_object
from habitus.infrastructure.store.filesystem import restrict_file_to_owner
from tests.platform_helpers import configured_storage_root, grant_everyone_read

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
EXAMPLE_CONFIG = REPOSITORY_ROOT / "habitus" / "config" / "example.yaml"


def valid_mapping(tmp_path: Path) -> dict[str, object]:
    payload = yaml.safe_load(EXAMPLE_CONFIG.read_text(encoding="utf-8"))
    payload["storage"]["root"] = str(tmp_path / "data")
    return payload


def test_example_yaml_declares_a_complete_cross_domain_configuration(tmp_path) -> None:
    config = HabitusConfig.from_mapping(valid_mapping(tmp_path))
    assert config.storage_root == configured_storage_root(tmp_path / "data")
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
    assert config.memory.vector_index.lock_ttl_seconds == 30
    assert config.memory.vector_index.lock_wait_timeout_seconds == 30.0
    assert config.memory.vector_index.lock_retry_delay_seconds == 0.02
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
    payload["credentials"]["chat"]["api_key"] = "chat-secret"
    payload["credentials"]["embedding"]["api_key"] = "embedding-secret"
    payload["credentials"]["dashscope"]["api_key"] = "dashscope-secret"

    config = HabitusConfig.from_mapping(payload)

    assert config.models.chat.route.credential_ref == "chat"
    assert config.models.embedding.route.credential_ref == "embedding"
    assert config.models.rerank is not None
    assert config.models.rerank.route.credential_ref == "dashscope"
    assert dict(config.credentials.resolve("dashscope")) == {"api_key": "dashscope-secret"}
    assert config.memory.vector_store.route.credential_ref == ""
    assert config.conversation.summary_vector_store.route.credential_ref == ""
    rendered = repr(config)
    assert "chat-secret" not in rendered
    assert "dashscope-secret" not in rendered


def test_secret_bearing_yaml_requires_private_file_permissions(tmp_path) -> None:
    payload = valid_mapping(tmp_path)
    payload["credentials"]["chat"]["api_key"] = "private-secret"
    path = tmp_path / "habitus.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    grant_everyone_read(path)

    with pytest.raises(ConfigError, match="group or other"):
        HabitusConfig.from_file(path)

    restrict_file_to_owner(path)
    assert HabitusConfig.from_file(path).credentials.resolve("chat")["api_key"] == "private-secret"


def test_credential_registry_rejects_missing_references_and_unsafe_values_without_owning_adapter_fields(
    tmp_path,
) -> None:
    payload = valid_mapping(tmp_path)
    payload["models"]["chat"]["route"]["credential_ref"] = "missing-provider"
    with pytest.raises(ConfigError, match="does not exist"):
        HabitusConfig.from_mapping(payload)

    payload = valid_mapping(tmp_path)
    payload["credentials"]["chat"].pop("api_key")
    payload["credentials"]["chat"]["token"] = "secret"
    config = HabitusConfig.from_mapping(payload)
    assert dict(config.credentials.resolve("chat")) == {"token": "secret"}

    payload = valid_mapping(tmp_path)
    payload["credentials"]["chat"]["api_key"] = " secret "
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


def test_memory_editor_retrieval_group_is_named_for_the_write_side_prefetch(tmp_path) -> None:
    """写入侧旧记忆预取的配置节叫 ``memory.editor_retrieval``；旧键 ``memory.retrieval`` 不再被接受。"""

    payload = valid_mapping(tmp_path)
    payload["memory"]["editor_retrieval"]["search_limit"] = 4
    assert HabitusConfig.from_mapping(payload).memory.editor_retrieval.search_limit == 4

    payload = valid_mapping(tmp_path)
    payload["memory"]["retrieval"] = payload["memory"].pop("editor_retrieval")
    with pytest.raises(ConfigError, match="config.memory.retrieval"):
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
            lambda p: p["conversation"]["summary_vector_store"].update(
                collection=p["memory"]["vector_store"]["collection"]
            ),
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
        ({"context_lookback_seconds": 59}, "context_lookback_seconds"),
        ({"context_lookback_seconds": 90_000}, "context_lookback_seconds"),
        ({"reduction_sweep_interval_seconds": 0.5}, "reduction_sweep_interval_seconds"),
        ({"worker_shutdown_timeout_seconds": 0}, "worker_shutdown_timeout_seconds"),
        ({"reduction_sweep_lock_ttl_seconds": 59}, "reduction_sweep_lock_ttl_seconds"),
        ({"reduction_sweep_lock_ttl_seconds": 86_401}, "reduction_sweep_lock_ttl_seconds"),
        ({"retry_backlog_max": 0}, "retry_backlog_max"),
    ):
        with pytest.raises(ValueError, match=match):
            BehaviorConfig(**kwargs)
    # bool 冒充整数在 YAML 读入处拒（W16：类型检查在解码器，不在构造器）
    for name in ("reduction_sweep_lock_ttl_seconds", "retry_backlog_max"):
        with pytest.raises(ConfigError, match=rf"config\.behavior\.{name} must be an integer"):
            BehaviorConfig.from_mapping({name: True})


def test_behavior_kinds_fields_are_typed_and_derived_by_prefix() -> None:
    """kinds_* 与 behavior/kinds/config.py 的字段一一对应；配置层只查类型，覆盖项按前缀派生，不双份维护。"""

    from dataclasses import fields

    from habitus.behavior.kinds.config import BehaviorKindConfig
    from habitus.config.behavior import BehaviorConfig

    assert BehaviorConfig().overrides_for("kinds_") == {}
    domain = {field.name for field in fields(BehaviorKindConfig)}
    exposed = {field.name.removeprefix("kinds_") for field in fields(BehaviorConfig) if field.name.startswith("kinds_")}
    assert exposed == domain  # 每个门槛都能从配置调（裁定 6），而且不多出领域里没有的字段
    config = BehaviorConfig(kinds_batch_size=5, kinds_revision_weekday=4, kinds_max_anchor_pull=0.05)
    overrides = config.overrides_for("kinds_")
    assert overrides == {"batch_size": 5, "revision_weekday": 4, "max_anchor_pull": 0.05}
    assert BehaviorKindConfig(**overrides).batch_size == 5
    # 配置层只查类型；值域归 BehaviorKindConfig，越界的值在组装时拒（见 test_out_of_range_overrides_are_refused_at_startup）
    assert BehaviorConfig(kinds_batch_size=0).overrides_for("kinds_") == {"batch_size": 0}
    for kwargs, match in (
        ({"kinds_batch_size": 1.5}, "kinds_batch_size must be an integer"),
        ({"kinds_recurrence_min_count": True}, "kinds_recurrence_min_count must be an integer"),
        ({"kinds_transient_retry_delay_seconds": "5"}, "kinds_transient_retry_delay_seconds must be a number"),
        ({"kinds_max_anchor_pull": False}, "kinds_max_anchor_pull must be a number"),
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


def test_relation_thresholds_come_from_config_and_the_clock_from_temporal(tmp_path) -> None:
    """写了的门槛覆盖默认值、没写的用默认值；槽宽、转移窗口只认 ``config.temporal`` 那一组（语义树不另配，裁定 59）。"""

    from habitus.config.scene import SceneConfig
    from habitus.runtime.scene_config import relation_config
    from habitus.scene.relations import RelationThresholds

    scene = SceneConfig.from_mapping({"enabled": True, "relations": {"fdr": 0.2, "forward_max_antecedents": 30}})
    assert scene.relations.overrides() == {"fdr": 0.2, "forward_max_antecedents": 30}
    mapping = valid_mapping(tmp_path)
    mapping["behavior"] = {"primary_subject": "用户"}
    mapping["scene"] = {"enabled": True, "relations": {"fdr": 0.2, "forward_max_antecedents": 30}}
    config = HabitusConfig.from_mapping(mapping)
    built = relation_config(config)
    assert built.thresholds == RelationThresholds(fdr=0.2, forward_max_antecedents=30)
    # 时间尺子只认 config.temporal（裁定 59）
    assert (built.slot_minutes, built.transition_window_slots) == (
        config.temporal.slot_minutes,
        config.temporal.transition_window_slots,
    )
    # 配置层只查类型；值域归 RelationThresholds，越界的值在组装时拒（见 test_out_of_range_overrides_are_refused_at_startup）
    assert SceneConfig.from_mapping({"relations": {"fdr": 1.0}}).relations.overrides() == {"fdr": 1.0}
    for bad, match in (
        ({"fdr": "0.1"}, "scene.relations.fdr must be a number"),
        ({"max_blind_share": True}, "scene.relations.max_blind_share must be a number"),
        ({"condition_ask_min_antecedents": 4.0}, "scene.relations.condition_ask_min_antecedents must be an integer"),
        ({"slot_minutes": 15}, "unknown"),  # 时间的数不在这一组
    ):
        with pytest.raises(ConfigError, match=match):
            SceneConfig.from_mapping({"relations": bad})


def test_locale_is_one_group_shared_by_the_scene_and_foresight_layers(tmp_path) -> None:
    """时区与当地日历只有一份：情景读侧标日型、预测层造此刻时钟，两处分别配会对不上。"""

    from habitus.config.locale import LocaleConfig

    config = HabitusConfig.from_mapping(valid_mapping(tmp_path))
    # 不写 locale 时默认新加坡（W3，用户 11-04 定）
    assert config.locale.timezone == "Asia/Singapore" and config.locale.region == "SG"
    assert str(config.locale.zone()) == "Asia/Singapore"
    assert config.locale.calendar_path is None  # 只用内置的放假安排；某一年没有数据，那一年的日型是未知
    # 时区名区分大小写：macOS 上 ZoneInfo("asia/singapore") 打得开，配置搬到 Linux 就读不出，所以加载时就拒
    for field_name, bad in (
        ("timezone", "Nowhere/Nothing"),
        ("timezone", " "),
        ("timezone", "asia/singapore"),
        ("timezone", "ASIA/SHANGHAI"),
        ("region", "cn"),
        ("region", "CHN"),
    ):
        with pytest.raises(ConfigError, match=f"locale.{field_name}"):
            LocaleConfig.from_mapping({field_name: bad})
    # 日历跟着这个人在哪（裁定 59）：州一级的地区码也认；名义周末可配
    assert LocaleConfig.from_mapping({"region": "US-CA", "nominal_weekend": [6, 5]}).nominal_weekend == (5, 6)
    for bad in ({"nominal_weekend": []}, {"nominal_weekend": [7]}, {"nominal_weekend": [5, 5]}, {"region": "us-ca"}):
        with pytest.raises(ConfigError, match="locale"):
            LocaleConfig.from_mapping(bad)
    # 覆盖目录（裁定 59）：给了就收下，目录在不在由读取时查；空串直接拒。
    assert LocaleConfig.from_mapping({"calendar_path": "/srv/calendar"}).calendar_path == "/srv/calendar"
    with pytest.raises(ConfigError, match="locale.calendar_path"):
        LocaleConfig.from_mapping({"calendar_path": " "})
    with pytest.raises(ConfigError, match="unknown"):
        LocaleConfig.from_mapping({"zone": "UTC"})


def test_foresight_carries_the_loss_ratios_and_protective_limits_and_needs_both_derived_trees(tmp_path) -> None:
    """预测层这一组没有算法参数：时窗来自历史那几次的实际时刻，转移窗住在 temporal 那边，升不升从账上算（裁定 49）。
    这里只有开关、两档可接受误提醒率（产品决定）、重轻多久重判、几道保护闸，以及一个还没定的开关（默认关，裁定 46）。

    启用它却没启用语义树是**配置自相矛盾**：机会表、复现与关系全从语义树那一晚来（预测树删了，裁定 59）。
    """

    from habitus.config.foresight import ForesightConfig

    config = HabitusConfig.from_mapping(valid_mapping(tmp_path))
    assert config.foresight.enabled is False
    assert [field.name for field in dataclasses.fields(ForesightConfig)] == [
        "enabled",
        "false_alarm_rate",
        "shadow_uses_unestablished",
        "daily_reminder_cap",
        "open_reminders_judged",
        "unit_prior_share",
        "judge_transient_retries",
        "judge_transient_retry_delay_seconds",
        "tick_interval_seconds",
        "worker_shutdown_timeout_seconds",
        "half_life_opportunities",
        "outlet_liveness_hours",
        "outlet_lease_minutes",
        "outlet_max_attempts",
        "outlet_claim_limit",
        "outlet_match_similarity",
        "unseen_day_types",
        "store",
    ]
    assert config.foresight.shadow_uses_unestablished is True
    assert config.foresight.unseen_day_types == "skip"  # 待用户定之前不外推（裁定 59 第 7 条）
    assert config.foresight.false_alarm_rate == pytest.approx(0.33)  # 线 0.67，所有账同一条（裁定 58 第 1 条）
    for field_name, bad in (
        ("false_alarm_rate", 1.0),
        ("false_alarm_rate", True),
        ("daily_reminder_cap", 0),
        ("open_reminders_judged", 0),
        ("shadow_uses_unestablished", "yes"),
        ("judge_transient_retries", -1),
        ("judge_transient_retry_delay_seconds", -1.0),
        ("worker_shutdown_timeout_seconds", 0),
        ("unseen_day_types", "guess"),
        ("outlet_liveness_hours", 0.5),
        ("outlet_lease_minutes", 0),
        ("outlet_max_attempts", 0),
        ("outlet_claim_limit", 21),
        ("outlet_match_similarity", 1.5),
    ):
        with pytest.raises(ConfigError, match=f"foresight.{field_name}"):
            ForesightConfig.from_mapping({field_name: bad})
    # 落盘的容量闸是运维边界（段 6 Q17）：整段可配、未知字段与越界都拒
    assert config.foresight.store.record_max_bytes == 1_000_000
    tuned = ForesightConfig.from_mapping({"store": {"opportunity_table_max_bytes": 32 * 1024 * 1024}})
    assert tuned.store.opportunity_table_max_bytes == 32 * 1024 * 1024
    with pytest.raises(ConfigError, match="foresight.store.record_max_bytes"):
        ForesightConfig.from_mapping({"store": {"record_max_bytes": 10}})
    with pytest.raises(ConfigError, match="unknown"):
        ForesightConfig.from_mapping({"store": {"max_files": 3}})
    with pytest.raises(ConfigError, match="unknown"):
        ForesightConfig.from_mapping({"window_days": 3})
    with pytest.raises(ConfigError, match="unknown"):
        ForesightConfig.from_mapping({"speak_after_net_hits": 1})  # 节奏项已删（裁定 48、49）
    for removed in ("holdout_rate", "line_floor", "cost_ratio_unjudged", "severity_review_days"):
        with pytest.raises(ConfigError, match="unknown"):  # 扣发与急缓这一版不做（裁定 58 第 1 条）
            ForesightConfig.from_mapping({removed: 0.1})

    mapping = valid_mapping(tmp_path)
    mapping["foresight"] = {"enabled": True}
    with pytest.raises(ConfigError, match="config.foresight is enabled but config.scene is not"):
        HabitusConfig.from_mapping(mapping)
    # 预测树删了（裁定 59）：预测层只要语义树开着，不再要求 config.prediction
    mapping["behavior"] = {"primary_subject": "家庭成员A"}
    mapping["scene"] = {"enabled": True}
    assert HabitusConfig.from_mapping(mapping).foresight.enabled is True


def test_the_nightly_run_has_no_clock_time_only_a_shutdown_wait() -> None:
    """整晚那一趟前一天定稿完成就接着跑，不挂固定时刻（裁定 94 第 1 条）：语义树那一段只留停机等多久（默认 30 秒），越界硬拒；
    写了旧的 ``nightly_at`` 是未知字段。"""

    from habitus.config.scene import SceneConfig

    assert SceneConfig().nightly_shutdown_timeout_seconds == 30.0
    for field_name, bad in (
        ("nightly_shutdown_timeout_seconds", 0.5),
        ("nightly_shutdown_timeout_seconds", "30"),
    ):
        with pytest.raises(ConfigError, match=f"scene.{field_name}"):
            SceneConfig.from_mapping({field_name: bad})
    with pytest.raises(ConfigError, match="nightly_at"):
        SceneConfig.from_mapping({"nightly_at": "04:00"})


def startup_yaml(tmp_path: Path, *, behavior: dict[str, object], scene: dict[str, object]) -> Path:
    """example.yaml 换上 fake 路由、写进 behavior / scene 两段，落成真实的 YAML 文件。"""

    payload = valid_mapping(tmp_path)
    for route in (
        payload["models"]["chat"]["route"],
        payload["models"]["embedding"]["route"],
        payload["models"]["rerank"]["route"],
        payload["memory"]["vector_store"]["route"],
        payload["conversation"]["summary_vector_store"]["route"],
    ):
        route.update(provider="fake", credential_ref="")
    payload["models"]["chat"]["route"]["adapter"] = "fake_chat"
    payload["models"]["embedding"]["route"]["adapter"] = "fake_embedding"
    payload["models"]["rerank"]["route"]["adapter"] = "fake_rerank"
    payload["memory"]["vector_store"]["route"]["adapter"] = "fake_vector"
    payload["conversation"]["summary_vector_store"]["route"]["adapter"] = "fake_vector"
    payload["behavior"] = behavior
    payload["scene"] = scene
    path = tmp_path / "habitus.yaml"
    path.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("behavior", "scene", "key"),
    [
        ({"primary_subject": "用户", "kinds_batch_size": 0}, {}, "config.behavior.kinds_batch_size"),
        ({"primary_subject": "用户", "kinds_max_anchor_pull": 1.5}, {}, "config.behavior.kinds_max_anchor_pull"),
        # 几项之间的约束（复现天数不超过复现条数）同样在组装时拒、点得出写了的那一项
        ({"primary_subject": "用户", "kinds_recurrence_min_days": 5}, {}, "config.behavior.kinds_recurrence_min_days"),
        # 行为侧没开：写错的值照样在启动时拒，不被静默收下
        ({"kinds_revision_weekday": 7}, {}, "config.behavior.kinds_revision_weekday"),
        ({"primary_subject": "用户"}, {"enabled": True, "relations": {"fdr": 1.0}}, "config.scene.relations.fdr"),
        ({}, {"relations": {"min_relative_lift": -0.1}}, "config.scene.relations.min_relative_lift"),
        ({}, {"relations": {"max_combined_conditions": 0}}, "config.scene.relations.max_combined_conditions"),
    ],
)
def test_out_of_range_overrides_are_refused_at_startup(
    tmp_path: Path, behavior: dict[str, object], scene: dict[str, object], key: str
) -> None:
    """kinds_* 与 scene.relations 的值域只在领域配置（配置层不 import 领域、不抄第二份）：配置文件照常读进来，
    ``build_runtime`` 组装时由领域配置拒，报错点得出是哪个配置键；不论那一侧开没开都拒。"""

    from habitus.infrastructure.store.contracts import PathLock
    from habitus.runtime import build_runtime
    from tests.integration.test_runtime_assembly import runtime_dependencies
    from tests.lock_store import ProcessLocalLockStore

    config = HabitusConfig.from_file(startup_yaml(tmp_path, behavior=behavior, scene=scene))
    providers, vectors = runtime_dependencies()
    with pytest.raises(ConfigError, match=f"invalid '{key}'"):
        build_runtime(config, providers=providers, vector_stores=vectors, path_lock=PathLock(ProcessLocalLockStore()))


def test_the_scene_relations_group_lists_every_relation_threshold() -> None:
    """配置组与 ``RelationThresholds`` 两边各列一份字段：钉住一致，给门槛加一项而忘了加进配置就报红（第四轮评审 E17）。"""

    from habitus.config.scene import SceneRelationsConfig
    from habitus.scene.relations import RelationThresholds

    assert {item.name for item in dataclasses.fields(SceneRelationsConfig)} == {
        item.name for item in dataclasses.fields(RelationThresholds)
    }


def test_behaviour_store_capacity_limits_come_through_the_config_boundary() -> None:
    """26 号清单 8.2：存储的容量闸从 ``Config.behavior`` 进来，按前缀派生到领域配置的同名字段；没写的取领域默认值。"""

    from habitus.config.behavior import BehaviorConfig

    config = BehaviorConfig(observation_max_files=10, fusion_max_judgement_files=20, tree_max_children_per_directory=30)
    assert config.overrides_for("observation_") == {"max_files": 10}
    assert config.overrides_for("fusion_") == {"max_judgement_files": 20}
    assert config.overrides_for("tree_") == {"max_children_per_directory": 30}
    with pytest.raises(ValueError, match="observation_max_files must be an integer"):
        BehaviorConfig(observation_max_files=1.5)  # type: ignore[arg-type]


def _behavior_domains() -> dict[str, type]:
    from habitus.behavior.document import BehaviorDocumentConfig
    from habitus.behavior.fusion.config import BehaviorFusionConfig
    from habitus.behavior.fusion.lanes.physical import PhysicalLaneConfig
    from habitus.behavior.fusion.lanes.session import SessionLaneConfig
    from habitus.behavior.kinds.config import BehaviorKindConfig
    from habitus.behavior.observation.config import BehaviorObservationConfig
    from habitus.behavior.semantic import BehaviorSemanticConfig
    from habitus.behavior.tree.config import BehaviorTreeConfig
    from habitus.behavior.tree.writer import BehaviorWriteConfig

    return {
        "session_": SessionLaneConfig,
        "physical_": PhysicalLaneConfig,
        "kinds_": BehaviorKindConfig,
        "observation_": BehaviorObservationConfig,
        "fusion_": BehaviorFusionConfig,
        "tree_": BehaviorTreeConfig,
        "document_": BehaviorDocumentConfig,
        "write_": BehaviorWriteConfig,
        "semantic_": BehaviorSemanticConfig,
    }


# 存储的内容类上限（一次交付多少条、每条多长）随上游契约一起定，这一版只接与生命周期相关的容量闸（26 号清单 8.2）
_PARTIAL_PREFIXES = frozenset({"observation_", "fusion_", "tree_"})


def test_every_behaviour_override_prefix_maps_onto_its_domain_config() -> None:
    """E18：每一组覆盖项都按前缀对到一份领域配置的同名字段——会话 / 物理 lane、词表、文档、写锁、语义概览的每个字段都调得到，
    不多出领域里没有的字段；组合根把它们真的交给领域配置。"""

    from habitus.config.behavior import OVERRIDE_PREFIXES, BehaviorConfig
    from habitus.runtime.behavior import behavior_domain_configs

    domains = _behavior_domains()
    assert set(OVERRIDE_PREFIXES) == set(domains)
    exposed_all = {item.name for item in dataclasses.fields(BehaviorConfig)}
    for prefix, domain in domains.items():
        names = {item.name for item in dataclasses.fields(domain)}
        exposed = {name.removeprefix(prefix) for name in exposed_all if name.startswith(prefix)}
        assert exposed <= names, prefix
        if prefix not in _PARTIAL_PREFIXES:
            assert exposed == names, prefix
    built = behavior_domain_configs(
        BehaviorConfig(
            session_max_step_chars=99,
            physical_max_segments=7,
            document_max_encoded_bytes=4096,
            write_lock_ttl_seconds=45,
            semantic_max_prompt_chars=5000,
        )
    )
    assert built.session.max_step_chars == 99
    assert built.physical.max_segments == 7
    assert built.document.max_encoded_bytes == 4096
    assert built.write.lock_ttl_seconds == 45
    assert built.semantic.max_prompt_chars == 5000


@pytest.mark.parametrize(
    ("field", "value", "accepted"),
    [
        ("session_max_steps", 60, True),  # 领域收（配置层以前另写了上限 50）
        ("tree_max_children_per_directory", 5_000_000, False),  # 领域拒（配置层以前另写了上限 1000 万）
        ("tree_max_children_per_directory", 1, False),
        ("write_lock_ttl_seconds", 0, False),
        ("physical_max_facts", 0, False),
    ],
)
def test_behaviour_override_ranges_are_the_domain_ranges(field: str, value: int, accepted: bool) -> None:
    """值域只在领域配置一处：配置层只查类型、照常读进来，组装时由领域配置判；拒了报的是配置键。"""

    from habitus.config.behavior import BehaviorConfig
    from habitus.runtime.behavior import behavior_domain_configs

    config = BehaviorConfig(**{field: value})
    if accepted:
        behavior_domain_configs(config)
    else:
        with pytest.raises(ConfigError, match=f"invalid 'config.behavior.{field}'"):
            behavior_domain_configs(config)


def test_ingress_ceilings_default_to_the_same_64_mib() -> None:
    """分块上传拼回的一轮（最大 http.remember_parts_max_bytes）要整份放进会话源单条信封与会话日志单文件：三者默认对齐。"""

    from habitus.config import ConversationSourceConfig, HTTPAPIConfig
    from habitus.memory.conversation import ConversationJournalConfig

    assert (
        ConversationSourceConfig().max_envelope_bytes
        == ConversationJournalConfig().max_file_bytes
        == HTTPAPIConfig().remember_parts_max_bytes
        == 64 * 1024 * 1024
    )
    example = yaml.safe_load((Path(__file__).resolve().parents[3] / "habitus/config/example.yaml").read_text("utf-8"))
    assert example["conversation"]["source"]["max_envelope_bytes"] == 64 * 1024 * 1024


@pytest.mark.parametrize("value", [0, -1, 1023, 64 * 1024 * 1024 + 1])
def test_http_max_request_bytes_outside_its_range_is_rejected(value: int) -> None:
    from habitus.config import HTTPAPIConfig

    with pytest.raises(ValueError, match="max_request_bytes must be between 1024 and 67108864"):
        HTTPAPIConfig(max_request_bytes=value)


@pytest.mark.parametrize("value", ["text", "multimodal"])
def test_embedding_input_mode_left_in_yaml_is_an_unknown_field(tmp_path, value: str) -> None:
    """``input_mode`` 只为一家厂商的专用协议存在过，W7 删掉：旧配置里还写着它要明确报未知字段，不能静默忽略。"""

    payload = valid_mapping(tmp_path)
    payload["models"]["embedding"]["input_mode"] = value

    with pytest.raises(ConfigError, match=r"unknown config field 'config\.models\.embedding\.input_mode'"):
        HabitusConfig.from_mapping(payload)


def test_embedding_prefixes_load_from_yaml_and_must_be_strings(tmp_path) -> None:
    payload = valid_mapping(tmp_path)
    payload["models"]["embedding"]["max_input_chars"] += 64  # 记录与查询上限按"减去前缀"算，要给前缀留出位置
    payload["models"]["embedding"]["query_prefix"] = "query: "
    payload["models"]["embedding"]["document_prefix"] = "passage: "
    embedding = HabitusConfig.from_mapping(payload).models.embedding
    assert (embedding.query_prefix, embedding.document_prefix) == ("query: ", "passage: ")

    for name, invalid in (("query_prefix", 1), ("document_prefix", ["passage: "]), ("document_prefix", None)):
        broken = valid_mapping(tmp_path)
        broken["models"]["embedding"][name] = invalid
        with pytest.raises(ConfigError, match=f"config.models.embedding.{name}"):
            HabitusConfig.from_mapping(broken)


@pytest.mark.parametrize(
    ("section", "field", "prefix"),
    [
        ("memory.vector_index", "max_record_chars", "document_prefix"),
        ("conversation.summary_vector_index", "max_record_chars", "document_prefix"),
        ("memory.editor_retrieval", "max_query_chars", "query_prefix"),
    ],
)
def test_prefix_counts_against_the_record_and_query_budgets(tmp_path, section: str, field: str, prefix: str) -> None:
    """记录与查询上限按"max_input_chars 减去对应前缀"校验，不然索引或召回时才因超长失败。每条单独钉：其余两项放宽。"""

    payload = valid_mapping(tmp_path)
    budget = payload["models"]["embedding"]["max_input_chars"]
    payload["memory"]["vector_index"]["max_record_chars"] = budget - 64
    payload["conversation"]["summary_vector_index"]["max_record_chars"] = budget - 64
    payload["models"]["rerank"]["max_query_chars"] = budget  # 查询上限还受 rerank 查询上限约束，这里放开
    group = payload
    for part in section.split("."):
        group = group[part]
    group[field] = budget
    HabitusConfig.from_mapping(payload)

    payload["models"]["embedding"][prefix] = "passage: "
    with pytest.raises(ConfigError, match=prefix):
        HabitusConfig.from_mapping(payload)


def test_prefix_budget_error_says_what_to_change(tmp_path) -> None:
    """默认 max_record_chars 与 max_input_chars 相等，照文档设非空 document_prefix 会加载失败；报错要能自说明：
    点名 max_input_chars、前缀，并给出改法（调小上限或调大 max_input_chars），不然用户只知道失败不知道怎么办。"""

    payload = valid_mapping(tmp_path)
    budget = payload["models"]["embedding"]["max_input_chars"]
    payload["memory"]["vector_index"]["max_record_chars"] = budget
    payload["conversation"]["summary_vector_index"]["max_record_chars"] = budget
    payload["models"]["embedding"]["document_prefix"] = "passage: "

    with pytest.raises(ConfigError) as error:
        HabitusConfig.from_mapping(payload)
    message = str(error.value)
    assert "max_input_chars" in message
    assert "document_prefix" in message
    assert str(budget) in message
    assert "lower" in message or "raise" in message


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("max_conversations_per_cycle", True),
        ("max_conversations_per_cycle", "100"),
        ("summary_compaction.recent_use_protection_days", 90.0),
        ("summary_compaction.cleanup_batch_size", False),
    ],
)
def test_conversation_lifecycle_integers_reject_bool_text_or_float_from_yaml(tmp_path, field: str, bad: object) -> None:
    """YAML 里 bool、文本、浮点冒充整数的生命周期字段在读入处就拒（W16：判定收进公共函数后行为不变）。"""

    payload = valid_mapping(tmp_path)
    group = payload["conversation"]["lifecycle"]
    *parents, name = field.split(".")
    for parent in parents:
        group = group[parent]
    group[name] = bad
    with pytest.raises(ConfigError, match=rf"'config\.conversation\.lifecycle\.{re.escape(field)}' must be an integer"):
        HabitusConfig.from_mapping(payload)


@pytest.mark.parametrize("raw_path", ["", "   ", "\n"])
def test_from_env_treats_a_blank_config_path_as_missing(raw_path: str) -> None:
    with pytest.raises(ConfigError, match="config file path is missing: HABITUS_CONFIG_FILE"):
        HabitusConfig.from_env(environ={"HABITUS_CONFIG_FILE": raw_path})


@pytest.mark.parametrize(
    ("section", "group", "message"),
    [
        ("temporal", {"slot_minutes": "15"}, r"config\.temporal\.slot_minutes must be an integer"),
        ("temporal", {"slot_minutes": True}, r"config\.temporal\.slot_minutes must be an integer"),
        ("foresight", {"enabled": "yes"}, r"config\.foresight\.enabled must be a boolean"),
        ("behavior", {"primary_subject": 1}, r"config\.behavior\.primary_subject must be text"),
        ("locale", {"nominal_weekend": [5, "6"]}, r"config\.locale\.nominal_weekend must be a list of integers"),
        ("locale", {"nominal_weekend": "5,6"}, r"config\.locale\.nominal_weekend must be a list of integers"),
        ("locale", {"calendar_path": 3}, r"config\.locale\.calendar_path must be text"),
    ],
)
def test_yaml_scalar_fields_with_wrong_types_are_refused_before_construction(
    tmp_path: Path, section: str, group: dict[str, object], message: str
) -> None:
    """YAML 里类型写错的标量字段在 ``construct_config`` 就被拒（W16：不靠各领域配置的构造器兜）。"""

    mapping = valid_mapping(tmp_path)
    mapping[section] = group
    with pytest.raises(ConfigError, match=message):
        HabitusConfig.from_mapping(mapping)


def test_yaml_number_tuples_with_wrong_items_are_refused(tmp_path: Path) -> None:
    mapping = valid_mapping(tmp_path)
    observability = mapping["observability"]
    assert isinstance(observability, dict)
    observability["metrics"]["duration_buckets_seconds"] = [0.1, "0.5"]
    with pytest.raises(ConfigError, match=r"duration_buckets_seconds must be a list of numbers"):
        HabitusConfig.from_mapping(mapping)


def test_field_type_check_skips_multi_member_unions_and_reports_text_tuples() -> None:
    """``require_field_types``：多成员联合不查（交给领域规则），``tuple[str, ...]`` 的坏值报 ``ConfigError``。"""

    from dataclasses import dataclass

    from habitus.config.loader import require_field_types

    @dataclass(frozen=True)
    class Probe:
        names: tuple[str, ...] = ()
        either: str | tuple[str, ...] = ""
        optional: int | None = None

    require_field_types(Probe, {"names": ["a"], "either": "abc", "optional": None}, path="config.probe")
    require_field_types(Probe, {"either": ("a",)}, path="config.probe")
    for data, message in (
        ({"names": [1]}, r"config\.probe\.names must be a list of texts"),
        ({"names": "ab"}, r"config\.probe\.names must be a list of texts"),
        ({"optional": True}, r"config\.probe\.optional must be an integer"),
    ):
        with pytest.raises(ConfigError, match=message):
            require_field_types(Probe, data, path="config.probe")


@pytest.mark.parametrize(
    ("keys", "bad", "message"),
    [
        (("chat", "route", "timeout_seconds"), "300", r"config\.models\.chat\.route\.timeout_seconds must be a number"),
        (("chat", "route", "timeout_seconds"), True, r"config\.models\.chat\.route\.timeout_seconds must be a number"),
        (("chat", "route", "max_retries"), 1.5, r"config\.models\.chat\.route\.max_retries must be an integer"),
        (("chat", "route", "max_concurrent"), None, r"config\.models\.chat\.route\.max_concurrent must be an integer"),
        (("chat", "max_output_tokens"), True, r"config\.models\.chat\.max_output_tokens must be an integer"),
        (("chat", "reasoning"), "false", r"config\.models\.chat\.reasoning must be a boolean"),
        (("embedding", "dimension"), "1024", r"config\.models\.embedding\.dimension must be an integer"),
        (("embedding", "max_batch_size"), 1.5, r"config\.models\.embedding\.max_batch_size must be an integer"),
        (("rerank", "max_documents"), True, r"config\.models\.rerank\.max_documents must be an integer"),
    ],
)
def test_model_route_yaml_with_wrong_types_is_refused_before_construction(
    tmp_path: Path, keys: tuple[str, ...], bad: object, message: str
) -> None:
    """模型路由与能力配置里类型写错的字段在解码时就被拒（W16：不靠 ``ProviderConfig`` 等构造器兜）。"""

    mapping = valid_mapping(tmp_path)
    group = mapping["models"]
    for key in keys[:-1]:
        assert isinstance(group, dict)
        group = group[key]
    assert isinstance(group, dict)
    group[keys[-1]] = bad
    with pytest.raises(ConfigError, match=message):
        HabitusConfig.from_mapping(mapping)
