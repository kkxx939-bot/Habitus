"""行为管线（观测 → 融合 → 归约 → 行为树）的运行配置。

本模块只有纯标量、不 import ``behavior``：上下文窗口的**默认值**唯一出处仍是
``behavior/fusion/config.py``（此处留 ``None``，由 Runtime 组合根解析）——组合根用同一份
数值喂融合与归约两处，这正是 BHV-FUSION-003 点名要关死的分叉之门。
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from habitus.config.loader import construct_config

# (字段, 下界, 上界)：与 behavior/kinds/config.py 的校验一致。
_KINDS_INT_BOUNDS = (
    ("kinds_batch_size", 1, 100),
    ("kinds_validation_rounds", 0, 10),
    ("kinds_transient_retries", 0, 20),
    ("kinds_max_encoded_bytes", 1, 256 * 1024 * 1024),
    ("kinds_max_examples", 1, 10),
    ("kinds_recurrence_min_count", 1, 1_000),
    ("kinds_recurrence_min_days", 1, 365),
    ("kinds_max_pending", 1, 1_000_000),
    ("kinds_nightly_max_new", 1, 10_000),
    ("kinds_recall_per_entry", 0, 50),
    ("kinds_nightly_hour", 0, 23),
    ("kinds_revision_weekday", 0, 6),
    ("kinds_revision_hour", 0, 23),
    ("kinds_merge_evidence_runs", 1, 20),
    ("kinds_cooldown_days", 0, 365),
    ("kinds_max_changes_per_revision", 1, 100),
    ("kinds_lane_candidate_cap", 1, 10_000),
    ("kinds_anchors_per_class", 1, 100),
    ("kinds_revision_samples_per_class", 1, 100),
    ("kinds_revision_window_days", 7, 730),
    ("kinds_grouping_min_members", 2, 10_000),
    ("kinds_grouping_batch_size", 2, 500),
    ("kinds_checkpoint_every", 1, 100_000),
    ("kinds_prompt_max_steps", 0, 200),
    ("kinds_sample_max_steps", 0, 200),
)
_CAPACITY_FIELDS = (
    "observation_max_files",
    "fusion_max_receipt_files",
    "fusion_max_judgement_files",
    "tree_max_children_per_directory",
)
_KINDS_FLOAT_BOUNDS = (
    ("kinds_transient_retry_delay_seconds", 0.0, 600.0),
    ("kinds_max_anchor_pull", 0.0, 1.0),
)


@dataclass(frozen=True)
class BehaviorConfig:
    """行为侧的产品事实与运行节奏。

    ``primary_subject`` 是这套系统跟踪谁的**产品事实**（两条 lane 写的判断都记在他名下）；
    留空表示行为侧未启用，组合根将完全跳过行为组件的组装与 Worker 启动。
    """

    primary_subject: str = ""
    # 归约的封口视界（秒）：一条判断的证据齐备之后过了这么久才封口落树；留空取 behavior/fusion/config.py 的默认。
    context_lookback_seconds: float | None = None
    # 归约 sweep 节奏（用户裁定 5 分钟）；纯运维数，不影响正确性。
    # Worker 优雅停止的等待上界也在这里（运维参数不藏 Python 构造器）。
    reduction_sweep_interval_seconds: float = 300.0
    # 归约 sweep 的租约时长；留空取 behavior/reduction 的默认（600s）。周尺度回填（万级小文件写入把
    # fseventsd 顶到 240% CPU、单篇 publish 被文件系统拖到秒级）需要更长的租约——运维参数，不是常量。
    reduction_sweep_lock_ttl_seconds: int | None = None
    worker_shutdown_timeout_seconds: float = 30.0
    # 融合覆盖索引 / 回执 / 消费账本的过期窗口（天）：同一个单位在这么多天之内再送来认得出来。
    # 它是：重做去重的判据、覆盖记录/回执/账本的过期期限。覆盖记录在其凭据仍在存储里时**不**过期
    # （否则会被当成没判过重做），所以它不限制下游停机时长、不丢数据。
    # 树上的数据不受它影响（撞车消歧的占用集合以树上既有地址为最终真相，不依赖账本窗口）。
    coverage_window_days: int = 7
    # 基础词表（裁定 6：门槛做成配置）：全部可调字段，留空取 ``behavior/kinds/config.py`` 的默认值
    # （默认值只在领域模块）。字段名 = ``kinds_`` + BehaviorKindConfig 的同名字段，组合根按前缀派生注入。
    # 会话 lane 的保护上限（方案第五节）：留空取 ``behavior/fusion/lanes/session/config.py`` 的默认值。
    # 字段名 = ``session_`` + SessionLaneConfig 的同名字段，组合根按前缀派生注入。
    session_max_turn_chars: int | None = None
    session_max_instruction_chars: int | None = None
    session_max_steps: int | None = None
    kinds_batch_size: int | None = None
    kinds_validation_rounds: int | None = None
    kinds_transient_retries: int | None = None
    kinds_transient_retry_delay_seconds: float | None = None
    kinds_max_encoded_bytes: int | None = None
    kinds_max_examples: int | None = None
    kinds_recurrence_min_count: int | None = None
    kinds_recurrence_min_days: int | None = None
    kinds_max_pending: int | None = None
    kinds_nightly_max_new: int | None = None
    kinds_recall_per_entry: int | None = None
    kinds_nightly_hour: int | None = None
    kinds_revision_weekday: int | None = None
    kinds_revision_hour: int | None = None
    kinds_max_anchor_pull: float | None = None
    kinds_merge_evidence_runs: int | None = None
    kinds_cooldown_days: int | None = None
    kinds_max_changes_per_revision: int | None = None
    kinds_lane_candidate_cap: int | None = None
    kinds_anchors_per_class: int | None = None
    kinds_revision_samples_per_class: int | None = None
    kinds_revision_window_days: int | None = None
    kinds_grouping_min_members: int | None = None
    kinds_grouping_batch_size: int | None = None
    kinds_checkpoint_every: int | None = None
    kinds_prompt_max_steps: int | None = None
    kinds_sample_max_steps: int | None = None
    # 存储的容量闸（26 号清单 8.2：生命周期相关、原来写死在各存储的默认值里；只挪不改值，不写就是领域模块的默认值）。
    # 字段名 = 前缀 + 那份配置的同名字段：``observation_`` → BehaviorObservationConfig、``fusion_`` → BehaviorFusionConfig、
    # ``tree_`` → BehaviorTreeConfig。
    observation_max_files: int | None = None
    fusion_max_receipt_files: int | None = None
    fusion_max_judgement_files: int | None = None
    tree_max_children_per_directory: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.primary_subject, str):
            raise TypeError("behavior.primary_subject must be a string")
        if self.context_lookback_seconds is not None and (
            isinstance(self.context_lookback_seconds, bool)
            or not isinstance(self.context_lookback_seconds, int | float)
            or not 60.0 <= float(self.context_lookback_seconds) <= 86_400.0
        ):
            raise ValueError("behavior.context_lookback_seconds must be between 60 and 86400")
        if (
            isinstance(self.coverage_window_days, bool)
            or not isinstance(self.coverage_window_days, int)
            or not 1 <= self.coverage_window_days <= 365
        ):
            raise ValueError("behavior.coverage_window_days must be between 1 and 365")
        for name, upper in (
            ("session_max_turn_chars", 1_000_000),
            ("session_max_instruction_chars", 1_000_000),
            ("session_max_steps", 50),
        ):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= upper):
                raise ValueError(f"behavior.{name} must be an integer between 1 and {upper}")
        # 下界与 behavior/kinds/config.py 一致：非法值在配置层就拒，不让它拖到组装期以裸异常爆。
        for name, lower, upper in _KINDS_INT_BOUNDS:
            value = getattr(self, name)
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"behavior.{name} must be an integer between {lower} and {upper}")
        for name, low, high in _KINDS_FLOAT_BOUNDS:
            number = getattr(self, name)
            if number is not None and (
                isinstance(number, bool) or not isinstance(number, int | float) or not low <= float(number) <= high
            ):
                raise ValueError(f"behavior.{name} must be between {low:g} and {high:g}")
        for name in _CAPACITY_FIELDS:
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 10_000_000
            ):
                raise ValueError(f"behavior.{name} must be an integer between 1 and 10000000")
        if self.reduction_sweep_lock_ttl_seconds is not None and (
            isinstance(self.reduction_sweep_lock_ttl_seconds, bool)
            or not isinstance(self.reduction_sweep_lock_ttl_seconds, int)
            or not 60 <= self.reduction_sweep_lock_ttl_seconds <= 86_400
        ):
            raise ValueError("behavior.reduction_sweep_lock_ttl_seconds must be an integer between 60 and 86400")
        for name, value, minimum, maximum in (
            ("reduction_sweep_interval_seconds", self.reduction_sweep_interval_seconds, 1.0, 3_600.0),
            ("worker_shutdown_timeout_seconds", self.worker_shutdown_timeout_seconds, 1.0, 3_600.0),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float) or not minimum <= float(value) <= maximum:
                raise ValueError(f"behavior.{name} must be between {minimum:g} and {maximum:g}")

    @property
    def enabled(self) -> bool:
        return bool(self.primary_subject.strip())

    def session_overrides(self) -> dict[str, Any]:
        """非空的 ``session_*`` 字段 → ``SessionLaneConfig`` 的同名字段。"""

        return {
            field.name[len("session_") :]: getattr(self, field.name)
            for field in fields(self)
            if field.name.startswith("session_") and getattr(self, field.name) is not None
        }

    def overrides_for(self, prefix: str) -> dict[str, Any]:
        """非空的 ``<prefix>*`` 字段 → 那份领域配置的同名字段（存储的容量闸用，按前缀派生，不双份维护）。"""

        return {
            field.name[len(prefix) :]: getattr(self, field.name)
            for field in fields(self)
            if field.name.startswith(prefix) and getattr(self, field.name) is not None
        }

    def kinds_overrides(self) -> dict[str, Any]:
        """非空的 ``kinds_*`` 字段 → ``BehaviorKindConfig`` 的同名字段（按前缀派生，不双份维护）。"""

        return {
            field.name[len("kinds_") :]: getattr(self, field.name)
            for field in fields(self)
            if field.name.startswith("kinds_") and getattr(self, field.name) is not None
        }

    @classmethod
    def from_mapping(cls, value: object) -> BehaviorConfig:
        return construct_config(cls, value, "config.behavior")


__all__ = ["BehaviorConfig"]
