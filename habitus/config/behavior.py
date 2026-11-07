"""行为管线（观测 → 融合 → 归约 → 行为树）的运行配置。

本模块只有纯标量、不 import ``behavior``：上下文窗口的**默认值**唯一出处仍是
``behavior/fusion/config.py``（此处留 ``None``，由 Runtime 组合根解析）——组合根用同一份
数值喂融合与归约两处，这正是 BHV-FUSION-003 点名要关死的分叉之门。
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from habitus.config.loader import construct_config, require_override_types

# 按前缀派生到领域配置的覆盖项（字段名 = 前缀 + 领域配置的同名字段）。这里只查类型；值域只在领域配置，组合根构造时拒
# （带配置键）——同一条边界只定义一处。
OVERRIDE_PREFIXES = (
    "session_",
    "physical_",
    "kinds_",
    "observation_",
    "fusion_",
    "tree_",
    "document_",
    "write_",
    "semantic_",
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
    # 每晚重试的堆每一支最多留几项（裁定 67 第 8 条，保护闸）：满了新来的只落「没读懂」、不进堆，计数留信号。
    # 量级按一天的轮数给：模型断一整天也大多装得下，又不至于让一晚的重试把刚恢复的额度一下烧光。
    retry_backlog_max: int = 500
    # 基础词表（裁定 6：门槛做成配置）：全部可调字段，留空取 ``behavior/kinds/config.py`` 的默认值
    # （默认值只在领域模块）。字段名 = ``kinds_`` + BehaviorKindConfig 的同名字段，组合根按前缀派生注入。
    # 会话 lane 的保护上限（方案第五节）：留空取 ``behavior/fusion/lanes/session/config.py`` 的默认值。
    # 字段名 = ``session_`` + SessionLaneConfig 的同名字段，组合根按前缀派生注入。
    session_max_turn_chars: int | None = None
    session_max_instruction_chars: int | None = None
    session_tool_argument_chars: int | None = None
    session_tool_error_chars: int | None = None
    session_kept_tool_lines: int | None = None
    session_kept_reply_lines: int | None = None
    session_max_steps: int | None = None
    session_max_summary_chars: int | None = None
    session_max_step_chars: int | None = None
    # 物理 lane 的保护上限：留空取 ``behavior/fusion/lanes/physical/config.py`` 的默认值。
    # 字段名 = ``physical_`` + PhysicalLaneConfig 的同名字段。
    physical_max_segments: int | None = None
    physical_max_coverage_spans: int | None = None
    physical_max_facts: int | None = None
    physical_max_fact_chars: int | None = None
    physical_max_summary_chars: int | None = None
    physical_max_device_chars: int | None = None
    physical_max_segment_id_chars: int | None = None
    physical_max_future_seconds: int | None = None
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
    kinds_revision_weekday: int | None = None
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
    # 行为树文档的物理边界（``document_`` → BehaviorDocumentConfig）、文档写锁的租约（``write_`` → BehaviorWriteConfig）、
    # 每天的语义概览（``semantic_`` → BehaviorSemanticConfig）：都是保护闸，留空取领域默认值。
    document_max_markdown_body_chars: int | None = None
    document_max_encoded_bytes: int | None = None
    write_lock_ttl_seconds: int | None = None
    semantic_max_direct_entries: int | None = None
    semantic_max_prompt_chars: int | None = None
    semantic_max_narrative_chars: int | None = None
    semantic_max_entry_summary_chars: int | None = None
    semantic_max_overview_chars: int | None = None
    semantic_max_abstract_chars: int | None = None

    def __post_init__(self) -> None:
        if self.context_lookback_seconds is not None and not 60.0 <= float(self.context_lookback_seconds) <= 86_400.0:
            raise ValueError("behavior.context_lookback_seconds must be between 60 and 86400")
        if not 1 <= self.coverage_window_days <= 365:
            raise ValueError("behavior.coverage_window_days must be between 1 and 365")
        if not 1 <= self.retry_backlog_max <= 100_000:
            raise ValueError("behavior.retry_backlog_max must be an integer between 1 and 100000")
        for prefix in OVERRIDE_PREFIXES:
            require_override_types(self, path="behavior", prefix=prefix)
        if (
            self.reduction_sweep_lock_ttl_seconds is not None
            and not 60 <= self.reduction_sweep_lock_ttl_seconds <= 86_400
        ):
            raise ValueError("behavior.reduction_sweep_lock_ttl_seconds must be an integer between 60 and 86400")
        for name, value, minimum, maximum in (
            ("reduction_sweep_interval_seconds", self.reduction_sweep_interval_seconds, 1.0, 3_600.0),
            ("worker_shutdown_timeout_seconds", self.worker_shutdown_timeout_seconds, 1.0, 3_600.0),
        ):
            if not minimum <= float(value) <= maximum:
                raise ValueError(f"behavior.{name} must be between {minimum:g} and {maximum:g}")

    @property
    def enabled(self) -> bool:
        return bool(self.primary_subject.strip())

    def overrides_for(self, prefix: str) -> dict[str, Any]:
        """非空的 ``<prefix>*`` 字段 → 那份领域配置的同名字段（``OVERRIDE_PREFIXES``，按前缀派生，不双份维护）。"""

        if prefix not in OVERRIDE_PREFIXES:
            raise ValueError(f"unknown behavior override prefix {prefix!r}")
        return {
            field.name[len(prefix) :]: getattr(self, field.name)
            for field in fields(self)
            if field.name.startswith(prefix) and getattr(self, field.name) is not None
        }

    @classmethod
    def from_mapping(cls, value: object) -> BehaviorConfig:
        return construct_config(cls, value, "config.behavior")


__all__ = ["OVERRIDE_PREFIXES", "BehaviorConfig"]
