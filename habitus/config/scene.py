"""语义关联层的运行配置。

本模块只有纯标量、不 import ``scene``（与 ``config/behavior.py`` 同一纪律）。

旧语义树（按候选累积的关联）那一组参数——提示词字符上限、重试、每轮调用数、每批目标数、前因/前提
行数预算——随它整块删掉（2026-09-26）。新语义树按功能分支重建，每支的旋钮在那一支落地时再进配置。

``relations``：关系检验的统计门槛（裁定 26）。每一项都可以不写，不写就是 ``scene.relations.config.RelationThresholds``
的默认值（组合根只把写了的换算过去，默认值不在这里抄第二份）。和时间有关的尺子（槽宽、合并窗口）不在这一组：
它们只认 ``config.temporal``（裁定 59），语义树不另配一份。
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any

from habitus.config.loader import construct_config, require_override_types


@dataclass(frozen=True)
class SceneRelationsConfig:
    """关系检验的统计门槛覆盖值；``None`` = 用默认值。"""

    fdr: float | None = None
    forward_alpha: float | None = None
    forward_power: float | None = None
    forward_max_antecedents: int | None = None
    min_absolute_lift: float | None = None
    min_relative_lift: float | None = None
    heterogeneity_alpha: float | None = None
    interval_level: float | None = None
    bootstrap_rounds: int | None = None
    exact_strata_limit: int | None = None
    first_seen_min_days: int | None = None
    first_seen_min_blocks: int | None = None
    condition_ask_min_antecedents: int | None = None
    max_blind_share: float | None = None
    max_combined_conditions: int | None = None

    def __post_init__(self) -> None:
        # 这里只查类型；值域只在 scene/relations/config.py，组合根构造 RelationThresholds 时拒（带配置键）。
        require_override_types(self, path="scene.relations")

    def overrides(self) -> dict[str, Any]:
        """写了的字段 → ``RelationThresholds`` 的同名字段。"""

        return {item.name: getattr(self, item.name) for item in fields(self) if getattr(self, item.name) is not None}


@dataclass(frozen=True)
class SceneConfig:
    """语义关联层是否启用、关系检验的统计门槛，以及换代与模型队列的运维数（语义树没开就没有它们，所以归这里）。"""

    enabled: bool = False
    relations: SceneRelationsConfig = field(default_factory=SceneRelationsConfig)
    # 常态按峰分份时，峰的两边各放宽几槽的容差（待定档）。原来借预测树的 pool_half_width，预测树删除后归语义树自己（裁定 59）。
    baseline_slack_slots: int = 3
    # 换代什么时候做不是配置：预测 worker 每个槽边界看一眼，连续定稿到昨天、预测拍水位过了余量就做（实时化阶段 1、2）。
    # 完成标记、每代报告、trace 文件留几天（一年）：结算认"那天没有这一代"只往回看到过期待决的预期还没裁的那一段。
    generation_kept_days: int = 366
    # 三把锁（预测根租约、语义树写锁、模型队列租约）共用的租期：多久过期、多久续一次（两个进程挂同一个 storage root 时
    # 同一天只换代一次、账本只有一个写者、同一时刻只有一个进程在做队列的活）。
    generation_lease_ttl_seconds: float = 120.0
    generation_heartbeat_seconds: float = 30.0
    # 连续几槽看到"定稿日不连续"（较早一天还没定稿、较晚一天已定稿）就在健康面报出卡住的那一天；正常只持续一轮归约。
    generation_stuck_ticks: int = 3
    # 必须成的那一步（改树、读序列、机会表、日子预期、完成标记）没成之后，同一天按退避一直重试：首次等多久、最多等多久
    # （逐次加倍，用户 11-10）。退避只在进程里，重启后从头。
    generation_retry_base_seconds: float = 300.0
    generation_retry_max_seconds: float = 3_600.0
    # 模型队列（原来换代里调模型的活都在它那里做，一项一项）：没活时隔多久从盘上看一眼有没有新欠账。
    model_queue_poll_seconds: float = 60.0
    # 模型整个调不通时队列整体退避：首次等多久、最多等多久（逐次加倍；单项失败也按它各自等；到点接着做，不另探模型）。退避只在进程里。
    model_queue_backoff_base_seconds: float = 60.0
    model_queue_backoff_max_seconds: float = 3_600.0
    # 连着这么多项（不同的项）都是模型失败、中间没有任何一项做成，队列就整体退避（不分失败原因）。
    model_queue_unreachable_after_failures: int = 3
    # 停机时等模型队列当前这一项多久，超时取消（取消掉的那一项下次从盘上重新认出来）。
    model_queue_shutdown_timeout_seconds: float = 30.0
    # 映射按槽：一条记录最后所见所在的槽过后再留几槽就可以映射（不等那一天封口；"过了几槽"按预测拍水位算）。留小了不会出错——
    # 之后才落树的记录让当天时间线变了，受影响的那一条按现有规则重判，只是多花调用。
    mapping_margin_slots: int = 2
    # 欠账到这么多项、或最老一项等了这么久，健康面报降级（non-critical；清完自动消掉）。
    model_queue_backlog_warning_items: int = 50
    model_queue_backlog_warning_seconds: float = 21_600.0

    def __post_init__(self) -> None:
        if not 0 <= self.baseline_slack_slots <= 96:
            raise ValueError("scene.baseline_slack_slots must be an integer between 0 and 96")
        if not 1 <= self.generation_kept_days <= 3_660:
            raise ValueError("scene.generation_kept_days must be an integer between 1 and 3660")
        if not 5.0 <= float(self.generation_lease_ttl_seconds) <= 3_600.0:
            raise ValueError("scene.generation_lease_ttl_seconds must be between 5 and 3600")
        if not 1.0 <= float(self.generation_heartbeat_seconds) < float(self.generation_lease_ttl_seconds):
            raise ValueError("scene.generation_heartbeat_seconds must be at least 1 and below the lease ttl")
        if not 1 <= self.generation_stuck_ticks <= 10_000:
            raise ValueError("scene.generation_stuck_ticks must be an integer between 1 and 10000")
        base, cap = float(self.generation_retry_base_seconds), float(self.generation_retry_max_seconds)
        if not 1.0 <= base <= cap <= 86_400.0:
            raise ValueError(
                "scene.generation_retry_base_seconds must be at least 1 and at most generation_retry_max_seconds (≤ 86400)"
            )

        if not 1.0 <= float(self.model_queue_poll_seconds) <= 86_400.0:
            raise ValueError("scene.model_queue_poll_seconds must be between 1 and 86400")
        base, cap = float(self.model_queue_backoff_base_seconds), float(self.model_queue_backoff_max_seconds)
        if not 1.0 <= base <= cap <= 86_400.0:
            raise ValueError(
                "scene.model_queue_backoff_base_seconds must be at least 1 and at most model_queue_backoff_max_seconds (≤ 86400)"
            )
        margin = self.mapping_margin_slots
        if isinstance(margin, bool) or not isinstance(margin, int) or not 0 <= margin <= 96:
            raise ValueError("scene.mapping_margin_slots must be an integer between 0 and 96")
        if not 1 <= self.model_queue_unreachable_after_failures <= 1_000:
            raise ValueError("scene.model_queue_unreachable_after_failures must be an integer between 1 and 1000")
        if not 1.0 <= float(self.model_queue_shutdown_timeout_seconds) <= 3_600.0:
            raise ValueError("scene.model_queue_shutdown_timeout_seconds must be between 1 and 3600")
        if not 1 <= self.model_queue_backlog_warning_items <= 1_000_000:
            raise ValueError("scene.model_queue_backlog_warning_items must be an integer between 1 and 1000000")
        if not 60.0 <= float(self.model_queue_backlog_warning_seconds) <= 2_592_000.0:
            raise ValueError("scene.model_queue_backlog_warning_seconds must be between 60 and 2592000")

    @classmethod
    def from_mapping(cls, value: Any) -> SceneConfig:
        data = dict(value) if isinstance(value, dict) else value
        if isinstance(data, dict) and "relations" in data:
            data["relations"] = construct_config(SceneRelationsConfig, data["relations"], "config.scene.relations")
        return construct_config(cls, data, "config.scene")


__all__ = ["SceneConfig", "SceneRelationsConfig"]
