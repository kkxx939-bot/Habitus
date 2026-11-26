"""预测层的运行配置：开关、可接受的误提醒率、保护闸与账本遗忘的半衰期。

时窗与回看跨度从数据算，不在这里配；时间尺子在 ``config.temporal``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from habitus.config.loader import construct_config, group_fields

_INT_BOUNDS = (
    ("daily_reminder_cap", 1, 100),
    ("open_reminders_judged", 1, 50),
    ("outlet_max_attempts", 1, 20),
    ("outlet_claim_limit", 1, 20),
    ("unwatched_tolerance_slots", 1, 96),
    ("generation_margin_slots", 0, 96),
)
_FLOAT_BOUNDS = (
    ("false_alarm_rate", 0.01, 0.99),
    ("tick_interval_seconds", 60.0, 86_400.0),
    ("worker_shutdown_timeout_seconds", 1.0, 3_600.0),
    ("half_life_opportunities", 1.0, 200.0),
    ("outlet_liveness_hours", 1.0, 720.0),
    ("outlet_lease_minutes", 1.0, 1_440.0),
    ("outlet_match_similarity", 0.1, 1.0),
)
_STORE_BOUNDS = (
    ("record_max_bytes", 1_024, 64 * 1024 * 1024),
    ("directory_max_entries", 1_000, 10_000_000),
    ("unreadable_max_paths", 1, 1_000_000),
    ("outlet_max_entries", 100, 1_000_000),
    ("unit_moment_max_records", 1_000, 100_000_000),
    ("unit_moment_max_record_bytes", 1_024, 16 * 1024 * 1024),
    ("unit_history_max_bytes", 1_024 * 1_024, 1_024 * 1_024 * 1_024),
    ("opportunity_table_max_bytes", 1_024 * 1_024, 1_024 * 1_024 * 1_024),
    ("nightly_max_entries", 100, 1_000_000),
    ("read_cache_max_records", 1, 10_000_000),
)


@dataclass(frozen=True)
class ForesightStoreConfig:
    """预测层落盘的容量闸。只做保护；撞上了说明该检查压实、退出与释放是否在推进。"""

    # 账本与出口的一份记录最多多少字节（写入就查，超限的不落盘）。
    record_max_bytes: int = 1_000_000
    # 账本一个目录最多列多少条（日目录、按账、按行为单元的目录）。
    directory_max_entries: int = 100_000
    # 两次取走之间最多记多少条坏文件路径（再多的只计不记）。
    unreadable_max_paths: int = 1_000
    # 出口一个目录（取走记录、问过的提问、宿主活性）最多列多少条。
    outlet_max_entries: int = 10_000
    # 单元历史的每刻记录一条 lane 最多多少份、一份最多多少字节。
    unit_moment_max_records: int = 200_000
    unit_moment_max_record_bytes: int = 64 * 1024
    # 一代的单元历史快照、机会表最多多少字节。
    unit_history_max_bytes: int = 32 * 1024 * 1024
    opportunity_table_max_bytes: int = 16 * 1024 * 1024
    # 每代派生物一条 lane 的目录最多列多少条。
    nightly_max_entries: int = 10_000
    # 账本在内存里最多留多少条解码好的记录（只影响快慢，不影响读到什么；一条约 2.3 KB）。
    read_cache_max_records: int = 20_000

    def __post_init__(self) -> None:
        for name, lower, upper in _STORE_BOUNDS:
            value = getattr(self, name)
            if not lower <= value <= upper:
                raise ValueError(f"foresight.store.{name} must be an integer between {lower} and {upper}")

    @classmethod
    def from_mapping(cls, value: Any) -> ForesightStoreConfig:
        return construct_config(cls, value, "config.foresight.store")


@dataclass(frozen=True)
class ForesightConfig:
    """预测层是否启用、可接受的误提醒率、几道保护闸。"""

    enabled: bool = False
    # 影子 → 提醒的线 θ = 1 − 可接受的误提醒率，所有账同一条线（裁定 58）。0.33 → θ = 0.67。
    false_alarm_rate: float = 0.33
    # 影子预期也用还没成立的候选关系，让账早些开始攒；开口仍只用成立的关系（裁定 56）。
    shadow_uses_unestablished: bool = True
    # ── 保护闸 ──
    # 每天最多主动提醒几次；应当很少触发，常触发说明升级条件该重看。
    daily_reminder_cap: int = 10
    # 每段会话进融合时最多带几条开着的提醒给模型判回应，按截止由近到远；没带进去的只按做没做结算（裁定 61）。
    open_reminders_judged: int = 5
    # 行为单元的先验分量，占有效样本上限 1 / (1 − 2^(−1/H)) 的比例（裁定 56）：没见过的组合贴着更粗的那一层。
    # 写成比例而不是固定数，是因为按机会数遗忘后有效样本有上限（H = 8 时约 12），固定数会让样本充足的组合也过不了线。
    unit_prior_share: float = 1 / 3
    # 预测 worker 一拍的间隔与停机时限。一拍一槽、对齐槽边界，必须等于 ``temporal.slot_minutes`` × 60（配置根交叉校验）。
    tick_interval_seconds: float = 900.0
    worker_shutdown_timeout_seconds: float = 30.0
    # 两拍之间隔了超过这么多槽，那一段就记成会话一支的「未观测」空白（关机、合盖、进程没跑）。
    unwatched_tolerance_slots: int = 2
    # 换代要等预测拍水位过了 max(今天零点, 最后一段跨过零点的「未观测」的结尾) 再这么多槽，给归约留时间把昨天的空白
    # 落树，结算与读序列才看得到它（裁定 99）。
    generation_margin_slots: int = 2
    # 账本遗忘的半衰期，按机会数而不按天（裁定 57）：一个月一次的事不会被压到只剩两三次有效样本。
    # 行为单元、temporal 的时段读数（最近 M = ⌈H · log₂(1/α)⌉ 次）和账的久别判定（空白至少 H 个典型间隔）用同一个 H。
    half_life_opportunities: float = 8.0
    # ── 提醒出口（裁定 64）──
    # 最近这么多小时内有宿主来取过提醒，才算有宿主在；有才问模型写话、记「待取」，没有就记「不可达」、不调模型。
    outlet_liveness_hours: float = 48.0
    # 一条请求被一个会话取走后多久内别的会话取不到（不超过这条的最晚时限）。
    outlet_lease_minutes: float = 30.0
    # 一条请求最多被放回（取走了没问出口、租约过期）几次，到了就不再给。
    outlet_max_attempts: int = 3
    # 一次取最多给几条（插件来取时带的 limit 不得超过它）。
    outlet_claim_limit: int = 3
    # 认提问：题目与那句话去空白后的相似度不低于它才算配上（题目含类名、或一取一问的直接配上，不看这个数）。
    outlet_match_similarity: float = 0.6
    # ── 没见过的日型怎么建预期（裁定 59）──
    # "skip"：日历关系只在见过他做的那几种日型上建（只在工作日的周三做过，国庆的周三不建）；
    # "official"：没见过的日型按外接放假数据的官方含义归组（调休上班算上班、节假日算休息）。只管建预期，不管关系检验。
    unseen_day_types: str = "skip"
    # ── 落盘的容量闸 ──
    store: ForesightStoreConfig = field(default_factory=ForesightStoreConfig)

    def __post_init__(self) -> None:
        if self.unseen_day_types not in ("skip", "official"):
            raise ValueError("foresight.unseen_day_types must be 'skip' or 'official'")
        for name, lower, upper in _INT_BOUNDS:
            value = getattr(self, name)
            if not lower <= value <= upper:
                raise ValueError(f"foresight.{name} must be an integer between {lower} and {upper}")
        share = self.unit_prior_share
        if not 0.0 < float(share) < 1.0:
            raise ValueError("foresight.unit_prior_share must be a number strictly between 0 and 1")
        for name, low, high in _FLOAT_BOUNDS:
            number = getattr(self, name)
            if not low <= float(number) <= high:
                raise ValueError(f"foresight.{name} must be a number between {low:g} and {high:g}")

    @classmethod
    def from_mapping(cls, value: Any) -> ForesightConfig:
        data = dict(group_fields(cls, value, "config.foresight"))
        data["store"] = ForesightStoreConfig.from_mapping(data.get("store", {}))
        return construct_config(cls, data, "config.foresight")


__all__ = ["ForesightConfig", "ForesightStoreConfig"]
