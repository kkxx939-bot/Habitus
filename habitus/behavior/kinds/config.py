"""词表的全部门槛与运维参数（裁定 6：门槛做成配置，不写死）。

数值由 ``Config.behavior`` 的 ``kinds_*`` 字段经组合根注入（``runtime/behavior.py``）；这里的默认值是代码内
唯一出处，YAML 留空即取之。下面标"待定档"的数值是起步值，要等真实数据（待定池的复现分布、锚点被拉走的实测比例）重定。
"""

from __future__ import annotations

from dataclasses import dataclass

_INTEGER_BOUNDS: tuple[tuple[str, int, int], ...] = (
    ("batch_size", 1, 100),
    ("validation_rounds", 0, 10),
    ("transient_retries", 0, 20),
    ("max_encoded_bytes", 1, 256 * 1024 * 1024),
    ("max_examples", 1, 10),
    ("recurrence_min_count", 1, 1_000),
    ("recurrence_min_days", 1, 365),
    ("max_pending", 1, 1_000_000),
    ("nightly_hour", 0, 23),
    ("revision_weekday", 0, 6),
    ("revision_hour", 0, 23),
    ("merge_evidence_runs", 1, 20),
    ("cooldown_days", 0, 365),
    ("max_changes_per_revision", 1, 100),
    ("lane_candidate_cap", 1, 10_000),
    ("anchors_per_class", 1, 100),
    ("revision_samples_per_class", 1, 100),
    ("revision_window_days", 7, 730),
    ("grouping_min_members", 2, 10_000),
    ("grouping_batch_size", 2, 500),
    ("checkpoint_every", 1, 100_000),
    ("prompt_max_steps", 0, 200),
    ("sample_max_steps", 0, 200),
)


@dataclass(frozen=True)
class BehaviorKindConfig:
    """词表的门槛与运维边界（标"待定档"的是起步值，等真实数据重定）。

    - 归类调用：一次判几条、结构不对重问几轮、瞬态错误重试几次。
    - 每晚新增：每天几点（本地时间）；待定池里一组至少几条、跨几天才建类（待定档）；待定池上限是保护闸。
    - 定期拆改（裁定 17）：每周几、几点；放行 = 证据规则（拆分两边各至少 ``recurrence_min_count`` 条、跨
      ``recurrence_min_days`` 天，提醒句不同；合并要在 ``merge_evidence_runs`` 个不同周期里都被提出）+ 锚点自检
      （每个没动过的类取 ``anchors_per_class`` 个站得稳的锚点，新清单下被拉走的比例不超过 ``max_anchor_pull``）；
      改过的类冷却几天；一次最多改几处；每类给模型看几条原话（只给合并 / 改判据的提案用）。
    - 拆分（2026-10-12 起由分组 + 复现定，`18` 第四节）：只看最近 ``revision_window_days`` 天的成员——按月做的事要攒够
      ``recurrence_min_count`` 次才切得出来，窗口不能短于它的几个周期；成员不足 ``grouping_min_members`` 的类不分组
      （实际门槛 ``grouping_floor`` 至少是两边各 ``recurrence_min_count`` 条）；一次分组最多给模型 ``grouping_batch_size`` 条，多了分批、同名的组合并。
    - 候选：同 lane 在用类超过上限时报警（保护闸，见 ``classify``）。
    - 提示索引：给当日实况用的"原话 → 编号"回看多少天。
    - 续约间隔：扫树、重打树时每处理多少条续一次 sweep 租约。
    - 给模型看的步骤：归类与待定池每条最多列几步；拆改时每条抽样原话最多列几步（抽样多，列得短）。
    """

    batch_size: int = 10
    validation_rounds: int = 2
    transient_retries: int = 5
    transient_retry_delay_seconds: float = 5.0
    max_encoded_bytes: int = 16 * 1024 * 1024
    max_examples: int = 3
    recurrence_min_count: int = 3
    recurrence_min_days: int = 3
    max_pending: int = 5_000
    nightly_hour: int = 2
    revision_weekday: int = 2
    revision_hour: int = 3
    max_anchor_pull: float = 0.02
    merge_evidence_runs: int = 2
    cooldown_days: int = 14
    max_changes_per_revision: int = 3
    lane_candidate_cap: int = 80
    anchors_per_class: int = 8
    revision_samples_per_class: int = 8
    revision_window_days: int = 120
    grouping_min_members: int = 6
    grouping_batch_size: int = 60
    checkpoint_every: int = 200
    prompt_max_steps: int = 12
    sample_max_steps: int = 4

    def __post_init__(self) -> None:
        for name, lower, upper in _INTEGER_BOUNDS:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer between {lower} and {upper}")
        _require_number("transient_retry_delay_seconds", self.transient_retry_delay_seconds, 0.0, 600.0)
        _require_number("max_anchor_pull", self.max_anchor_pull, 0.0, 1.0)
        if self.recurrence_min_days > self.recurrence_min_count:
            raise ValueError("recurrence_min_days cannot exceed recurrence_min_count")

    @property
    def grouping_floor(self) -> int:
        """分组至少要装得下两边各 ``recurrence_min_count`` 条，否则怎么分都切不出两个够格的组——取两者里大的那个。"""

        return max(self.grouping_min_members, 2 * self.recurrence_min_count)


def _require_number(name: str, value: object, lower: float, upper: float) -> None:
    if isinstance(value, bool) or not isinstance(value, int | float) or not lower <= float(value) <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")


__all__ = ["BehaviorKindConfig"]
