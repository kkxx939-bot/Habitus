"""行为观测清洗层的统一容量边界。

这些是纯运维参数：它们只回答"一次交付最多接受多少"，不回答"这条观测有没有价值"。
清洗层不做任何语义判断，因此这里没有、也不应该出现置信度阈值之类的过滤开关——
低置信度观测照常收下，是否采信由后续融合层在语义里裁量。

容量上限里和生命周期相关的 ``max_files`` 从 ``Config.behavior.observation_max_files`` 进来（26 号清单 8.2，只挪不改值）；
内容类上限（一次交付多少条、每条多长）仍用这里的默认值，接真实观测源时随上游契约一起定。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BehaviorObservationConfig:
    max_observations_per_batch: int = 512
    max_semantics_chars: int = 2_000
    max_participants: int = 32
    max_evidence_refs: int = 64
    max_reference_chars: int = 512
    max_identifier_chars: int = 256
    max_file_bytes: int = 2_000_000
    max_files: int = 100_000
    # 上游用自己的时钟标注 available_at，Habitus 用自己的时钟标注 recorded_at；两者之间允许
    # 的最大反向偏差。超过它说明交付携带了"尚未可用"的语义，属于结构性矛盾而非时钟抖动。
    max_clock_skew_seconds: int = 300

    def __post_init__(self) -> None:
        for name, value in {
            "max_observations_per_batch": self.max_observations_per_batch,
            "max_semantics_chars": self.max_semantics_chars,
            "max_participants": self.max_participants,
            "max_evidence_refs": self.max_evidence_refs,
            "max_reference_chars": self.max_reference_chars,
            "max_identifier_chars": self.max_identifier_chars,
            "max_file_bytes": self.max_file_bytes,
            "max_files": self.max_files,
            "max_clock_skew_seconds": self.max_clock_skew_seconds,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


__all__ = ["BehaviorObservationConfig"]
