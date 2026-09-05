"""行为观测清洗层的统一容量边界。

这些是纯运维参数：它们只回答"一次交付最多接受多少"，不回答"这条观测有没有价值"。
清洗层不做任何语义判断，因此这里没有、也不应该出现置信度阈值之类的过滤开关——
低置信度观测照常收下，是否采信由后续融合层在语义里裁量。

TODO(BHV-FUSION-003·余项): 这些参数仍只有构造器默认值，未接入唯一的 ``Config/`` 边界。
- 现状更新：BHV-RUNTIME-001 已完成——``Config/`` 有了 behavior 段（主体/窗口/节奏五个标量），
  但**容量类** Config（本文件与 fusion/kinds/tree/document 各容量组）当批刻意未并入，原定
  "与接入主链同批"的时机已失约。
- 影响大小：中，不变。"可调运维参数必须经由 ``Config/`` 进入主链"仍是明文约束，现状仍是
  暂时偏离而非豁免。
- 改造方案：不变（容量组并入 ``Config.behavior``、示例 YAML 完整列出、Store 的 ``config``
  参数改必填）。
- 新时机：与 BHV-LIFECYCLE-001 的存储生命周期改造同批——保留期、容量上限、时间分区读取
  本来就要一起进配置，分两批做会改两次同一张表。
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
